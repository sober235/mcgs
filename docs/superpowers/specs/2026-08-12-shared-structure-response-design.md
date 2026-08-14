# Design — Is the shared *dictionary* enough, or does the response need shared *structure*?

- **Date:** 2026-08-12
- **Status:** Draft (awaiting user review)
- **Scope decision (user, this session):** restricted to the **Gaussian representation model itself**.
  CAIPI (ky-kz lattice, g-factor) is acquisition-side; J-LORAKS / SPIRiT (Hankel low-rank, coil
  self-consistency) is a different reconstruction mechanism. Neither touches the Gaussian
  parameterisation, so both are **out of scope** — they are orthogonal to "can the Gaussian
  representation be made better".
- **No algorithm change ships from this spec.** It defines one measurement.

## 1. The question, and why it is the only one left

Capacity is closed. G0's representation-ceiling gate has the Gaussian passing the 2x line on both
contrasts, so the representation is not the bottleneck; forced densification to 900k / 1.1M is
*monotonically worse* (`.claude/memory.md` §8b), so the model is data-limited rather than
capacity-limited; and §14b's `lambda=0` control removes the artefact from both the Gaussian
representation and the sampling mask and pins it on the regulariser. "More Gaussians, bigger
capacity, a different kernel" has no room left.

What has never been tested is the **structure of the sharing**. `JointGaussianModel3D` shares
`mu`, `Sigma` (positions / scales / rotations) while `density_plain` and `density_contrast` are two
fully independent complex tables. That is a shared *dictionary with independent coefficients*, not an
identifiable shared/private decomposition: nothing in the model constrains which information ought to
be shared and which ought to stay private.

ARC does add a latent and a decoder, but the arithmetic shows it is not a bottleneck. From
`paired_arc_gs/modules/response_decoder.py`:

```
z_A : (N, latent_dim = 12)                 per-Gaussian free latent
e_m : (n_contrasts = 2, condition_dim = 4) learned per-contrast condition code
trunk : Linear(12+4 -> 24) -> Tanh -> Linear(24 -> 24) -> Tanh      # "24/2" = hidden 24, 2 layers
heads : Linear(24 -> 2) per contrast                                # the 2 is (real, imag)
```

Each Gaussian carries **12 free reals** and the two contrasts together emit **4 reals**. 12 -> 4 is an
expansion, not a bottleneck. This also fixes the vocabulary: **`decoder-24/2` means hidden 24 /
2 layers, not "outputs 2 contrasts", and `K` is `model.anatomy.latent_dim`.**

## 2. The four arms

A ladder in what is shared, all with identical geometry and identical data:

| arm | free reals per Gaussian | what is shared across contrasts | role |
|---|---|---|---|
| `independent` | 4 (two complex tables) | geometry only | baseline = the shipped MC-GS parameterisation |
| `K12` | 12 -> decoder | geometry **+ the decoder's weights** | current ARC default; **no bottleneck** |
| `K3` | 3 -> decoder | as above, plus a real bottleneck | |
| `K2` | 2 -> decoder | as above, tighter | |

The ladder answers two separable questions: does a **shared response map** help at all
(`independent` vs `K12`), and does a **real bottleneck** help (`K12` vs `K3` vs `K2`)?

⚠️ **The first question has already been asked, and the honest reading of its answer is "no
difference".** `docs/validation_layer1.md` ran exactly this comparison under nearly these conditions —
geometry frozen, 200k Gaussians, held-out k-space NRMSE as the primary metric, both the T1/FLAIR and
the CE pair:

| pair | contrast | independent | decoder (K12) | delta |
|---|---|---|---|---|
| T1/FLAIR | T1 | 0.5485 | 0.5483 | −0.0002 |
| T1/FLAIR | FLAIR | 0.8364 | 0.8332 | −0.0032 |
| CE | pre | 0.3548 | 0.3527 | −0.0021 |
| CE | post | 0.3361 | 0.3336 | −0.0026 |

It was reported as "4/4 contrasts: decoder <= independent", which is a pass against its own stated
requirement ("this coupling costs **no** representational power"). But the deltas are 0.0002-0.0032 and
the run was **n=1 with no seeds**, so they carry no scale: this project has just measured a
run-to-run spread of 0.031 on a normalised index while the difference being claimed was 0.025
(`.claude/memory.md` §16b). Re-running this rung is therefore worth doing only for the seed floor it
adds — it is not new information.

**All of the new information lives in the bottleneck arms.** That is what this spec is for.

⚠️ **A degeneracy that would have wasted half the runs, and why it does not apply.** If `K` indexed the
rank of the `(Gaussian x contrast)` response matrix, then with only two contrasts `K=2` would be full
rank and therefore *identical to `independent`* — two of four arms would be the same thing, and the
resulting "no difference" would read as "the constraint does nothing". `K` is instead the latent width
feeding a shared non-linear decoder, so the arms are genuinely distinct. Verified against the code,
not inferred from prose.

## 3. Scoring: held-out acquired-k, because in-sample scoring cannot answer this

`independent` has the most freedom, so it can only tie or win *in sample*. The value of a constraint,
if any, is in extrapolation. The measurement must therefore be on acquired-k points the model never
saw.

This is already implemented and already understands the trap. `PairedARCTrainer.holdout_kspace_nrmse`
(`paired_arc_gs/train/trainer.py:520`):

```
NRMSE_ho = || M_test (F(S x) - y) || / || M_test y ||          per contrast
m_seen   = (mfull - mtest).clamp(0, 1)                          the render sees everything EXCEPT M_test
```

with the docstring's own note that `M_test` is "never fit, never audited, so this **cannot be gamed by
hard data-consistency replacement**". That property is not optional here. Under the scoring path
`k_dc = m*y + (1-m)*k_model`, any metric computed on points the DC projection is allowed to overwrite
degenerates into comparing the ground truth with itself as sampling density rises — which is exactly
how a "fully sampled" control in this project scored 156 dB while containing no information
(`.claude/memory.md` §16a). Held-out points must be excluded from **both** the loss and the DC
projection; `holdout_kspace_nrmse` already does this, and it is GT-free.

**Primary metric:** post-training `holdout_nrmse`, reported per contrast, never averaged across them.
`.claude/memory.md` §16's rule stands: a gain on one contrast must not be allowed to hide a loss on
the other.

## 4. Noise floor: three seeds per arm, and the arms are compared against within-arm spread

Four arms x 3 seeds = **12 training runs**. A between-arm difference counts only if it exceeds the
within-arm spread across seeds; otherwise the reading is "cannot distinguish".

This is not boilerplate. In the immediately preceding work, two nominally identical runs differed by
**0.031** on a normalised index while the difference driving a conclusion was **0.025** — the
"effect" sat inside run-to-run variation and the conclusion had to be withdrawn
(`.claude/memory.md` §16b). Without a within-arm floor, a four-way ranking is uninterpretable. The
prior Layer-1 comparison in §2 is exactly the shape this prevents: four deltas in the third decimal
place, reported as a verdict, from a single run each.

**Convergence is a gate, not an assumption.** The arms do not have equal parameter counts, so they do
not converge at equal rates: an arm with 2 free reals per Gaussian can still be climbing when one with
4 has plateaued. Comparing them at a fixed iteration count would then measure **optimisation speed**
rather than the value of the constraint — and it would do so with a bias that always favours the
freest arm, which is the very hypothesis under test. The harness smoke run showed this risk concretely:
at 300 iterations `K2` read 0.6696 (T1) against `independent`'s 0.3454, a gap far more consistent with
underfitting than with a representational limit.
**Gate:** every run records the held-out NRMSE trajectory, and an arm counts only if that trajectory has
plateaued before the stopping point. An arm still improving at the stop is reported as **not converged,
not comparable** — never as a loss.

"Plateaued" is defined against a quantity this experiment already measures rather than a fresh constant:
**an arm has converged when the improvement over its final 20% of iterations is smaller than that arm's
own between-seed spread.**

⚠️ **That improvement must be computed ROBUSTLY, and the first draft of this spec did not say so — the
omission immediately produced a false reading.** Differencing the window's two endpoints reports a
single-point excursion as "improvement": in wave 1, `K12` spikes to 0.2008 at iter 1600 and returns to
0.1627 by 2000, so the endpoint difference reads 0.038 and the arm looks badly unconverged, while its
full trajectory is flat from about iter 1000 (0.1642 -> 0.1627). These runs do have excursions — `K2`
oscillates by +/-0.006 between adjacent checkpoints — so this is not a corner case.
**Compute the figure as the median of the window's first half minus the median of its second half, per
contrast**, and report the raw endpoint difference alongside it so the excursions stay visible: they are
real behaviour worth recording, just not evidence of non-convergence. The reasoning is that further improvement below the noise floor cannot be
distinguished from noise anyway, so it cannot change any comparison the experiment is entitled to make.
This deliberately avoids inventing a convergence threshold — the failure mode that has cost this project
twelve pre-registered bounds (`.claude/memory.md` §16c). If an arm fails this gate, the remedy is more
iterations for **every** arm, never for the failing arm alone: extending one arm's budget to rescue it
would reintroduce exactly the optimisation-speed confound the gate exists to exclude.

## 5. Controls and gates

- **Geometry frozen.** `mu`, `Sigma`, rotations are loaded from the shipped working point,
  `results_arc/tvs4k_e040/tvs4k_e040/checkpoints/best.pth`, via `model.warm_start_path`, and held
  fixed with `training.lr.geometry = 0.0`
  (`paired_arc_gs/model/paired_arc_model.py:110` reads `lr_cfg.get("geometry", 3e-4)`). The arms must
  then differ *only* in the response parameterisation.
  The trainer's warm-start path already has exactly the semantics this needs
  (`paired_arc_gs/train/trainer.py:96-106`): it copies `positions`, `exp(scales)` and rotations from an
  MC-GS checkpoint and **does not transfer densities** — "ARC produces them from the response decoder"
  — so the geometry is inherited while every arm starts its response from scratch.
  **Gate:** `max|delta|` on each of positions, scales and rotations must be exactly 0 — not merely
  small. **Fallback if that checkpoint will not load:** the ARC default warm start
  (`.../results_espirit_winner_v2/mcgs_espirit_winner_v2_tv06hi22/checkpoints/best.pth`, the iso-TV
  winner) is known to load. Any fixed geometry serves the comparison, but all four arms must use the
  same one, and which was used must be recorded.
- **Aligned mask.** Use `mask3d_vdpois_af12_transverse_ALIGNED.mat`. The existing ablation skeleton
  `paired_arc_gs/train/_ablation_layer1.py` points at the **misregistered** mask, whose ACS sits ~22 px
  off the data's k-space DC and which captures only 27% (T1) of k-space energy (§8). Inheriting it
  would reproduce a known bug inside a new experiment.
- **Arms are actually distinct.** Print the resolved `model.anatomy.latent_dim` and
  `model.decoder.enable` per arm and record the per-Gaussian free-parameter count. Identical held-out
  scores between `independent` and `K2` mean the config never reached the model, not that the
  constraint is inert.
- **Leakage.** Inject a synthetic lesion into the **FLAIR measurement only** and measure whether it
  appears in T1. Specified concretely so the reading is not a matter of taste:
  - a sphere of radius **6 voxels**, centred in white matter at least 15 voxels from any tissue
    boundary (so the check is not confounded by the boundary error the diagnosis already
    characterised), added to the FLAIR image **before** the forward operator so it enters FLAIR's
    measured k-space and nothing else;
  - amplitude **+30% of the local FLAIR tissue value**, large enough to be unambiguous at AF=12 and
    small enough to stay inside the intensity range the model already represents;
  - each arm is run twice, with and without the lesion, at a fixed seed; the reading is the RMS of
    `T1_with − T1_without` inside the lesion sphere, normalised by the same T1 peak used everywhere
    else.
  **Gate:** that reading must not exceed the within-arm seed spread from §4. Exceeding it means the arm
  moved FLAIR-only information into T1. This matters most for the tight bottlenecks — an
  over-constrained shared latent has no way to represent a FLAIR-only feature except by pushing it into
  T1, so `K2` is where leakage is most likely and where the gate earns its cost.

## 5b. A guardrail is deliberately relaxed, with its justification on the record

`paired_arc_gs/train/config_schema.py:187` rejects `anatomy.latent_dim` outside "recommended 8-16
(hard bounds 4-32, §6.2)". **`K3` and `K2` both violate that lower bound**, so this spec relaxes it to
`K >= 2` — for this experiment's configs only, leaving the default `latent_dim = 12` and the validator's
shipped bounds untouched. The user approved this after the following was established, and it is
recorded here rather than in a session log because deviating from a bound someone else wrote is exactly
the kind of thing that should not rest on memory:

1. **The lower bound has no recorded justification.** `git log -S` traces it to `a01ecf1`, "Layer 0:
   paired_arc_gs scaffold" — it arrived with the scaffolding, not from an experiment, and no ablation on
   `latent_dim` exists anywhere in the repository.
2. **§6.2's actual argument is an UPPER bound.** Where `docs/validation_layer1.md` invokes §6.2 it is to
   argue the decoder is *small enough*: "a 516-parameter decoder (for 200k Gaussians — far too small to
   memorize either image, §6.2)". The concern being encoded is over-capacity, not under-capacity.
3. **The lower bound removes precisely the only new information.** With two contrasts emitting 4 reals
   in total, a genuine bottleneck requires `K < 4`; `K = 4` is break-even and `K >= 8` is expansion. A
   lower bound of 4 therefore permits every arm except the ones that test the hypothesis, reducing the
   experiment to a rerun of the already-answered first rung (§2).

Note also that `validate_config()` is not called by the trainer itself — the smoke run proceeded with
`latent_dim = 2` and the config demonstrably reached the model. The relaxation is therefore about being
explicit rather than about unblocking: any harness that *does* wire the validator in (as
`paired_arc_gs/train/_smoke_layer1.py` does, via `assert`) must carry this exemption, or the `K2` arm
will be blocked silently.

## 5c. The harness has been validated before committing to the runs

A smoke run (`scripts/smoke_fourarm_harness.py`, report
`.superpowers/sdd/2026-08-12-fourarm-harness-smoke.md`) established the three properties without which
the experiment measures nothing. 200k Gaussians, 300 iters, the **ALIGNED** mask:

- **The held-out metric works.** T1 1.0009 -> 0.3376 and FLAIR 1.0002 -> 0.4586, with the DC loss
  falling 0.7295 -> 0.0525 over the same run: a null model scores ~1.0 as documented, and this is real
  fitting rather than noise.
- **The geometry freeze is exact.** With `training.lr.geometry = 0.0`, `max|delta|` on positions,
  scales and rotations is **0.0** — not merely small — while the metrics moved substantially.
  (Caveat recorded: `s2_response_fit`'s `opt_groups` already excludes the `geometry` group, so two
  freeze mechanisms are stacked. This validates the configuration the real runs will use; it does not
  isolate whether `lr.geometry = 0` alone would hold under a stage that reopens geometry.)
- **The arms are genuinely distinct.** `independent` read T1 0.3454 / FLAIR 0.4925 against `K2`'s
  0.6696 / 0.5613, and the resolved `latent_dim` / `decoder.enable` were printed per arm to confirm the
  config reaches the model. Identical scores here would have meant the twelve planned runs were a
  four-way comparison of one thing with itself.

## 6. What each outcome licenses

```
K3 or K2 beats independent on held-out NRMSE, on BOTH contrasts, by more than the seed spread,
   with no leakage                      -> shared structure carries real information.
                                           Proceed to an identifiable shared/private factorisation.

K12 beats independent but K3/K2 do not  -> the shared decoder helps; the bottleneck does not.
                                           The gain is the shared map, not the compression.

nothing beats independent               -> the shared dictionary is already sufficient at this
                                           acquisition. The sharing-structure line closes, and the
                                           remaining levers are outside the representation.

any arm wins on one contrast and loses on the other, or the leakage gate trips
                                        -> report as such; do NOT average the contrasts.
```

A null result here is a real result: it would close the last open direction *inside* the Gaussian
representation, which is worth knowing before anything larger is built on it.

## 7. Out of scope, recorded so it is not silently re-added

CAIPI and J-LORAKS/SPIRiT (see the scope decision above). The three remaining model-vNext items from
the prompting audit — bounded per-scan rigid + low-frequency phase inside the forward operator,
local cross-fitted shared/private/absent routing, and wiring `confidence/pi` and `lambda_route` into
the render and the loss — are **deferred**, not rejected: they are only worth building if §6 returns
the first outcome. Note also that `model.deformation.enable` must stay `False` for any ARC run
(`.claude/memory.md` §13, user decision).

## 8. Deliverables and cost

One driver script, its JSON records, and a verdict table carrying, per arm: per-contrast held-out
NRMSE, the within-arm seed spread, the convergence-gate reading (§4), the geometry-freeze gate reading
(§5), the leakage reading (§5), and the resolved `latent_dim` / `decoder.enable` that produced it.

**Run budget: 16 training runs.** 4 arms x 3 seeds = 12 for the main comparison and the noise floor,
plus 4 lesion runs (one per arm, at a fixed seed; the matching no-lesion run is reused from the 12).
At the smoke run's scale this is roughly 18 minutes each, so about an hour wall-clock across the five
currently idle GPUs.

No change to `paired_gs/`, `arc/`, `losses/` or `paired_arc_gs/` — every arm is expressible in existing
config keys, and the one deviation from a shipped default (§5b's `latent_dim` bound) is confined to this
experiment's configs.
