# Where does the MC-GS margin come from? Coupling attribution — design (2026-09-20)

Branch `coupling-attribution`, worktree `mcgs_hparam_wt`, cut from `hparam-sensitivity` 69e7daa.
Status of this document: **draft, not yet approved by the user.**

## 1. The question this answers

A reviewer asks: *how do you know the MC-GS margin comes from joint modelling and not from the fact
that you tuned the joint arm?* Today the line cannot answer it, because the whole margin was
manufactured by the tuning ladder:

| jiangyizhou, head-only PSNR, 4000 it | T1 | FLAIR | source |
|---|---|---|---|
| MC-GS − GS-single at the phase-1 recipe (P6) | +0.037 | **−0.003** | `T1FLAIR_AF16_v3_jiangyizhou/README.md:195` |
| MC-GS − GS-single at the phase-3 recipe | **+0.148** | **+0.042** | same line |
| the same margin at AF 12, P13 → P14 | +0.066 → +0.150 | **−0.038 → +0.036** | `T1FLAIR_AF12_v3_jiangyizhou/README.md:643` |
| seed floor (repeat at the same recipe) | 0.001 | 0.000 | `…AF16…/README.md:195` |

The margin is 40 to 150 times the seed floor, so it is real. But it did not exist before the ladder
added two λ, and on FLAIR it changed sign. "We tuned until the joint arm won" is a fair reading of
that table, and nothing currently in the topic rules it out.

## 2. What is actually coupled — verified, not assumed

Read out of `paired_gs/trainers/joint_trainer.py` on this branch and confirmed against the training
logs of the two registered runs. MC-GS has exactly **two** cross-contrast channels:

1. **Shared geometry (structural).** `JointGaussianModel3D` shares positions, scales and rotations
   across the two channels and keeps densities per channel (`paired_gs/gaussian/joint_gaussian_model.py:1-6`).
   There is no λ on it. It cannot be turned off without turning MC-GS into two independent clouds.
2. **R_edge, the symmetric common-edge cross-gradient prior**, `_edge_align_loss`
   (`joint_trainer.py:550-575`), entered at `λ_edge = 0.001` (`joint_trainer.py:959-961`). This is a
   tunable λ, and it is the one the phase-3 ladder added.

Everything else in the recipe is **per channel** and applies identically to the single arm:

| term | where | coupled? |
|---|---|---|
| R_geo edge-preserving TV | `joint_trainer.py:938-941`, `edge_preserving_tv(vol_p, κ_p) + edge_preserving_tv(vol_c, κ_c)`, κ from each channel's **own** ZF | no — self-guided per channel |
| R_ptv phase-only TV | `joint_trainer.py:973-976`, `phase_tv(vol_p) + phase_tv(vol_c)` | no |
| R_wav wavelet L1 | per-channel λ (T1 0.12 / FLAIR 0.10) | no |
| R_dtv plain-guided directional TV | `joint_trainer.py:950-953` | **off**: every registered log prints `lambda_dtv=0.0` |
| R_vtv vectorial TV | `joint_trainer.py:576-586` | **off**: `lambda_vtv` default 0, never set |

So the attribution question has a clean, cheap answer: **set `--lambda_edge 0` and the only tunable
coupling is gone, while every per-channel knob keeps its tuned value.** What survives of the margin
is what shared geometry alone buys.

### 2.1 GS-single is already a matched control, and λ_edge is not a no-op in it

`configs/t1flair_v3_af16_jyz_singleT1.yaml` is the joint config with **T1 in both channel slots**
(same `data_path`, same `csm_path`, `known_zero` dropped). GS-single therefore runs the same code,
the same recipe, the same number of iterations and the same capacity as MC-GS; the only difference
between the arms is what sits in the second channel. That is a stronger baseline than the topic
README claims for it, and it should be said plainly in the paper.

It also means R_edge is **not** inert in the single arm. Measured, iteration 150 of each run:

```
hp_n36 (joint)                R_edge = 6.6012e-01
p3_r3_singleT1_full4k (single) R_edge = 4.0620e-01
```

The two channels of the single arm carry independent densities, so their gradients do not align by
construction and the prior still pulls on them. Consequence for the design: **λ_edge = 0 changes both
arms, so both must be re-run at λ_edge = 0.** A joint-only ablation would compare against a baseline
that still pays a prior the joint arm no longer pays.

## 3. Runs

Prefix `ca_`. Base configs as in the AF16 line; recipe block byte-identical to `run_hp_mcgs.sh`
except that `--lambda_edge` becomes a positional knob. 4000 iterations, 3.6M points, seed 42,
`SAVE_VOLUMES=1`, jiangyizhou, house AF16 mask (md5 `d08179b8e74f35846455983dbe299294`).

### Stage 1 — the decisive ablation (5 runs, ~5.5 GPU-h, 5 idle cards ≈ 70 min wall clock)

| exp | arm | config | λ_edge | role |
|---|---|---|---|---|
| (`hp_n36`, exists) | joint | `t1flair_v3_af16_jyz.yaml` | 0.001 | J1 anchor, 35.706 / 34.928 |
| `ca_s_t1_p3` | single T1 | `…_singleT1.yaml` | 0.001 | S1, re-run **in this worktree** |
| `ca_s_fl_p3` | single FLAIR | `…_singleFLAIR.yaml` | 0.001 | S1, re-run in this worktree |
| `ca_j_edge0` | joint | `t1flair_v3_af16_jyz.yaml` | **0** | J0 |
| `ca_s_t1_edge0` | single T1 | `…_singleT1.yaml` | **0** | S0 |
| `ca_s_fl_edge0` | single FLAIR | `…_singleFLAIR.yaml` | **0** | S0 |

The two S1 runs are re-run rather than read from `mcgs_af16_wt/results_arc/p3_r3_single*_full4k`
so that all four cells of the 2×2 come out of one code tree. `hp_n36` already reproduced the
registered joint value to 0.000 dB across worktrees, so the joint anchor is not re-run; the S1
re-runs double as the check that the same is true of the single arm (gate: within 0.02 dB of
35.558 / 34.886).

**Reported quantity.** Per contrast, head-only PSNR:

```
M1 = J1 - S1        the published margin           (expect +0.148 / +0.042)
M0 = J0 - S0        shared geometry alone
M1 - M0             what the common-edge prior buys
```

### Stage 2 — dose-response on λ_edge (5 runs)

Joint at λ_edge ∈ {0.0003, 0.003, 0.01} and singles at λ_edge = 0.003. With Stage 1 this gives the
margin at λ_edge ∈ {0, 0.001, 0.003} and the joint absolute curve at five values. A coupling term
that is doing real work should move the margin monotonically over at least part of that range; a
margin that is flat in λ_edge but positive is shared geometry, and a margin that peaks exactly at the
tuned 0.001 and falls off on both sides is the signature the reviewer is worried about.

### Stage 3 — the placebo (1 run + data prep)

Stage 1 and 2 separate "shared geometry" from "coupling prior", but both are still joint modelling.
What they do not test is whether the gain needs **shared anatomy** at all, or merely a second
well-posed channel. Placebo: keep jiangyizhou's own FLAIR k-space, coils, SNR and elliptical support,
but flip the FLAIR volume so that it is a real brain that does not register to T1.

Built in image space so the flip is self-consistent (arrays coil-last):

```
ksp_flip = fft3c( flip(ifft3c(ksp_FLAIR), axis=AXIS) )
csm_flip = flip(csm_FLAIR, axis=AXIS)
```

which makes the SENSE-combined ground truth exactly `flip(GT_FLAIR, axis=AXIS)`.

**Which axis — corrected 2026-09-22, against the measured shutter.** This section first said axis 1
(ky) and claimed "a one-sample re-centring roll keeps the elliptical support identical, so
`known_zero` stays valid". **That claim is false on this data and the sentence is withdrawn.** Two
measurements, both made before any Stage-3 reconstruction was scored:

- FLAIR's acquired support on the PE plane is 34139 of 45056 lines (frac 0.757702) and is modelled to
  **1 line in 45056** by `ellipse(centre 128, 88; half-axes 128, 88) ∧ ky ≥ 20`. So the shutter is a
  DC-centred ellipse **with a one-sided partial-Fourier cut** in ky, and is *not* symmetric about the
  DC sample in ky.
- An image-space flip always sends the k-space support to its reflection about the **DC sample**, not
  to `np.flip` of it: the two differ by a one-voxel image translation, which multiplies k-space by a
  unit-modulus linear phase and therefore cannot move a zero. Under that reflection **2454 ky lines
  move** (2806 under a plain flip). No placement of a roll rescues it — a roll in image space is a
  translation and cannot move a zero; a roll in k-space would move DC off the array centre, which is
  the ACS/DC misregistration this line has already paid ~1 dB for once.

The axis actually used is the **readout** — array axis 0 once the coil axis is moved last, i.e. the
axis this config records as `readout_axis: 0`. FLAIR's support is complete along it (34139 PE points
at every one of the 256 x indices), so it is trivially symmetric about that axis' DC sample and the
flip returns it with **0 differing samples**; the trainer's `[physics:contrast]` line for the placebo
is then identical to the real-FLAIR anchor's **in every known-zero field** (`cuts`, `ellipse`,
`frac_z_used`, `zero_lines`, `explained`, `nonzero_in_z`, `outliers`). Exactly one of its fifteen
fields differs — `noise_level_used`, 5.1730e-04 relative, from the float32 FFT round trip — and it
carries no known-zero content. The `x = const` plane is axial, so the readout
is the superior–inferior direction and the flip is a **head–foot flip** — the most adversarial of the
three axes, not the least: NCC(|T1|, flipped |FLAIR|) is 0.349 for readout, 0.427 for ky, 0.678 for
kz, against 0.801 unflipped (head-masked: −0.096 readout vs 0.058 ky).

kz is in fact the one PE axis whose support *is* DC-symmetric, so it would pass the support test —
but it is still rejected for the reason already given above: at NCC 0.678 it is nearly registered,
because it is the left–right stack direction of this sagittal acquisition and a head is near-symmetric
across it. A placebo that still registers would make the stage meaningless while looking fine.

**Controller ruling, 2026-09-22 — recorded before any Stage-3 score existed.**

- **`ca_j_placebo` (readout flip) is the primary arm.** The §4 Stage-3 row applies to it and to it
  alone.
- **`ca_j_placebo_ky` (ky flip) is secondary and is reported as context, not against the §4 rule.**
  Its support is demonstrably not preserved (2454 PE lines move; its `[physics:contrast]` line
  explains 199 of 229 zero lines against the anchor's 233 of 233, frac_z 0.2151 vs 0.2423), so it
  cannot satisfy a rule that assumes a valid `known_zero`.
- The primary arm was chosen on support preservation and NCC measured **before** any Stage-3
  reconstruction scored, and it is the more adversarial of the two.

Run `ca_j_placebo` (joint, λ_edge 0.001, channel 2 = flipped FLAIR) and read the **T1** channel only.
If T1 still beats `ca_s_t1_p3` by something close to +0.148 dB, the margin is not coming from shared
anatomy and the paper's mechanism sentence is wrong.

### Stage 4 — equal-budget search on the single arm (gated, ~13 GPU-h, not started without the user)

Stages 1–3 hold the recipe fixed for both arms. They do not answer the strict form of the question:
the recipe itself was *selected* by joint performance. The strict control is to give GS-single its
own search of the same size over the knobs that mean anything to it (λ_geo, wavelet λ, λ_ptv,
edge-TV w_floor / κ_q — λ_edge does not belong to a single-contrast method), then quote
MC-GS(best joint) − GS-single(best single). Six trials × 2 contrasts ≈ 13 GPU-h.

This stage is listed so the plan is honest about what Stages 1–3 do and do not cover. It is **not**
scheduled here; the user decides after reading Stage 1.

## 4. Pre-registered decision rule

Written before the runs. Indistinguishability band = 0.02 dB (this line's anchor tolerance, 20× the
measured seed floor).

| outcome | reading | what the paper must say |
|---|---|---|
| M0 ≥ 0.6 · M1 on both contrasts | shared geometry carries the margin | "the gain is structural: one geometry, two densities" — the current claim stands |
| 0.02 dB < M0 < 0.6 · M1 | both mechanisms contribute | report the split; name R_edge as part of the method, not as a tuning detail |
| M0 ≤ 0.02 dB on a contrast | on that contrast the margin **is** the common-edge prior | say so. It is still cross-contrast modelling, but the mechanism sentence changes and the shared-geometry claim must be dropped for that contrast |
| M0 < 0 on a contrast | shared geometry costs that contrast, the prior pays it back | report it; this is the AF 12 FLAIR story (−0.038 → +0.036) and must not be hidden |
| Stage 3, **primary arm `ca_j_placebo` only**: placebo T1 margin within 0.02 dB of M1 | the second channel's anatomy is irrelevant | the joint-modelling claim fails; the effect is regularisation / conditioning |

None of these outcomes is a failure of the study. The one failure mode is running it and then not
writing down the unflattering cell.

## 5. What this does not cover

- **One subject, one mask, one acceleration.** Stage 1 is jiangyizhou at AF 16 only. The AF 12 line
  is where the FLAIR sign flip happened, so a repeat of Stage 1 at AF 12 is the obvious extension and
  is not in this design.
- **The peds counter-example is already evidence and is not re-run.** `Peds_TSC_MTP_AF12_retro/README.md:398`:
  T1w + T2* are "the two least similar contrasts of the ten (r = 0.57)", MC-GS is not ahead there, and
  Joint SENSE moves the same way — a cross-family dose-response that tuning does not explain. It
  belongs in the rebuttal as-is.
- **Noise.** The frozen recipe already fails at 2σ₀ (`T1FLAIR_AF16_noise2x_v3_jiangyizhou`). Attribution
  is measured at 1σ₀ here.
- **Stage 3 cannot separate "wrong anatomy" from "twice the support to cover".** `init_mode=head` draws
  the shared cloud from the union of the two channels' ZF head supports, so both placebo arms grow it
  from `support_frac` 0.4264 to **0.4801** — the same 3.6 M points must now cover ~12.6 % more volume.
  This is intrinsic to any misregistered partner and is axis-independent (both arms measure 0.4801 /
  0.4802), not an artefact of how the flip was built. So a negative Stage-3 result reads as **"the
  margin needs the two volumes to register"**, which is broader than "needs the partner's anatomy",
  and the write-up must use the broader wording. For this model the two are arguably the same
  statement — one geometry shared between two volumes only helps if they occupy the same geometry —
  but the spec should not claim the narrower one.
- Lesion contrast: the T1+FLAIR subjects are healthy volunteers, as in the hparam study.

## 6. Fixed factors

- Code: this branch. Build cache `TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_hparam_wt.torch_ext`
  (already built for this code; reusing this worktree avoids a rebuild).
- Recipe, unchanged from `run_hp_mcgs.sh` except λ_edge: P6 cold start (`init_mode=head`,
  `init_scale_mode=3nn`, `edge_tv_kappa_source=zf`), edge-TV λ_geo 0.10 / κ_q 0.85 / w_floor 0.10 /
  ε 0.04, `kspace_high` 2.2, warm-up 500, wavelet db4 finest level (T1 0.12 / FLAIR 0.10),
  phase TV 0.03, 3.6M points, 4000 iterations, seed 42 with `deterministic=False`.
- Scoring: head-only (`|GT| > 0.05 · P99.9|GT|`) PSNR / SSIM / HFEN / NMSE, head NMSE recomputed from
  saved volumes by the summariser as in the hparam study.
- Outputs only under `mcgs_hparam_wt/results_arc/` (gitignored). Nothing is written into
  `mcgs_af16_wt` or into any existing `Results_20260908/` topic until the line closes.
