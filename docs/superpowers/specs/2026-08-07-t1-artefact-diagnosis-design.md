# Design — Adjudicating the "aliasing + patchiness" the user sees in the T1 recon

- **Date:** 2026-08-07
- **Status:** Draft (awaiting user review)
- **Trigger:** User, on `results_arc/LOOK_T1_axial.png`: the T1 reconstruction has "a lot of aliasing
  artefacts, plus the kind of patchy artefact TV produces — what is the cause?"
- **Terminal state (user decision, this session):** *establish the phenomenon first, then discuss cause.*
  The deliverable is a falsifiable diagnostic verdict. **No algorithm change is in scope.**
- **Control budget (user approved):** new reconstruction runs are allowed for controls A and B.
- **Base:** MC-GS edge-TV + Charbonnier, memory `§0`. Candidate checkpoints are the existing
  `results_arc/*_volumes.npz`.

## 1. Why this needs a design rather than an answer

The two artefact classes the user names stand in **opposite** evidential positions in the project record,
so answering from memory would be wrong in two different directions at once.

**The patchiness already has a verified root cause** (memory `§14` / `§14b`, 2026-08-01), established with
controls rather than argued. Two layers:

```
edge weight  w = clamp( 1/(1+(|grad u|/kappa)^2), w_floor, 1 )      w_floor = 0.1
  -> flat regions are penalised up to 10x more than edges           <- layer 1, by design
L1's marginal cost stays at 1 however small g gets, never eases off  <- layer 2, the actual defect
```

Anchored by controls: zero-fill (a pure k-space truncation, i.e. a pure low-pass) gives a **flat** gradient
ratio curve at 0.97, so a rising curve cannot be blur; `lambda=0` flips the whole curve **above 1**
(0.39–0.83 becomes 1.67–2.00), which refutes the Gaussian representation and the sampling mask as sources
and pins it on the regulariser. Charbonnier removes only layer 2, and `selectivity` falls monotonically
2.09 -> 1.46 -> **1.18 (eps=0.04)** -> 1.13 -> 1.06 across the eps series.

**The aliasing claim contradicts the record.** Memory `§8b` states the residual is *incoherent grain /
detail-loss, NOT coherent wrap-around ghosts*, background clean. But `§8b` was written **before** the mask
registration fix (`§8`), when the pipeline captured only 27% (T1) of k-space energy, so it cannot simply be
cited as current. The user, asked where the aliasing sits, selected **all four** offered locations —
including ghosts in the air, which is the one that would overturn `§8b` outright.

**And the figure's provenance is unknown**, which decides which numbers even apply. No `*_volumes.npz` was
written after 2026-08-01 17:05, while `LOOK_*.png` is dated 2026-08-03 16:42 — so the figure was re-drawn
from an existing volume. It is also not from `scripts/fig_recon_view.py` (that script has no zero-fill
column and slices at z=128, the figure has one and slices at z=160), i.e. an ad-hoc script that was never
written to disk. If the figure came from `rc_lam000` (the `§14b` no-TV control) then patchiness *should not
exist* and speckle *should* be worse than shipped — a different question entirely.

Three facts that shape the design, all established while exploring context for this spec:

1. **This mask cannot draw classical ghosts.** `§10`: Bridson Poisson-disc (blue-noise) variable density,
   NN spacing 2.54 vs 1.85 for random. Discrete wrap-around replicas are a *uniform*-undersampling
   phenomenon; a blue-noise PSF has diffuse sidelobes. So "snow-like speckle" is not a misreading — it is
   what incoherent aliasing is supposed to look like.
2. **Nothing suppresses air on this pipeline.** `§11`: `ecalib -c 0.0` (no crop) gives `SoS == 1` over the
   whole FOV *including air*, and the conjugate-SENSE combine therefore weights air fully. Critically the
   **GT is built by the same combine** (`dataset.py:313–315`), so the GT's air already carries full-amplitude
   noise. Background grain in the recon may simply be present in the GT too.
3. **The pure-aliasing reference is already on disk.** `zf* = conj-SENSE(M . y)` is mask-only, prior-free, so
   `zf - gt` **is** what this mask's aliasing looks like. Comparing `rec - gt` against it is free and
   ablative. This converts much of control B into a zero-cost measurement.

## 2. Pre-registered propositions

The user's report is translated into four falsifiable claims, each with its verdict condition fixed
**before** any measurement is taken.

| # | Proposition | True iff |
|---|---|---|
| **P1** | WM speckle = residual incoherent undersampling aliasing | `rec-gt` autocorrelation inside WM falls **inside the white-noise surrogate's band** (M2), i.e. no long-range structure, AND its structure is same-family as `zf-gt`, only smaller in amplitude |
| **P2** | Outer-rim / skull-scalp band = aliasing | the band's error pattern **moves** when the mask instance changes (control B); if it does not move it is not aliasing |
| **P3** | Ventricle / midline streaks = displaced structure | M4's `corr(rec-gt, d(gt)/dx)` is non-zero on at least one axis beyond what the half-pixel-shift positive control calibrates as detectable — displacement makes the residual sign-paired across a structure, which is exactly a correlation with the GT gradient |
| **P4** | The background contains recon-introduced ghosts | `E_air(rec)/E_air(gt) >> 1` AND air-region error is same-family as `zf`'s |

P4's criterion is fixed as an inequality because it is the only **binary** one of the four and it alone
decides whether `§8b` must be overturned:

```
E_air(rec)/E_air(gt) ~= 1  -> the background grain is in the GT too
                              (SoS==1 over air, zero air suppression, GT combined the same way)
                           -> not recon-introduced; §8b stands
                     >> 1  -> energy really is leaking into air -> §8b overturned, go find the path
```

## 3. Adjudication rule

**Every metric must first pass calibration on a control whose mechanism is known, before it is allowed to
support a verdict.**

This is not ceremony. It is the lesson of `§14`'s own correction: the `S` index ordered the runs exactly as
the eye does — which made it look trustworthy — while actually measuring **global smoothing rather than
staircasing**. Re-measured with a self-normalised threshold the shape-only excess is **+14%**, not the
+758% that `S = 8.58` implies, and there is **no atom of exactly-zero gradients** at all. Ranking correctly
is not the same as measuring the named thing.

Calibration stakes:

```
zf         (mask only, no prior)  -> aliasing-class metrics must read FULL SCALE on it; if not, metric is blind
rc_lam000  (no TV)                -> patchiness metrics must read "no patchiness" on it
                                     (§14b measured: selectivity 1.20, whole ratio curve above 1)
gt                                -> every metric normalised to 1.00
```

A metric that fails its stake is **discarded on the spot** and may not be used to support a verdict. A
proposition returning "insufficient evidence" is a legitimate outcome and is preferred over a verdict
propped up by an uncalibrated number.

## 4. Measurement matrix

`M0` gates everything else.

| Metric | Decides | Method | Calibration stake | Cost |
|---|---|---|---|---|
| **M0** provenance | all | re-render T1 z=160 from each candidate npz, compare pixelwise against the PNG panel | a unique match | zero |
| **M1** mask PSF | P1, P2 | `psf = ifftn(mask)` over the PE plane; is out-of-mainlobe energy **discrete** or **diffuse**? | a uniform-undersampling mask as positive control — it must yield discrete replicas | zero |
| **M2** error same-family | P1, P2 | `NCC(rec-gt, zf-gt)` plus the 2D autocorrelation decay length of each | `zf` full-scale against itself; `rc_lam000` should sit closest to pure aliasing; **white-noise surrogate** (residual permuted within the WM mask) supplies the delta reference | zero |
| **M3** air energy ratio | **P4** | `E_air(rec)/E_air(gt)`, `E_air(zf)/E_air(gt)` | GT normalised to 1.00 | zero |
| **M4** displacement | P3 | `corr(rec-gt, d(gt)/dx)` on all three axes; displacement makes error proportional to the GT gradient | half-pixel shifted GT as positive control | zero |
| **M5** selectivity | patchiness | `§14b`'s calibrated `ratio@q0.999 / ratio@q0.10`, white matter only | zf = 0.97 flat; `lambda=0` -> 1.20, curve above 1 | zero |
| **M6** mask dependence | **P2** | `corr(E_rec^maskA, E_rec^maskB)`: low = follows the mask = aliasing | — | **control B** |
| **M7** representation-intrinsic patchiness | **patchiness attribution** | patchiness surviving at full sampling ⇒ unrelated to the mask | — | **control A** |

M0–M5 cover P1, P3, P4 and the patchiness magnitude at zero GPU cost. Only P2 and the final attribution of
the patchiness need to wait. That is the whole reason for running the two tracks in parallel.

**No metric here carries a hard-coded threshold.** Each verdict boundary is read off that metric's own
calibration stake — the surrogate band for "near-delta", the `zf` reading for full scale, `lambda=0` for
"no patchiness". Absolute thresholds are precisely what broke the `S` index (`§3`), so they are not used.

**A prior pass at M2's autocorrelation already exists** in `results_crisp/diagnosis_stats.json` (committed
2026-08-05): T1 winner `[0.197, 0.041, 0.066]` vs zero-fill `[1.450, 1.208, 1.087]`. It **must be
re-derived, not reused as-is** — the zero-fill values exceed 1, which a variance-normalised autocorrelation
cannot do, so those numbers are on some other normalisation whose definition is not recoverable (the
producing script was never written to disk). Treat the file as a hint about direction, not as a reading.

## 5. Controls

### Control A — fully sampled

Mask all-ones, everything else identical to the checkpoint M0 identifies, 1500 iters (this is an
adjudication run, not a metric-chasing run).

**Known bias, stated up front:** the warm start (576 902 Gaussians) was fitted against *undersampled* data,
so it is mismatched at full sampling and will understate the quality full sampling can reach. The bias is
**conservative for our purpose** — it can only make "less patchiness at full sampling" harder to observe —
so if patchiness survives anyway, the conclusion is firmer, not weaker.

**Confound that would fake the desired result, and its fix.** Replacing the mask with all-ones multiplies
the measurement count by ~12, which raises the data term's relative weight and therefore **dilutes TV's
effective strength**. "Less patchiness at full sampling" then becomes a foregone conclusion that would be
misread as "the patchiness came from the mask." Run A as-specified would manufacture its own answer.

Fix: run control A at **two lambdas** — the original `lambda_geo = 0.10` and a measurement-count-scaled
`lambda_geo ~= 1.2` — and read both with M5's selectivity (a ratio of ratios, hence relatively robust to
overall strength). **The verdict counts only if both lambdas agree.** If they disagree, control A is
reported as unable to decide; picking whichever arm is convenient is forbidden. Cost: one extra 1500-iter
run, about +18 min.

### Control B — different mask instance

Same AF=12 vd-Poisson, same parameters, **seed changed only**.

**Trap that must be blocked before the run.** A fresh `vdPoisMex` mask puts its ACS at the matrix centre,
but this data's k-space DC sits at `axis1 = 132` while the matrix centre is 110. That 22 px offset is
exactly what fabricated the whole "AF=12 ceiling" in `§8` (only 27% of central energy captured). **The new
mask must be rolled onto DC by the same `(2, 24, -6)`**, or control B measures a registration bug instead of
a mask-instance difference and P2's verdict is void.

Pre-run assertions (both must pass or the run does not start):

```
mask @ DC == 1                     # ACS actually lands on the data's DC
|net_AF - 11.83| < 0.05            # sampling rate matched to the shipped mask, else not comparable
```

## 6. Execution order and early-stop conditions

```
M0 provenance                       <- gate; nothing downstream is interpretable without it
  |- track 1 (GPU): launch control A (two lambdas) and control B, after the mask assertions pass
  |- track 2 (CPU): M1 -> M3 -> M2 -> M5 -> M4
join: M6 / M7 arrive -> four-proposition verdict table
```

Early stop (these do not need to wait for GPU):

```
M1 shows a diffuse PSF with no discrete sidelobes  AND  M3 gives E_air(rec)/E_air(gt) ~= 1
   -> P4 FALSE; classical wrap-around ghosts declared structurally impossible under this mask; closed

M2 gives high NCC with near-delta autocorrelation  AND  M5's selectivity lands in the known eps-series band
   -> P1 TRUE (it is incoherent aliasing, correctly named), and the patchiness magnitude falls out with it
```

## 7. Failure modes and fallbacks

- **M0 finds no unique match** (the figure may come from a deleted npz or an intermediate checkpoint) →
  fall back to the shipped `tvs4k_e040`, bind every conclusion **explicitly to that checkpoint**, and record
  the figure's provenance as unrecoverable. Do not pretend to know.
- **E_air is sensitive to how the air mask is thresholded** → report the ratio as a **curve over thresholds**,
  never a single point.
- **Control B's net AF misses 11.83** → mismatched sampling rate is not comparable; the assertion above
  rejects the seed and another is drawn.
- **Control A's two lambdas disagree** → control A reported as undecided (see §5).

## 8. Deliverables

- A JSON holding every metric's raw numbers **and its calibration stake's reading** alongside it.
- One adjudication figure: a row per location (GT / `zf` error / `rec` error, shared window), error panels
  in `jet` at fixed `[0, 0.1]` per the global convention.
- The four-proposition verdict table: **true / false / insufficient evidence** for each, each carrying the
  measurement reading and its stake reading.
- Figures copied to a stable directory and reported by **absolute path**.

## 9. Out of scope

- **Any algorithm change.** This spec ends at a verdict. Whether to act on it is a separate decision.
- Re-tuning lambda, kappa_q or eps. `§0` fixes the working point and `§14` records that lambda only slides
  along the staircase-vs-grain trade, both ends of which the user rejected.
- 2-map CSM / CSM recalibration (`§12`: ruled out, decisively, do not re-try).
- T1-FLAIR spatial registration (`§13`: user ruled it out of scope twice). M4 detects displacement as
  *evidence*; it does not license adding a registration module.
- Re-inventing the dropped indices — the "isolated jump" fraction (0.000 for every volume including GT) and
  the mid-range gradient deficit (merely tracks the distribution shift). `§14` dropped both as
  degenerate/redundant.
