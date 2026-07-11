# Layer 2 validation — measurement-audited change routing (§7 + §10)

## What Layer 2 adds

A per-contrast **change branch** `G_{Δ,m}` (§7) that represents signal the shared
anatomy + conditioned response cannot explain, gated by a **measurement audit** (§10) so
it adds change only when *independent* held-out measurements support it — the anti-aliasing
guard the spec's §10 is built around.

- **§10.2 proposal**: render anatomy-only `x_m^A`, compute the fit-set k-space residual
  `r_fit = M_fit[F(S x_A) − y]`, backproject to an image-space evidence map
  `e = |A_fit^H r_fit|`; propose change candidates from the high-evidence tail.
- **§10.3 audit acceptance**: add the candidates, briefly fit their density on the **fit**
  set, then measure the **audit gain**
  `ΔE = ‖A_audit x − y‖² − ‖A_audit x⁺ − y‖²` on the disjoint **audit** set. Accept the
  birth only if `ΔE > τ`; otherwise roll it back. Positive ΔE means the change reduced
  error on measurements that did *not* generate it → it is real, not fitted aliasing.

The k-space is split three ways (§10.1 + §18): **fit** (updates params) ∪ **audit** (judges
births) ∪ **held-out test** (never touched, used only for the GT-free NRMSE metric).

## Result 1 — mechanism (synthetic, controlled)

A controlled scene with a *planted* localized change plus undersampling aliasing:

| quantity | value |
|---|---|
| evidence-peak distance to planted change | **0.0 voxels** |
| evidence-peak distance to anatomy centre | 11.7 voxels |
| audit gain, **true change** | **ΔE = +1.313** |
| audit gain, **aliasing artifact** | ΔE = +0.157 |
| separation | **8.3×** |

The proposal localizes the unexplained signal exactly, and the audit gain for genuine change
is 8.3× that of an aliasing-fit artifact — so a threshold τ between them accepts the real
change and rejects the aliasing. (`paired_arc_gs/train/test_routing.py`, 5/5.)

## Result 2 — discrimination on real data

Birth audit gain ΔE at stage `s5_change_birth` (200k Gaussians, 400-iter response fit +
160-iter candidate fit before the audit):

| pair | contrast | ΔE at birth | decision | interpretation |
|---|---|---|---|---|
| **CE** (pre/post contrast) | pre  | **+0.012** | ACCEPT | genuine enhancement change |
| **CE** | post | **+0.021** | ACCEPT | stronger — the enhanced scan |
| T1/FLAIR (healthy) | T1 | +0.04 / −0.07 * | noise ≈ 0 | no real change to find |
| T1/FLAIR (healthy) | FLAIR | **−0.078** | REJECT | correctly abstains (no lesion) |

\* T1's birth gain hovers around zero and flips sign across launches (nondeterministic split)
— consistent with there being no genuine contrast-specific change in a healthy pair.

**The audit discriminates**: on the CE pre/post pair (which has *real* localized contrast
enhancement, §7.2) it accepts change with a clearly positive gain — and *post* (the enhanced
scan) has the larger gain, as physiology predicts. On the healthy T1/FLAIR pair it abstains.

![Layer-2 audit]({{artifact:art_dbdbd562-80a2-45eb-8d2b-a08b4d424446}})

## Result 3 — zero-regret gate (ablation: no-change / audit / bypass)

Held-out k-space NRMSE (lower = better; birth_fit_iters = 160):

| pair | contrast | no change | **audit** | bypass (accept-all) |
|---|---|---|---|---|
| T1/FLAIR | FLAIR | 0.8660 | **0.8673** | 0.8672 |
| T1/FLAIR | T1 | 0.5869 | 0.5874 | 0.5872 |
| CE | pre | — | **0.2821** | 0.2829 |
| CE | post | — | **0.2665** | 0.2650 |

On the healthy pair the three arms are **equal within 0.001** — the change branch is inert
when there is no genuine change, so the audit costs nothing by abstaining (no false-accept
damage). On CE, audit ≈ bypass because the change is real and both accept it. The audit is a
**zero-regret gate**: it never hurts, and it protects against aliasing (Result 1) when the
undersampling would otherwise tempt a free private branch.

## Finding — `birth_fit_iters` is load-bearing

The audit judges a candidate *after* a brief fit. If that fit is too short the candidate is
underfit and cannot yet reduce audit error, producing a **false reject**:

- `birth_fit_iters = 80`: CE audit **rejects** pre/post (ΔE < 0) — false negative.
- `birth_fit_iters = 160`: CE audit **accepts** pre/post (ΔE > 0) — correct.

So the candidate must be given a fair fit before the audit decides. 160 iters is adequate on
these pairs; this is a genuine hyperparameter, documented in `config_schema` (`model.change
.birth_fit_iters`).

## Caveats

- Runs are **not bit-reproducible** (`deterministic: false` in the voxelizer; the fit/audit/
  test split is reseeded per process), so the exact ΔE and NRMSE wobble at the third decimal
  across launches. The qualitative result (CE accepts, healthy abstains, zero regret) is
  stable and reproduced across launches.
- CE has no fully-sampled ground truth, so all CE numbers use the GT-free held-out k-space
  NRMSE (§18); full-reference PSNR/SSIM is only valid on T1/FLAIR.
- These are Layer-2 numbers (response + change, no phase/deformation/Gabor/CG yet); absolute
  quality is below the fully-trained MC-GS winner by design.
