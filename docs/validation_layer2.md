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

Two results are robustly reproducible; a third (real-data per-contrast birth gains) is at the
noise floor and is reported honestly as such rather than as a discrimination claim.

![Layer-2 audit]({{artifact:art_dbdbd562-80a2-45eb-8d2b-a08b4d424446}})

## Robust result 1 — mechanism (synthetic, controlled)

A controlled scene with a *planted* localized change plus undersampling aliasing:

| quantity | value |
|---|---|
| evidence-peak distance to planted change | **0.0 voxels** |
| evidence-peak distance to anatomy centre | 11.7 voxels |
| audit gain, **true change** | **ΔE = +1.313** |
| audit gain, **aliasing artifact** | ΔE = +0.157 |
| separation | **8.3×** |

The proposal localizes the unexplained signal exactly, and the audit gain for genuine
(strong, localized) change is 8.3× that of an aliasing-fit artifact — so a threshold τ between
them accepts the real change and rejects the aliasing. This is the mechanism the audit is
designed to provide, verified where ground truth is known. (`paired_arc_gs/train/test_routing.py`, 5/5.)

## Robust result 2 — zero-regret gate on a healthy pair

FLAIR held-out k-space NRMSE, healthy T1/FLAIR pair, 3-way ablation (single process,
birth_fit_iters = 160):

| arm | FLAIR held-out NRMSE |
|---|---|
| no change (branch disabled) | 0.8660 |
| **audit** (branch gated) | 0.8673 |
| bypass (accept-all) | 0.8679 |

The three arms are **equal within 0.002**. On a healthy pair with no genuine contrast-specific
change the branch is inert; refusing it (audit), forcing it (bypass) or disabling it all give
the same held-out error. The audit therefore **costs nothing by abstaining** — there is no
false-accept damage to avoid *and* no benefit foregone. On CE the same holds: `ce_audit` and
`ce_bypass` held-out NRMSE agree within 0.002 (pre 0.282/0.283, post 0.267/0.265).

## Real-data birth gains are at the noise floor (reported honestly)

Unlike the controlled synthetic case, the birth audit gain ΔE on the **real** pairs is small
and **straddles zero across nondeterministic launches** — it is not a reproducible
per-contrast discrimination:

| contrast | ΔE range across launches |
|---|---|
| CE pre | [−0.037, +0.012] |
| CE post | [−0.005, +0.021] |
| T1 | [−0.071, +0.041] |
| FLAIR (healthy) | [−0.105, +0.008] (trends negative → reject) |

At this scale (200k Gaussians, AF≈12/5.3×) the audit is **conservative** on real data: it does
not confidently fire, and healthy FLAIR trends toward reject. We do **not** claim the audit
demonstrably accepts genuine CE enhancement on this single pair — the gains are within
run-to-run noise. The mechanism is proven in the controlled setting (result 1); the real-data
behavior is a zero-regret, conservative gate (result 2).

## Finding — `birth_fit_iters` matters

The audit judges a candidate *after* a brief fit; too short and the candidate is underfit,
pushing ΔE more negative (a bias toward reject). At 80 iters CE birth gains are more negative
than at 160. So the candidate must be given a fair fit before the audit decides
(`config_schema` → `model.change.birth_fit_iters`). Even at 160, real-data gains remain at the
noise floor (above) — this hyperparameter removes an underfit *bias* but does not manufacture a
signal that isn't there.

## Caveats

- Runs are **not bit-reproducible** (`deterministic: false` in the voxelizer; the
  fit/audit/test split is reseeded per process), so ΔE and NRMSE wobble across launches — hence
  the noise-floor framing for the real-data gains. The two robust results above are stable
  across launches.
- CE has no fully-sampled ground truth, so all CE numbers use the GT-free held-out k-space
  NRMSE (§18); full-reference PSNR/SSIM is only valid on T1/FLAIR.
- These are Layer-2 numbers (response + change, no phase/deformation/Gabor/CG yet); absolute
  quality is below the fully-trained MC-GS winner by design.
