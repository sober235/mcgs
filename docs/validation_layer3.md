# Layer 3 validation — deformation (§8) + phase (§9)

## What Layer 3 adds

- **§8 cross-scan deformation** `T_m(r) = R_m r + t_m + u_m(r)`: a rigid part (rotation
  `R_m` via axis-angle→Rodrigues, translation `t_m`) plus a coarse-grid nonrigid field
  `u_m` that aligns each contrast to the reference scan. The reference contrast is pinned to
  identity; only the other contrast deforms.
- **§9 low-frequency phase field** `φ_m`: a per-contrast coarse control grid trilinearly
  upsampled to a complex phasor `e^{jφ_m}`, low-frequency by construction; applied as
  `x_m = e^{jφ_m}[…]` in the §4 render.

The **§8 caveat** is the crux: *the nonrigid field must be strictly limited in amplitude and
spatial frequency, or the deformation model will absorb genuine lesion / enhancement.* The
implementation caps `|u| ≤ u_max` (default 0.05 normalized units) via a `tanh`.

## Controlled recovery tests

Because these modules must be verified against *known* ground truth, the validation injects
known perturbations on a synthetic phantom and checks recovery
(`paired_arc_gs/modules/validate_layer3.py`, 3/3).

![Layer-3 validation]({{artifact:art_daa2d657-4c3b-42ed-a100-e6e29a53b807}})

### A — deformation recovers a known rigid shift (should help)

A reference point set is shifted by a known translation; the deformation is optimized to
align contrast-1 back to the reference.

| injected (z, y, x) | recovered | max error |
|---|---|---|
| (0.060, −0.030, 0.020) | (0.0571, −0.0286, 0.0190) | **0.0029** |

The rigid translation is recovered to < 0.003 normalized units — mild acquisition
misalignment is corrected.

### B — amplitude cap prevents absorbing change (the §8 caveat)

The nonrigid field is driven *hard* toward a displacement of 0.5 — 10× the cap. The achieved
maximum `|u|` is:

| requested | u_max cap | achieved max \|u\| |
|---|---|---|
| 0.50 | 0.05 | **0.0316** |

Even under a drive 10× past the cap, the focal nonrigid displacement stays bounded (≤ u_max).
So a *localized* signal change (a lesion or enhancement, which requires changing intensity,
not just moving geometry) **cannot be absorbed** by the nonrigid field — it is left for the
Layer-2 change branch. (The rigid part is unbounded but global, so it cannot fake a *focal*
change either.) This is the discrimination the plan step calls 轻度错位区分: deformation fixes
mild misalignment (A) but is capped so it cannot eat real change (B).

### C — phase field recovers a known low-frequency ramp

A smooth phase ramp `φ = 1.2z + 0.7y − 0.5x` is injected; the phase field is fit to the
resulting phasor.

| phasor agreement ⟨Re(e^{jφ̂} e^{−jφ})⟩ |
|---|
| **1.00** |

Agreement 1.00 (1.0 = perfect) — the low-frequency phase is recovered exactly, and the
wrap-aware construction (`e^{jφ}` optimized directly) handles the ±π branch with no artifact.

## Notes

- These are module-level controlled tests (known injected ground truth), complementing the
  integration smoke (`L3_SMOKE_OK`: both modules train end-to-end inside the full trainer,
  all §12 loss terms finite). Unit tests: `test_layer3.py` 11/11.
- The amplitude cap `u_max` is the single most important §8 hyperparameter; it is exposed in
  `config_schema` (`model.deformation.u_max`).
