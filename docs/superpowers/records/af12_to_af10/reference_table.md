# Reference table — AF12 baseline vs AF10 (Phase 0, spec §3)

| run | contrast | head PSNR | head SSIM | head HFEN | N | whole PSNR | whole SSIM | whole HFEN | e_unobs |
|---|---|---|---|---|---|---|---|---|---|
| tvs4k_e040 | T1 | 33.922 | 0.9047 | 0.1694 | 0.960 | 38.379 | 0.9103 | 0.7703 | — |
| tvs4k_e040 | FLAIR | 33.624 | 0.9258 | 0.1365 | 0.949 | 38.221 | 0.9137 | 0.6732 | — |
| af10_joint_edgetv_e040 | T1 | 34.604 | 0.9152 | 0.1536 | 0.952 | 38.944 | 0.9123 | 0.7314 | — |
| af10_joint_edgetv_e040 | FLAIR | 34.400 | 0.9352 | 0.1223 | 0.949 | 38.866 | 0.9162 | 0.6309 | — |
| af10_joint_edgetv_e040_rerun20260904 | T1 | 34.605 | 0.9152 | 0.1535 | 0.952 | 38.945 | 0.9124 | 0.7314 | 0.6353 |
| af10_joint_edgetv_e040_rerun20260904 | FLAIR | 34.401 | 0.9352 | 0.1222 | 0.949 | 38.867 | 0.9162 | 0.6309 | 0.6298 |

## Gates (from gates.json)

- **T1**: head_psnr ≥ 34.605 − 0.0050; head_ssim ≥ 0.9152 − 0.0005; head_hfen ≤ 0.1535 + 0.0010; e_unobs ≤ 0.6488; N ≤ 1.00
- **FLAIR**: head_psnr ≥ 34.401 − 0.0050; head_ssim ≥ 0.9352 − 0.0005; head_hfen ≤ 0.1222 + 0.0010; e_unobs ≤ 0.6444; N ≤ 1.00

## Hand annotations (Phase 0)

- **Grain-ratio sanity check (Step 1):** measured N (grain_ratio) of `tvs4k_e040` T1 = 0.960; expectation was 0.944 ± 0.02 (memory §0 / §14c "grain 0.944 of GT"). |0.960 − 0.944| = 0.016 ≤ 0.02 → **PASS**, N definition matches; continued to Step 2.
- **Head-PSNR gap to close (AF10 reference − AF12 baseline):**
  - T1: 34.605 − 33.922 = **0.683 dB**
  - FLAIR: 34.401 − 33.624 = **0.777 dB**
