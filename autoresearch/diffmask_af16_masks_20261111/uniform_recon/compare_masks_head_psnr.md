# Diff-mask study: every arm on three sampling patterns, one gauge (head_psnr)

Same subject, same k-space, same coil maps, same ground truth, same scoring code, same frozen AF16 phase-3
MC-GS recipe; matched point budget (2964 points, T1 15.20x; the house mask 2966, 15.19x). Only the sampling
PATTERN differs, so every difference across a row is the pattern's effect.

**T1** (head_psnr; higher is better unless the metric says otherwise)

| arm | vd-Poisson (house) | Uniform density | radial | Uniform density − vd-Poisson | radial − vd-Poisson |
|---|---|---|---|---|---|
| MC-GS | 35.706 | 34.903 | 31.113 | -0.80 | -4.59 |
| GS-single | 35.558 | 34.396 | 30.913 | -1.16 | -4.65 |
| ZS-SSDU | 34.150 | 32.274 | 30.854 | -1.88 | -3.30 |
| CS-SENSE (TV, converged) | 34.945 | 30.924 | 29.955 | -4.02 | -4.99 |
| joint CS-SENSE (JTV) | 34.831 | 30.755 | 29.862 | -4.08 | -4.97 |
| CS-SENSE | 34.409 | 30.361 | 29.961 | -4.05 | -4.45 |
| L1-SPIRiT | 33.788 | 32.732 | 30.004 | -1.06 | -3.78 |
| CG-SENSE | 32.354 | 28.561 | 29.109 | -3.79 | -3.24 |
| MC-INR | 31.590 | 27.671 | 28.952 | -3.92 | -2.64 |
| 3D-DIP | 30.994 | 29.894 | 29.855 | -1.10 | -1.14 |
| 3D-INR | 30.375 | 29.491 | 27.868 | -0.88 | -2.51 |
| ZF | 25.841 | 24.408 | 25.406 | -1.43 | -0.43 |

Best arm per pattern: **vd-Poisson** MC-GS 35.706;  **Uniform density** MC-GS 34.903;  **radial** MC-GS 31.113

**FLAIR** (head_psnr; higher is better unless the metric says otherwise)

| arm | vd-Poisson (house) | Uniform density | radial | Uniform density − vd-Poisson | radial − vd-Poisson |
|---|---|---|---|---|---|
| MC-GS | 34.928 | 34.984 | 31.729 | +0.06 | -3.20 |
| GS-single | 34.886 | 34.842 | 31.579 | -0.04 | -3.31 |
| ZS-SSDU | 34.005 | 34.221 | 31.635 | +0.22 | -2.37 |
| CS-SENSE (TV, converged) | 34.302 | 32.072 | 30.777 | -2.23 | -3.52 |
| joint CS-SENSE (JTV) | 34.355 | 32.075 | 30.609 | -2.28 | -3.75 |
| CS-SENSE | 33.806 | 31.929 | 30.932 | -1.88 | -2.87 |
| L1-SPIRiT | 33.720 | 33.823 | 30.432 | +0.10 | -3.29 |
| CG-SENSE | 32.826 | 30.781 | 30.301 | -2.04 | -2.53 |
| MC-INR | 32.243 | 29.642 | 29.966 | -2.60 | -2.28 |
| 3D-DIP | 30.650 | 30.046 | 30.290 | -0.60 | -0.36 |
| 3D-INR | 30.257 | 29.312 | 28.247 | -0.94 | -2.01 |
| ZF | 27.879 | 26.264 | 27.377 | -1.62 | -0.50 |

Best arm per pattern: **vd-Poisson** MC-GS 34.928;  **Uniform density** MC-GS 34.984;  **radial** MC-GS 31.729

## MC-GS against the strongest comparison arm, per pattern

| contrast | pattern | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | vd-Poisson | 35.706 | GS-single 35.558 | +0.148 |
| T1 | Uniform density | 34.903 | GS-single 34.396 | +0.507 |
| T1 | radial | 31.113 | GS-single 30.913 | +0.200 |
| FLAIR | vd-Poisson | 34.928 | GS-single 34.886 | +0.042 |
| FLAIR | Uniform density | 34.984 | GS-single 34.842 | +0.142 |
| FLAIR | radial | 31.729 | ZS-SSDU 31.635 | +0.093 |

