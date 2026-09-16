# Diff-mask study: every arm on three sampling patterns, one gauge (head_psnr)

Same subject, same k-space, same coil maps, same ground truth, same scoring code, same frozen AF16 phase-3
MC-GS recipe; matched point budget (2964 points, T1 15.20x; the house mask 2966, 15.19x). Only the sampling
PATTERN differs, so every difference across a row is the pattern's effect.

**T1** (head_psnr; higher is better unless the metric says otherwise)

| arm | vd-Poisson (house) | Gaussian | radial | Gaussian − vd-Poisson | radial − vd-Poisson |
|---|---|---|---|---|---|
| MC-GS | 35.706 | 34.909 | 31.113 | -0.80 | -4.59 |
| GS-single | 35.558 | 34.771 | 30.913 | -0.79 | -4.65 |
| ZS-SSDU | 34.150 | 34.375 | 30.854 | +0.22 | -3.30 |
| CS-SENSE (TV, converged) | 34.945 | 33.896 | 29.955 | -1.05 | -4.99 |
| joint CS-SENSE (JTV) | 34.831 | 33.841 | 29.862 | -0.99 | -4.97 |
| CS-SENSE | 34.409 | 33.637 | 29.961 | -0.77 | -4.45 |
| L1-SPIRiT | 33.788 | 32.995 | 30.004 | -0.79 | -3.78 |
| CG-SENSE | 32.354 | 31.899 | 29.109 | -0.45 | -3.24 |
| MC-INR | 31.590 | 31.558 | 28.952 | -0.03 | -2.64 |
| 3D-DIP | 30.994 | 31.141 | 29.855 | +0.15 | -1.14 |
| 3D-INR | 30.375 | 29.443 | 27.868 | -0.93 | -2.51 |
| ZF | 25.841 | 25.412 | 25.406 | -0.43 | -0.43 |

Best arm per pattern: **vd-Poisson** MC-GS 35.706;  **Gaussian** MC-GS 34.909;  **radial** MC-GS 31.113

**FLAIR** (head_psnr; higher is better unless the metric says otherwise)

| arm | vd-Poisson (house) | Gaussian | radial | Gaussian − vd-Poisson | radial − vd-Poisson |
|---|---|---|---|---|---|
| MC-GS | 34.928 | 34.317 | 31.729 | -0.61 | -3.20 |
| GS-single | 34.886 | 34.294 | 31.579 | -0.59 | -3.31 |
| ZS-SSDU | 34.005 | 33.972 | 31.635 | -0.03 | -2.37 |
| CS-SENSE (TV, converged) | 34.302 | 33.683 | 30.777 | -0.62 | -3.52 |
| joint CS-SENSE (JTV) | 34.355 | 33.731 | 30.609 | -0.62 | -3.75 |
| CS-SENSE | 33.806 | 33.447 | 30.932 | -0.36 | -2.87 |
| L1-SPIRiT | 33.720 | 32.849 | 30.432 | -0.87 | -3.29 |
| CG-SENSE | 32.826 | 32.557 | 30.301 | -0.27 | -2.53 |
| MC-INR | 32.243 | 32.102 | 29.966 | -0.14 | -2.28 |
| 3D-DIP | 30.650 | 30.918 | 30.290 | +0.27 | -0.36 |
| 3D-INR | 30.257 | 29.807 | 28.247 | -0.45 | -2.01 |
| ZF | 27.879 | 27.451 | 27.377 | -0.43 | -0.50 |

Best arm per pattern: **vd-Poisson** MC-GS 34.928;  **Gaussian** MC-GS 34.317;  **radial** MC-GS 31.729

## MC-GS against the strongest comparison arm, per pattern

| contrast | pattern | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | vd-Poisson | 35.706 | GS-single 35.558 | +0.148 |
| T1 | Gaussian | 34.909 | GS-single 34.771 | +0.138 |
| T1 | radial | 31.113 | GS-single 30.913 | +0.200 |
| FLAIR | vd-Poisson | 34.928 | GS-single 34.886 | +0.042 |
| FLAIR | Gaussian | 34.317 | GS-single 34.294 | +0.023 |
| FLAIR | radial | 31.729 | ZS-SSDU 31.635 | +0.093 |

## MC-GS minus GS-single (the coupling gain), per pattern

| contrast | pattern | MC-GS | GS-single | gain |
|---|---|---|---|---|
| T1 | vd-Poisson | 35.706 | 35.558 | +0.148 |
| T1 | Gaussian | 34.909 | 34.771 | +0.138 |
| T1 | radial | 31.113 | 30.913 | +0.200 |
| FLAIR | vd-Poisson | 34.928 | 34.886 | +0.042 |
| FLAIR | Gaussian | 34.317 | 34.294 | +0.023 |
| FLAIR | radial | 31.729 | 31.579 | +0.150 |

