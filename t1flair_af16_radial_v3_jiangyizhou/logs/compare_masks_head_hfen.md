# Diff-mask study: every arm on three sampling patterns, one gauge (head_hfen)

Same subject, same k-space, same coil maps, same ground truth, same scoring code, same frozen AF16 phase-3
MC-GS recipe; matched point budget (2964 points, T1 15.20x; the house mask 2966, 15.19x). Only the sampling
PATTERN differs, so every difference across a row is the pattern's effect.

**T1** (head_hfen; higher is better unless the metric says otherwise)

| arm | vd-Poisson (house) | Gaussian | radial | Gaussian − vd-Poisson | radial − vd-Poisson |
|---|---|---|---|---|---|
| MC-GS | 0.114 | 0.134 | 0.342 | +0.02 | +0.23 |
| GS-single | 0.119 | 0.141 | 0.359 | +0.02 | +0.24 |
| ZS-SSDU | 0.165 | 0.170 | 0.352 | +0.01 | +0.19 |
| CS-SENSE (TV, converged) | 0.138 | 0.181 | 0.408 | +0.04 | +0.27 |
| joint CS-SENSE (JTV) | 0.143 | 0.183 | 0.419 | +0.04 | +0.28 |
| CS-SENSE | 0.153 | 0.185 | 0.405 | +0.03 | +0.25 |
| L1-SPIRiT | 0.145 | 0.165 | 0.368 | +0.02 | +0.22 |
| CG-SENSE | 0.224 | 0.252 | 0.458 | +0.03 | +0.23 |
| MC-INR | 0.232 | 0.260 | 0.486 | +0.03 | +0.25 |
| 3D-DIP | 0.302 | 0.305 | 0.430 | +0.00 | +0.13 |
| 3D-INR | 0.213 | 0.221 | 0.457 | +0.01 | +0.24 |
| ZF | 0.695 | 0.732 | 0.754 | +0.04 | +0.06 |

Best arm per pattern: **vd-Poisson** MC-GS 0.114;  **Gaussian** MC-GS 0.134;  **radial** MC-GS 0.342

**FLAIR** (head_hfen; higher is better unless the metric says otherwise)

| arm | vd-Poisson (house) | Gaussian | radial | Gaussian − vd-Poisson | radial − vd-Poisson |
|---|---|---|---|---|---|
| MC-GS | 0.112 | 0.128 | 0.281 | +0.02 | +0.17 |
| GS-single | 0.114 | 0.129 | 0.299 | +0.02 | +0.19 |
| ZS-SSDU | 0.153 | 0.148 | 0.277 | -0.00 | +0.12 |
| CS-SENSE (TV, converged) | 0.131 | 0.148 | 0.374 | +0.02 | +0.24 |
| joint CS-SENSE (JTV) | 0.130 | 0.148 | 0.390 | +0.02 | +0.26 |
| CS-SENSE | 0.136 | 0.148 | 0.355 | +0.01 | +0.22 |
| L1-SPIRiT | 0.121 | 0.136 | 0.314 | +0.02 | +0.19 |
| CG-SENSE | 0.161 | 0.180 | 0.406 | +0.02 | +0.25 |
| MC-INR | 0.171 | 0.195 | 0.428 | +0.02 | +0.26 |
| 3D-DIP | 0.277 | 0.261 | 0.342 | -0.02 | +0.06 |
| 3D-INR | 0.200 | 0.205 | 0.430 | +0.01 | +0.23 |
| ZF | 0.660 | 0.689 | 0.719 | +0.03 | +0.06 |

Best arm per pattern: **vd-Poisson** MC-GS 0.112;  **Gaussian** MC-GS 0.128;  **radial** ZS-SSDU 0.277

## MC-GS against the strongest comparison arm, per pattern

| contrast | pattern | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | vd-Poisson | 0.114 | GS-single 0.119 | -0.005 |
| T1 | Gaussian | 0.134 | GS-single 0.141 | -0.007 |
| T1 | radial | 0.342 | ZS-SSDU 0.352 | -0.010 |
| FLAIR | vd-Poisson | 0.112 | GS-single 0.114 | -0.002 |
| FLAIR | Gaussian | 0.128 | GS-single 0.129 | -0.002 |
| FLAIR | radial | 0.281 | ZS-SSDU 0.277 | +0.004 |

## MC-GS minus GS-single (the coupling gain), per pattern

| contrast | pattern | MC-GS | GS-single | gain |
|---|---|---|---|---|
| T1 | vd-Poisson | 0.114 | 0.119 | -0.005 |
| T1 | Gaussian | 0.134 | 0.141 | -0.007 |
| T1 | radial | 0.342 | 0.359 | -0.017 |
| FLAIR | vd-Poisson | 0.112 | 0.114 | -0.002 |
| FLAIR | Gaussian | 0.128 | 0.129 | -0.002 |
| FLAIR | radial | 0.281 | 0.299 | -0.018 |

