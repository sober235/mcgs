# Diff-mask study: every arm on three sampling patterns, one gauge (head_hfen)

Same subject, same k-space, same coil maps, same ground truth, same scoring code, same frozen AF16 phase-3
MC-GS recipe; matched point budget (2964 points, T1 15.20x; the house mask 2966, 15.19x). Only the sampling
PATTERN differs, so every difference across a row is the pattern's effect.

**T1** (head_hfen; higher is better unless the metric says otherwise)

| arm | vd-Poisson (house) | Uniform density | radial | Uniform density − vd-Poisson | radial − vd-Poisson |
|---|---|---|---|---|---|
| MC-GS | 0.114 | 0.218 | 0.342 | +0.10 | +0.23 |
| GS-single | 0.119 | 0.244 | 0.359 | +0.13 | +0.24 |
| ZS-SSDU | 0.165 | 0.325 | 0.352 | +0.16 | +0.19 |
| CS-SENSE (TV, converged) | 0.138 | 0.399 | 0.408 | +0.26 | +0.27 |
| joint CS-SENSE (JTV) | 0.143 | 0.398 | 0.419 | +0.25 | +0.28 |
| CS-SENSE | 0.153 | 0.425 | 0.405 | +0.27 | +0.25 |
| L1-SPIRiT | 0.145 | 0.290 | 0.368 | +0.14 | +0.22 |
| CG-SENSE | 0.224 | 0.510 | 0.458 | +0.29 | +0.23 |
| MC-INR | 0.232 | 0.577 | 0.486 | +0.34 | +0.25 |
| 3D-DIP | 0.302 | 0.456 | 0.430 | +0.15 | +0.13 |
| 3D-INR | 0.213 | 0.472 | 0.457 | +0.26 | +0.24 |
| ZF | 0.695 | 0.870 | 0.754 | +0.17 | +0.06 |

Best arm per pattern: **vd-Poisson** MC-GS 0.114;  **Uniform density** MC-GS 0.218;  **radial** MC-GS 0.342

**FLAIR** (head_hfen; higher is better unless the metric says otherwise)

| arm | vd-Poisson (house) | Uniform density | radial | Uniform density − vd-Poisson | radial − vd-Poisson |
|---|---|---|---|---|---|
| MC-GS | 0.112 | 0.208 | 0.281 | +0.10 | +0.17 |
| GS-single | 0.114 | 0.220 | 0.299 | +0.11 | +0.19 |
| ZS-SSDU | 0.153 | 0.240 | 0.277 | +0.09 | +0.12 |
| CS-SENSE (TV, converged) | 0.131 | 0.360 | 0.374 | +0.23 | +0.24 |
| joint CS-SENSE (JTV) | 0.130 | 0.360 | 0.390 | +0.23 | +0.26 |
| CS-SENSE | 0.136 | 0.367 | 0.355 | +0.23 | +0.22 |
| L1-SPIRiT | 0.121 | 0.244 | 0.314 | +0.12 | +0.19 |
| CG-SENSE | 0.161 | 0.453 | 0.406 | +0.29 | +0.25 |
| MC-INR | 0.171 | 0.505 | 0.428 | +0.33 | +0.26 |
| 3D-DIP | 0.277 | 0.376 | 0.342 | +0.10 | +0.06 |
| 3D-INR | 0.200 | 0.492 | 0.430 | +0.29 | +0.23 |
| ZF | 0.660 | 0.845 | 0.719 | +0.19 | +0.06 |

Best arm per pattern: **vd-Poisson** MC-GS 0.112;  **Uniform density** MC-GS 0.208;  **radial** ZS-SSDU 0.277

## MC-GS against the strongest comparison arm, per pattern

| contrast | pattern | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | vd-Poisson | 0.114 | GS-single 0.119 | -0.005 |
| T1 | Uniform density | 0.218 | GS-single 0.244 | -0.026 |
| T1 | radial | 0.342 | ZS-SSDU 0.352 | -0.010 |
| FLAIR | vd-Poisson | 0.112 | GS-single 0.114 | -0.002 |
| FLAIR | Uniform density | 0.208 | GS-single 0.220 | -0.012 |
| FLAIR | radial | 0.281 | ZS-SSDU 0.277 | +0.004 |

