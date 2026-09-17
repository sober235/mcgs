# Noise study: every arm without and with 0.5 sigma0 of added measurement noise (head_psnr)

Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3
MC-GS recipe. Only the measurements differ; the classical lambdas are re-selected on the noisy data.

**T1**

| arm | no noise | 0.5 sigma0 | change |
|---|---|---|---|
| MC-GS | 35.706 | 35.280 | -0.426 |
| GS-single | 35.558 | 35.109 | -0.449 |
| ZS-SSDU | 34.150 | 33.836 | -0.314 |
| CS-SENSE (TV, converged) (λ 0.001) | 34.945 | 34.636 | -0.309 |
| Joint SENSE (JTV) (λ 0.001) | 34.831 | 34.551 | -0.281 |
| CS-SENSE (35 outer) | 34.409 | 34.223 | -0.187 |
| L1-SPIRiT (λ 0.001) | 33.788 | 33.404 | -0.384 |
| CG-SENSE | 32.354 | 32.156 | -0.197 |
| MC-INR | 31.590 | 31.129 | -0.460 |
| 3D-DIP | 30.994 | 30.898 | -0.097 |
| 3D-INR | 30.375 | 30.198 | -0.177 |
| ZF | 25.841 | 25.840 | -0.000 |

**FLAIR**

| arm | no noise | 0.5 sigma0 | change |
|---|---|---|---|
| MC-GS | 34.928 | 34.617 | -0.311 |
| GS-single | 34.886 | 34.581 | -0.305 |
| ZS-SSDU | 34.005 | 33.836 | -0.169 |
| CS-SENSE (TV, converged) (λ 0.0007) | 34.302 | 34.071 | -0.231 |
| Joint SENSE (JTV) (λ 0.001) | 34.355 | 34.126 | -0.229 |
| CS-SENSE (35 outer) | 33.806 | 33.683 | -0.123 |
| L1-SPIRiT (λ 0.002) | 33.720 | 33.394 | -0.326 |
| CG-SENSE | 32.826 | 32.622 | -0.205 |
| MC-INR | 32.243 | 31.881 | -0.361 |
| 3D-DIP | 30.650 | 30.591 | -0.058 |
| 3D-INR | 30.257 | 30.168 | -0.089 |
| ZF | 27.879 | 27.878 | -0.001 |

## MC-GS against the strongest comparison arm (GS-single counts)

| contrast | data | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 35.706 | GS-single 35.558 | +0.1481 |
| T1 | 0.5 sigma0 | 35.280 | GS-single 35.109 | +0.1714 |
| FLAIR | no noise | 34.928 | GS-single 34.886 | +0.0417 |
| FLAIR | 0.5 sigma0 | 34.617 | GS-single 34.581 | +0.0355 |

## Rank of MC-GS on the three head metrics (1 = best among the listed arms)

| contrast | data | head PSNR | head SSIM | head HFEN |
|---|---|---|---|---|
| T1 | no noise | 1/12 | 1/12 | 1/12 |
| T1 | 0.5 sigma0 | 1/12 | 1/12 | 1/12 |
| FLAIR | no noise | 1/12 | 1/12 | 1/12 |
| FLAIR | 0.5 sigma0 | 1/12 | 1/12 | 1/12 |

## Frozen-lambda view: each swept family at its zero-noise lambda, on the noisy data

| contrast | family | zero-noise λ | noisy data at that λ | noisy data at the re-selected λ |
|---|---|---|---|---|
| T1 | cssense_o200 | 0.0007 | 34.633 | 34.636 (λ 0.001) |
| T1 | jcssense | 0.001 | 34.551 | 34.551 (λ 0.001) |
| T1 | spirit | 0.001 (kernel 5) | 33.383 | 33.404 (λ 0.001) |
| FLAIR | cssense_o200 | 0.0005 | 34.024 | 34.071 (λ 0.0007) |
| FLAIR | jcssense | 0.0007 | 34.064 | 34.126 (λ 0.001) |
| FLAIR | spirit | 0.001 (kernel 5) | 33.364 | 33.394 (λ 0.002) |

## Where the re-selected lambdas sit on their grids (0.5 sigma0)

- T1 cssense_o200: λ 0.001 of [0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028] -> interior
- T1 jcssense: λ 0.001 of [0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028] -> interior
- T1 spirit: λ 0.001 of [0, 0.0005, 0.001, 0.002, 0.004, 0.008] -> interior
- FLAIR cssense_o200: λ 0.0007 of [0.00025, 0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002] -> interior
- FLAIR jcssense: λ 0.001 of [0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028] -> interior
- FLAIR spirit: λ 0.002 of [0, 0.0005, 0.001, 0.002, 0.004, 0.008] -> interior

## Diagnostic: GS arms with and without hard data consistency (both through this topic's Pair)

| contrast | arm | with DC (as in the tables) | render only | render − with DC |
|---|---|---|---|---|
| T1 | MC-GS | 35.280 | 35.290 | +0.0099 |
| T1 | GS-single | 35.109 | 35.119 | +0.0103 |
| FLAIR | MC-GS | 34.617 | 34.601 | -0.0156 |
| FLAIR | GS-single | 34.581 | 34.564 | -0.0177 |

