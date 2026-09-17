# Noise study: every arm without and with 0.75 sigma0 of added measurement noise (head_psnr)

Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3
MC-GS recipe. Only the measurements differ; the classical lambdas are re-selected on the noisy data.

**T1**

| arm | no noise | 0.75 sigma0 | change |
|---|---|---|---|
| MC-GS | 35.706 | 34.749 | -0.957 |
| GS-single | 35.558 | 34.555 | -1.003 |
| ZS-SSDU | 34.150 | 33.710 | -0.440 |
| CS-SENSE (TV, converged) (λ 0.001) | 34.945 | 34.367 | -0.578 |
| Joint SENSE (JTV) (λ 0.0014) | 34.831 | 34.248 | -0.583 |
| CS-SENSE (35 outer) | 34.409 | 34.007 | -0.402 |
| L1-SPIRiT (λ 0.002) | 33.788 | 33.113 | -0.675 |
| CG-SENSE | 32.354 | 31.922 | -0.432 |
| MC-INR | 31.590 | 30.545 | -1.044 |
| 3D-DIP | 30.994 | 31.071 | +0.077 |
| 3D-INR | 30.375 | 29.806 | -0.569 |
| ZF | 25.841 | 25.840 | -0.001 |

**FLAIR**

| arm | no noise | 0.75 sigma0 | change |
|---|---|---|---|
| MC-GS | 34.928 | 34.213 | -0.715 |
| GS-single | 34.886 | 34.188 | -0.698 |
| ZS-SSDU | 34.005 | 33.845 | -0.160 |
| CS-SENSE (TV, converged) (λ 0.0007) | 34.302 | 33.826 | -0.476 |
| Joint SENSE (JTV) (λ 0.001) | 34.355 | 33.872 | -0.483 |
| CS-SENSE (35 outer) | 33.806 | 33.538 | -0.268 |
| L1-SPIRiT (λ 0.002) | 33.720 | 33.157 | -0.563 |
| CG-SENSE | 32.826 | 32.377 | -0.449 |
| MC-INR | 32.243 | 31.414 | -0.829 |
| 3D-DIP | 30.650 | 30.681 | +0.032 |
| 3D-INR | 30.257 | 30.083 | -0.174 |
| ZF | 27.879 | 27.877 | -0.001 |

## MC-GS against the strongest comparison arm (GS-single counts)

| contrast | data | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 35.706 | GS-single 35.558 | +0.1481 |
| T1 | 0.75 sigma0 | 34.749 | GS-single 34.555 | +0.1941 |
| FLAIR | no noise | 34.928 | GS-single 34.886 | +0.0417 |
| FLAIR | 0.75 sigma0 | 34.213 | GS-single 34.188 | +0.0254 |

## Rank of MC-GS on the three head metrics (1 = best among the listed arms)

| contrast | data | head PSNR | head SSIM | head HFEN |
|---|---|---|---|---|
| T1 | no noise | 1/12 | 1/12 | 1/12 |
| T1 | 0.75 sigma0 | 1/12 | 1/12 | 1/12 |
| FLAIR | no noise | 1/12 | 1/12 | 1/12 |
| FLAIR | 0.75 sigma0 | 1/12 | 2/12 | 1/12 |

## Frozen-lambda view: each swept family at its zero-noise lambda, on the noisy data

| contrast | family | zero-noise λ | noisy data at that λ | noisy data at the re-selected λ |
|---|---|---|---|---|
| T1 | cssense_o200 | 0.0007 | 34.275 | 34.367 (λ 0.001) |
| T1 | jcssense | 0.001 | 34.229 | 34.248 (λ 0.0014) |
| T1 | spirit | 0.001 (kernel 5) | 32.929 | 33.113 (λ 0.002) |
| FLAIR | cssense_o200 | 0.0005 | 33.696 | 33.826 (λ 0.0007) |
| FLAIR | jcssense | 0.0007 | 33.720 | 33.872 (λ 0.001) |
| FLAIR | spirit | 0.001 (kernel 5) | 32.950 | 33.157 (λ 0.002) |

## Where the re-selected lambdas sit on their grids (0.75 sigma0)

- T1 cssense_o200: λ 0.001 of [0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028] -> interior
- T1 jcssense: λ 0.0014 of [0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028] -> interior
- T1 spirit: λ 0.002 of [0, 0.0005, 0.001, 0.002, 0.004, 0.008] -> interior
- FLAIR cssense_o200: λ 0.0007 of [0.00025, 0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002] -> interior
- FLAIR jcssense: λ 0.001 of [0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028] -> interior
- FLAIR spirit: λ 0.002 of [0, 0.0005, 0.001, 0.002, 0.004, 0.008] -> interior

## Diagnostic: GS arms with and without hard data consistency (both through this topic's Pair)

| contrast | arm | with DC (as in the tables) | render only | render − with DC |
|---|---|---|---|---|
| T1 | MC-GS | 34.749 | 34.774 | +0.0245 |
| T1 | GS-single | 34.555 | 34.579 | +0.0239 |
| FLAIR | MC-GS | 34.213 | 34.213 | -0.0002 |
| FLAIR | GS-single | 34.188 | 34.186 | -0.0019 |

