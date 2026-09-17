# Noise study: every arm without and with 2 sigma0 of added measurement noise (head_ssim)

Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3
MC-GS recipe. Only the measurements differ; the classical lambdas are re-selected on the noisy data.

**T1**

| arm | no noise | 2 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.9396 | 0.7771 | -0.1625 |
| GS-single | 0.9371 | 0.7671 | -0.1700 |
| ZS-SSDU | 0.9216 | 0.8945 | -0.0271 |
| CS-SENSE (TV, converged) (λ 0.002) | 0.9317 | 0.8951 | -0.0366 |
| Joint SENSE (JTV) (λ 0.0028) | 0.9311 | 0.8949 | -0.0362 |
| CS-SENSE (35 outer) | 0.9276 | 0.8711 | -0.0565 |
| L1-SPIRiT (λ 0.004) | 0.9083 | 0.8601 | -0.0481 |
| CG-SENSE | 0.8825 | 0.7646 | -0.1178 |
| MC-INR | 0.8355 | 0.5522 | -0.2833 |
| 3D-DIP | 0.8549 | 0.8525 | -0.0024 |
| 3D-INR | 0.7951 | 0.7224 | -0.0727 |
| ZF | 0.6795 | 0.6782 | -0.0013 |

**FLAIR**

| arm | no noise | 2 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.9261 | 0.7644 | -0.1617 |
| GS-single | 0.9249 | 0.7699 | -0.1549 |
| ZS-SSDU | 0.9185 | 0.8731 | -0.0454 |
| CS-SENSE (TV, converged) (λ 0.002) | 0.9180 | 0.8809 | -0.0371 |
| Joint SENSE (JTV) (λ 0.0028) | 0.9177 | 0.8801 | -0.0377 |
| CS-SENSE (35 outer) | 0.9116 | 0.8635 | -0.0482 |
| L1-SPIRiT (λ 0.004) | 0.9022 | 0.8493 | -0.0529 |
| CG-SENSE | 0.8860 | 0.7656 | -0.1204 |
| MC-INR | 0.8590 | 0.5950 | -0.2640 |
| 3D-DIP | 0.8290 | 0.8269 | -0.0021 |
| 3D-INR | 0.7889 | 0.7483 | -0.0405 |
| ZF | 0.7126 | 0.7111 | -0.0015 |

## MC-GS against the strongest comparison arm (GS-single counts)

| contrast | data | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 0.9396 | GS-single 0.9371 | +0.0025 |
| T1 | 2 sigma0 | 0.7771 | CS-SENSE (TV, converged) 0.8951 | -0.1180 |
| FLAIR | no noise | 0.9261 | GS-single 0.9249 | +0.0013 |
| FLAIR | 2 sigma0 | 0.7644 | CS-SENSE (TV, converged) 0.8809 | -0.1165 |

## Rank of MC-GS on the three head metrics (1 = best among the listed arms)

| contrast | data | head PSNR | head SSIM | head HFEN |
|---|---|---|---|---|
| T1 | no noise | 1/12 | 1/12 | 1/12 |
| T1 | 2 sigma0 | 8/12 | 7/12 | 5/12 |
| FLAIR | no noise | 1/12 | 1/12 | 1/12 |
| FLAIR | 2 sigma0 | 9/12 | 9/12 | 7/12 |

## Frozen-lambda view: each swept family at its zero-noise lambda, on the noisy data

| contrast | family | zero-noise λ | noisy data at that λ | noisy data at the re-selected λ |
|---|---|---|---|---|
| T1 | cssense_o200 | 0.0007 | 0.8373 | 0.8951 (λ 0.002) |
| T1 | jcssense | 0.001 | 0.8449 | 0.8949 (λ 0.0028) |
| T1 | spirit | 0.001 (kernel 5) | 0.7786 | 0.8601 (λ 0.004) |
| FLAIR | cssense_o200 | 0.0005 | 0.7924 | 0.8809 (λ 0.002) |
| FLAIR | jcssense | 0.0007 | 0.7911 | 0.8801 (λ 0.0028) |
| FLAIR | spirit | 0.001 (kernel 5) | 0.7693 | 0.8493 (λ 0.004) |

## Where the re-selected lambdas sit on their grids (2 sigma0)

- T1 cssense_o200: λ 0.002 of [0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028] -> interior
- T1 jcssense: λ 0.0028 of [0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028, 0.004, 0.0056] -> interior
- T1 spirit: λ 0.004 of [0, 0.0005, 0.001, 0.002, 0.004, 0.008] -> interior
- FLAIR cssense_o200: λ 0.002 of [0.00025, 0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028, 0.004] -> interior
- FLAIR jcssense: λ 0.0028 of [0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028, 0.004, 0.0056] -> interior
- FLAIR spirit: λ 0.004 of [0, 0.0005, 0.001, 0.002, 0.004, 0.008] -> interior

## Diagnostic: GS arms with and without hard data consistency (both through this topic's Pair)

| contrast | arm | with DC (as in the tables) | render only | render − with DC |
|---|---|---|---|---|
| T1 | MC-GS | 0.7771 | 0.7796 | +0.0025 |
| T1 | GS-single | 0.7671 | 0.7695 | +0.0024 |
| FLAIR | MC-GS | 0.7644 | 0.7668 | +0.0024 |
| FLAIR | GS-single | 0.7699 | 0.7724 | +0.0025 |

