# Noise study: every arm without and with 2 sigma0 of added measurement noise (head_psnr)

Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3
MC-GS recipe. Only the measurements differ; the classical lambdas are re-selected on the noisy data.

**T1**

| arm | no noise | 2 sigma0 | change |
|---|---|---|---|
| MC-GS | 35.706 | 29.802 | -5.904 |
| GS-single | 35.558 | 29.571 | -5.987 |
| ZS-SSDU | 34.150 | 33.131 | -1.019 |
| CS-SENSE (TV, converged) (λ 0.002) | 34.945 | 32.766 | -2.179 |
| Joint SENSE (JTV) (λ 0.0028) | 34.831 | 32.697 | -2.134 |
| CS-SENSE (35 outer) | 34.409 | 32.227 | -2.183 |
| L1-SPIRiT (λ 0.004) | 33.788 | 31.453 | -2.335 |
| CG-SENSE | 32.354 | 29.924 | -2.430 |
| MC-INR | 31.590 | 25.034 | -6.555 |
| 3D-DIP | 30.994 | 30.951 | -0.043 |
| 3D-INR | 30.375 | 28.698 | -1.677 |
| ZF | 25.841 | 25.835 | -0.005 |

**FLAIR**

| arm | no noise | 2 sigma0 | change |
|---|---|---|---|
| MC-GS | 34.928 | 29.946 | -4.982 |
| GS-single | 34.886 | 30.062 | -4.824 |
| ZS-SSDU | 34.005 | 32.348 | -1.657 |
| CS-SENSE (TV, converged) (λ 0.002) | 34.302 | 32.526 | -1.776 |
| Joint SENSE (JTV) (λ 0.0028) | 34.355 | 32.553 | -1.802 |
| CS-SENSE (35 outer) | 33.806 | 32.172 | -1.635 |
| L1-SPIRiT (λ 0.004) | 33.720 | 31.636 | -2.084 |
| CG-SENSE | 32.826 | 30.297 | -2.530 |
| MC-INR | 32.243 | 26.507 | -5.735 |
| 3D-DIP | 30.650 | 30.611 | -0.039 |
| 3D-INR | 30.257 | 29.389 | -0.868 |
| ZF | 27.879 | 27.868 | -0.010 |

## MC-GS against the strongest comparison arm (GS-single counts)

| contrast | data | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 35.706 | GS-single 35.558 | +0.1481 |
| T1 | 2 sigma0 | 29.802 | ZS-SSDU 33.131 | -3.3283 |
| FLAIR | no noise | 34.928 | GS-single 34.886 | +0.0417 |
| FLAIR | 2 sigma0 | 29.946 | Joint SENSE (JTV) 32.553 | -2.6063 |

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
| T1 | cssense_o200 | 0.0007 | 31.347 | 32.766 (λ 0.002) |
| T1 | jcssense | 0.001 | 31.565 | 32.697 (λ 0.0028) |
| T1 | spirit | 0.001 (kernel 5) | 29.872 | 31.453 (λ 0.004) |
| FLAIR | cssense_o200 | 0.0005 | 30.654 | 32.526 (λ 0.002) |
| FLAIR | jcssense | 0.0007 | 30.633 | 32.553 (λ 0.0028) |
| FLAIR | spirit | 0.001 (kernel 5) | 29.892 | 31.636 (λ 0.004) |

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
| T1 | MC-GS | 29.802 | 29.859 | +0.0569 |
| T1 | GS-single | 29.571 | 29.625 | +0.0543 |
| FLAIR | MC-GS | 29.946 | 30.005 | +0.0587 |
| FLAIR | GS-single | 30.062 | 30.125 | +0.0631 |

