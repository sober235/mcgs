# Noise study: every arm without and with 1 sigma0 of added measurement noise (head_psnr)

Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3
MC-GS recipe. Only the measurements differ; the classical lambdas are re-selected on the noisy data.

**T1**

| arm | no noise | 1 sigma0 | change |
|---|---|---|---|
| MC-GS | 35.706 | 34.016 | -1.690 |
| GS-single | 35.558 | 33.799 | -1.759 |
| ZS-SSDU | 34.150 | 33.950 | -0.200 |
| CS-SENSE (TV, converged) (λ 0.001) | 34.945 | 34.018 | -0.927 |
| Joint SENSE (JTV) (λ 0.0014) | 34.831 | 33.935 | -0.896 |
| CS-SENSE (35 outer) | 34.409 | 33.730 | -0.679 |
| L1-SPIRiT (λ 0.002) | 33.788 | 32.772 | -1.016 |
| CG-SENSE | 32.354 | 31.613 | -0.741 |
| MC-INR | 31.590 | 29.713 | -1.877 |
| 3D-DIP | 30.994 | 30.944 | -0.050 |
| 3D-INR | 30.375 | 29.538 | -0.837 |
| ZF | 25.841 | 25.839 | -0.001 |

**FLAIR**

| arm | no noise | 1 sigma0 | change |
|---|---|---|---|
| MC-GS | 34.928 | 33.635 | -1.293 |
| GS-single | 34.886 | 33.628 | -1.258 |
| ZS-SSDU | 34.005 | 33.289 | -0.715 |
| CS-SENSE (TV, converged) (λ 0.001) | 34.302 | 33.556 | -0.746 |
| Joint SENSE (JTV) (λ 0.0014) | 34.355 | 33.602 | -0.753 |
| CS-SENSE (35 outer) | 33.806 | 33.345 | -0.461 |
| L1-SPIRiT (λ 0.002) | 33.720 | 32.848 | -0.872 |
| CG-SENSE | 32.826 | 32.055 | -0.771 |
| MC-INR | 32.243 | 30.734 | -1.509 |
| 3D-DIP | 30.650 | 30.637 | -0.013 |
| 3D-INR | 30.257 | 29.900 | -0.356 |
| ZF | 27.879 | 27.876 | -0.002 |

## MC-GS against the strongest comparison arm (GS-single counts)

| contrast | data | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 35.706 | GS-single 35.558 | +0.1481 |
| T1 | 1 sigma0 | 34.016 | CS-SENSE (TV, converged) 34.018 | -0.0026 |
| FLAIR | no noise | 34.928 | GS-single 34.886 | +0.0417 |
| FLAIR | 1 sigma0 | 33.635 | GS-single 33.628 | +0.0073 |

## Rank of MC-GS on the three head metrics (1 = best among the listed arms)

| contrast | data | head PSNR | head SSIM | head HFEN |
|---|---|---|---|---|
| T1 | no noise | 1/12 | 1/12 | 1/12 |
| T1 | 1 sigma0 | 2/12 | 5/12 | 1/12 |
| FLAIR | no noise | 1/12 | 1/12 | 1/12 |
| FLAIR | 1 sigma0 | 1/12 | 5/12 | 1/12 |

## Frozen-lambda view: each swept family at its zero-noise lambda, on the noisy data

| contrast | family | zero-noise λ | noisy data at that λ | noisy data at the re-selected λ |
|---|---|---|---|---|
| T1 | cssense_o200 | 0.0007 | 33.816 | 34.018 (λ 0.001) |
| T1 | jcssense | 0.001 | 33.818 | 33.935 (λ 0.0014) |
| T1 | spirit | 0.001 (kernel 5) | 32.369 | 32.772 (λ 0.002) |
| FLAIR | cssense_o200 | 0.0005 | 33.257 | 33.556 (λ 0.001) |
| FLAIR | jcssense | 0.0007 | 33.266 | 33.602 (λ 0.0014) |
| FLAIR | spirit | 0.001 (kernel 5) | 32.424 | 32.848 (λ 0.002) |

## Where the re-selected lambdas sit on their grids (1 sigma0)

- T1 cssense_o200: λ 0.001 of [0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028] -> interior
- T1 jcssense: λ 0.0014 of [0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028] -> interior
- T1 spirit: λ 0.002 of [0, 0.0005, 0.001, 0.002, 0.004, 0.008] -> interior
- FLAIR cssense_o200: λ 0.001 of [0.00025, 0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002] -> interior
- FLAIR jcssense: λ 0.0014 of [0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028] -> interior
- FLAIR spirit: λ 0.002 of [0, 0.0005, 0.001, 0.002, 0.004, 0.008] -> interior

## Diagnostic: GS arms with and without hard data consistency (both through this topic's Pair)

| contrast | arm | with DC (as in the tables) | render only | render − with DC |
|---|---|---|---|---|
| T1 | MC-GS | 34.016 | 34.055 | +0.0388 |
| T1 | GS-single | 33.799 | 33.836 | +0.0369 |
| FLAIR | MC-GS | 33.635 | 33.653 | +0.0175 |
| FLAIR | GS-single | 33.628 | 33.644 | +0.0162 |

