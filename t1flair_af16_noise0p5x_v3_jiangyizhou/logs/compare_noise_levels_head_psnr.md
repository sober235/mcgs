# Noise study: every arm at every level of added measurement noise run so far (head_psnr)

Levels: no noise, 0.5 sigma0, 1 sigma0, 2 sigma0. Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth,
scorer and frozen AF16 phase-3 MC-GS recipe at every level; the classical lambdas are re-selected per level; every
noisy level is the same realization, scaled.

**T1**

| arm | no noise | 0.5 sigma0 | 1 sigma0 | 2 sigma0 | 0.5 sigma0 − none | 1 sigma0 − none | 2 sigma0 − none |
|---|---|---|---|---|---|---|---|
| MC-GS | 35.706 | 35.280 | 34.016 | 29.802 | -0.426 | -1.690 | -5.904 |
| GS-single | 35.558 | 35.109 | 33.799 | 29.571 | -0.449 | -1.759 | -5.987 |
| ZS-SSDU | 34.150 | 33.836 | 33.950 | 33.131 | -0.314 | -0.200 | -1.019 |
| CS-SENSE (TV, converged) | 34.945 | 34.636 | 34.018 | 32.766 | -0.309 | -0.927 | -2.179 |
| Joint SENSE (JTV) | 34.831 | 34.551 | 33.935 | 32.697 | -0.281 | -0.896 | -2.134 |
| CS-SENSE (35 outer) | 34.409 | 34.223 | 33.730 | 32.227 | -0.187 | -0.679 | -2.183 |
| L1-SPIRiT | 33.788 | 33.404 | 32.772 | 31.453 | -0.384 | -1.016 | -2.335 |
| CG-SENSE | 32.354 | 32.156 | 31.613 | 29.924 | -0.197 | -0.741 | -2.430 |
| MC-INR | 31.590 | 31.129 | 29.713 | 25.034 | -0.460 | -1.877 | -6.555 |
| 3D-DIP | 30.994 | 30.898 | 30.944 | 30.951 | -0.097 | -0.050 | -0.043 |
| 3D-INR | 30.375 | 30.198 | 29.538 | 28.698 | -0.177 | -0.837 | -1.677 |
| ZF | 25.841 | 25.840 | 25.839 | 25.835 | -0.000 | -0.001 | -0.005 |

**FLAIR**

| arm | no noise | 0.5 sigma0 | 1 sigma0 | 2 sigma0 | 0.5 sigma0 − none | 1 sigma0 − none | 2 sigma0 − none |
|---|---|---|---|---|---|---|---|
| MC-GS | 34.928 | 34.617 | 33.635 | 29.946 | -0.311 | -1.293 | -4.982 |
| GS-single | 34.886 | 34.581 | 33.628 | 30.062 | -0.305 | -1.258 | -4.824 |
| ZS-SSDU | 34.005 | 33.836 | 33.289 | 32.348 | -0.169 | -0.715 | -1.657 |
| CS-SENSE (TV, converged) | 34.302 | 34.071 | 33.556 | 32.526 | -0.231 | -0.746 | -1.776 |
| Joint SENSE (JTV) | 34.355 | 34.126 | 33.602 | 32.553 | -0.229 | -0.753 | -1.802 |
| CS-SENSE (35 outer) | 33.806 | 33.683 | 33.345 | 32.172 | -0.123 | -0.461 | -1.635 |
| L1-SPIRiT | 33.720 | 33.394 | 32.848 | 31.636 | -0.326 | -0.872 | -2.084 |
| CG-SENSE | 32.826 | 32.622 | 32.055 | 30.297 | -0.205 | -0.771 | -2.530 |
| MC-INR | 32.243 | 31.881 | 30.734 | 26.507 | -0.361 | -1.509 | -5.735 |
| 3D-DIP | 30.650 | 30.591 | 30.637 | 30.611 | -0.058 | -0.013 | -0.039 |
| 3D-INR | 30.257 | 30.168 | 29.900 | 29.389 | -0.089 | -0.356 | -0.868 |
| ZF | 27.879 | 27.878 | 27.876 | 27.868 | -0.001 | -0.002 | -0.010 |

## MC-GS against the strongest comparison arm (GS-single counts)

| contrast | level | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 35.706 | GS-single 35.558 | +0.1481 |
| T1 | 0.5 sigma0 | 35.280 | GS-single 35.109 | +0.1714 |
| T1 | 1 sigma0 | 34.016 | CS-SENSE (TV, converged) 34.018 | -0.0026 |
| T1 | 2 sigma0 | 29.802 | ZS-SSDU 33.131 | -3.3283 |
| FLAIR | no noise | 34.928 | GS-single 34.886 | +0.0417 |
| FLAIR | 0.5 sigma0 | 34.617 | GS-single 34.581 | +0.0355 |
| FLAIR | 1 sigma0 | 33.635 | GS-single 33.628 | +0.0073 |
| FLAIR | 2 sigma0 | 29.946 | Joint SENSE (JTV) 32.553 | -2.6063 |

## Rank of MC-GS on the three head metrics (1 = best among the listed arms)

| contrast | level | head PSNR | head SSIM | head HFEN |
|---|---|---|---|---|
| T1 | no noise | 1/12 | 1/12 | 1/12 |
| T1 | 0.5 sigma0 | 1/12 | 1/12 | 1/12 |
| T1 | 1 sigma0 | 2/12 | 5/12 | 1/12 |
| T1 | 2 sigma0 | 8/12 | 7/12 | 5/12 |
| FLAIR | no noise | 1/12 | 1/12 | 1/12 |
| FLAIR | 0.5 sigma0 | 1/12 | 1/12 | 1/12 |
| FLAIR | 1 sigma0 | 1/12 | 5/12 | 1/12 |
| FLAIR | 2 sigma0 | 9/12 | 9/12 | 7/12 |

## Lambda chosen by each swept family, per level

| contrast | family | no noise | 0.5 sigma0 | 1 sigma0 | 2 sigma0 |
|---|---|---|---|---|---|
| T1 | cssense_o200 | 0.0007 | 0.001 | 0.001 | 0.002 |
| T1 | jcssense | 0.001 (upper edge) | 0.001 | 0.0014 | 0.0028 |
| T1 | spirit | 0.001, kernel 5 | 0.001, kernel 3 | 0.002, kernel 5 | 0.004, kernel 3 |
| FLAIR | cssense_o200 | 0.0005 | 0.0007 | 0.001 | 0.002 |
| FLAIR | jcssense | 0.0007 (lower edge) | 0.001 | 0.0014 | 0.0028 |
| FLAIR | spirit | 0.001, kernel 5 | 0.002, kernel 5 | 0.002, kernel 5 | 0.004, kernel 5 |

## Diagnostic: GS arms with and without hard data consistency (both through each topic's Pair)

| contrast | level | arm | with DC | render only | render − with DC |
|---|---|---|---|---|---|
| T1 | 0.5 sigma0 | MC-GS | 35.280 | 35.290 | +0.0099 |
| T1 | 0.5 sigma0 | GS-single | 35.109 | 35.119 | +0.0103 |
| FLAIR | 0.5 sigma0 | MC-GS | 34.617 | 34.601 | -0.0156 |
| FLAIR | 0.5 sigma0 | GS-single | 34.581 | 34.564 | -0.0177 |
| T1 | 1 sigma0 | MC-GS | 34.016 | 34.055 | +0.0388 |
| T1 | 1 sigma0 | GS-single | 33.799 | 33.836 | +0.0369 |
| FLAIR | 1 sigma0 | MC-GS | 33.635 | 33.653 | +0.0175 |
| FLAIR | 1 sigma0 | GS-single | 33.628 | 33.644 | +0.0162 |
| T1 | 2 sigma0 | MC-GS | 29.802 | 29.859 | +0.0569 |
| T1 | 2 sigma0 | GS-single | 29.571 | 29.625 | +0.0543 |
| FLAIR | 2 sigma0 | MC-GS | 29.946 | 30.005 | +0.0587 |
| FLAIR | 2 sigma0 | GS-single | 30.062 | 30.125 | +0.0631 |

