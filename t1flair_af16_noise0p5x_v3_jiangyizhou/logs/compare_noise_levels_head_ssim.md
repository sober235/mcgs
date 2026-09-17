# Noise study: every arm at every level of added measurement noise run so far (head_ssim)

Levels: no noise, 0.5 sigma0, 1 sigma0, 2 sigma0. Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth,
scorer and frozen AF16 phase-3 MC-GS recipe at every level; the classical lambdas are re-selected per level; every
noisy level is the same realization, scaled.

**T1**

| arm | no noise | 0.5 sigma0 | 1 sigma0 | 2 sigma0 | 0.5 sigma0 − none | 1 sigma0 − none | 2 sigma0 − none |
|---|---|---|---|---|---|---|---|
| MC-GS | 0.9396 | 0.9336 | 0.9109 | 0.7771 | -0.0060 | -0.0287 | -0.1625 |
| GS-single | 0.9371 | 0.9306 | 0.9062 | 0.7671 | -0.0065 | -0.0309 | -0.1700 |
| ZS-SSDU | 0.9216 | 0.9166 | 0.9167 | 0.8945 | -0.0050 | -0.0049 | -0.0271 |
| CS-SENSE (TV, converged) | 0.9317 | 0.9284 | 0.9175 | 0.8951 | -0.0033 | -0.0142 | -0.0366 |
| Joint SENSE (JTV) | 0.9311 | 0.9264 | 0.9171 | 0.8949 | -0.0047 | -0.0140 | -0.0362 |
| CS-SENSE (35 outer) | 0.9276 | 0.9242 | 0.9140 | 0.8711 | -0.0034 | -0.0136 | -0.0565 |
| L1-SPIRiT | 0.9083 | 0.9001 | 0.8913 | 0.8601 | -0.0082 | -0.0170 | -0.0481 |
| CG-SENSE | 0.8825 | 0.8735 | 0.8481 | 0.7646 | -0.0090 | -0.0343 | -0.1178 |
| MC-INR | 0.8355 | 0.8135 | 0.7492 | 0.5522 | -0.0219 | -0.0863 | -0.2833 |
| 3D-DIP | 0.8549 | 0.8512 | 0.8518 | 0.8525 | -0.0037 | -0.0031 | -0.0024 |
| 3D-INR | 0.7951 | 0.7889 | 0.7601 | 0.7224 | -0.0062 | -0.0350 | -0.0727 |
| ZF | 0.6795 | 0.6794 | 0.6792 | 0.6782 | -0.0001 | -0.0003 | -0.0013 |

**FLAIR**

| arm | no noise | 0.5 sigma0 | 1 sigma0 | 2 sigma0 | 0.5 sigma0 − none | 1 sigma0 − none | 2 sigma0 − none |
|---|---|---|---|---|---|---|---|
| MC-GS | 0.9261 | 0.9204 | 0.8982 | 0.7644 | -0.0057 | -0.0279 | -0.1617 |
| GS-single | 0.9249 | 0.9193 | 0.8977 | 0.7699 | -0.0056 | -0.0271 | -0.1549 |
| ZS-SSDU | 0.9185 | 0.9135 | 0.9019 | 0.8731 | -0.0050 | -0.0165 | -0.0454 |
| CS-SENSE (TV, converged) | 0.9180 | 0.9142 | 0.9043 | 0.8809 | -0.0039 | -0.0137 | -0.0371 |
| Joint SENSE (JTV) | 0.9177 | 0.9140 | 0.9039 | 0.8801 | -0.0038 | -0.0138 | -0.0377 |
| CS-SENSE (35 outer) | 0.9116 | 0.9093 | 0.9018 | 0.8635 | -0.0023 | -0.0098 | -0.0482 |
| L1-SPIRiT | 0.9022 | 0.8989 | 0.8842 | 0.8493 | -0.0033 | -0.0180 | -0.0529 |
| CG-SENSE | 0.8860 | 0.8769 | 0.8511 | 0.7656 | -0.0091 | -0.0349 | -0.1204 |
| MC-INR | 0.8590 | 0.8410 | 0.7839 | 0.5950 | -0.0180 | -0.0751 | -0.2640 |
| 3D-DIP | 0.8290 | 0.8265 | 0.8296 | 0.8269 | -0.0025 | +0.0006 | -0.0021 |
| 3D-INR | 0.7889 | 0.7841 | 0.7712 | 0.7483 | -0.0048 | -0.0177 | -0.0405 |
| ZF | 0.7126 | 0.7125 | 0.7122 | 0.7111 | -0.0001 | -0.0004 | -0.0015 |

## MC-GS against the strongest comparison arm (GS-single counts)

| contrast | level | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 0.9396 | GS-single 0.9371 | +0.0025 |
| T1 | 0.5 sigma0 | 0.9336 | GS-single 0.9306 | +0.0030 |
| T1 | 1 sigma0 | 0.9109 | CS-SENSE (TV, converged) 0.9175 | -0.0066 |
| T1 | 2 sigma0 | 0.7771 | CS-SENSE (TV, converged) 0.8951 | -0.1180 |
| FLAIR | no noise | 0.9261 | GS-single 0.9249 | +0.0013 |
| FLAIR | 0.5 sigma0 | 0.9204 | GS-single 0.9193 | +0.0012 |
| FLAIR | 1 sigma0 | 0.8982 | CS-SENSE (TV, converged) 0.9043 | -0.0061 |
| FLAIR | 2 sigma0 | 0.7644 | CS-SENSE (TV, converged) 0.8809 | -0.1165 |

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
| T1 | 0.5 sigma0 | MC-GS | 0.9336 | 0.9340 | +0.0004 |
| T1 | 0.5 sigma0 | GS-single | 0.9306 | 0.9309 | +0.0004 |
| FLAIR | 0.5 sigma0 | MC-GS | 0.9204 | 0.9204 | -0.0001 |
| FLAIR | 0.5 sigma0 | GS-single | 0.9193 | 0.9191 | -0.0002 |
| T1 | 1 sigma0 | MC-GS | 0.9109 | 0.9121 | +0.0011 |
| T1 | 1 sigma0 | GS-single | 0.9062 | 0.9073 | +0.0011 |
| FLAIR | 1 sigma0 | MC-GS | 0.8982 | 0.8991 | +0.0009 |
| FLAIR | 1 sigma0 | GS-single | 0.8977 | 0.8985 | +0.0008 |
| T1 | 2 sigma0 | MC-GS | 0.7771 | 0.7796 | +0.0025 |
| T1 | 2 sigma0 | GS-single | 0.7671 | 0.7695 | +0.0024 |
| FLAIR | 2 sigma0 | MC-GS | 0.7644 | 0.7668 | +0.0024 |
| FLAIR | 2 sigma0 | GS-single | 0.7699 | 0.7724 | +0.0025 |

