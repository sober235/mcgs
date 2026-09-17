# Noise study: every arm without and with 0.5 sigma0 of added measurement noise (head_ssim)

Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3
MC-GS recipe. Only the measurements differ; the classical lambdas are re-selected on the noisy data.

**T1**

| arm | no noise | 0.5 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.9396 | 0.9336 | -0.0060 |
| GS-single | 0.9371 | 0.9306 | -0.0065 |
| ZS-SSDU | 0.9216 | 0.9166 | -0.0050 |
| CS-SENSE (TV, converged) (λ 0.001) | 0.9317 | 0.9284 | -0.0033 |
| Joint SENSE (JTV) (λ 0.001) | 0.9311 | 0.9264 | -0.0047 |
| CS-SENSE (35 outer) | 0.9276 | 0.9242 | -0.0034 |
| L1-SPIRiT (λ 0.001) | 0.9083 | 0.9001 | -0.0082 |
| CG-SENSE | 0.8825 | 0.8735 | -0.0090 |
| MC-INR | 0.8355 | 0.8135 | -0.0219 |
| 3D-DIP | 0.8549 | 0.8512 | -0.0037 |
| 3D-INR | 0.7951 | 0.7889 | -0.0062 |
| ZF | 0.6795 | 0.6794 | -0.0001 |

**FLAIR**

| arm | no noise | 0.5 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.9261 | 0.9204 | -0.0057 |
| GS-single | 0.9249 | 0.9193 | -0.0056 |
| ZS-SSDU | 0.9185 | 0.9135 | -0.0050 |
| CS-SENSE (TV, converged) (λ 0.0007) | 0.9180 | 0.9142 | -0.0039 |
| Joint SENSE (JTV) (λ 0.001) | 0.9177 | 0.9140 | -0.0038 |
| CS-SENSE (35 outer) | 0.9116 | 0.9093 | -0.0023 |
| L1-SPIRiT (λ 0.002) | 0.9022 | 0.8989 | -0.0033 |
| CG-SENSE | 0.8860 | 0.8769 | -0.0091 |
| MC-INR | 0.8590 | 0.8410 | -0.0180 |
| 3D-DIP | 0.8290 | 0.8265 | -0.0025 |
| 3D-INR | 0.7889 | 0.7841 | -0.0048 |
| ZF | 0.7126 | 0.7125 | -0.0001 |

## MC-GS against the strongest comparison arm (GS-single counts)

| contrast | data | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 0.9396 | GS-single 0.9371 | +0.0025 |
| T1 | 0.5 sigma0 | 0.9336 | GS-single 0.9306 | +0.0030 |
| FLAIR | no noise | 0.9261 | GS-single 0.9249 | +0.0013 |
| FLAIR | 0.5 sigma0 | 0.9204 | GS-single 0.9193 | +0.0012 |

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
| T1 | cssense_o200 | 0.0007 | 0.9267 | 0.9284 (λ 0.001) |
| T1 | jcssense | 0.001 | 0.9264 | 0.9264 (λ 0.001) |
| T1 | spirit | 0.001 (kernel 5) | 0.8990 | 0.9001 (λ 0.001) |
| FLAIR | cssense_o200 | 0.0005 | 0.9124 | 0.9142 (λ 0.0007) |
| FLAIR | jcssense | 0.0007 | 0.9118 | 0.9140 (λ 0.001) |
| FLAIR | spirit | 0.001 (kernel 5) | 0.8932 | 0.8989 (λ 0.002) |

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
| T1 | MC-GS | 0.9336 | 0.9340 | +0.0004 |
| T1 | GS-single | 0.9306 | 0.9309 | +0.0004 |
| FLAIR | MC-GS | 0.9204 | 0.9204 | -0.0001 |
| FLAIR | GS-single | 0.9193 | 0.9191 | -0.0002 |

