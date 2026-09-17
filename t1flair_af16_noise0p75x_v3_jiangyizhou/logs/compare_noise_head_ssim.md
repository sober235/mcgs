# Noise study: every arm without and with 0.75 sigma0 of added measurement noise (head_ssim)

Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3
MC-GS recipe. Only the measurements differ; the classical lambdas are re-selected on the noisy data.

**T1**

| arm | no noise | 0.75 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.9396 | 0.9250 | -0.0146 |
| GS-single | 0.9371 | 0.9213 | -0.0158 |
| ZS-SSDU | 0.9216 | 0.9125 | -0.0091 |
| CS-SENSE (TV, converged) (λ 0.001) | 0.9317 | 0.9240 | -0.0078 |
| Joint SENSE (JTV) (λ 0.0014) | 0.9311 | 0.9231 | -0.0080 |
| CS-SENSE (35 outer) | 0.9276 | 0.9199 | -0.0076 |
| L1-SPIRiT (λ 0.002) | 0.9083 | 0.8998 | -0.0084 |
| CG-SENSE | 0.8825 | 0.8627 | -0.0198 |
| MC-INR | 0.8355 | 0.7865 | -0.0490 |
| 3D-DIP | 0.8549 | 0.8556 | +0.0007 |
| 3D-INR | 0.7951 | 0.7716 | -0.0235 |
| ZF | 0.6795 | 0.6793 | -0.0002 |

**FLAIR**

| arm | no noise | 0.75 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.9261 | 0.9121 | -0.0140 |
| GS-single | 0.9249 | 0.9111 | -0.0137 |
| ZS-SSDU | 0.9185 | 0.9126 | -0.0059 |
| CS-SENSE (TV, converged) (λ 0.0007) | 0.9180 | 0.9091 | -0.0089 |
| Joint SENSE (JTV) (λ 0.001) | 0.9177 | 0.9088 | -0.0089 |
| CS-SENSE (35 outer) | 0.9116 | 0.9063 | -0.0053 |
| L1-SPIRiT (λ 0.002) | 0.9022 | 0.8928 | -0.0094 |
| CG-SENSE | 0.8860 | 0.8659 | -0.0201 |
| MC-INR | 0.8590 | 0.8176 | -0.0415 |
| 3D-DIP | 0.8290 | 0.8298 | +0.0008 |
| 3D-INR | 0.7889 | 0.7810 | -0.0078 |
| ZF | 0.7126 | 0.7124 | -0.0002 |

## MC-GS against the strongest comparison arm (GS-single counts)

| contrast | data | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 0.9396 | GS-single 0.9371 | +0.0025 |
| T1 | 0.75 sigma0 | 0.9250 | CS-SENSE (TV, converged) 0.9240 | +0.0011 |
| FLAIR | no noise | 0.9261 | GS-single 0.9249 | +0.0013 |
| FLAIR | 0.75 sigma0 | 0.9121 | ZS-SSDU 0.9126 | -0.0005 |

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
| T1 | cssense_o200 | 0.0007 | 0.9202 | 0.9240 (λ 0.001) |
| T1 | jcssense | 0.001 | 0.9205 | 0.9231 (λ 0.0014) |
| T1 | spirit | 0.001 (kernel 5) | 0.8875 | 0.8998 (λ 0.002) |
| FLAIR | cssense_o200 | 0.0005 | 0.9047 | 0.9091 (λ 0.0007) |
| FLAIR | jcssense | 0.0007 | 0.9038 | 0.9088 (λ 0.001) |
| FLAIR | spirit | 0.001 (kernel 5) | 0.8819 | 0.8928 (λ 0.002) |

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
| T1 | MC-GS | 0.9250 | 0.9257 | +0.0007 |
| T1 | GS-single | 0.9213 | 0.9220 | +0.0007 |
| FLAIR | MC-GS | 0.9121 | 0.9124 | +0.0004 |
| FLAIR | GS-single | 0.9111 | 0.9114 | +0.0003 |

