# Noise study: every arm without and with 1 sigma0 of added measurement noise (head_ssim)

Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3
MC-GS recipe. Only the measurements differ; the classical lambdas are re-selected on the noisy data.

**T1**

| arm | no noise | 1 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.9396 | 0.9109 | -0.0287 |
| GS-single | 0.9371 | 0.9062 | -0.0309 |
| ZS-SSDU | 0.9216 | 0.9167 | -0.0049 |
| CS-SENSE (TV, converged) (λ 0.001) | 0.9317 | 0.9175 | -0.0142 |
| Joint SENSE (JTV) (λ 0.0014) | 0.9311 | 0.9171 | -0.0140 |
| CS-SENSE (35 outer) | 0.9276 | 0.9140 | -0.0136 |
| L1-SPIRiT (λ 0.002) | 0.9083 | 0.8913 | -0.0170 |
| CG-SENSE | 0.8825 | 0.8481 | -0.0343 |
| MC-INR | 0.8355 | 0.7492 | -0.0863 |
| 3D-DIP | 0.8549 | 0.8518 | -0.0031 |
| 3D-INR | 0.7951 | 0.7601 | -0.0350 |
| ZF | 0.6795 | 0.6792 | -0.0003 |

**FLAIR**

| arm | no noise | 1 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.9261 | 0.8982 | -0.0279 |
| GS-single | 0.9249 | 0.8977 | -0.0271 |
| ZS-SSDU | 0.9185 | 0.9019 | -0.0165 |
| CS-SENSE (TV, converged) (λ 0.001) | 0.9180 | 0.9043 | -0.0137 |
| Joint SENSE (JTV) (λ 0.0014) | 0.9177 | 0.9039 | -0.0138 |
| CS-SENSE (35 outer) | 0.9116 | 0.9018 | -0.0098 |
| L1-SPIRiT (λ 0.002) | 0.9022 | 0.8842 | -0.0180 |
| CG-SENSE | 0.8860 | 0.8511 | -0.0349 |
| MC-INR | 0.8590 | 0.7839 | -0.0751 |
| 3D-DIP | 0.8290 | 0.8296 | +0.0006 |
| 3D-INR | 0.7889 | 0.7712 | -0.0177 |
| ZF | 0.7126 | 0.7122 | -0.0004 |

## MC-GS against the strongest comparison arm (GS-single counts)

| contrast | data | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 0.9396 | GS-single 0.9371 | +0.0025 |
| T1 | 1 sigma0 | 0.9109 | CS-SENSE (TV, converged) 0.9175 | -0.0066 |
| FLAIR | no noise | 0.9261 | GS-single 0.9249 | +0.0013 |
| FLAIR | 1 sigma0 | 0.8982 | CS-SENSE (TV, converged) 0.9043 | -0.0061 |

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
| T1 | cssense_o200 | 0.0007 | 0.9109 | 0.9175 (λ 0.001) |
| T1 | jcssense | 0.001 | 0.9120 | 0.9171 (λ 0.0014) |
| T1 | spirit | 0.001 (kernel 5) | 0.8716 | 0.8913 (λ 0.002) |
| FLAIR | cssense_o200 | 0.0005 | 0.8928 | 0.9043 (λ 0.001) |
| FLAIR | jcssense | 0.0007 | 0.8916 | 0.9039 (λ 0.0014) |
| FLAIR | spirit | 0.001 (kernel 5) | 0.8661 | 0.8842 (λ 0.002) |

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
| T1 | MC-GS | 0.9109 | 0.9121 | +0.0011 |
| T1 | GS-single | 0.9062 | 0.9073 | +0.0011 |
| FLAIR | MC-GS | 0.8982 | 0.8991 | +0.0009 |
| FLAIR | GS-single | 0.8977 | 0.8985 | +0.0008 |

