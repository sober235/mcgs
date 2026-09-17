# Noise study: every arm without and with 0.5 sigma0 of added measurement noise (head_hfen)

Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3
MC-GS recipe. Only the measurements differ; the classical lambdas are re-selected on the noisy data.

**T1**

| arm | no noise | 0.5 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.1139 | 0.1197 | +0.0058 |
| GS-single | 0.1186 | 0.1245 | +0.0059 |
| ZS-SSDU | 0.1646 | 0.1710 | +0.0064 |
| CS-SENSE (TV, converged) (λ 0.001) | 0.1380 | 0.1436 | +0.0055 |
| Joint SENSE (JTV) (λ 0.001) | 0.1429 | 0.1496 | +0.0067 |
| CS-SENSE (35 outer) | 0.1531 | 0.1583 | +0.0053 |
| L1-SPIRiT (λ 0.001) | 0.1448 | 0.1527 | +0.0079 |
| CG-SENSE | 0.2239 | 0.2280 | +0.0041 |
| MC-INR | 0.2325 | 0.2406 | +0.0081 |
| 3D-DIP | 0.3024 | 0.3083 | +0.0059 |
| 3D-INR | 0.2125 | 0.2316 | +0.0191 |
| ZF | 0.6950 | 0.6950 | +0.0000 |

**FLAIR**

| arm | no noise | 0.5 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.1121 | 0.1177 | +0.0056 |
| GS-single | 0.1137 | 0.1190 | +0.0054 |
| ZS-SSDU | 0.1529 | 0.1520 | -0.0009 |
| CS-SENSE (TV, converged) (λ 0.0007) | 0.1309 | 0.1351 | +0.0042 |
| Joint SENSE (JTV) (λ 0.001) | 0.1302 | 0.1344 | +0.0041 |
| CS-SENSE (35 outer) | 0.1360 | 0.1405 | +0.0046 |
| L1-SPIRiT (λ 0.002) | 0.1207 | 0.1351 | +0.0144 |
| CG-SENSE | 0.1608 | 0.1662 | +0.0054 |
| MC-INR | 0.1711 | 0.1788 | +0.0077 |
| 3D-DIP | 0.2769 | 0.2803 | +0.0034 |
| 3D-INR | 0.1998 | 0.2075 | +0.0077 |
| ZF | 0.6595 | 0.6596 | +0.0000 |

## MC-GS against the strongest comparison arm (GS-single counts)

| contrast | data | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 0.1139 | GS-single 0.1186 | +0.0047 |
| T1 | 0.5 sigma0 | 0.1197 | GS-single 0.1245 | +0.0048 |
| FLAIR | no noise | 0.1121 | GS-single 0.1137 | +0.0016 |
| FLAIR | 0.5 sigma0 | 0.1177 | GS-single 0.1190 | +0.0013 |

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
| T1 | cssense_o200 | 0.0007 | 0.1449 | 0.1436 (λ 0.001) |
| T1 | jcssense | 0.001 | 0.1496 | 0.1496 (λ 0.001) |
| T1 | spirit | 0.001 (kernel 5) | 0.1533 | 0.1527 (λ 0.001) |
| FLAIR | cssense_o200 | 0.0005 | 0.1383 | 0.1351 (λ 0.0007) |
| FLAIR | jcssense | 0.0007 | 0.1377 | 0.1344 (λ 0.001) |
| FLAIR | spirit | 0.001 (kernel 5) | 0.1284 | 0.1351 (λ 0.002) |

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
| T1 | MC-GS | 0.1197 | 0.1204 | +0.0006 |
| T1 | GS-single | 0.1245 | 0.1252 | +0.0007 |
| FLAIR | MC-GS | 0.1177 | 0.1182 | +0.0005 |
| FLAIR | GS-single | 0.1190 | 0.1197 | +0.0006 |

