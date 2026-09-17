# Noise study: every arm without and with 1 sigma0 of added measurement noise (head_hfen)

Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3
MC-GS recipe. Only the measurements differ; the classical lambdas are re-selected on the noisy data.

**T1**

| arm | no noise | 1 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.1139 | 0.1385 | +0.0246 |
| GS-single | 0.1186 | 0.1438 | +0.0252 |
| ZS-SSDU | 0.1646 | 0.1701 | +0.0054 |
| CS-SENSE (TV, converged) (λ 0.001) | 0.1380 | 0.1586 | +0.0205 |
| Joint SENSE (JTV) (λ 0.0014) | 0.1429 | 0.1645 | +0.0217 |
| CS-SENSE (35 outer) | 0.1531 | 0.1729 | +0.0198 |
| L1-SPIRiT (λ 0.002) | 0.1448 | 0.1836 | +0.0388 |
| CG-SENSE | 0.2239 | 0.2397 | +0.0158 |
| MC-INR | 0.2325 | 0.2663 | +0.0338 |
| 3D-DIP | 0.3024 | 0.3086 | +0.0062 |
| 3D-INR | 0.2125 | 0.2519 | +0.0394 |
| ZF | 0.6950 | 0.6951 | +0.0001 |

**FLAIR**

| arm | no noise | 1 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.1121 | 0.1360 | +0.0239 |
| GS-single | 0.1137 | 0.1364 | +0.0227 |
| ZS-SSDU | 0.1529 | 0.1644 | +0.0115 |
| CS-SENSE (TV, converged) (λ 0.001) | 0.1309 | 0.1488 | +0.0179 |
| Joint SENSE (JTV) (λ 0.0014) | 0.1302 | 0.1483 | +0.0181 |
| CS-SENSE (35 outer) | 0.1360 | 0.1535 | +0.0175 |
| L1-SPIRiT (λ 0.002) | 0.1207 | 0.1506 | +0.0298 |
| CG-SENSE | 0.1608 | 0.1813 | +0.0206 |
| MC-INR | 0.1711 | 0.2038 | +0.0327 |
| 3D-DIP | 0.2769 | 0.2740 | -0.0029 |
| 3D-INR | 0.1998 | 0.2294 | +0.0296 |
| ZF | 0.6595 | 0.6596 | +0.0001 |

## MC-GS against the strongest comparison arm (GS-single counts)

| contrast | data | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 0.1139 | GS-single 0.1186 | +0.0047 |
| T1 | 1 sigma0 | 0.1385 | GS-single 0.1438 | +0.0053 |
| FLAIR | no noise | 0.1121 | GS-single 0.1137 | +0.0016 |
| FLAIR | 1 sigma0 | 0.1360 | GS-single 0.1364 | +0.0004 |

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
| T1 | cssense_o200 | 0.0007 | 0.1640 | 0.1586 (λ 0.001) |
| T1 | jcssense | 0.001 | 0.1679 | 0.1645 (λ 0.0014) |
| T1 | spirit | 0.001 (kernel 5) | 0.1763 | 0.1836 (λ 0.002) |
| FLAIR | cssense_o200 | 0.0005 | 0.1597 | 0.1488 (λ 0.001) |
| FLAIR | jcssense | 0.0007 | 0.1594 | 0.1483 (λ 0.0014) |
| FLAIR | spirit | 0.001 (kernel 5) | 0.1505 | 0.1506 (λ 0.002) |

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
| T1 | MC-GS | 0.1385 | 0.1386 | +0.0001 |
| T1 | GS-single | 0.1438 | 0.1440 | +0.0002 |
| FLAIR | MC-GS | 0.1360 | 0.1362 | +0.0002 |
| FLAIR | GS-single | 0.1364 | 0.1367 | +0.0003 |

