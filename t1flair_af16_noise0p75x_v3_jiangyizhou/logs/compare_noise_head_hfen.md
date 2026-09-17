# Noise study: every arm without and with 0.75 sigma0 of added measurement noise (head_hfen)

Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3
MC-GS recipe. Only the measurements differ; the classical lambdas are re-selected on the noisy data.

**T1**

| arm | no noise | 0.75 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.1139 | 0.1272 | +0.0133 |
| GS-single | 0.1186 | 0.1323 | +0.0137 |
| ZS-SSDU | 0.1646 | 0.1717 | +0.0071 |
| CS-SENSE (TV, converged) (λ 0.001) | 0.1380 | 0.1499 | +0.0119 |
| Joint SENSE (JTV) (λ 0.0014) | 0.1429 | 0.1561 | +0.0133 |
| CS-SENSE (35 outer) | 0.1531 | 0.1646 | +0.0115 |
| L1-SPIRiT (λ 0.002) | 0.1448 | 0.1740 | +0.0293 |
| CG-SENSE | 0.2239 | 0.2330 | +0.0091 |
| MC-INR | 0.2325 | 0.2508 | +0.0183 |
| 3D-DIP | 0.3024 | 0.3004 | -0.0020 |
| 3D-INR | 0.2125 | 0.2413 | +0.0288 |
| ZF | 0.6950 | 0.6951 | +0.0000 |

**FLAIR**

| arm | no noise | 0.75 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.1121 | 0.1250 | +0.0129 |
| GS-single | 0.1137 | 0.1260 | +0.0124 |
| ZS-SSDU | 0.1529 | 0.1489 | -0.0039 |
| CS-SENSE (TV, converged) (λ 0.0007) | 0.1309 | 0.1424 | +0.0115 |
| Joint SENSE (JTV) (λ 0.001) | 0.1302 | 0.1417 | +0.0115 |
| CS-SENSE (35 outer) | 0.1360 | 0.1460 | +0.0101 |
| L1-SPIRiT (λ 0.002) | 0.1207 | 0.1417 | +0.0210 |
| CG-SENSE | 0.1608 | 0.1726 | +0.0119 |
| MC-INR | 0.1711 | 0.1890 | +0.0180 |
| 3D-DIP | 0.2769 | 0.2734 | -0.0035 |
| 3D-INR | 0.1998 | 0.2241 | +0.0243 |
| ZF | 0.6595 | 0.6596 | +0.0000 |

## MC-GS against the strongest comparison arm (GS-single counts)

| contrast | data | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 0.1139 | GS-single 0.1186 | +0.0047 |
| T1 | 0.75 sigma0 | 0.1272 | GS-single 0.1323 | +0.0051 |
| FLAIR | no noise | 0.1121 | GS-single 0.1137 | +0.0016 |
| FLAIR | 0.75 sigma0 | 0.1250 | GS-single 0.1260 | +0.0010 |

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
| T1 | cssense_o200 | 0.0007 | 0.1531 | 0.1499 (λ 0.001) |
| T1 | jcssense | 0.001 | 0.1574 | 0.1561 (λ 0.0014) |
| T1 | spirit | 0.001 (kernel 5) | 0.1633 | 0.1740 (λ 0.002) |
| FLAIR | cssense_o200 | 0.0005 | 0.1473 | 0.1424 (λ 0.0007) |
| FLAIR | jcssense | 0.0007 | 0.1468 | 0.1417 (λ 0.001) |
| FLAIR | spirit | 0.001 (kernel 5) | 0.1378 | 0.1417 (λ 0.002) |

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
| T1 | MC-GS | 0.1272 | 0.1276 | +0.0004 |
| T1 | GS-single | 0.1323 | 0.1328 | +0.0005 |
| FLAIR | MC-GS | 0.1250 | 0.1254 | +0.0003 |
| FLAIR | GS-single | 0.1260 | 0.1265 | +0.0005 |

