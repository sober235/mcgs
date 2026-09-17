# Noise study: every arm at no noise, 1 sigma0 and 2 sigma0 of added measurement noise (head_hfen)

Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3
MC-GS recipe at every level; the classical lambdas are re-selected per level; the 2 sigma0 noise is the 1 sigma0
noise doubled (one realization).

**T1**

| arm | no noise | 1 sigma0 | 2 sigma0 | 1 sigma0 − none | 2 sigma0 − none |
|---|---|---|---|---|---|
| MC-GS | 0.1139 | 0.1385 | 0.2252 | +0.0246 | +0.1113 |
| GS-single | 0.1186 | 0.1438 | 0.2326 | +0.0252 | +0.1140 |
| ZS-SSDU | 0.1646 | 0.1701 | 0.1870 | +0.0054 | +0.0224 |
| CS-SENSE (TV, converged) | 0.1380 | 0.1586 | 0.1967 | +0.0205 | +0.0587 |
| Joint SENSE (JTV) | 0.1429 | 0.1645 | 0.2044 | +0.0217 | +0.0616 |
| CS-SENSE (35 outer) | 0.1531 | 0.1729 | 0.2221 | +0.0198 | +0.0690 |
| L1-SPIRiT | 0.1448 | 0.1836 | 0.2427 | +0.0388 | +0.0979 |
| CG-SENSE | 0.2239 | 0.2397 | 0.2801 | +0.0158 | +0.0562 |
| MC-INR | 0.2325 | 0.2663 | 0.3746 | +0.0338 | +0.1421 |
| 3D-DIP | 0.3024 | 0.3086 | 0.3062 | +0.0062 | +0.0038 |
| 3D-INR | 0.2125 | 0.2519 | 0.3397 | +0.0394 | +0.1272 |
| ZF | 0.6950 | 0.6951 | 0.6955 | +0.0001 | +0.0004 |

**FLAIR**

| arm | no noise | 1 sigma0 | 2 sigma0 | 1 sigma0 − none | 2 sigma0 − none |
|---|---|---|---|---|---|
| MC-GS | 0.1121 | 0.1360 | 0.2205 | +0.0239 | +0.1085 |
| GS-single | 0.1137 | 0.1364 | 0.2161 | +0.0227 | +0.1024 |
| ZS-SSDU | 0.1529 | 0.1644 | 0.1867 | +0.0115 | +0.0339 |
| CS-SENSE (TV, converged) | 0.1309 | 0.1488 | 0.1847 | +0.0179 | +0.0538 |
| Joint SENSE (JTV) | 0.1302 | 0.1483 | 0.1847 | +0.0181 | +0.0545 |
| CS-SENSE (35 outer) | 0.1360 | 0.1535 | 0.2004 | +0.0175 | +0.0645 |
| L1-SPIRiT | 0.1207 | 0.1506 | 0.2030 | +0.0298 | +0.0822 |
| CG-SENSE | 0.1608 | 0.1813 | 0.2319 | +0.0206 | +0.0711 |
| MC-INR | 0.1711 | 0.2038 | 0.3081 | +0.0327 | +0.1370 |
| 3D-DIP | 0.2769 | 0.2740 | 0.2816 | -0.0029 | +0.0047 |
| 3D-INR | 0.1998 | 0.2294 | 0.2965 | +0.0296 | +0.0966 |
| ZF | 0.6595 | 0.6596 | 0.6600 | +0.0001 | +0.0004 |

## MC-GS against the strongest comparison arm (GS-single counts)

| contrast | level | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 0.1139 | GS-single 0.1186 | +0.0047 |
| T1 | 1 sigma0 | 0.1385 | GS-single 0.1438 | +0.0053 |
| T1 | 2 sigma0 | 0.2252 | ZS-SSDU 0.1870 | -0.0382 |
| FLAIR | no noise | 0.1121 | GS-single 0.1137 | +0.0016 |
| FLAIR | 1 sigma0 | 0.1360 | GS-single 0.1364 | +0.0004 |
| FLAIR | 2 sigma0 | 0.2205 | Joint SENSE (JTV) 0.1847 | -0.0359 |

## Rank of MC-GS on the three head metrics (1 = best among the listed arms)

| contrast | level | head PSNR | head SSIM | head HFEN |
|---|---|---|---|---|
| T1 | no noise | 1/12 | 1/12 | 1/12 |
| T1 | 1 sigma0 | 2/12 | 5/12 | 1/12 |
| T1 | 2 sigma0 | 8/12 | 7/12 | 5/12 |
| FLAIR | no noise | 1/12 | 1/12 | 1/12 |
| FLAIR | 1 sigma0 | 1/12 | 5/12 | 1/12 |
| FLAIR | 2 sigma0 | 9/12 | 9/12 | 7/12 |

## Lambda chosen by each swept family, per level

| contrast | family | no noise | 1 sigma0 | 2 sigma0 |
|---|---|---|---|---|
| T1 | cssense_o200 | 0.0007 | 0.001 | 0.002 |
| T1 | jcssense | 0.001 (upper edge) | 0.0014 | 0.0028 |
| T1 | spirit | 0.001, kernel 5 | 0.002, kernel 5 | 0.004, kernel 3 |
| FLAIR | cssense_o200 | 0.0005 | 0.001 | 0.002 |
| FLAIR | jcssense | 0.0007 (lower edge) | 0.0014 | 0.0028 |
| FLAIR | spirit | 0.001, kernel 5 | 0.002, kernel 5 | 0.004, kernel 5 |

## Diagnostic: GS arms with and without hard data consistency (both through each topic's Pair)

| contrast | level | arm | with DC | render only | render − with DC |
|---|---|---|---|---|---|
| T1 | 1 sigma0 | MC-GS | 0.1385 | 0.1386 | +0.0001 |
| T1 | 1 sigma0 | GS-single | 0.1438 | 0.1440 | +0.0002 |
| FLAIR | 1 sigma0 | MC-GS | 0.1360 | 0.1362 | +0.0002 |
| FLAIR | 1 sigma0 | GS-single | 0.1364 | 0.1367 | +0.0003 |
| T1 | 2 sigma0 | MC-GS | 0.2252 | 0.2246 | -0.0006 |
| T1 | 2 sigma0 | GS-single | 0.2326 | 0.2321 | -0.0005 |
| FLAIR | 2 sigma0 | MC-GS | 0.2205 | 0.2205 | -0.0000 |
| FLAIR | 2 sigma0 | GS-single | 0.2161 | 0.2161 | +0.0001 |

