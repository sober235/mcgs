# Noise study: every arm without and with 2 sigma0 of added measurement noise (head_hfen)

Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3
MC-GS recipe. Only the measurements differ; the classical lambdas are re-selected on the noisy data.

**T1**

| arm | no noise | 2 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.1139 | 0.2252 | +0.1113 |
| GS-single | 0.1186 | 0.2326 | +0.1140 |
| ZS-SSDU | 0.1646 | 0.1870 | +0.0224 |
| CS-SENSE (TV, converged) (λ 0.002) | 0.1380 | 0.1967 | +0.0587 |
| Joint SENSE (JTV) (λ 0.0028) | 0.1429 | 0.2044 | +0.0616 |
| CS-SENSE (35 outer) | 0.1531 | 0.2221 | +0.0690 |
| L1-SPIRiT (λ 0.004) | 0.1448 | 0.2427 | +0.0979 |
| CG-SENSE | 0.2239 | 0.2801 | +0.0562 |
| MC-INR | 0.2325 | 0.3746 | +0.1421 |
| 3D-DIP | 0.3024 | 0.3062 | +0.0038 |
| 3D-INR | 0.2125 | 0.3397 | +0.1272 |
| ZF | 0.6950 | 0.6955 | +0.0004 |

**FLAIR**

| arm | no noise | 2 sigma0 | change |
|---|---|---|---|
| MC-GS | 0.1121 | 0.2205 | +0.1085 |
| GS-single | 0.1137 | 0.2161 | +0.1024 |
| ZS-SSDU | 0.1529 | 0.1867 | +0.0339 |
| CS-SENSE (TV, converged) (λ 0.002) | 0.1309 | 0.1847 | +0.0538 |
| Joint SENSE (JTV) (λ 0.0028) | 0.1302 | 0.1847 | +0.0545 |
| CS-SENSE (35 outer) | 0.1360 | 0.2004 | +0.0645 |
| L1-SPIRiT (λ 0.004) | 0.1207 | 0.2030 | +0.0822 |
| CG-SENSE | 0.1608 | 0.2319 | +0.0711 |
| MC-INR | 0.1711 | 0.3081 | +0.1370 |
| 3D-DIP | 0.2769 | 0.2816 | +0.0047 |
| 3D-INR | 0.1998 | 0.2965 | +0.0966 |
| ZF | 0.6595 | 0.6600 | +0.0004 |

## MC-GS against the strongest comparison arm (GS-single counts)

| contrast | data | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 0.1139 | GS-single 0.1186 | +0.0047 |
| T1 | 2 sigma0 | 0.2252 | ZS-SSDU 0.1870 | -0.0382 |
| FLAIR | no noise | 0.1121 | GS-single 0.1137 | +0.0016 |
| FLAIR | 2 sigma0 | 0.2205 | Joint SENSE (JTV) 0.1847 | -0.0359 |

## Rank of MC-GS on the three head metrics (1 = best among the listed arms)

| contrast | data | head PSNR | head SSIM | head HFEN |
|---|---|---|---|---|
| T1 | no noise | 1/12 | 1/12 | 1/12 |
| T1 | 2 sigma0 | 8/12 | 7/12 | 5/12 |
| FLAIR | no noise | 1/12 | 1/12 | 1/12 |
| FLAIR | 2 sigma0 | 9/12 | 9/12 | 7/12 |

## Frozen-lambda view: each swept family at its zero-noise lambda, on the noisy data

| contrast | family | zero-noise λ | noisy data at that λ | noisy data at the re-selected λ |
|---|---|---|---|---|
| T1 | cssense_o200 | 0.0007 | 0.2327 | 0.1967 (λ 0.002) |
| T1 | jcssense | 0.001 | 0.2319 | 0.2044 (λ 0.0028) |
| T1 | spirit | 0.001 (kernel 5) | 0.2471 | 0.2427 (λ 0.004) |
| FLAIR | cssense_o200 | 0.0005 | 0.2411 | 0.1847 (λ 0.002) |
| FLAIR | jcssense | 0.0007 | 0.2399 | 0.1847 (λ 0.0028) |
| FLAIR | spirit | 0.001 (kernel 5) | 0.2235 | 0.2030 (λ 0.004) |

## Where the re-selected lambdas sit on their grids (2 sigma0)

- T1 cssense_o200: λ 0.002 of [0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028] -> interior
- T1 jcssense: λ 0.0028 of [0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028, 0.004, 0.0056] -> interior
- T1 spirit: λ 0.004 of [0, 0.0005, 0.001, 0.002, 0.004, 0.008] -> interior
- FLAIR cssense_o200: λ 0.002 of [0.00025, 0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028, 0.004] -> interior
- FLAIR jcssense: λ 0.0028 of [0.00035, 0.0005, 0.0007, 0.001, 0.0014, 0.002, 0.0028, 0.004, 0.0056] -> interior
- FLAIR spirit: λ 0.004 of [0, 0.0005, 0.001, 0.002, 0.004, 0.008] -> interior

## Diagnostic: GS arms with and without hard data consistency (both through this topic's Pair)

| contrast | arm | with DC (as in the tables) | render only | render − with DC |
|---|---|---|---|---|
| T1 | MC-GS | 0.2252 | 0.2246 | -0.0006 |
| T1 | GS-single | 0.2326 | 0.2321 | -0.0005 |
| FLAIR | MC-GS | 0.2205 | 0.2205 | -0.0000 |
| FLAIR | GS-single | 0.2161 | 0.2161 | +0.0001 |

