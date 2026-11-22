# 逐病例结果表（头部掩膜内 PSNR / SSIM / NMSE，全体积）

每个（扫描，加速倍数）一张表，T1 与 FLAIR 并排；粗体 = 该列最佳；`—` = 该方法的结果尚缺（见 table.json 的 missing）；
不完整的格子标注“不完整”，不计入名次与统计。NMSE 为头部 NMSE（越小越好）。

## vol1_huangyuwei，AF12（aligned组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | 23.703 | 0.6699 | 0.14297 | 26.738 | 0.6882 | 0.06751 |
| CG-SENSE | 31.941 | 0.8981 | 0.02145 | 32.192 | 0.8834 | 0.01923 |
| CS-SENSE | 35.182 | 0.9467 | 0.01017 | 33.873 | 0.9195 | 0.01306 |
| Joint SENSE | 34.851 | 0.9439 | 0.01097 | 33.929 | 0.9196 | 0.01289 |
| L1-SPIRiT | 33.738 | 0.9285 | 0.01418 | 33.423 | 0.9034 | 0.01448 |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | 36.053 | 0.9510 | 0.00832 | 34.743 | 0.9253 | 0.01069 |
| **MC-GS** | **36.200** | **0.9531** | **0.00804** | **34.773** | **0.9260** | **0.01061** |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol1_huangyuwei，AF16（aligned组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | 23.418 | 0.6470 | 0.15265 | 26.432 | 0.6653 | 0.07243 |
| CG-SENSE | 30.549 | 0.8794 | 0.02955 | 31.047 | 0.8617 | 0.02503 |
| CS-SENSE | 33.770 | 0.9323 | 0.01408 | 32.550 | 0.8979 | 0.01771 |
| Joint SENSE | 33.439 | 0.9292 | 0.01519 | 32.665 | 0.8988 | 0.01724 |
| L1-SPIRiT | 32.705 | 0.9104 | 0.01799 | 32.110 | 0.8804 | 0.01960 |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | 34.700 | 0.9390 | 0.01136 | 33.271 | 0.9059 | 0.01500 |
| **MC-GS** | **34.866** | **0.9418** | **0.01094** | **33.317** | **0.9073** | **0.01484** |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol1_huangyuwei，AF20（aligned组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | 23.212 | 0.6282 | 0.16009 | 26.208 | 0.6481 | 0.07626 |
| CG-SENSE | 29.453 | 0.8595 | 0.03803 | 30.327 | 0.8446 | 0.02955 |
| CS-SENSE | 32.669 | 0.9180 | 0.01814 | 31.733 | 0.8820 | 0.02137 |
| Joint SENSE | 32.345 | 0.9146 | 0.01955 | 31.846 | 0.8826 | 0.02082 |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | 33.747 | 0.9273 | 0.01415 | 32.546 | 0.8915 | 0.01773 |
| **MC-GS** | **33.939** | **0.9310** | **0.01354** | **32.595** | **0.8934** | **0.01752** |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol1_huangyuwei，AF24（aligned组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | 22.916 | 0.6024 | 0.17138 | 25.884 | 0.6225 | 0.08218 |
| CG-SENSE | 28.230 | 0.8320 | 0.05042 | 29.540 | 0.8203 | 0.03541 |
| CS-SENSE | 31.274 | 0.8946 | 0.02501 | 30.823 | 0.8588 | 0.02636 |
| Joint SENSE | 30.987 | 0.8912 | 0.02672 | 30.908 | 0.8585 | 0.02584 |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | 32.596 | 0.9107 | 0.01845 | 31.559 | 0.8701 | 0.02225 |
| **MC-GS** | **32.795** | **0.9154** | **0.01762** | **31.625** | **0.8726** | **0.02191** |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol2_jiangdian，AF12（aligned组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | 36.851 | 0.9556 | 0.00815 | 36.044 | 0.9332 | 0.01098 |
| **MC-GS** | **37.010** | **0.9578** | **0.00785** | **36.060** | **0.9332** | **0.01094** |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol2_jiangdian，AF16（aligned组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | 35.497 | 0.9445 | 0.01113 | 34.636 | 0.9157 | 0.01518 |
| **MC-GS** | **35.664** | **0.9474** | **0.01071** | **34.665** | **0.9164** | **0.01508** |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol2_jiangdian，AF20（aligned组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol2_jiangdian，AF24（aligned组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol4_tangbing，AF12（aligned组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol4_tangbing，AF16（aligned组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol4_tangbing，AF20（aligned组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol4_tangbing，AF24（aligned组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol5_sunxinyi，AF12（offset组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol5_sunxinyi，AF16（offset组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol5_sunxinyi，AF20（offset组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol5_sunxinyi，AF24（offset组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol6_jiangdian_790，AF12（offset组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol6_jiangdian_790，AF16（offset组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol6_jiangdian_790，AF20（offset组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol6_jiangdian_790，AF24（offset组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol7_tangbin_790，AF12（offset组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol7_tangbin_790，AF16（offset组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol7_tangbin_790，AF20（offset组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-

## vol7_tangbin_790，AF24（offset组）（不完整）

| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |
|---|---|---|---|---|---|---|
| ZF | — | — | — | — | — | — |
| CG-SENSE | — | — | — | — | — | — |
| CS-SENSE | — | — | — | — | — | — |
| Joint SENSE | — | — | — | — | — | — |
| L1-SPIRiT | — | — | — | — | — | — |
| ZS-SSDU | — | — | — | — | — | — |
| 3D-DIP | — | — | — | — | — | — |
| 3D-INR | — | — | — | — | — | — |
| MC-INR | — | — | — | — | — | — |
| GS-single | — | — | — | — | — | — |
| **MC-GS** | — | — | — | — | — | — |

MC-GS 名次（PSNR/SSIM/NMSE）：T1 -/-/-；FLAIR -/-/-
