
**T1** (sorted by head-only PSNR; vd-Poisson column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @vd-Poisson | this − vd-Poisson |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 35.047 | 0.9011 | **31.113** | **0.8702** | 0.3417 | 0.635 | 35.706 | -4.59 |
| GS-single | 34.840 | 0.9005 | 30.913 | 0.8629 | 0.3587 | 0.631 | 35.558 | -4.65 |

**FLAIR** (sorted by head-only PSNR; vd-Poisson column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @vd-Poisson | this − vd-Poisson |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 35.625 | 0.9012 | **31.729** | **0.8673** | 0.2809 | 0.615 | 34.928 | -3.20 |
| GS-single | 35.459 | 0.8833 | 31.579 | 0.8624 | 0.2991 | 0.617 | 34.886 | -3.31 |
| CS-SENSE | 34.829 | 0.8955 | 30.932 | 0.8455 | 0.3545 | 0.547 | 33.806 | -2.87 |
| CG-SENSE | 33.642 | 0.8065 | 30.301 | 0.8200 | 0.4060 | 0.642 | 32.826 | -2.53 |
| ZF | 31.230 | 0.8248 | 27.377 | 0.6851 | 0.7194 | 0.286 | 27.879 | -0.50 |

**MC-GS joint minus MC-GS single**

| channel | whole PSNR | head PSNR | head SSIM | (vd-Poisson: whole / head) |
|---|---|---|---|---|
