# Diff-mask study: every arm on three sampling patterns, one gauge (head_ssim)

Same subject, same k-space, same coil maps, same ground truth, same scoring code, same frozen AF16 phase-3
MC-GS recipe; matched point budget (2964 points, T1 15.20x; the house mask 2966, 15.19x). Only the sampling
PATTERN differs, so every difference across a row is the pattern's effect.

**T1** (head_ssim; higher is better unless the metric says otherwise)

| arm | vd-Poisson (house) | Gaussian | radial | Gaussian − vd-Poisson | radial − vd-Poisson |
|---|---|---|---|---|---|
| MC-GS | 0.940 | 0.933 | 0.870 | -0.01 | -0.07 |
| GS-single | 0.937 | 0.931 | 0.863 | -0.01 | -0.07 |
| ZS-SSDU | 0.922 | 0.927 | 0.862 | +0.01 | -0.06 |
| CS-SENSE (TV, converged) | 0.932 | 0.919 | 0.845 | -0.01 | -0.09 |
| joint CS-SENSE (JTV) | 0.931 | 0.919 | 0.846 | -0.01 | -0.09 |
| CS-SENSE | 0.928 | 0.917 | 0.844 | -0.01 | -0.08 |
| L1-SPIRiT | 0.908 | 0.901 | 0.824 | -0.01 | -0.08 |
| CG-SENSE | 0.882 | 0.878 | 0.814 | -0.00 | -0.07 |
| MC-INR | 0.835 | 0.846 | 0.784 | +0.01 | -0.05 |
| 3D-DIP | 0.855 | 0.855 | 0.813 | -0.00 | -0.04 |
| 3D-INR | 0.795 | 0.766 | 0.700 | -0.03 | -0.10 |
| ZF | 0.679 | 0.652 | 0.645 | -0.03 | -0.03 |

Best arm per pattern: **vd-Poisson** MC-GS 0.940;  **Gaussian** MC-GS 0.933;  **radial** MC-GS 0.870

**FLAIR** (head_ssim; higher is better unless the metric says otherwise)

| arm | vd-Poisson (house) | Gaussian | radial | Gaussian − vd-Poisson | radial − vd-Poisson |
|---|---|---|---|---|---|
| MC-GS | 0.926 | 0.921 | 0.867 | -0.01 | -0.06 |
| GS-single | 0.925 | 0.920 | 0.862 | -0.00 | -0.06 |
| ZS-SSDU | 0.918 | 0.920 | 0.869 | +0.00 | -0.05 |
| CS-SENSE (TV, converged) | 0.918 | 0.912 | 0.846 | -0.01 | -0.07 |
| joint CS-SENSE (JTV) | 0.918 | 0.911 | 0.842 | -0.01 | -0.08 |
| CS-SENSE | 0.912 | 0.908 | 0.845 | -0.00 | -0.07 |
| L1-SPIRiT | 0.902 | 0.892 | 0.819 | -0.01 | -0.08 |
| CG-SENSE | 0.886 | 0.883 | 0.820 | -0.00 | -0.07 |
| MC-INR | 0.859 | 0.860 | 0.797 | +0.00 | -0.06 |
| 3D-DIP | 0.829 | 0.839 | 0.814 | +0.01 | -0.01 |
| 3D-INR | 0.789 | 0.774 | 0.695 | -0.01 | -0.09 |
| ZF | 0.713 | 0.696 | 0.685 | -0.02 | -0.03 |

Best arm per pattern: **vd-Poisson** MC-GS 0.926;  **Gaussian** MC-GS 0.921;  **radial** ZS-SSDU 0.869

## MC-GS against the strongest comparison arm, per pattern

| contrast | pattern | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | vd-Poisson | 0.940 | GS-single 0.937 | +0.003 |
| T1 | Gaussian | 0.933 | GS-single 0.931 | +0.003 |
| T1 | radial | 0.870 | GS-single 0.863 | +0.007 |
| FLAIR | vd-Poisson | 0.926 | GS-single 0.925 | +0.001 |
| FLAIR | Gaussian | 0.921 | GS-single 0.920 | +0.001 |
| FLAIR | radial | 0.867 | ZS-SSDU 0.869 | -0.001 |

## MC-GS minus GS-single (the coupling gain), per pattern

| contrast | pattern | MC-GS | GS-single | gain |
|---|---|---|---|---|
| T1 | vd-Poisson | 0.940 | 0.937 | +0.003 |
| T1 | Gaussian | 0.933 | 0.931 | +0.003 |
| T1 | radial | 0.870 | 0.863 | +0.007 |
| FLAIR | vd-Poisson | 0.926 | 0.925 | +0.001 |
| FLAIR | Gaussian | 0.921 | 0.920 | +0.001 |
| FLAIR | radial | 0.867 | 0.862 | +0.005 |

