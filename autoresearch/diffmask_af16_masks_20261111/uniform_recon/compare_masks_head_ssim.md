# Diff-mask study: every arm on three sampling patterns, one gauge (head_ssim)

Same subject, same k-space, same coil maps, same ground truth, same scoring code, same frozen AF16 phase-3
MC-GS recipe; matched point budget (2964 points, T1 15.20x; the house mask 2966, 15.19x). Only the sampling
PATTERN differs, so every difference across a row is the pattern's effect.

**T1** (head_ssim; higher is better unless the metric says otherwise)

| arm | vd-Poisson (house) | Uniform density | radial | Uniform density − vd-Poisson | radial − vd-Poisson |
|---|---|---|---|---|---|
| MC-GS | 0.940 | 0.915 | 0.870 | -0.02 | -0.07 |
| GS-single | 0.937 | 0.904 | 0.863 | -0.03 | -0.07 |
| ZS-SSDU | 0.922 | 0.879 | 0.862 | -0.04 | -0.06 |
| CS-SENSE (TV, converged) | 0.932 | 0.847 | 0.845 | -0.08 | -0.09 |
| joint CS-SENSE (JTV) | 0.931 | 0.846 | 0.846 | -0.09 | -0.09 |
| CS-SENSE | 0.928 | 0.835 | 0.844 | -0.09 | -0.08 |
| L1-SPIRiT | 0.908 | 0.867 | 0.824 | -0.04 | -0.08 |
| CG-SENSE | 0.882 | 0.771 | 0.814 | -0.11 | -0.07 |
| MC-INR | 0.835 | 0.672 | 0.784 | -0.16 | -0.05 |
| 3D-DIP | 0.855 | 0.790 | 0.813 | -0.06 | -0.04 |
| 3D-INR | 0.795 | 0.756 | 0.700 | -0.04 | -0.10 |
| ZF | 0.679 | 0.565 | 0.645 | -0.11 | -0.03 |

Best arm per pattern: **vd-Poisson** MC-GS 0.940;  **Uniform density** MC-GS 0.915;  **radial** MC-GS 0.870

**FLAIR** (head_ssim; higher is better unless the metric says otherwise)

| arm | vd-Poisson (house) | Uniform density | radial | Uniform density − vd-Poisson | radial − vd-Poisson |
|---|---|---|---|---|---|
| MC-GS | 0.926 | 0.917 | 0.867 | -0.01 | -0.06 |
| GS-single | 0.925 | 0.913 | 0.862 | -0.01 | -0.06 |
| ZS-SSDU | 0.918 | 0.916 | 0.869 | -0.00 | -0.05 |
| CS-SENSE (TV, converged) | 0.918 | 0.866 | 0.846 | -0.05 | -0.07 |
| joint CS-SENSE (JTV) | 0.918 | 0.865 | 0.842 | -0.05 | -0.08 |
| CS-SENSE | 0.912 | 0.865 | 0.845 | -0.05 | -0.07 |
| L1-SPIRiT | 0.902 | 0.886 | 0.819 | -0.02 | -0.08 |
| CG-SENSE | 0.886 | 0.822 | 0.820 | -0.06 | -0.07 |
| MC-INR | 0.859 | 0.752 | 0.797 | -0.11 | -0.06 |
| 3D-DIP | 0.829 | 0.793 | 0.814 | -0.04 | -0.01 |
| 3D-INR | 0.789 | 0.735 | 0.695 | -0.05 | -0.09 |
| ZF | 0.713 | 0.615 | 0.685 | -0.10 | -0.03 |

Best arm per pattern: **vd-Poisson** MC-GS 0.926;  **Uniform density** MC-GS 0.917;  **radial** ZS-SSDU 0.869

## MC-GS against the strongest comparison arm, per pattern

| contrast | pattern | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | vd-Poisson | 0.940 | GS-single 0.937 | +0.003 |
| T1 | Uniform density | 0.915 | GS-single 0.904 | +0.011 |
| T1 | radial | 0.870 | GS-single 0.863 | +0.007 |
| FLAIR | vd-Poisson | 0.926 | GS-single 0.925 | +0.001 |
| FLAIR | Uniform density | 0.917 | ZS-SSDU 0.916 | +0.000 |
| FLAIR | radial | 0.867 | ZS-SSDU 0.869 | -0.001 |

