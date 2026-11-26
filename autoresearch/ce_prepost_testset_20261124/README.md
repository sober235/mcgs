# Plain-scan / enhanced test set (3D_PE_20261119): truth by traditional reconstruction, then AF12

User, 2026-11-24:
- "Here in the GPU server, there are more plain-scan and enhanced test sets. You can use all of them as the test sets
  to conduct the dataset tests: /data2/congcong/3D/3D_PE_20261119/" (actual path `/data2/congcong/data/3D/3D_PE_20261119`).
- Choices: start with the subjects whose multi-coil k-space is readable for both scans; MC-GS = the tuned final
  config of `../ce_af12_tune_20261122` (held-out check of that tuning).
- "You know that you need to first perform the traditional reconstruction to obtain the complete image, and then
  perform the 12-fold downsampling?" -- yes: the acquisitions are prospectively undersampled (about AF 5.3), so the
  AF12 mask asks for points that were never measured. As for the record subject, a traditional reconstruction of all
  acquired data is the complete image (truth); k-space is re-simulated from it and then sampled at AF12.

## Subjects

| subject | plain scan (pre) | enhanced (post, `<IsInjection>1`) | k-space used | coils |
|---|---|---|---|---|
| 9000103445 | 16884476230000 | 16884476230050 | `rawdata_T1_turmor_*` / `rawdata_CE_turmor_*` (cfl) | 42 |
| 9002944080 | 16886244540000 | 16886244540050 | `rawdata` (cfl) | 50 |
| 9003136585 | 16886147050000 | 16886147050050 | `T1` / `T1_CE` (cfl) | 18 |
| 9003135093 | 16887049370010 | 16887049370060 | record subject (recon_real work dirs); MC-GS was tuned on it, reported apart | 42 |

Not yet usable (UIH `.raw` only, or only one scan converted): 9003065128, 9003122771, 9003128786, 9003141620,
9003145528, 9003137160, 9003137288. They are added when their k-space exists as multi-coil cfl.

## Pipeline per subject (plain scan and enhanced each)

0. `probe_cases.py` (CPU, read only): array layout, k-space centre, prospective mask (AF, ACS), plain / enhanced
   label, zero-filled images.
1. Prepare: k-space and prospective mask in the record's layout (`ksp_full.npy` (coil, x, ky, kz), `mask2d.npy`);
   coil maps by ESPIRiT from the fully sampled centre.
2. Truth = CS-SENSE of all acquired data with the scripts of `Results_20260908/CE_pre_post_contrast` (ADMM, 200 outer
   iterations). lambda is chosen per scan exactly as for the record subject: CS-SENSE on 95 % of the acquired lines
   (split seed 0) for the grid 2.5e-4 ... 4e-3, the lambda with the lowest residual on the 5 % held-out lines wins
   (grid extended at an edge), then refit on all lines. No truth is involved. (The record's 2e-3 is not reused:
   `Results_20260908/CE_retro_gt` found the record lambda far too strong on other data.)
3. AF12 simulation as in `Results_20260908/CE_retro_cssense_af16` / `CE_retro_cssense_af12`: k-space = F(S x_truth)
   plus complex Gaussian noise at the level of the acquired data (sigma from the residual on the prospective mask),
   BART Poisson-disc mask, seed 0, ACS 24 x 24, exactly AF 12 on that subject's ky x kz grid, one mask for both scans.
4. Eleven methods, settings frozen as in the AF12 topic; MC-GS = tuned final config, seed 42.
5. Head-masked PSNR / SSIM / NMSE / HFEN against the truth; per-subject table and figure.

GPU work starts only on a free card within the project cap (4 cards for all mcgs sessions together).
