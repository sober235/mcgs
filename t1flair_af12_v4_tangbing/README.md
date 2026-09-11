# T1+FLAIR, retrospective AF12 — tangbing (MC_Jia vol4), the second held-out subject

## What this topic adds

A third person, held out from every choice in the MC-GS autoresearch campaign (framework repo
`autoresearch_t1t2.md`, Round 21, pre-registered before any of this subject's data was read). Its own T1 and FLAIR
k-space and its own two ESPIRiT maps (house recipe `ecalib -t 0.001 -c 0.0 -r 24 -m 1`); the same vd-Poisson AF12 mask
and geometry as the other subjects (the mask file is byte-identical across topics). Created with
`../T1FLAIR_AF12_retro/scripts/setup_subject.sh vol4_tangbing tangbing v4`.

Geometry matches the other subjects to the voxel: T1 support 1.0000, FLAIR support 0.7577 (elliptical shutter), DC on the array centre, AF 11.71 on T1 and 9.74 on FLAIR against their own support; both ESPIRiT maps have SoS == 1 over the FOV. T1/FLAIR co-registration: peak cross-correlation shift (0, 0, 0), brain-mask Dice 0.901.

Every comparison arm runs at the settings fixed on the other topics, nothing tuned here: CG-SENSE, converged TV CS-SENSE
(200 ADMM outer iterations, λ 1e-3 on T1, 7e-4 on FLAIR), joint CS-SENSE (200 outer, λ 1e-3, μ 0.1), L1-ESPIRiT (BART,
100 iterations, λ 0.0015 on T1, 0.005 on FLAIR, selected on huangyuwei), ZS-SSDU (100 epochs), 3D-DIP (5000 iterations),
3D-INR (1500 iterations). P-LORAKS is not run here (7 h of CPU per channel; last on every other topic). Three MC-GS
joint recipes run unchanged: v3, v3.1 (T1 wavelet 0.06) and v3.2 (T1 wavelet 0.06, FLAIR wavelet 0.12, edge-TV 0.07).

**Reading this topic.** Head PSNR here sits 2.7–3.4 dB below jiangyizhou's for every arm, zero-filling and CG-SENSE
included. Most of that offset is the PSNR peak, not reconstruction difficulty: this subject's max/P99.9 is 1.75 on T1
and 1.77 on FLAIR against jiangyizhou's 2.29 and 2.27, worth about 2.3 dB. The ground truth is clean (no visible motion
or ghosting) and its air background has the same noise floor as jiangyizhou's. Compare arms within this topic, not
absolute PSNR across subjects. At the λ selected on huangyuwei, L1-ESPIRiT is visibly under-regularized on this
subject's T1 (grainy, sharpness 1.10) and falls behind TV; every arm runs at its fixed setting, MC-GS included.

## Results

**T1** — sorted by head-only PSNR (head = |GT| > 0.05 P99.9):

| method | PSNR | SSIM | head PSNR | head SSIM | wall |
|---|---|---|---|---|---|
| (pending) | | | | | |

**FLAIR**:

| method | PSNR | SSIM | head PSNR | head SSIM | wall |
|---|---|---|---|---|---|
| (pending) | | | | | |

## Regenerating

```bash
R=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v4_tangbing
bash $R/scripts/prep_03_csm.sh          # both maps, CPU
# then the framework repo's scratchpad round21_arms.sh (export, map conversion, every arm on GPUs 0-5, L1-ESPIRiT on CPU),
# followed by scripts/extract_gs_volumes.py, scripts/summarize.py and ../T1T2_AF12_retro/scripts/refresh_topic.sh
```
