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
| MC-GS joint v3.1 | **37.197** | 0.8938 | **33.929** | 0.9301 | 56 min |
| MC-GS joint v3 | 37.083 | 0.8945 | 33.825 | 0.9281 | 56 min |
| **MC-GS joint v3.2** | 37.074 | 0.8938 | 33.791 | 0.9275 | 56 min |
| CS-SENSE (TV, converged) | 37.077 | 0.9307 | 33.580 | **0.9319** | 5 min |
| joint CS-SENSE (JTV) | 36.886 | 0.9397 | 33.419 | 0.9286 | 5 min |
| ZS-SSDU | 36.674 | **0.9445** | 33.091 | 0.9297 | 40 min |
| CS-SENSE (TV, 35 outer, as run 09-09) | 36.591 | 0.9302 | 33.058 | 0.9274 | 3 min |
| L1-ESPIRiT | 33.082 | 0.7309 | 31.865 | 0.8786 | 8 min |
| CG-SENSE | 32.446 | 0.7154 | 30.686 | 0.8684 | 2 min |
| 3D-INR | 32.103 | 0.7660 | 29.261 | 0.8027 | 60 min |
| 3D-DIP | 32.448 | 0.7961 | 28.661 | 0.8321 | 53 min |
| ZF | 27.281 | 0.7886 | 23.284 | 0.6524 | 2 min |

**FLAIR**:

| method | PSNR | SSIM | head PSNR | head SSIM | wall |
|---|---|---|---|---|---|
| **MC-GS joint v3.2** | **35.857** | 0.8571 | **32.591** | 0.9050 | 56 min |
| MC-GS joint v3 | 35.631 | 0.8673 | 32.403 | 0.9005 | 56 min |
| MC-GS joint v3.1 | 35.631 | 0.8675 | 32.403 | 0.9004 | 56 min |
| L1-ESPIRiT | 35.722 | 0.9212 | 32.351 | **0.9081** | 8 min |
| joint CS-SENSE (JTV) | 35.675 | 0.9215 | 32.271 | 0.9063 | 7 min |
| CS-SENSE (TV, converged) | 35.460 | 0.9130 | 32.180 | 0.9054 | 4 min |
| CS-SENSE (TV, 35 outer, as run 09-09) | 35.129 | 0.9118 | 31.678 | 0.8997 | 2 min |
| ZS-SSDU | 35.057 | **0.9250** | 31.530 | 0.9047 | 22 min |
| CG-SENSE | 32.356 | 0.7188 | 30.705 | 0.8604 | 2 min |
| 3D-INR | 30.893 | 0.7257 | 28.016 | 0.7617 | 60 min |
| 3D-DIP | 31.624 | 0.7419 | 27.964 | 0.7859 | 53 min |
| ZF | 29.199 | 0.7905 | 25.528 | 0.6666 | 2 min |

## MC-GS joint v3.2 — T1-channel wavelet 0.06 and lighter edge-TV 0.07 (2026-09-11)

v3.2 = v3 + a T1-channel wavelet strength of 0.06 (the second channel keeps 0.05) + edge-TV 0.07 instead of 0.10; on the
T1+FLAIR topics the FLAIR channel takes 0.12. Selected on the tuning pair by the pre-registered rules of the framework
program (Rounds 16–17, "## v3.2 — adopted"). Second held-out subject (Round 21): nothing in v3, v3.1 or v3.2 was selected on it.

```bash
cd $F && CONFIG=configs/t1flair_v4_af12_hyw.yaml SAVE_VOLUMES=1 ./autoresearch_t1t2.sh ar21_tf_v4_v32 0 4000 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=2400000 --set loss.lambda_wav=0.05 --set loss.lambda_wav_plain=0.06 --set loss.lambda_wav_contrast=0.12 --lambda_geo 0.07
```

Standing of v3.2 against every other method in this topic's table (single run, seed 42):

| metric | T1 | FLAIR |
|---|---|---|
| head PSNR | **first**, 33.791 (next: converged TV 33.580) | **first**, 32.591 (next: L1-ESPIRiT 32.351) |
| head SSIM | fourth, 0.9275 (best: converged TV 0.9319) | fourth, 0.9050 (best: L1-ESPIRiT 0.9081) |
| head HFEN | **first**, 0.1110 (next: converged TV 0.1274) | **first**, 0.1191 (next: L1-ESPIRiT 0.1193) |
| whole-volume PSNR | second, 37.074 (best: converged TV 37.077) | **first**, 35.857 (next: L1-ESPIRiT 35.722) |
| whole-volume NMSE | second, 0.01455 (best: converged TV 0.01454) | **first**, 0.01573 (next: L1-ESPIRiT 0.01623) |
| whole-volume HFEN | fourth, 0.6220 (best: converged TV 0.5974) | **first**, 0.6334 (next: joint CS-SENSE 0.6472) |
| whole-volume SSIM | fifth, 0.8938 (best: ZS-SSDU 0.9445) | sixth, 0.8571 (best: ZS-SSDU 0.9250) |

| arm | head PSNR T1 / FLAIR | head SSIM | head HFEN | whole PSNR | sharp |
|---|---|---|---|---|---|
| **MC-GS joint v3.2** | 33.791 / 32.591 | 0.9275 / 0.9050 | 0.1110 / 0.1191 | 37.074 / 35.857 | 0.905 / 0.824 |
| MC-GS joint v3.1 (T1 wavelet 0.06 only) | 33.929 / 32.403 | 0.9301 / 0.9004 | 0.1122 / 0.1189 | 37.197 / 35.631 | 0.869 / 0.849 |
| MC-GS joint v3 | 33.825 / 32.403 | 0.9281 / 0.9005 | 0.1129 / 0.1189 | 37.083 / 35.631 | 0.876 / 0.849 |
| L1-ESPIRiT | 31.865 / 32.351 | 0.8786 / 0.9081 | 0.1324 / 0.1193 | 33.082 / 35.722 | 1.097 / 0.800 |
| converged TV (best other head SSIM) | 33.580 / 32.180 | 0.9319 / 0.9054 | 0.1274 / 0.1278 | 37.077 / 35.460 | 0.828 / 0.780 |

- **T1**: first on head PSNR, head HFEN; not first on head SSIM (fourth), whole-volume PSNR (second), whole-volume NMSE (second), whole-volume HFEN (fourth), whole-volume SSIM (fifth).
- **FLAIR**: first on head PSNR, head HFEN, whole-volume PSNR, whole-volume NMSE, whole-volume HFEN; not first on head SSIM (fourth), whole-volume SSIM (sixth).
- **Against v3**: T1 -0.034 dB / head SSIM -0.0006, FLAIR +0.187 dB / head SSIM +0.0045.
- **Gate**: T1 sharpness 0.905 passes the pre-registered 0.874 floor.
- **Per slice** (`scripts/find_best_slice_v2.py`, full-brain slices, against every other arm; P-LORAKS not yet in on every channel): T1 first on 94/129, 128/128, 104/129 transverse / coronal / sagittal slices; FLAIR first on 120/120, 130/130, 128/128 transverse / coronal / sagittal slices.
- Figures (compact panels, v3.2 as the proposed method): `figures/t1flair_v4_af12_T1_best_axis0_170.png`, `figures/t1flair_v4_af12_FLAIR_best_axis0_71.png`, `figures/t1flair_v4_af12_{T1,FLAIR}_best3planes.png`.

## Regenerating

```bash
R=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v4_tangbing
bash $R/scripts/prep_03_csm.sh          # both maps, CPU
# then the framework repo's scratchpad round21_arms.sh (export, map conversion, every arm on GPUs 0-5, L1-ESPIRiT on CPU),
# followed by scripts/extract_gs_volumes.py, scripts/summarize.py and ../T1T2_AF12_retro/scripts/refresh_topic.sh
```
