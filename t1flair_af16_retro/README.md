# Retrospective AF 16 on the 3D T1 + FLAIR pair — same subject and data as the AF 12 study

Fourth topic in the series. **Read `../T1T2_AF12_retro/README.md` first** (data provenance, the
geometry adjudication, the ESPIRiT recipe, the known-zero region, the eight methods and the scoring
gauge), then `../T1FLAIR_AF12_retro/README.md` (this pairing on this subject at AF 12). Both are
documented there and not repeated. `../T1FLAIR_AF16_v3_jiangyizhou/` is this study on the second
subject.

## What differs from `T1FLAIR_AF12_retro`

Exactly one thing: **the undersampling mask**. Same subject (`vol1_huangyuwei`), same CFL k-space,
same two ESPIRiT maps, same ground truth, same frozen working point, same seed, same gauge, same
eight arms with the same hyper-parameters. The k-space and the maps are not copied: `data/ksp_*.h5`
and `data/csm_espirit_*.h5` are **symlinks into the AF 12 topic**, and `scripts/prep_04_export.py`
was changed so it can never write through them (it asserts the symlink and only writes the mask).
Nothing is reused from the AF 12 results: every arm depends on the mask, so all sixteen
reconstructions here are new.

| mask | generator call | PE points | of the plane | AF, full plane | T1 AF vs own support | FLAIR AF vs own support |
|---|---|---|---|---|---|---|
| AF 12 (`../T1FLAIR_AF12_retro`) | `vdPoisMex(256,176,256,176, 3, 4, 24, 0, 1)` | 3847 | 8.54 % | 11.71 | 11.71 | 9.74 |
| **AF 16 (this topic)** | `vdPoisMex(256,176,256,176, 4, 4, 24, 0, 1)` | **2966** | **6.58 %** | **15.19** | **15.19** | **12.49** |

Same house generator and convention (`accely = 4`, `accelx = AF/4`, `calib = 24`, `ellipse = 0`,
`pp = 1`): a nominal AF 16 lands at 15.19 on the full plane the way the nominal AF 12 landed at
11.71. FLAIR keeps its elliptical vendor shutter (support 0.7577), so against the data it actually
acquires its acceleration is 12.49. One mask for both contrasts and for both subjects
(`data/mask_vdpois_256x176_af16.mat`, md5 `2d7ed000…` in both topics). The 24x24 calibration
block keeps 81.9 % of its points.

## How it was run (matters for reading the wall times, not the numbers)

- Code: git worktree `/data0/congcong/code/GS/mcgs_af16_wt`, branch `t1flair-af16-retro`, checked
  out at `30e6775` — the exact commit the AF 12 topics were archived on — so that a concurrent
  autoresearch loop in the main checkout could keep editing the trainer without touching this
  study. The MC-GS runs, their `results_arc/`, and the JIT-built CUDA voxelizer
  (`TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_af16_wt.torch_ext`) all live with the
  worktree. New configs: `configs/t1flair_af16_hyw{,_singleT1,_singleFLAIR}.yaml`.
- GPUs: only cards 0 and 1 (2–7 belonged to the autoresearch loop), and another user's training
  job held ~33 GB and most of the compute on both cards until 01:29. `scripts/sched_af16.py` runs
  the thirty jobs of both subjects on the two cards: strict priority (classical → MC-GS ×3 →
  ZS-SSDU for every subject, then DIP / INR), a per-job VRAM need, at most two jobs per card, a
  card that drains for the next heavy job accepts only a filler short enough to finish first, and
  every job is skipped if its result file already exists or adopted if it is already running, so
  the scheduler can be restarted at any time. **Wall times in the logs are therefore not comparable
  with the AF 12 topic** (MC-GS joint took 137 min here against 25 min alone on a card; the same
  run on the second subject, after the other user left, took 29 min).
- Gauge: `scripts/check_gauge.py --contrast T1 --method cgsense` reproduces the classical numbers
  to the last digit (`logs/post_gauge.log`: `GAUGE OK`), as in the AF 12 topics.

## Results

`logs/summary.tsv`, one gauge (whole@max PSNR/SSIM + the head-only set, `|GT| > 0.05 P99.9`).
Head-only PSNR / SSIM are the numbers to quote (see the T1+T2 README on why whole-volume SSIM
misleads on this data). Per-arm AF 12 values come from `../T1FLAIR_AF12_retro/logs/summary.json`.

<!-- TABLES:BEGIN -->
**T1** (sorted by head-only PSNR; AF12 column = the same arm in `T1FLAIR_AF12_retro`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @AF12 | AF16 − AF12 |
|---|---|---|---|---|---|---|---|---|
| ZS-SSDU | 38.501 | 0.9562 | **33.665** | **0.9328** | 0.1430 | 0.982 | 34.134 | -0.47 |
| CS-SENSE | 37.911 | 0.9138 | 33.082 | 0.9270 | 0.1384 | 0.771 | 34.607 | -1.52 |
| MC-GS joint | 35.401 | 0.8594 | 32.028 | 0.8933 | 0.1592 | 0.914 | 33.181 | -1.15 |
| MC-GS single | 34.406 | 0.7901 | 31.978 | 0.8907 | 0.1470 | 0.948 | 33.296 | -1.32 |
| CG-SENSE | 33.875 | 0.7350 | 30.549 | 0.8794 | 0.2175 | 0.824 | 31.941 | -1.39 |
| 3D-DIP | 34.948 | 0.8433 | 29.910 | 0.8488 | 0.2597 | 0.670 | 30.106 | -0.20 |
| 3D-INR | 32.811 | 0.7197 | 29.281 | 0.7941 | 0.1759 | 1.332 | 30.838 | -1.56 |
| ZF | 28.787 | 0.8232 | 23.418 | 0.6470 | 0.6655 | 0.291 | 23.703 | -0.28 |

**FLAIR** (sorted by head-only PSNR; AF12 column = the same arm in `T1FLAIR_AF12_retro`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @AF12 | AF16 − AF12 |
|---|---|---|---|---|---|---|---|---|
| ZS-SSDU | 37.501 | 0.9441 | **32.685** | **0.9047** | 0.1517 | 1.020 | 33.269 | -0.58 |
| CS-SENSE | 36.773 | 0.9177 | 31.974 | 0.8892 | 0.1544 | 0.651 | 33.252 | -1.28 |
| CG-SENSE | 34.242 | 0.7487 | 31.047 | 0.8617 | 0.1806 | 0.746 | 32.192 | -1.14 |
| MC-GS joint | 34.517 | 0.8435 | 30.743 | 0.8598 | 0.1938 | 0.795 | 31.806 | -1.06 |
| MC-GS single | 33.999 | 0.8083 | 30.479 | 0.8463 | 0.1934 | 0.895 | 31.755 | -1.28 |
| 3D-INR | 32.759 | 0.7082 | 29.174 | 0.7786 | 0.2146 | 1.029 | 29.939 | -0.77 |
| 3D-DIP | 33.998 | 0.7933 | 29.069 | 0.7945 | 0.2771 | 0.506 | 29.397 | -0.33 |
| ZF | 31.370 | 0.8391 | 26.432 | 0.6653 | 0.6620 | 0.252 | 26.738 | -0.31 |

**MC-GS joint minus MC-GS single**

| channel | whole PSNR | head PSNR | head SSIM | (AF12: whole / head) |
|---|---|---|---|---|
| T1 | +0.995 | +0.050 | +0.0026 | +0.891 / -0.114 |
| FLAIR | +0.518 | +0.265 | +0.0135 | +0.290 / +0.051 |
<!-- TABLES:END -->

### Reading

1. **Sixteen-fold costs every regularised method 1.1 to 1.5 dB inside the head, except ZS-SSDU.**
   CS-SENSE loses 1.52 (T1) / 1.28 (FLAIR) dB, CG-SENSE 1.39 / 1.14, MC-GS single 1.32 / 1.28,
   MC-GS joint 1.15 / 1.06. ZS-SSDU loses only 0.47 / 0.58 dB and moves from a tie with CS-SENSE at
   AF 12 to a clear first place on both contrasts (+0.58 dB on T1, +0.71 dB on FLAIR over CS-SENSE).
   The self-supervised unrolled network is the arm most robust to the sparser mask; the hand-crafted
   priors (TV, the Gaussian representation) are the ones that pay.
2. **MC-GS joint is the arm that degrades least among the non-learned ones.** Its gap to CS-SENSE
   narrows from 1.43 / 1.44 dB at AF 12 to 1.05 / 1.23 dB, but the gap to the new leader is
   1.64 / 1.94 dB. It moves from fourth to third of eight on T1 (passing its own single arm) and
   stays fourth on FLAIR, where CG-SENSE remains 0.30 dB ahead of it (0.39 dB at AF 12).
3. **The shared geometry gains inside the head at AF 16, in both channels** — the first
   channel-pairing in the series where that happens: joint − single = +0.05 dB (T1) and +0.27 dB
   (FLAIR) head-only, +0.003 / +0.014 head SSIM, against −0.11 / +0.05 dB at AF 12. The whole-volume
   gain is unchanged in kind (+1.0 / +0.5 dB, background suppression). One subject and one seed; the
   second subject's table is in `../T1FLAIR_AF16_v3_jiangyizhou/`.
4. Per-slice sweep (`logs/per_slice.json`, `logs/per_slice_sweep.log`, all eight arms): MC-GS joint
   ranks first on 12 of 556 scored T1 slices and 0 of 570 FLAIR slices; ZS-SSDU wins 503 / 403 of
   them, CS-SENSE 30 / 162. The slices the joint model wins are peripheral, as at AF 12.
5. 3D-INR and 3D-DIP stay 3 to 4 dB behind the leaders (INR over-sharpens: sharpness 1.33 on T1);
   nothing about them changes the reading above.

## Figures

`figures/` (mirrored to `/home/congcongliu/figs/t1flair_af16/`, served at
`http://localhost:8765/t1flair_af16/<file>`):

- `t1flair_af16_{T1,FLAIR}_plane{0,1,2}.png` — the three mid-planes, all arms, magnitude row +
  error row (jet, fixed [0, 0.1] of the GT peak, never autoscaled).
- `t1flair_af16_{T1,FLAIR}_butterfly_axis0_154.png` — the same transverse slice (axis0 = 154, the
  lateral-ventricle level) the AF 12 topic uses, titles carrying that slice's own head-masked PSNR.

## Regenerating

```bash
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
R=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_retro
E="PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID"

bash $R/scripts/setup_af16_hyw.sh        # clones the AF12 topic: scripts, symlinked inputs, MATLAB mask, export, worktree configs
bash $R/scripts/setup_af16_jyz.sh        # the same onto vol3_jiangyizhou (its own AF12 topic's inputs, the same mask)
cd $R && setsid nohup env $E $PY -u scripts/sched_af16.py --subjects hyw,jyz --gpus 0,1 > logs/sched_stdout.log 2>&1 &
#   status: logs/sched_status.json; log: logs/sched.log; DONE_SCHED when the queue is empty; restart-safe
bash $R/scripts/postprocess.sh hyw 0     # extract MC-GS volumes, summarize, gauge proof, figures, per-slice sweep, AF12-vs-AF16, README tables
```

## Gotchas that cost time (do not re-discover)

- **A symlinked input is one `h5py.File(..., "w")` away from overwriting the parent topic.** The
  AF 12 export script rewrites `ksp_*.h5`; the copy here asserts the symlink and skips the write.
  Keep it that way in every clone.
- **A greedy "first job that fits" scheduler starves the heavy runs** when light jobs keep arriving
  and another user leaves 47 GB per card: two ZS-SSDU / INR jobs always fit, an MC-GS run never
  does. Strict priority with an explicit drain (and fillers only when they finish first) fixed it.
- **Another user's job on the same card is a 2 to 5x slowdown, not a memory problem.** MC-GS
  joint: 137 min shared vs 29 min alone; ZS-SSDU FLAIR epochs 32 s alone vs ~200 s shared with a
  ZS-SSDU T1 (time-slicing paces the light process to the heavy one). Do not pair the two ZS-SSDU
  contrasts on one card.
- **`nvidia-smi` between two restarts of a foreign job reads low.** The scheduler once saw 17 GB
  where 33 GB was coming back and put two MC-GS runs on one card; it now assumes a floor for other
  users' memory (34 GB while they were present, 10 GB after they left) and retries a failed job once.
- **ZS-SSDU T1 does not early-stop the way it did at AF 12** (87 epochs, 5 h shared, against 23
  epochs there): the validation loss keeps creeping down by 1e-4 per epoch, which resets the
  patience. Budget for it; do not "fix" the criterion — the AF 12 numbers used the same one.
- **The agent harness kills its own background tasks when the host's MemFree (not MemAvailable)
  dips**, even at 850 GB available. Anything that must survive — the scheduler, the post-processing
  — runs under `setsid nohup`, and completion is watched from the log.
- The trainer's own prints are block-buffered (see the T1+T2 README): read
  `results_arc/<exp>/<exp>/joint_train.log` in the worktree for progress, not the launcher log.
