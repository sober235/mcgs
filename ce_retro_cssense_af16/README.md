# CE Retro CS-SENSE AF16

Topic: `/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/`
Worktree / branch: `mcgs_ce_retro_wt` / `ce-retro-cssense-af16`
Design spec: `docs/superpowers/specs/2026-09-17-ce-retro-cssense-af16-design.md`

Variables used below: `$TOPIC` = the topic path above, `$WT` = `/data0/congcong/code/GS/mcgs_ce_retro_wt`,
`$PY` = `/home/congcongliu/anaconda3/envs/GSTMI/bin/python`.

## 1. Purpose

Treat the CS-SENSE reconstruction of the UIH 3D T1 GRE plain/enhanced pair as ground truth, synthesise k-space with
the real 42-coil maps, resample at a variable-density Poisson-disc AF16, and rerun 11 reconstruction arms against
metrics that now have a truth to compare to. MC-GS is the only proposed method; the other ten arms (GS-single
included) are baselines. The headline margin is **MC-GS minus GS-single**. This is a new, self-contained
retrospective experiment -- not an extension of the prospective CE topic (`CE_pre_post_contrast/`) and unrelated to
the parallel `CE_retro_gt/` session (UCSF-PDGM synthetic truth); none of the three write into another's directory.

Arms (figure order): ZF, CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR, GS-single,
MC-GS. Architecture and hyperparameters for MC-GS/GS-single are copied unmodified from the frozen jyz T1+FLAIR AF16
recipe (`Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/`) -- see the arms table in §3.

## 2. Data and synthesis

**Ground truth** (`sim/meta.json:gt`, one volume per scan, same grid):
`CE_pre_post_contrast/tv_sense/cssense_o200_lam0.002_full_{pre,post}_complex.npy`, shape (256, 232, 176) =
(readout S-I, A-P, L-R), complex64. Used as-is (already scaled to the measured k-space, see the alpha gate below).

**Coil maps** (`sim/meta.json:scans.*.csm`): the real 42-coil sensitivity maps, one set per scan --
`.../NV-Generate-CTMR/recon_real/work_ce_t1_16887049370010/csm.npy` (pre) and `...0060/csm.npy` (post), shape
(42, 256, 232, 176) complex64.

**Mask** (`sim/mask/mask_af16.json`): BART `poisson`, deterministic seed **0**, density parameter (BART `-y`/`-z`,
not the AF) **acc = 3.883**, found after **31** BART calls on a fine grid over `a`; the raw draw at that `a` had
2555 points, **trimmed 3** (dropped the points farthest from DC) to land exactly on **2552** points, i.e. **AF =
16.00** (= 232x176 / 2552, the user's 2026-09-14 AF definition). ACS **24x24** full-sampled, DC at (116, 88), **no
elliptical shutter** -- the full 232x176 rectangle is eligible and the four corners are sampled (user decision,
2026-09-17). Same mask for pre and post. BART v1.0.00 (`/home/congcongliu/src/bart/bart`); exact command:
`bart poisson -Y 232 -Z 176 -y 3.883 -z 3.883 -C 24 -v -s 0`.

**Noise** (`sim/meta.json:scans.*`, full derivation in design §4): per scan, sigma is fit from the residual between
the original prospective measurement and the alpha-rescaled truth, on the original prospective mask; the complex
scale alpha is a least-squares fit and is asserted to be within 1e-3 of 1 (**pre 0.99999997, post 1.00000006** --
both pass, confirming the truth is already on the measured k-space scale). The run is gated on reproducing the
archived CS-SENSE data-consistency residual `kres` to < 1e-3:

| scan | sigma | kres (reproduced) | kres (archived reference) | noise seed | WM SNR |
|---|---|---|---|---|---|
| pre | 1.36686 | 0.184868 | 0.18487 | 100 | 34.445 |
| post | 1.44523 | 0.149690 | 0.14969 | 200 | 33.098 |

Both reproductions are within 2e-6 of the archived reference -- the FFT/CSM convention gate. WM SNR =
`median_WM(|x_GT|) / sigma` (coil maps are unit-SoS here, so no separate SoS correction is needed in the formula
itself, only in its derivation -- see §7); the white-matter mask is the "正常白质" ROI of
`enhance_rois_cssense.npz`. Noise is added to the *entire* k-space grid (not just the sampled points) with the
seeds above, then the AF16 mask is applied when writing `sim/work_{pre,post}/ksp_full.npy` -- this is equivalent to
masking first for what the arms see, but also lets the trainer's non-presampled data path read a full, noisy grid.

Sigma is the data-consistency residual of a TV-regularised truth, so it upper-bounds the thermal noise. This is
the user-selected noise convention, not a bug.

**Operator gates** (`logs/check_sim.json`, `scripts/check_sim.py`, one GPU, design §10 items 1-2): on the
noiseless, fully-sampled k-space, the adjoint identity `x_hat = sum_c conj(S_c) ifft3c(F(S_c x)) / sum_c|S_c|^2 ==
x_GT` inside the head, and CG-SENSE (Tikhonov 1e-6, 60 iters) head PSNR > 60 dB. Both scans, all three gates
(adjoint, CG, sigma-written-vs-meta) pass -- `CHECK_SIM OK`:

| scan | adjoint rel err | CG-SENSE noiseless head PSNR | sigma written vs meta (rel dev) |
|---|---|---|---|
| pre | 3.81e-7 | 69.19 dB | 4.27e-5 |
| post | 3.84e-7 | 75.91 dB | 1.64e-5 |

(`check_sim.json` also records a ZF-on-AF16 head PSNR "for the record" -- left out of this table since it duplicates
the ZF row of §4's results table below, now that every arm is summarised there: 24.250 dB pre / 25.829 dB post,
identical to `check_sim.json`'s `zf_af16_head_psnr`, as it should be -- both come from the same `Pair.score`.)

The adjoint gate is restricted to head voxels where coil coverage `sum_c|S_c|^2 > 0.25` -- see §7 for why, and how
small that restriction is (< 0.006% of the head either way).

**Trainer snapshot** (`trainer/_trainer_e778bf2/`, `SOURCE.txt`): `git archive e778bf2` of the sober235/mcgs
lineage (`paired_gs`, `arc`, `configs`, `diag_artefact` -- the same trainer the jyz AF16 phase-3 runs used), with
the compiled `cuda_ext/build/tile_voxelize_cuda_0141c525_7aedd3b0.so` copied in from
`/data0/congcong/code/GS/mcgs_af16_wt/paired_gs/gaussian/cuda_ext/build/` (same source hash) rather than rebuilt.
Note: design §9's directory sketch names this `scripts/_trainer_e778bf2/`; what is actually on disk, and what
`scripts/tests/test_deps.py`'s `RETRO_TRAINER` and `run_gs.sh`'s `$TR` both point at, is the top-level
`trainer/_trainer_e778bf2/` -- use that path, not the spec sketch (see §7).

## 3. Arms and lambda protocol

Copied verbatim from design §5 (must stay word-for-word identical to the design doc so the two never drift):

| 臂 | 设置(照搬) | 本课题差别 / 风险 |
|---|---|---|
| ZF、CG-SENSE | `recon_classical.py`：ZF = SENSE 伴随；CG-SENSE 60 步、Tikhonov 0.01；真 3D，线圈分块 | 42 线圈 |
| CS-SENSE | 同脚本 `--admm_outer 200 --admm_cg 8 --mu 0.1`，各向同性 3D TV；λ 网格 {2.5e-4, 3.5e-4, 5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3} | λ 重扫 |
| Joint SENSE | `recon_joint_cssense.py`：矢量 TV（两对比度 6 分量联合收缩）、外层 200、CG 8、μ 0.1；λ 网格 {5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3} | 通道 pre,post |
| L1-SPIRiT | `recon_spirit.py`（mcgs_spirit_wt 版）：核 {3,5} × λ {0, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3}、Tikhonov 0.01、150 步、db2 四层、FISTA、`--calib acquired`、逐读出层 2D | ACS 真 24×24；42 线圈下核 5 的标定方程 (24−5+1)²=400 个 < 未知数 42×25=1050 个，欠定；核 3 未知数 378 个 < 484 个方程。网格里两种核都跑，按 GT 选；若核 5 发散或明显差，写进 README |
| ZS-SSDU | `recon_zsssdu.py`：ResNet 5 块 64 通道、展开 10、CG 6、batch 4、lr 5e-4、μ 0.05、≤100 epoch、patience 10、Γ 10% 早停、Θ/Λ 80/20 每步重采（seed 1234+step）、seed 0、逐读出层 2D | 42 线圈更慢 |
| 3D-DIP | `recon_dip_inr.py --method dip`：ConvDecoder3D 通道 [256,128,64,64]、固定 N(0,0.1) 噪声输入、BN、零初始化输出头、5000 步、lr 3e-3、每 100 步评、TV 0、留出 5% PE 线（seed 0）选 checkpoint | 低分辨种子网格按 (256,232,176)/16 取整 |
| 3D-INR | 同脚本 `--method inr`：hash 16 级、T=2^19、每级 2 特征、分辨率 16→512、MLP 64×3、1500 步、lr 5e-3、每 50 步评、留出 5% | 同上 |
| MC-INR | `/data0/congcong/code/GS/mcinr_niessen2025`（`run_volume.py`，已有 `--ksp_T1/--ksp_FLAIR` 外部 k 空间入口）：hash 16 级、log2T 20、MLP 32→64→64→2N、lr 1e-2、每层 200 步、加权 k 空间 MSE、最后一步、seed=层号；逐读出层，一网出两对比度 | pre 走 T1 槽、post 走 FLAIR 槽；输出用本课题评分器重算 |
| MC-GS | 训练器提交 e778bf2（`git archive e778bf2 paired_gs` 钉进 `scripts/_trainer_e778bf2/`）。基础配置从 `mcgs_af16_p3_wt/configs/t1flair_v3_af16_jyz.yaml` 派生为 `configs/ce_retro_af16.yaml`：data 段换本课题 h5，plain=pre、contrast=post，网格 (256,232,176)，**known_zero 两通道都关**（合成数据无厂商快门）。CLI 原文：`--iters 4000 --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf --set gaussian.initial_num_points=3600000 --set loss.wav_name=db4 --set loss.wav_levels=1 --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10 --lambda_edge 0.001 --lambda_ptv 0.03 --save_volumes` | 360 万点 × 42 线圈显存待探（jyz 26 线圈 32.9 GB）；若 OOM，先降 tile，再报告用户决定 |
| GS-single | 同配方同训练器；`configs/ce_retro_af16_single_pre.yaml`、`..._single_post.yaml` 把同一扫描喂进 plain 与 contrast 两通道；引用 plain 通道 | 同上 |

**λ 协议**：CS-SENSE、Joint SENSE、L1-SPIRiT 按本数据 GT 的头内 PSNR 取最优（每个扫描各选各的；Joint SENSE 一次 λ
运行同时产出两个扫描的体，每个扫描从这些运行里各挑自己最优的 λ；jyz 口径）；最优落在网格边缘时向外扩一档再扫，最多扩两次。
ZS-SSDU、3D-DIP、3D-INR 不看 GT（留出/早停）；MC-INR 无选择；MC-GS、GS-single 不调。

**不对称说明**：MC-GS 配方在 jyz 上冻结、搬来不调，对手 λ 却在本数据 GT 上扫。这对 MC-GS 偏保守，与 jyz 留出受试者的
做法一致。

MC-GS smoke test (design §10 item 4, `trainer/_trainer_e778bf2/results_arc/smoke_joint.log`, 50 steps, N =
3,600,000 points, 42 coils): **peak VRAM 43.8 GB**.

Live footnote on the MC-GS row above: the archive actually lives at `trainer/_trainer_e778bf2/` (see §2), not
`scripts/_trainer_e778bf2/` as the design sketch has it.

## 4. Results

All eleven arms are scored on both scans; every gauge cross-check (`scripts/score_volume.py --check`, CPU, double
precision) prints `GAUGE OK` against the arm's own recorded json (14/14: cgsense, zsssdu, dip, inr, mcinr,
gs_single, gs_joint x pre/post -- the seven arms scored outside `recon_classical.py`'s own internal check; full
stdout in `logs/gauge_check.txt`). Literal
output of `readme_tables.py` (2026-09-18), not hand-edited:

**pre** (sorted by head-only PSNR; truth = CS-SENSE of the prospective pair)

| method | head PSNR | head SSIM | head NMSE | head HFEN | complex PSNR | whole PSNR | λ / kernel |
|---|---|---|---|---|---|---|---|
| MC-GS | **33.623** | **0.9034** | **0.01305** | **0.2505** | 30.796 | 36.811 |  |
| GS-single | 33.429 | 0.8997 | 0.01365 | 0.2557 | 30.647 | 36.578 |  |
| ZS-SSDU | 32.096 | 0.8910 | 0.01855 | 0.3012 | 29.442 | 35.105 |  |
| L1-SPIRiT | 31.657 | 0.8489 | 0.02052 | 0.2731 | 28.750 | 34.405 | 0.0005 / ks5 |
| CS-SENSE | 31.031 | 0.8633 | 0.02371 | 0.3712 | 28.282 | 34.048 | 0.001 |
| Joint SENSE | 30.975 | 0.8616 | 0.02401 | 0.3744 | 28.238 | 34.009 | 0.0014 |
| 3D-DIP | 29.324 | 0.8093 | 0.03511 | 0.4215 | 26.485 | 18.444 |  |
| CG-SENSE | 28.937 | 0.7859 | 0.03839 | 0.4488 | 26.233 | 31.418 |  |
| 3D-INR | 28.084 | 0.6937 | 0.04672 | 0.4245 | 25.112 | 26.287 |  |
| MC-INR | 27.444 | 0.6837 | 0.05414 | 0.5532 | 24.454 | 28.661 |  |
| ZF | 24.250 | 0.6416 | 0.11296 | 0.8222 | 21.415 | 27.383 |  |

**post** (sorted by head-only PSNR; truth = CS-SENSE of the prospective pair)

| method | head PSNR | head SSIM | head NMSE | head HFEN | complex PSNR | whole PSNR | λ / kernel |
|---|---|---|---|---|---|---|---|
| MC-GS | **35.657** | **0.9253** | **0.01356** | **0.2335** | 32.941 | 38.865 |  |
| GS-single | 35.302 | 0.9196 | 0.01472 | 0.2495 | 32.675 | 38.481 |  |
| L1-SPIRiT | 34.069 | 0.8925 | 0.01955 | 0.2571 | 31.285 | 36.913 | 0.0005 / ks5 |
| ZS-SSDU | 33.504 | 0.9070 | 0.02227 | 0.3119 | 30.955 | 36.687 |  |
| CS-SENSE | 32.725 | 0.8829 | 0.02664 | 0.3614 | 30.093 | 35.766 | 0.001 |
| Joint SENSE | 32.605 | 0.8796 | 0.02739 | 0.3807 | 29.935 | 35.499 | 0.001 |
| CG-SENSE | 30.643 | 0.8232 | 0.04303 | 0.4663 | 28.191 | 33.239 |  |
| 3D-INR | 30.609 | 0.7677 | 0.04337 | 0.3644 | 27.680 | 27.968 |  |
| 3D-DIP | 30.235 | 0.7949 | 0.04728 | 0.4534 | 27.525 | 21.062 |  |
| MC-INR | 29.739 | 0.7496 | 0.05299 | 0.5193 | 26.681 | 31.120 |  |
| ZF | 25.829 | 0.6466 | 0.13038 | 0.8418 | 23.289 | 28.937 |  |

**MC-GS minus GS-single** (the headline margin)

| scan | head PSNR | head SSIM | head NMSE | head HFEN |
|---|---|---|---|---|
| pre | +0.194 | +0.0036 | -0.00060 | -0.0052 |
| post | +0.354 | +0.0057 | -0.00115 | -0.0160 |

MC-GS also leads the best non-GS arm on head PSNR: pre +1.527 dB over ZS-SSDU (32.096); post +1.588 dB over
L1-SPIRiT (34.069). L1-SPIRiT kernel 5 (calibration-underdetermined per §7) did not diverge -- it beat kernel 3 at
every swept λ except one (post, λ=0, a negligible 0.09 dB gap far from the winning λ=0.0005) and was the kernel
`summarize.py` kept for both scans; kept as-is, not dropped from the tables (resolves the §7/STATUS open question).

**Enhancement (post − pre, white-matter units); err = arm − truth**

| arm | vessels | plexus | cortex | WM | lesion core | rim 0–3 mm | background(post) |
|---|---|---|---|---|---|---|---|
| truth (CS-SENSE) | +2.841 | +1.080 | +0.104 | +0.003 | +0.021 | +0.028 | 0.007 |
| ZF | +1.544 (-1.297) | +0.362 (-0.718) | +0.097 (-0.007) | +0.009 (+0.006) | +0.018 (-0.003) | +0.021 (-0.006) | 0.024 |
| CG-SENSE | +2.190 (-0.651) | +0.657 (-0.423) | +0.098 (-0.006) | +0.004 (+0.001) | +0.024 (+0.003) | +0.030 (+0.002) | 0.040 |
| CS-SENSE | +2.550 (-0.291) | +0.715 (-0.365) | +0.104 (-0.000) | +0.008 (+0.005) | +0.019 (-0.003) | +0.027 (-0.001) | 0.018 |
| Joint SENSE | +2.469 (-0.372) | +0.676 (-0.404) | +0.100 (-0.005) | +0.007 (+0.004) | +0.018 (-0.004) | +0.027 (-0.001) | 0.022 |
| L1-SPIRiT | +2.648 (-0.193) | +0.788 (-0.292) | +0.103 (-0.002) | +0.004 (+0.001) | +0.019 (-0.002) | +0.028 (+0.000) | 0.018 |
| ZS-SSDU | +3.103 (+0.262) | +0.814 (-0.266) | +0.103 (-0.001) | +0.002 (-0.001) | +0.014 (-0.007) | +0.030 (+0.003) | 0.043 |
| 3D-DIP | +2.337 (-0.504) | +0.481 (-0.599) | +0.103 (-0.002) | +0.004 (+0.001) | +0.012 (-0.009) | +0.028 (+0.000) | 0.783 |
| 3D-INR | +2.429 (-0.411) | +0.723 (-0.358) | +0.096 (-0.008) | +0.003 (-0.000) | +0.037 (+0.016) | +0.042 (+0.015) | 0.369 |
| MC-INR | +2.290 (-0.551) | +0.693 (-0.387) | +0.099 (-0.006) | +0.005 (+0.001) | +0.028 (+0.006) | +0.030 (+0.002) | 0.162 |
| GS-single | +2.695 (-0.145) | +0.764 (-0.316) | +0.098 (-0.006) | +0.003 (-0.000) | +0.019 (-0.002) | +0.028 (-0.000) | 0.010 |
| MC-GS | +2.715 (-0.126) | +0.833 (-0.247) | +0.100 (-0.004) | +0.005 (+0.001) | +0.021 (-0.000) | +0.026 (-0.002) | 0.009 |

Reproduce with:

```
cd $WT/ce_retro_cssense_af16 && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/readme_tables.py
```

## 5. Figures

**Mask (approved, design §3 gate)** -- `figures/mask_af16.png`:
http://localhost:8765/ce_retro_af16/mask_af16.png
`/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/figures/mask_af16.png`

**Three-plane comparison** (design §7; 11 columns ZF...MC-GS, slices z201/y131/x48, jet error 0-0.1 head-only,
PDF 600 dpi) -- one file **per scan** (`fig_three_planes_compact.py` takes `--scan {pre,post}`; it is not one
scan-less figure):

`figures/ce_retro_af16_three_planes_pre_201_131_48.{png,pdf}` (copied/renamed from
`fig_three_planes_compact.py`'s own `t1flair_v3_af16_pre_row3planes_compact_201_131_48_inset48_headerr_201_131_48_spirit_zf_caps.{png,pdf}`,
left alongside it under both names):
http://localhost:8765/ce_retro_af16/ce_retro_af16_three_planes_pre_201_131_48.png
`/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/figures/ce_retro_af16_three_planes_pre_201_131_48.png`

`figures/ce_retro_af16_three_planes_post_201_131_48.{png,pdf}` (same renaming, `_post_`):
http://localhost:8765/ce_retro_af16/ce_retro_af16_three_planes_post_201_131_48.png
`/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/figures/ce_retro_af16_three_planes_post_201_131_48.png`

MC-GS ranks #1 of all eleven arms on PSNR, SSIM and NMSE on every one of the six panels (3 planes x 2 scans) at
these slices.

**Enhancement figure** (design §7, `fig_enhancement.py` 9-row layout) --
`figures/ce_retro_af16_enhancement.{png,pdf}` (this script writes the final name directly):
http://localhost:8765/ce_retro_af16/ce_retro_af16_enhancement.png
`/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/figures/ce_retro_af16_enhancement.png`

## 6. Reproduction commands

All python commands take the mandatory prefix `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID`;
omitted below for width. Run from `$TOPIC` (equivalently `$WT/ce_retro_cssense_af16`, `scripts`/`configs` are
symlinked). `<gpu>` = an index picked with `nvidia-smi` per the GPU etiquette in constraints.md.

```bash
# 1. mask (CPU, no args)
$PY scripts/make_mask_af16.py

# 2. synthetic k-space (CPU is fine: --gpu -1; the archived run used the default GPU 0)
$PY scripts/sim_cssense_af16.py --gpu <gpu-or--1>

# 3. operator gates (one GPU, required)
$PY scripts/check_sim.py --gpu <gpu>

# 4. every arm -- either the scheduler (recommended: handles the lambda grids, retries, and the
#    already-running guard for the six jobs that were hand-launched before it existed):
cd $TOPIC && setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
  $PY scripts/retro_orchestrate.py > logs/orch/orchestrate.log 2>&1 &
$PY scripts/retro_orchestrate.py --dry      # print the job graph without running anything

#    ... or by hand, one arm at a time, e.g.:
$PY scripts/recon_classical.py --scan pre --methods zf,cgsense --gpu <gpu>
$PY scripts/recon_classical.py --scan pre --methods cssense --admm_outer 200 --lam 0.001 --suffix _o200_lam0.001 --gpu <gpu>
$PY scripts/recon_joint_cssense.py --topic $TOPIC --contrasts pre,post --admm_outer 200 --lam 0.001 --gpu <gpu>
$PY scripts/recon_spirit.py --scan pre --ksize 5 --lam 0.001 --gpu <gpu>
$PY scripts/recon_zsssdu.py --scan pre --gpu <gpu>
$PY scripts/recon_dip_inr.py --method dip --scan post --iters 5000 --lr 3e-3 --eval_every 100 --gpu <gpu>
$PY scripts/recon_dip_inr.py --method inr --scan pre --iters 1500 --eval_every 50 --gpu <gpu>
$PY scripts/run_mcinr.py --gpu <gpu>
bash scripts/run_gs.sh ce_retro_af16_joint <gpu> 4000                          # MC-GS
SINGLE=pre  bash scripts/run_gs.sh ce_retro_af16_single_pre  <gpu> 4000        # GS-single, pre channel
SINGLE=post bash scripts/run_gs.sh ce_retro_af16_single_post <gpu> 4000        # GS-single, post channel

# 5. pull the GS npz results into recon/<key>_<scan>_complex.npy (CPU)
$PY scripts/extract_gs_volumes.py --exp ce_retro_af16_joint       --key gs_joint  --map pre:plain,post:contrast
$PY scripts/extract_gs_volumes.py --exp ce_retro_af16_single_pre  --key gs_single --map pre:plain
$PY scripts/extract_gs_volumes.py --exp ce_retro_af16_single_post --key gs_single --map post:plain

# 6. aggregate (CPU)
$PY scripts/summarize.py                 # -> logs/summary.json, logs/summary.tsv
$PY scripts/enhancement_fidelity.py      # -> logs/enhancement_fidelity.json
$PY scripts/find_best_slice.py           # -> logs/slice_scores.json
$PY scripts/best_slice_mcgs.py           # -> logs/best_slice_mcgs.json

# 7. README tables (paste the stdout into §4 above)
$PY scripts/readme_tables.py

# 8. figures -- one file per scan (--scan {pre,post}); flags per the Task 16 brief
for s in pre post; do
  $PY scripts/fig_three_planes_compact.py --scan $s --slices 201 131 48 --inset 48 --inset_corner tr bl bl \
    --mask_error --gray_cbar --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 --with_zf --caps --l1_arm spirit \
    --pdf --pdf_dpi 600 --tag _201_131_48
done
$PY scripts/fig_enhancement.py --slices 201,131,48
# fig_three_planes_compact.py writes its own native name (t1flair_v3_af16_<scan>_row3planes_compact_...) into
# figures/ and into /home/congcongliu/figs/ce_retro_af16/ (see its docstring) -- copy/rename the two PNG+PDF
# pairs it wrote to figures/ce_retro_af16_three_planes_<scan>_201_131_48.{png,pdf} before citing them in §5.

# 9. export bundle (--no_raw skips ~26 GB of h5 / work_* k-space; the design doc's "~14 GB" guess was low -- see §7)
$PY scripts/export_bundle.py --no_raw
$PY scripts/export_bundle.py               # full bundle, raw k-space included: 113 files, 28.36 GiB (Task 16)
```

The lambda grids and per-scan/per-kernel invocations are exactly what `retro_orchestrate.py --dry` prints (job
name, GPU need, dependency, full command) -- read that instead of hand-transcribing every grid point here.

## 7. Gotchas

**check_sim.py's adjoint identity is singular at low coil coverage (adaptation, pre-authorised by the task-5
brief).** The raw identity `x_hat = sum_c conj(S_c) ifft3c(F(S_c x)) / sum_c|S_c|^2` blows up wherever
`sum_c|S_c|^2` is near zero. The first run failed with relative error 1.90e-3 (pre) / 1.13e-3 (post) -- exactly the
failure mode the brief predicted. Per the brief, the gate is restricted to head voxels with `sum_c|S_c|^2 > 0.25`
(recorded as `adjoint_voxel_condition` / `adjoint_head_voxels_total` / `adjoint_head_voxels_used` in
`logs/check_sim.json`); this excludes 223/4,495,671 head voxels for pre and 74/4,520,978 for post (< 0.006% of the
head either way). With the restriction the gate passes cleanly (§2 table). This is a restriction of *which voxels
the check inspects*, not a change to any pass/fail threshold.

**`sigma_from_residual`'s sample count `n` had a missing-axis bug (found and fixed, not just documented).** As
given in the task-5 brief, `n` (used only inside `sigma = sqrt(res / (2n))`) was `m0.sum() * S.shape[0]` (mask
points times coils), omitting the readout axis: `[..., m0]` fancy-indexing with a 2D mask on a 4D
`(coil, readout, ky, kz)` tensor keeps the readout axis, so `res` actually sums over
`m0.sum() * S.shape[0] * S.shape[1]` terms. Left as given, sigma would be inflated by `sqrt(S.shape[1])` (16x for
the real 256-readout grid), while `kres` (which does not depend on `n`) and the check_sim sigma gate (a
self-consistency check between the written noise and `meta.json`, both derived from the same sigma) would both
still pass -- so the bug would have silently written ~16x-too-noisy synthetic k-space into the 14 GB of study data,
undetected by either named gate. Confirmed by dimensional analysis, an empirical `torch` indexing check, and the
brief's own unit test (`test_sigma_from_residual_recovers_known_noise`), which failed against the unfixed code
(measured sigma/sigma0 = 3.96 = sqrt(16), matching the missing factor exactly) and passes once fixed. The one-line
fix `n = m0.sum() * S.shape[0] * S.shape[1]` was applied identically in `sim_cssense_af16.py` and in the test's own
`expect_kres` formula (`test_sim.py`), which had the same omission.

**Trainer snapshot lives at `trainer/_trainer_e778bf2/`, not `scripts/_trainer_e778bf2/`.** Design §9's directory
sketch put it under `scripts/`; the actual `git archive` in commit `18c9aac` landed it at the topic-root
`trainer/_trainer_e778bf2/`, and everything downstream (`scripts/tests/test_deps.py`'s `RETRO_TRAINER` default,
`run_gs.sh`'s `$TR`, `t1t2_common.py`'s `TRAINER`) already points there. Use the real path; do not "fix" it to
match the sketch.

**`logs/summary.json` is a stale partial snapshot while production is in flight -- do not run `readme_tables.py`
against it as-is.** As of Task 15 it had 4 rows (zf/cgsense x pre/post) from an early `summarize.py` run;
`recon/` already had more arms finished than that (CS-SENSE's full lambda grid, most of L1-SPIRiT, ZS-SSDU, one
DIP) that just hadn't been re-summarised. Re-run `scripts/summarize.py` after the GPU queue drains, not before.
**Resolved (Task 16):** the scheduler ran to completion (`logs/orch/ALL_DONE_retro16`); `summarize.py` now writes
22 rows (11 arms x 2 scans), matching the §4 tables above.

**42-coil L1-SPIRiT kernel 5 is calibration-underdetermined (known risk, not yet observed).** ACS is a true 24x24;
at 42 coils the kernel-5 calibration has `(24-5+1)^2 = 400` equations for `42*25 = 1050` unknowns. Kernel 3 is
fine (`378` unknowns, `484` equations). Both kernels are run; if kernel 5 diverges or is clearly worse once scored,
say so in §4/§8 rather than silently dropping it. **Resolved (Task 16):** it did not diverge -- ks5 beat ks3 at
every swept λ except one negligible point (post, λ=0, 0.09 dB, nowhere near the winning λ), and `summarize.py` kept
ks5 for both scans' L1-SPIRiT row. Both kernels stay in the raw per-lambda logs; only the winning one is tabulated.

**`find_best_slice.py`'s "cssense" never resolved to a file (found and fixed, Task 16).** Flagged as a review note
in the script itself since 2026-09-17: `METHODS` includes `"cssense"`, but this topic's CS-SENSE volumes are named
`recon/cssense_o200_lam<L>_<c>_complex.npy` (swept lambda), and the flat loader only ever tried
`recon/cssense_<c>_complex.npy`, which does not exist. Harmless while `gs_joint`/`gs_single` were not yet
registered (the script would have hit a different missing key first); once every arm existed this raised
`KeyError: 'cssense'` exactly as predicted. Fixed by resolving "cssense" through `logs/summary.json`'s
`cssense_o200` row for that scan (the λ `summarize.py` kept), the same pattern `arms.py`/`best_slice_mcgs.py`
already use -- the other seven methods in `METHODS` were already fine.

**Both per-slice scripts still only covered 8 of 11 registered arms (found and fixed, whole-branch review
2026-09-18).** The cssense fix above resolved the `KeyError` but `find_best_slice.py`'s `METHODS` and
`best_slice_mcgs.py`'s `arms()` were each their own hand-written list, and each separately dropped a different
subset of the field (`find_best_slice.py` was also missing Joint SENSE, L1-SPIRiT and MC-INR; `best_slice_mcgs.py`
was also missing ZF, L1-SPIRiT and MC-INR) -- so every per-slice claim below had silently been a comparison
against 8 of this topic's 11 arms, not the whole field. Both scripts now resolve every arm through
`arms.arm_files()` (arms.py's `ORDER`/`LABEL`, the single source of truth for which file each arm's registered
result lives in) instead of a hand-written list. Re-run against all eleven: MC-GS joint ranks first on 602/604
scored slices (pre) and 604/604 (post) of all ten other arms -- unchanged from the 8-arm count; Joint SENSE,
L1-SPIRiT and MC-INR do not win any additional slice on either scan. On axis 0 it still leads every other arm
(GS-single included) on all 190/190 eligible slices, both scans, also unchanged (`logs/best_slice_mcgs.json`;
best slice still 218, margin +0.402 dB pre / +0.646 dB post over GS-single). What *did* change at slice 218: the
strongest non-GS arm is ZS-SSDU on pre (+1.517 dB, unchanged) but **L1-SPIRiT** on post (+1.550 dB) -- the
previously reported post figure (CS-SENSE, +2.409 dB) had simply never been compared against L1-SPIRiT, which
beats CS-SENSE by 1.34 dB on post at the volume level (§4).

**The design doc's "~14 GB" for `export_bundle.py --no_raw`'s skipped raw data was a guess, not a measurement.**
The real raw payload (`sim/gs/` 4 files + `sim/work_{pre,post}/` 3 files each) is ~26 GB, not 14 -- confirmed by
running both bundle variants (Task 16): `--no_raw` is 103 files / 2.11 GiB, the full bundle is 113 files / 28.36
GiB, so raw itself is 28.36 - 2.11 = 26.25 GiB. Both `export_bundle.py`'s docstring and this README's §6 are
corrected; the number was never load-bearing for any gate, only for deciding whether `/data0` has room (it does).

**Shell is zsh.** Quote every variable (`"$TOPIC"`, `"$PY"`, ...); an unquoted `$VAR:more/path` silently eats
everything after the colon.

## 8. Wording constraints (design §11)

- The ground truth is a TV-regularised solution; the CS-SENSE family has a home-turf advantage on this data.
  Every conclusion carries that caveat.
- Noise is defined from the data-consistency residual, so it runs slightly above pure thermal noise (§2).
- Under one shared mask, the post-minus-pre coherent artefact partly cancels; a difference map alone is not
  image-quality evidence.
- Permitted claims: per-arm head-only metrics against this ground truth at AF16 Poisson-disc; the MC-GS-minus-GS-
  single margin.
- Not permitted: "better than the prospective experiment" (different gauge entirely), any perfusion language,
  lesion-detection claims.
- The truth is a CS-SENSE reconstruction of a prospective AF 5.33 acquisition whose PE support covers only
  7663/40832 = 18.8% of the plane; 1599 of the 2552 AF16 sample locations (62.7%) were never measured in that
  acquisition. In k-space energy terms this is much less severe: 20.3% (pre) / 21.9% (post) of the truth volume's
  own k-space energy lies outside that original support, and only 1.7% (pre) / 1.8% (post) of the energy the AF16
  mask actually samples falls at those never-measured locations. At those (mostly high-frequency) locations the
  synthetic measurement is the TV-ADMM truth's own extrapolation, not independently acquired data -- and those
  are precisely the locations HFEN, SSIM and sharpness respond to.
- The synthesis and every SENSE-based arm use the identical coil maps: `sim_cssense_af16.py` writes
  `sim/gs/<scan>_csm.h5` straight from the same `work_ce_t1_*/csm.npy` that `t1t2_common.Pair` reads for every
  arm (verified byte-identical), and those maps are unit sum-of-squares everywhere they have coil coverage. So
  the forward operator here has no coil-map mismatch at all, one of the two dominant real-world error sources --
  this favours every model-based arm uniformly, not just MC-GS. L1-SPIRiT is the only arm whose reconstruction
  does not consume the maps: it calibrates its interpolation kernel from the ACS alone, and only touches the maps
  in the final linear coil combine (the same step every multi-coil method needs to go from coil images to one
  image).
