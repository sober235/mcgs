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

Sigma is the data-consistency residual of a TV-regularised truth, so it upper-bounds the thermal noise; the
pre/post spread (kres 0.185 vs 0.150 at nearly equal sigma) bounds the model-error contamination (CSM error + detail
the TV prior removed) at roughly 6% in amplitude. This is the user-selected noise convention, not a bug.

**Operator gates** (`logs/check_sim.json`, `scripts/check_sim.py`, one GPU, design §10 items 1-2): on the
noiseless, fully-sampled k-space, the adjoint identity `x_hat = sum_c conj(S_c) ifft3c(F(S_c x)) / sum_c|S_c|^2 ==
x_GT` inside the head, and CG-SENSE (Tikhonov 1e-6, 60 iters) head PSNR > 60 dB. Both scans, all three gates
(adjoint, CG, sigma-written-vs-meta) pass -- `CHECK_SIM OK`:

| scan | adjoint rel err | CG-SENSE noiseless head PSNR | sigma written vs meta (rel dev) |
|---|---|---|---|
| pre | 3.81e-7 | 69.19 dB | 4.27e-5 |
| post | 3.84e-7 | 75.91 dB | 1.64e-5 |

(`check_sim.json` also records a ZF-on-AF16 head PSNR "for the record" -- left out of this table and out of §4:
it is a real per-arm quality number, the same kind §4 defers until every arm is summarised, so it is `NOT_RUN_YET`
here too rather than reported early just because it happens to sit next to the gate fields.)

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

`NOT_RUN_YET` -- production GPU runs are still in flight (see STATUS.md for a timestamped snapshot of what has
finished). `logs/summary.json` today has only 4 of the eventual ~40+ rows (zf/cgsense, both scans); running
`readme_tables.py` against it now would crash on the `MC-GS minus GS-single` step (no `gs_joint`/`gs_single` rows
yet) and would not be representative even if it didn't. Once every arm has a row in `logs/summary.json` and
`logs/enhancement_fidelity.json` exists, replace this whole section with the literal output of:

```
cd $WT/ce_retro_cssense_af16 && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/readme_tables.py
```

That prints, per scan (pre, post): the eleven arms sorted by head-only PSNR with the best value per column in
**bold**, then the MC-GS-minus-GS-single line (the headline margin), then the enhancement-fidelity table
(post - pre in white-matter units, arm error vs the truth in six reference regions plus extracranial background).
Paste it here verbatim -- do not hand-edit the numbers.

## 5. Figures

**Mask (approved, design §3 gate)** -- `figures/mask_af16.png`:
http://localhost:8765/ce_retro_af16/mask_af16.png
`/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/figures/mask_af16.png`

**Three-plane comparison** (design §7; 11 columns ZF...MC-GS, slices z201/y131/x48, jet error 0-0.1 head-only,
PDF 600 dpi) -- one file **per scan** (`fig_three_planes_compact.py` takes `--scan {pre,post}`; it is not one
scan-less figure):

`figures/ce_retro_af16_three_planes_pre_201_131_48.{png,pdf}`:
`NOT_RUN_YET` (not generated yet; needs every arm's volume registered first)

`figures/ce_retro_af16_three_planes_post_201_131_48.{png,pdf}`:
`NOT_RUN_YET` (not generated yet; needs every arm's volume registered first)

**Enhancement figure** (design §7, `fig_enhancement.py` 9-row layout) --
`figures/ce_retro_af16_enhancement.{png,pdf}`:
`NOT_RUN_YET` (not generated yet)

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
$PY scripts/extract_gs_volumes.py --exp ce_retro_af16_single_post --key gs_single --map post:contrast

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

# 9. export bundle (--no_raw skips ~14 GB of h5 / work_* k-space)
$PY scripts/export_bundle.py --no_raw
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
against it as-is.** As of this task it has 4 rows (zf/cgsense x pre/post) from an early `summarize.py` run;
`recon/` already has more arms finished than that (CS-SENSE's full lambda grid, most of L1-SPIRiT, ZS-SSDU, one
DIP) that just haven't been re-summarised. Re-run `scripts/summarize.py` after the GPU queue drains, not before.

**42-coil L1-SPIRiT kernel 5 is calibration-underdetermined (known risk, not yet observed).** ACS is a true 24x24;
at 42 coils the kernel-5 calibration has `(24-5+1)^2 = 400` equations for `42*25 = 1050` unknowns. Kernel 3 is
fine (`378` unknowns, `484` equations). Both kernels are run; if kernel 5 diverges or is clearly worse once scored,
say so in §4/§8 rather than silently dropping it.

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
