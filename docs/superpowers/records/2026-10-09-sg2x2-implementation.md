# Record — Shared-geometry 2x2 ablation: implementation (jiangyizhou AF16, 2026-10-09)

Branch `exp/shared-geometry-2x2-ablation-20261009` (worktree `mcgs_sg2x2_wt`, start 9df60e7). Spec
`docs/experiments/2026-10-09-mcgs-shared-geometry-2x2-ablation.md`, plan
`docs/superpowers/plans/2026-10-09-sg2x2-shared-geometry-ablation.md`. Execution ledger with every ruling:
`.superpowers/sdd/2026-10-09-sg2x2-shared-geometry-ablation/progress.md` (gitignored, local to this worktree).

**State at the tag `handoff/2026-10-09-sg2x2-pilot`: code complete, pilot pending.** The six pilot runs were not
launched; they wait for the user's go on the GPU time. Nothing in this record is a result.

## 1. What was built

Four cells: A = MC-GS + Reg, B = GS-single + Reg (per contrast), C = MC-GS - Reg, D = GS-single - Reg (per contrast).
"Reg" is the frozen per-channel phase-3 recipe (edge-preserving TV, phase-only TV, wavelet L1). No cell carries a
cross-contrast weight (`lambda_edge`, `lambda_dtv`, `lambda_vtv`, `lambda_nl`, guide alphas all 0).

| piece | file |
|---|---|
| cell table (the only place cells are defined) | `scripts/sg2x2_cells.py`, `tests/test_sg2x2_cells.py` |
| single-contrast trainer, one contrast per process | `paired_gs/trainers/single_contrast_trainer.py`, `arc/cycle.py --trainer single`, `tests/test_sg2x2_single_trainer.py` |
| frozen init support (`init_mode=support_file`) | `paired_gs/gaussian/init_cloud.py` (+40), `scripts/sg2x2_make_init_support.py`, `configs/sg2x2/init_support_af16_jyz.npz`, `tests/test_sg2x2_init_support.py` |
| base configs per cell kind | `configs/sg2x2/af16_jyz_{joint,single_T1,single_FLAIR}.yaml`, `tests/test_sg2x2_configs.py` |
| launcher | `run_sg2x2.sh EXP GPU CELL [ITERS] [SEED] [extra args]` (env `NPTS`, `INIT`, `SUPPORT`, `SAVE_VOLUMES`, `WARMUP`) |
| manifest + regulariser audit | `scripts/sg2x2_manifest.py` (exit code 3 when the audit fails), `tests/test_sg2x2_manifest.py` |
| result table | `scripts/sg2x2_table.py`, `tests/test_sg2x2_table.py` |
| regression gate | `scripts/sg2x2_regress_compare.py` |
| queue | `scripts/sg2x2_queue.sh`, `scripts/sg2x2_pilot_s42.txt` |

Three existing files were touched, additively: `paired_gs/trainers/joint_trainer.py` (+23 lines, 0 removed, includes
the R9 kappa fix below), `paired_gs/gaussian/init_cloud.py` (+40), `arc/cycle.py` (+39 / -7, the 7 deletions are the
permitted guard / move lines). `git diff --stat 9df60e7`: 3 files, 102 insertions, 7 deletions.

Design choices:

- **One contrast per process.** The only reading of "truly independent GS-single" that needs no new model class: the
  single trainer runs a one-channel model and shares nothing with the other contrast. The single objective equals the
  joint objective's per-channel half; the gradient-ratio test pins this (residuals about 1e-7 against a bound of
  1e-4; ratios exactly 2 : 1 for geometry : density).
- **Frozen support file instead of an ellipsoid.** Measured on the development subject (2026-10-09, CPU): the union
  of the two ZF supports covers 0.4273 of the box, its bounding box is the whole volume, and the inscribed ellipsoid
  covers only 0.8954 of the box, so an ellipsoid would put most points in empty air and move the operating point.
  The file `configs/sg2x2/init_support_af16_jyz.npz` (md5 `310a6520631a330e59cb5deb390e9035`, support_frac 0.4273)
  is made with `--dilate 0` (dilate 1 / 2 / 3 gave 0.4811 / 0.5431 / 0.6035; a 3-voxel ring would park about 29 % of
  the cloud outside the own support). With dilate 0 the support equals what `init_mode=head` draws from, so cell A's
  cloud equals the registered recipe's own head init (parity test: the `support_file` draw equals the `head` draw on
  the same support). Ruling R7. This equality rests on the smoke's identical iter-1 numbers (`dc_p` 7.8237e-02 and
  the `R_geo` decomposition in A against the registered run) and on the parity test on an identical support; it is not
  a pinned positions equality between the CPU-made file and the trainer's own GPU support. The `pos_sum` checksum
  (below) lets the pilot check that all cells drew the same cloud.
- **A and B share the identical frozen phase-3 recipe.** The spec §4 equal-budget per-arm lambda search is out of scope.
- **`in_own_zf_support` in single cells reads below 1** because it is measured against one contrast's own ZF support
  while the file holds the union (ruling R12). Smoke values: A 1.0000, C 1.0000, T1 single 0.9697, FLAIR single 0.9206.
- **`--warmup` / `training.shared_warmup_iters` is inert in both trainers** (the joint trainer does `del warmup`, the
  single trainer never reads it). It is passed only so the launcher's argument list matches the registered recipe; it
  changes nothing and must not be read as a frozen knob that matters.
- **The audit pins the frozen knobs, not only the switches.** `scripts/sg2x2_manifest.py` compares, with the trainer's
  own default when a key is absent: k-space weighting (`kspace_high_weight_plain` unset, curriculum off, centre fraction
  0.05 / weight 0.3, mid weight 1.0), no whitening (loss and per channel), `lambda_private_sp` 0, no private points, no
  adaptive control, `max_radius` 20, `init_scale_mode` 3nn, `tv_eps` 0.04, `edge_tv_kappa_source` zf, `known_zero` by
  channel name (FLAIR plain and every contrast block true, T1 plain false), and in reg-on cells db4 / 1 level /
  `edge_tv_kappa_q` 0.85 / `edge_tv_wfloor` 0.1. The launcher's `--seed --npts --init --support --iters` are compared
  with the resolved config. `loss_ok` fails on a missing or non-finite `dc_p` / `R_*` (reg-off: `R_geo`, `R_wav`
  required, `R_ptv` only when the trainer logs it, which the joint trainer does not).
- **Pre-existing bug fixed (ruling R9).** Since f413c26 (2026-09-26) any joint run with `loss.edge_tv=false` died with
  `AttributeError` at iteration 1 (`_edge_kappa_c` / `_edge_kappa_p` read every iteration, set only inside
  `if self._edge_tv:`). Cell C needs that path. Fix: two lines setting both to `None` before the `if` (the block
  overwrites them when edge-TV is on, so registered runs are bit-identical; `None` is never dereferenced). Side effect:
  the iso-TV reproduction commands in the project memory (configs without edge_tv) have been broken since 09-26.

## 2. Regression reference and result

The unchanged joint path was run before (tree 9df60e7, `results_arc/sg2x2_regress_before.log`) and after the edits
(`results_arc/sg2x2_regress_after.log`, card 7, 50 iterations). The four reference lines:

```
[iter     1/50] loss=1.3397e-01 dc_p=7.8237e-02 dc_c=5.4249e-02 R_geo=6.1685e-03 R_dtv=0.0000e+00 R_edge=0.0000e+00 R_vtv=0.0000e+00 R_dtvp=0.0000e+00 R_wav=7.8157e-03 R_nl=0.0000e+00 balance_violation=False
[iter    50/50] loss=4.5198e-02 dc_p=2.0174e-02 dc_c=2.0549e-02 R_geo=1.1108e-02 R_dtv=0.0000e+00 R_edge=0.0000e+00 R_vtv=0.0000e+00 R_dtvp=0.0000e+00 R_wav=2.8066e-02 R_nl=0.0000e+00 balance_violation=False
T1    psnr=37.579 ssim=0.9078 nmse=0.0227 hfen=0.7701 e_unobs=0.6379 e_unobs_xz=0.6379 head_psnr=33.807 head_ssim=0.9201 head_hfen=0.1491 N=1.099 kz_energy=0.000e+00 sharp=0.803 grain=0.0104 head_psnr_cplx=30.631 cos_phase=0.9888 comp=29.696 beat=True
FLAIR psnr=36.660 ssim=0.8922 nmse=0.0253 hfen=0.8195 e_unobs=0.7606 e_unobs_xz=0.7605 head_psnr=32.772 head_ssim=0.8968 head_hfen=0.1420 N=0.945 kz_energy=0.000e+00 sharp=0.703 grain=0.0108 head_psnr_cplx=30.002 cos_phase=0.9896 comp=22.924 beat=True
```

Result (Task 7 report, re-run by the reviewer): iter 1 / iter 50 `dc_p`, `dc_c`, `R_geo`, `R_wav` all rel = 0.00e+00;
T1 head_psnr 33.807 / psnr 37.579 and FLAIR head_psnr 32.772 / psnr 36.660, all differences +0.0000.
Output: `REGRESSION PASS` (bit-identical).

## 3. Test counts

- Baseline in this worktree before any edit (2026-10-09, card 4): `223 passed, 1 warning in 445.81s`.
- After all tasks (card 7): 277 passed + 1 failed, the failure being `tests/test_nonlocal_red.py::test_trainer_smoke_af16_with_fake_blur` with
  `torch.OutOfMemoryError` because another session took card 7 mid-run; re-run alone on card 5: `1 passed in 285.86s`.
  Net 278 = 223 + 55 new, per-file counts as the plan expected (cells 9, init support 8, single trainer 23, configs 4, manifest 7, table 4).
- Smokes (4 runs, card 1: A, C, B_T1, D_FLAIR): all four manifests audit ok.
- Final-review wave (card 0): 94 passed (manifest 46 incl. the mutant and NaN tests, init support, configs, cells, table,
  single trainer). Launcher to manifest end to end on the final tree: `WARMUP=0 bash run_sg2x2.sh sg2x2_smoke2_D_T1 0 D_T1 2 42`
  finished (rc 0, 3.6M points, 25.4 GB, ~3 min), `sg2x2_runs.log` has no `manifest FAILED` line for it, its manifest audits
  `ok: True` with the new checks; the four earlier smokes re-audited with `--iters 2`, all ok. Start-up line now ends in
  `pos_sum=319918.032643` for T1 single.

## 4. Pilot: not launched yet (awaiting the user's go) — the acceptance commands below fill this section

Plan: six runs (A, C, B_T1, B_FLAIR, D_T1, D_FLAIR), jiangyizhou AF16, seed 42, 4000 iterations, 3.6M points per
contrast, about 1 h each on an idle A100; cards 0-4 allowed by the queue. Launch (only after the go):

```bash
setsid nohup bash scripts/sg2x2_queue.sh scripts/sg2x2_pilot_s42.txt > results_arc/sg2x2_queue_pilot_s42.log 2>&1 < /dev/null &
```

When the queue drains, run (with `PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python`):

```bash
for e in A C B_T1 B_FLAIR D_T1 D_FLAIR; do echo -n "$e: "; $PY -c "import json;m=json.load(open('results_arc/sg2x2_pilot_af16_s42_$e'+'_manifest.json'));print(m['audit'], 'dirty' if m['git_dirty'] else 'clean')"; done
grep -h -E '^\[init_cloud\] mode=support_file' results_arc/sg2x2_pilot_af16_s42_*.log | sort | uniq -c
grep -l -i -E 'nan|out of memory' results_arc/sg2x2_pilot_af16_s42_*.log || echo "no NaN / OOM"
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/sg2x2_table.py --prefix sg2x2_pilot_af16 --seeds 42
```

Expected: all six audits `ok: True` and `clean`; the six `[init_cloud]` lines agree on `n=3600000 support_frac=0.4273`,
`in_own_zf_support=1.0000` in A and C, the single-cell values recorded as diagnostics (T1 cells share one value,
FLAIR cells the other; the smokes gave 0.9697 and 0.9206); all six `pos_sum=` values in the `[init_cloud]` lines equal
(float64 sum of the cloud's positions, the same-cloud check); no NaN / OOM; the table prints eight cell rows, A-B, C-D
and the interaction, for both contrasts. Resource row per cell (B and D: sum the two runs of the cell):
`jq -r '[.exp,.npts,.vram_gb,.elapsed_s]|@tsv' results_arc/sg2x2_pilot_af16_s42_*_score.json`.
Table to paste here: `PILOT, one subject, seed 42, NOT EVIDENCE`
(placeholder, empty until the pilot runs).

## 5. What is not done

- The pilot itself (needs the user's go), seeds 43 / 44.
- The 1.8M capacity supplement (`NPTS=1800000 bash run_sg2x2.sh ...`).
- The `INIT=head` end-to-end control (the parity test covers the draw, not a full run).
- The 16-pair core test: another grid, needs its own support file, a per-subject DC-centring check and its own masks. The core test's support file must come from a development subject on that grid (or a fixed-FOV / atlas support), never from the test subject's own ZF (spec §4); and `INIT=head` is not a matched init for single cells (a single `head` draw uses one contrast's own support, the joint uses the union).
- Raw-render (pre hard-DC) image metrics of spec §5.2 are not produced; every image metric in `arc/cycle.py` is on the hard-DC volume and the saved volumes are hard-DC; the raw render enters only through `e_unobs` / `nmse_unobs`; render it post hoc from `checkpoints/latest.pth` if wanted.
- The spec §4 equal-budget per-arm lambda search (A and B use the identical frozen recipe).
- Deferred minors from the ledger (none changes a result; fix in a cleanup wave):
  - Cells / launcher: `sg2x2_cells.py` docstring says COMMON but the block is inline in `cell_args`; the plan's path label is line 1 of the new files and pushes the shebang to line 2 (dead); recipe constants exist twice (argv strings and floats) with no cross-check test; tests never pass non-default iters / warmup; `_sets` keeps the last duplicate key. `run_sg2x2.sh` treats `--set` as ITERS when ITERS / SEED are omitted before extra args; the `mapfile` guard sees empty output only, not a non-zero exit.
  - Init support: `scripts/sg2x2_make_init_support.py` reads the plain slot's mask for both contrasts (correct here, assert equality before reuse); `--config` resolves against ROOT, `--out` against CWD; unclosed handle. The `in_own_zf_support` log test pins the format only; same-cloud identity rests on the CUDA generator; the start-up line now carries `pos_sum`, but the manifest does not assert it; `support_frac` means the file's fraction in the new line, the subject's own in the head line.
  - Single trainer: `--prewhiten_only contrast` raises KeyError (should be `ap.error`); inherited `export_results()` raises under the single trainer; `loss.get(k) not in (None, "", False)` lets `kspace_high_weight_plain: 0` through; `lambda_wav_plain > 0` with `lambda_wav == 0` silently disables the wavelet; `R_wav` is logged unweighted; refusal tests carry an unnecessary `@cuda`; the cycle test writes checkpoints into `results_arc/` and falls back to card 0 without `CUDA_VISIBLE_DEVICES`; the aliasing test misses `csm_fit_c` / `kspace_dc_c` / `_dc_norm_c`; the `tv_weight` compatibility branch is untested.
  - Manifest: only first / last logged iterations are checked; no test for the `[nl]` / `[guide]` / `[dtv-proj]` start-up rejection.
  - Table: `np.load` handles never closed; `row["contrast"]` would raise if a joint-named run were single; no test for the volumes / NMSE path (the pilot exercises it) or the C-D / FLAIR interaction cells.
  - Regression compare: should exit 4 with a message on a missing iter / score line instead of a KeyError; several `[iter 50]` lines, last wins silently; unclosed handle.

## 6. Rulings that shaped the code (full text in the ledger)

R1 equivalence tested on gradients, not Adam-stepped parameters; R2 launcher validates CELL and refuses empty argument
lists; R5 `.superpowers/` ignored; R6 audit also fixes `lambda_nl*` = 0, `guide_path_*` unset, `kspace_freq_adaptive`
true, `kspace_high_weight` 2.2; R7 support with dilate 0; R9 the `edge_tv=false` kappa fix; R10 smoke covers all four
cell kinds; R12 `in_own_zf_support` in single cells; R14 the launcher test keeps a permanent marker file instead of
removing one; R15 table prints `(n=k/N)` for incomplete cells and `n/a` SD at one seed.
