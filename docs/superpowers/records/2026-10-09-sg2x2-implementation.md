# Record — Shared-geometry 2x2 ablation: implementation (jiangyizhou AF16, 2026-10-09)

Branch `exp/shared-geometry-2x2-ablation-20261009` (worktree `mcgs_sg2x2_wt`, start 9df60e7). Spec
`docs/experiments/2026-10-09-mcgs-shared-geometry-2x2-ablation.md`, plan
`docs/superpowers/plans/2026-10-09-sg2x2-shared-geometry-ablation.md`. Execution ledger with every ruling:
`.superpowers/sdd/2026-10-09-sg2x2-shared-geometry-ablation/progress.md` (gitignored, local to this worktree).

**State at the tag `handoff/2026-10-09-sg2x2-pilot-r3`: code complete, pilot run (seed 42, one subject) — §4.**
Tags `-pilot` (53becc8) and `-pilot-r2` (53d38db) marked "code complete, pilot pending"; the pilot was launched on the
user's go at 12:39 and drained at 21:13 the same day. Nothing in this record is a result: §4 is a pilot on one subject
and one seed, labelled NOT EVIDENCE.

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

## 4. Pilot results: PILOT, one subject, seed 42, NOT EVIDENCE

**PILOT, one subject, seed 42, NOT EVIDENCE.** One subject (jiangyizhou), one seed (42), one acceleration (AF16).
Everything in this section checks that the machinery works (audit, init, table) and records what this single run
gave; none of it is a finding about shared geometry. This section supersedes the "pilot pending" state line at the top
of this record, which describes only the tag `handoff/2026-10-09-sg2x2-pilot` (53becc8). The pilot ran on the later
commit 53d38db (tag `handoff/2026-10-09-sg2x2-pilot-r2`).

Plan as run: six runs (A, C, B_T1, B_FLAIR, D_T1, D_FLAIR), jiangyizhou AF16, seed 42, 4000 iterations, 3.6M points per
contrast, cards 0-4 allowed by the queue. Launched on the user's go:

```bash
setsid nohup bash scripts/sg2x2_queue.sh scripts/sg2x2_pilot_s42.txt > results_arc/sg2x2_queue_pilot_s42.log 2>&1 < /dev/null &
```

### 4.1 Timeline (2026-10-09)

Sources: `results_arc/sg2x2_queue_pilot_s42.log` (launches, drain), `results_arc/sg2x2_runs.log` (run ends, return codes),
the run logs, and the section "Pilot" of the execution ledger.

- 12:39 queue launched on the user's go. A, C and B_T1 started at 12:39 / 12:40 / 12:42 on cards 0 / 1 / 3.
- From about 15:30 jobs of another session (the CE line, `ar_queue.py`) shared those three cards, and both sides
  stalled. Our logs stopped advancing at iter 1800 (A, last write 16:11), 2200 (C, 18:31) and 2900 (B_T1, 16:46) of
  4000. The GPU read "100 %" utilisation with 1 % memory utilisation and 85 W (ledger).
- 16:46 B_FLAIR started on card 2 (the queue launches only on an idle card). 18:34 B_FLAIR finished (6410 s) and D_T1
  started on card 2.
- 18:44 (ruling R17) our three stalled runs were stopped, and only those (rc=143 for B_T1, C and A in
  `results_arc/sg2x2_runs.log`; `arc/cycle.py` has no resume support). They were re-queued as
  `sg2x2_pilot2_af16_s42_{A,C,B_T1}` by appending three lines to `scripts/sg2x2_pilot_s42.txt`, which the queue runner
  re-reads. The CE jobs went back to full speed the moment ours stopped (ledger: 182-229 W, 42-77 % memory
  utilisation).
- 19:06 D_T1 finished (rc=0) and D_FLAIR started on card 2; 19:35 D_FLAIR finished.
- Re-runs under the new prefix: A on card 2, 19:35 to 20:39 (3810 s); C on card 3, 19:50 to 20:45 (3261 s); B_T1 on
  card 2, 20:40 to 21:13 (1962 s). The queue drained at 21:13.

Outputs by prefix. Original prefix `sg2x2_pilot_af16_s42_`: B_FLAIR, D_T1 and D_FLAIR finished (rc=0). The three stopped
runs left logs only (no score, manifest or volumes), contain no NaN / OOM line, and are not results. New prefix
`sg2x2_pilot2_af16_s42_`: A, C and B_T1. For the table the three finished original-prefix outputs are symlinked under
the `sg2x2_pilot2_af16` names (`_score.json`, `_volumes.npz`, `_manifest.json`, `.log`), so
`scripts/sg2x2_table.py --prefix sg2x2_pilot2_af16 --seeds 42` reads all six. The `exp` field inside those three score
files still carries the original-prefix name.

### 4.2 Acceptance checks (plan Task 8 step 3, run with the `sg2x2_pilot2_af16` prefix)

Run on 2026-10-09 after the queue drained, with `PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python`. Audits:

```bash
for e in A C B_T1 B_FLAIR D_T1 D_FLAIR; do echo -n "$e: "; $PY -c "import json;m=json.load(open('results_arc/sg2x2_pilot2_af16_s42_$e'+'_manifest.json'));print(m['audit'], 'dirty' if m['git_dirty'] else 'clean')"; done
```
```
A: {'resolved_ok': True, 'startup_ok': True, 'loss_ok': True, 'ok': True} dirty
C: {'resolved_ok': True, 'startup_ok': True, 'loss_ok': True, 'ok': True} dirty
B_T1: {'resolved_ok': True, 'startup_ok': True, 'loss_ok': True, 'ok': True} dirty
B_FLAIR: {'resolved_ok': True, 'startup_ok': True, 'loss_ok': True, 'ok': True} clean
D_T1: {'resolved_ok': True, 'startup_ok': True, 'loss_ok': True, 'ok': True} dirty
D_FLAIR: {'resolved_ok': True, 'startup_ok': True, 'loss_ok': True, 'ok': True} dirty
```

Commit and dirty paths (same manifests):

```bash
for e in A C B_T1 B_FLAIR D_T1 D_FLAIR; do $PY -c "import json;m=json.load(open('results_arc/sg2x2_pilot2_af16_s42_$e'+'_manifest.json'));print('$e', m['git_head'][:7], m['git_dirty'])"; done
```
```
A 53d38db ['scripts/sg2x2_pilot_s42.txt']
C 53d38db ['scripts/sg2x2_pilot_s42.txt']
B_T1 53d38db ['scripts/sg2x2_pilot_s42.txt']
B_FLAIR 53d38db []
D_T1 53d38db ['scripts/sg2x2_pilot_s42.txt']
D_FLAIR 53d38db ['scripts/sg2x2_pilot_s42.txt']
```

Initial cloud (the `uniq -c` count is the number of logs carrying that line):

```bash
grep -h -E '^\[init_cloud\] mode=support_file' results_arc/sg2x2_pilot2_af16_s42_*.log | sort | uniq -c
```
```
      2 [init_cloud] mode=support_file path=/data0/congcong/code/GS/mcgs_sg2x2_wt/configs/sg2x2/init_support_af16_jyz.npz n=3600000 support_frac=0.4273 in_own_zf_support=0.9206 pos_sum=319918.032643
      2 [init_cloud] mode=support_file path=/data0/congcong/code/GS/mcgs_sg2x2_wt/configs/sg2x2/init_support_af16_jyz.npz n=3600000 support_frac=0.4273 in_own_zf_support=0.9697 pos_sum=319918.032643
      2 [init_cloud] mode=support_file path=/data0/congcong/code/GS/mcgs_sg2x2_wt/configs/sg2x2/init_support_af16_jyz.npz n=3600000 support_frac=0.4273 in_own_zf_support=1.0000 pos_sum=319918.032643
```

NaN / out of memory, then the table (its output is the table in 4.4):

```bash
grep -l -i -E 'nan|out of memory' results_arc/sg2x2_pilot2_af16_s42_*.log || echo "no NaN / OOM"
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/sg2x2_table.py --prefix sg2x2_pilot2_af16 --seeds 42
```
```
no NaN / OOM
```

What the checks say, against the values the plan expected:

- Audits: all six `ok: True` (resolved, startup and loss all True). All six manifests carry `git_head` 53d38db.
- The plan expected `clean`; five read `dirty`. The only dirty path in each is `scripts/sg2x2_pilot_s42.txt`, the queue
  file with the three re-run lines appended at 18:44. It is not code. `git_dirty` is read when the manifest is written,
  which is the end of a run (`scripts/sg2x2_manifest.py`, called by `run_sg2x2.sh` after `cycle.py` returns), so every
  run that ended after 18:44 carries the mark: A, C, B_T1, D_FLAIR, and also D_T1 (started 18:34, ended 19:06). Only
  B_FLAIR (manifest written 18:34) reads `[]`. The queue file is passed on the command line to
  `scripts/sg2x2_queue.sh` and no `.py` or `.sh` file names it, and the dirty list of each manifest holds only that
  file, so the code that ran is the commit 53d38db in all six.
- Initial cloud: all six `[init_cloud]` lines read `n=3600000 support_frac=0.4273 pos_sum=319918.032643`, the identical
  initial cloud in all six cells (same count, same support fraction, same float64 position sum). `in_own_zf_support` is
  1.0000 in A and C, 0.9697 in the T1 single cells (B_T1, D_T1) and 0.9206 in the FLAIR single cells (B_FLAIR,
  D_FLAIR): below 1 in single cells by construction (ruling R12), and the same three values as the smokes.
- No NaN / out-of-memory line in any of the six logs.
- The table has the eight cell rows, A-B, C-D and the interaction, for both contrasts (4.4). Run again into a scratch
  file with `--out`, it reproduces `results_arc/sg2x2_pilot2_af16_table.md` byte for byte.

### 4.3 Resources

From the score jsons (`jq -r '[.exp,.npts,.vram_gb,.elapsed_s]|@tsv' results_arc/sg2x2_pilot2_af16_s42_*_score.json`;
`vram_gb` is the process's `torch.cuda.max_memory_allocated()`, `elapsed_s` the wall time of the run):

| run | cell | points | peak VRAM (GB) | elapsed (s) | card |
|---|---|---|---|---|---|
| `sg2x2_pilot2_af16_s42_A` | A | 3.6M | 32.9 | 3810 | 2 |
| `sg2x2_pilot2_af16_s42_C` | C | 3.6M | 32.9 | 3261 | 3 |
| `sg2x2_pilot2_af16_s42_B_T1` | B | 3.6M | 25.4 | 1962 | 2 |
| `sg2x2_pilot_af16_s42_B_FLAIR` | B | 3.6M | 27.9 | 6410 | 2 |
| `sg2x2_pilot_af16_s42_D_T1` | D | 3.6M | 25.4 | 1871 | 2 |
| `sg2x2_pilot_af16_s42_D_FLAIR` | D | 3.6M | 27.9 | 1701 | 2 |

B_FLAIR's 6410 s was measured on a shared card and is not a speed measurement (D_FLAIR, the same shape, took 1701 s).
A and C hold one shared cloud of 3.6M points for both contrasts. B and D hold one 3.6M cloud per contrast, so 7.2M
points per arm over two runs (spec §4 asks to report that the independent arms are larger). GPU time per cell, B and D
summed over their two runs: B 8373 s (includes the shared-card B_FLAIR), D 3572 s. The other five runs took 28 to 64 min
each; the plan assumed about 1 h. Failures: no NaN / OOM; three runs were stopped by us at 18:44 and re-run (4.1).

### 4.4 Table (verbatim from `results_arc/sg2x2_pilot2_af16_table.md`)

**PILOT, one subject, seed 42, NOT EVIDENCE**

Head-only metrics (|GT| > 0.05 P99.9), mean ± SD over 1 seed(s) [42]; differences are per-seed paired differences. PILOT numbers on one subject are engineering checks, not evidence.

| cell | contrast | PSNR | SSIM | NMSE | HFEN |
|---|---|---|---|---|---|
| A | T1 | 35.645 ± n/a | 0.9388 ± n/a | 0.01314 ± n/a | 0.1151 ± n/a |
| A | FLAIR | 34.876 ± n/a | 0.9253 ± n/a | 0.01389 ± n/a | 0.1134 ± n/a |
| B | T1 | 35.639 ± n/a | 0.9387 ± n/a | 0.01315 ± n/a | 0.1157 ± n/a |
| B | FLAIR | 34.898 ± n/a | 0.9256 ± n/a | 0.01382 ± n/a | 0.1139 ± n/a |
| C | T1 | 23.775 ± n/a | 0.4952 ± n/a | 0.20202 ± n/a | 0.3886 ± n/a |
| C | FLAIR | 24.854 ± n/a | 0.5402 ± n/a | 0.13958 ± n/a | 0.3339 ± n/a |
| D | T1 | 22.212 ± n/a | 0.4274 ± n/a | 0.28953 ± n/a | 0.4723 ± n/a |
| D | FLAIR | 23.805 ± n/a | 0.4930 ± n/a | 0.17769 ± n/a | 0.3828 ± n/a |
| A−B | T1 | +0.006 ± n/a | +0.0001 ± n/a | −0.00002 ± n/a | −0.0006 ± n/a |
| A−B | FLAIR | −0.022 ± n/a | −0.0003 ± n/a | +0.00007 ± n/a | −0.0004 ± n/a |
| C−D | T1 | +1.563 ± n/a | +0.0679 ± n/a | −0.08751 ± n/a | −0.0837 ± n/a |
| C−D | FLAIR | +1.049 ± n/a | +0.0472 ± n/a | −0.03812 ± n/a | −0.0489 ± n/a |
| (A−B)−(C−D) | T1 | −1.557 ± n/a | −0.0678 ± n/a | +0.08749 ± n/a | +0.0831 ± n/a |
| (A−B)−(C−D) | FLAIR | −1.071 ± n/a | −0.0476 ± n/a | +0.03819 ± n/a | +0.0485 ± n/a |

### 4.5 Sanity check against the registered numbers

The registered numbers are in `docs/superpowers/records/2026-09-20-coupling-attribution.md` of the worktree
`/data0/congcong/code/GS/mcgs_hparam_wt` (section 4 table, lines 90-95; the lambda_edge on/off table, line 210). Same
subject, AF16, head-only PSNR in dB at full precision:

| | T1 | FLAIR |
|---|---|---|
| J0: joint at lambda_edge 0, phase-3 recipe, head init (`ca_j_edge0`) | 35.644656 | 34.875808 |
| cell A, measured here (`support_file` init) | 35.644773 | 34.875728 |
| A − J0 | +0.0001 | −0.0001 |
| S0: old single, same contrast in both slots, lambda_edge 0 (`ca_s_t1_edge0`, `ca_s_fl_edge0`) | 35.598942 | 34.904343 |
| cell B, measured here (true single, one contrast per process) | 35.639252 | 34.897953 |
| B − S0 | +0.040 | −0.006 |

In that record J0 = S0 + M0 = 35.598942 + 0.045714 (T1) and 34.904343 − 0.028535 (FLAIR).

- Cell A reproduces the registered recipe's joint arm: +0.0001 / −0.0001 dB. So the `support_file` init reproduces the
  `head` init on this subject, at the level of the final head PSNR after 4000 iterations. This pins the claim that
  section 1 (frozen support) left indirect: identical iter-1 numbers plus a parity test on an identical support, not a
  positions equality. The iteration-1 loss values of both A runs (the stopped one and the re-run) also equal those of the
  regression reference's iteration-1 line digit for digit. Runs are not bit-deterministic (`deterministic=False`, TF32
  on), so this is agreement to 1e-4 dB, not identity.
- The true singles against the old duplicate-slot proxy S0: B_T1 is +0.040 dB (it starts from the same union-support
  cloud as A, and the duplicated slot of the proxy doubled the loss), B_FLAIR is −0.006 dB (the proxy scored FLAIR in
  the plain slot, which carried T1's wavelet weight 0.12 instead of FLAIR's own 0.10; `run_ca_mcgs.sh` lines 37-38,
  `scripts/ca_summarize.py` lines 36-48, spec line 76). These reasons follow from how the proxy was built; this pilot
  does not test them separately.

### 4.6 Reading, in the spec's own terms (§5.3 table)

PILOT, one subject, seed 42, NOT EVIDENCE. The pilot falls in the row "A ≈ B but C > D" (spec: without regularisation
the shared structure has an effect, but strong single-sequence regularisation may replace part of the shared benefit).

- Per-channel priors on (A−B): shared geometry buys +0.006 dB on T1 and −0.022 dB on FLAIR. T1 is inside the
  pre-registered 0.02 dB indistinguishability band of the coupling-attribution study (its lines 115-118); FLAIR is at
  the edge of it, 0.002 dB beyond. The signs differ between the contrasts, so by the spec's last row they are reported
  separately and not averaged.
- Every explicit prior off (C−D): shared geometry is worth +1.56 dB on T1 and +1.05 dB on FLAIR.
- Interaction ((A−B)−(C−D)): −1.56 dB on T1 and −1.07 dB on FLAIR. On this subject the per-channel priors replace the
  whole measurable shared-geometry benefit, where the spec row allows for part of it.
- Level without priors: C 23.8 / 24.9 dB and D 22.2 / 23.8 dB (T1 / FLAIR) are 10 to 13 dB below the regularised
  cells (A−C 11.9 / 10.0 dB, B−D 13.4 / 11.1 dB).

### 4.7 Caveats

- n = 1 subject and 1 seed: the SD is `n/a` and there is no interval. The spec makes the subject the statistical unit,
  so nothing here supports a claim; seeds 43 / 44 and more subjects are not run (section 5).
- A and B share the identical frozen recipe, tuned historically on the joint arm. The equal-budget own-lambda search of
  spec §4 is out of scope here, which departs from the spec's rule that the single arm gets the same search. B could
  only improve under its own search, which makes A−B ≤ 0 more likely, not less: the measured A−B is, if anything,
  tilted toward A.
- The no-regulariser cells (C, D) sit 10 to 13 dB below the regularised cells, far from any practical operating point.
  C−D says what shared geometry does when nothing else constrains the fit, not what it does in a usable
  reconstruction, so the interaction compares a usable operating point (A, B) with an unusable one (C, D).
- Capacity is matched per contrast (3.6M each), not in total: B and D hold 7.2M points per arm against 3.6M in A and C.
  The 1.8M capacity-matched supplement is not run.
- The registered seed-to-seed floor of this recipe on this subject is 0.001 / 0.000 dB (same record, lines 115-118),
  so A−B of +0.006 / −0.022 dB is larger than that registered noise. This pilot did not measure its own run-to-run
  noise, and whether the difference is an effect of the arm or of this one subject cannot be told from one subject.

### 4.8 Development subject, seeds 42 / 43 / 44, both capacity conventions (2026-10-10; one subject, NOT EVIDENCE)

Runs: seeds 43 and 44 of the six cells (queue `scripts/sg2x2_dev_seeds.txt`, 02:52 to 09:01); seed 42 is the pilot,
symlinked under the `sg2x2_dev_af16` prefix. Capacity-matched convention of spec §4 ("资源容量匹配"): B and D at
`NPTS=1800000` per contrast against A and C at 3.6M shared (A/C symlinked). `bd18_s43_B_FLAIR` was stopped at 1300/4000
when another user's job landed on its card in the same minute, and re-run as `bd18b_s43_B_FLAIR`; the table prefix
`sg2x2_dev_af16_bd18b` holds 68 links to the other bd18 outputs plus that re-run. Every run rc=0 (except the stopped
one), every manifest audit ok, one `pos_sum` per (seed, point count) across all cells. The spec asks for the two
conventions to be reported separately and never mixed.

Main convention (B and D at 3.6M per contrast), verbatim from `results_arc/sg2x2_dev_af16_table.md`:

Head-only metrics (|GT| > 0.05 P99.9), mean ± SD over 3 seed(s) [42, 43, 44]; differences are per-seed paired differences. PILOT numbers on one subject are engineering checks, not evidence.

| cell | contrast | PSNR | SSIM | NMSE | HFEN |
|---|---|---|---|---|---|
| A | T1 | 35.641 ± 0.007 | 0.9387 ± 0.0001 | 0.01315 ± 0.00002 | 0.1152 ± 0.0002 |
| A | FLAIR | 34.873 ± 0.005 | 0.9252 ± 0.0001 | 0.01390 ± 0.00002 | 0.1136 ± 0.0002 |
| B | T1 | 35.635 ± 0.007 | 0.9386 ± 0.0001 | 0.01317 ± 0.00002 | 0.1159 ± 0.0002 |
| B | FLAIR | 34.895 ± 0.006 | 0.9255 ± 0.0001 | 0.01383 ± 0.00002 | 0.1140 ± 0.0002 |
| C | T1 | 23.785 ± 0.010 | 0.4955 ± 0.0004 | 0.20157 ± 0.00044 | 0.3887 ± 0.0004 |
| C | FLAIR | 24.856 ± 0.003 | 0.5403 ± 0.0001 | 0.13950 ± 0.00008 | 0.3337 ± 0.0002 |
| D | T1 | 22.222 ± 0.010 | 0.4275 ± 0.0003 | 0.28891 ± 0.00066 | 0.4721 ± 0.0004 |
| D | FLAIR | 23.809 ± 0.004 | 0.4931 ± 0.0001 | 0.17752 ± 0.00017 | 0.3824 ± 0.0004 |
| A−B | T1 | +0.006 ± 0.001 | +0.0001 ± 0.0000 | −0.00002 ± 0.00000 | −0.0007 ± 0.0000 |
| A−B | FLAIR | −0.022 ± 0.002 | −0.0003 ± 0.0000 | +0.00007 ± 0.00000 | −0.0004 ± 0.0000 |
| C−D | T1 | +1.563 ± 0.001 | +0.0680 ± 0.0001 | −0.08734 ± 0.00023 | −0.0834 ± 0.0002 |
| C−D | FLAIR | +1.047 ± 0.002 | +0.0472 ± 0.0001 | −0.03802 ± 0.00009 | −0.0487 ± 0.0004 |
| (A−B)−(C−D) | T1 | −1.557 ± 0.001 | −0.0679 ± 0.0001 | +0.08732 ± 0.00023 | +0.0828 ± 0.0003 |
| (A−B)−(C−D) | FLAIR | −1.069 ± 0.002 | −0.0475 ± 0.0001 | +0.03810 ± 0.00009 | +0.0482 ± 0.0004 |

Capacity-matched convention (B and D at 1.8M per contrast, A and C at 3.6M shared), verbatim from
`results_arc/sg2x2_dev_af16_bd18b_table.md`:

Head-only metrics (|GT| > 0.05 P99.9), mean ± SD over 3 seed(s) [42, 43, 44]; differences are per-seed paired differences. PILOT numbers on one subject are engineering checks, not evidence.

| cell | contrast | PSNR | SSIM | NMSE | HFEN |
|---|---|---|---|---|---|
| A | T1 | 35.641 ± 0.007 | 0.9387 ± 0.0001 | 0.01315 ± 0.00002 | 0.1152 ± 0.0002 |
| A | FLAIR | 34.873 ± 0.005 | 0.9252 ± 0.0001 | 0.01390 ± 0.00002 | 0.1136 ± 0.0002 |
| B | T1 | 35.585 ± 0.005 | 0.9385 ± 0.0001 | 0.01332 ± 0.00001 | 0.1150 ± 0.0002 |
| B | FLAIR | 34.730 ± 0.001 | 0.9233 ± 0.0000 | 0.01436 ± 0.00000 | 0.1143 ± 0.0001 |
| C | T1 | 23.785 ± 0.010 | 0.4955 ± 0.0004 | 0.20157 ± 0.00044 | 0.3887 ± 0.0004 |
| C | FLAIR | 24.856 ± 0.003 | 0.5403 ± 0.0001 | 0.13950 ± 0.00008 | 0.3337 ± 0.0002 |
| D | T1 | 26.694 ± 0.007 | 0.6251 ± 0.0002 | 0.10317 ± 0.00018 | 0.2897 ± 0.0007 |
| D | FLAIR | 28.188 ± 0.006 | 0.6855 ± 0.0003 | 0.06477 ± 0.00010 | 0.2370 ± 0.0003 |
| A−B | T1 | +0.056 ± 0.002 | +0.0002 ± 0.0000 | −0.00017 ± 0.00001 | +0.0002 ± 0.0000 |
| A−B | FLAIR | +0.143 ± 0.004 | +0.0019 ± 0.0000 | −0.00046 ± 0.00001 | −0.0007 ± 0.0000 |
| C−D | T1 | −2.909 ± 0.003 | −0.1296 ± 0.0002 | +0.09840 ± 0.00027 | +0.0990 ± 0.0008 |
| C−D | FLAIR | −3.332 ± 0.005 | −0.1452 ± 0.0003 | +0.07473 ± 0.00008 | +0.0967 ± 0.0001 |
| (A−B)−(C−D) | T1 | +2.964 ± 0.003 | +0.1298 ± 0.0002 | −0.09857 ± 0.00027 | −0.0988 ± 0.0008 |
| (A−B)−(C−D) | FLAIR | +3.475 ± 0.007 | +0.1471 ± 0.0002 | −0.07519 ± 0.00009 | −0.0974 ± 0.0001 |

Reading (one subject, three seeds):
- Seed spread is at most 0.01 dB in every cell; seeds 43 and 44 reproduce the pilot.
- With the per-channel priors, A against B is about zero in the main convention (+0.006 / −0.022 dB) and small but
  positive at matched budget (+0.056 / +0.143 dB): halving B's points costs it 0.05 / 0.17 dB, so with priors on, shared
  geometry spends a given point budget slightly better.
- Without priors, the sign of C − D flips with the convention. D at 1.8M (26.694 / 28.188 dB) is 4.5 / 4.4 dB above D at
  3.6M (22.222 / 23.809 dB): without explicit priors, fewer points act as a regulariser. C − D is +1.563 / +1.047 dB in
  the main convention and −2.909 / −3.332 dB at matched budget. The main-convention "C > D" of 4.6 is therefore at
  least as much a capacity effect as a shared-geometry effect. The unregularised cells stay far from any practical
  operating point (4.7).
- The core-test plan (2026-10-10) runs the main convention only; the matched convention on the test subjects is not
  planned (it would add four runs per subject).

## 5. What is not done

- The pilot is done (section 4: one subject, seed 42, NOT EVIDENCE). Not done: seeds 43 / 44 (the spec's repeat design is
  42 / 43 / 44, six cells each) and more subjects.
- The 1.8M capacity supplement (`NPTS=1800000 bash run_sg2x2.sh ...`; B and D at 1.8M per contrast against the 3.6M of
  A and C).
- The `INIT=head` end-to-end control in this tree (the parity test covers the draw, not a full run). The pilot's cell A
  against the registered J0 (4.5) is an indirect check across trees, not this control.
- The 16-pair core test: another grid, needs its own support file, a per-subject DC-centring check and its own masks. The core test's support file must come from a development subject on that grid (or a fixed-FOV / atlas support), never from the test subject's own ZF (spec §4); and `INIT=head` is not a matched init for single cells (a single `head` draw uses one contrast's own support, the joint uses the union).
- Raw-render (pre hard-DC) image metrics of spec §5.2 are not produced; every image metric in `arc/cycle.py` is on the hard-DC volume and the saved volumes are hard-DC; the raw render enters only through `e_unobs` / `nmse_unobs`; render it post hoc from `checkpoints/latest.pth` if wanted.
- The spec §4 equal-budget per-arm lambda search (A and B use the identical frozen recipe; see the second caveat in 4.7).
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
