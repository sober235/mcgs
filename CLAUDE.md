# Paired-ARC-GS / MC-GS framework — session entry

Read in this order: this file → `STATUS.md` (current state, what to re-run first) →
`autoresearch_t1t2.md` (the live campaign: program, verdicts, round log) →
`docs/superpowers/records/` (dated session records, newest first) → `.claude/memory.md`
(the long project memory, §0–19; do not re-derive anything it already closed).

## Environment
- Python: `/home/congcongliu/anaconda3/envs/GSTMI/bin/python` (torch 2.7.1+cu118), always with
  `PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- Tests: `PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests -q --ignore=tests/smoke_test.py` (174, ~2 min, needs a GPU).
- Data roots: `/data0/congcong/code/GS/Results_20260908/<TOPIC>/data` (retrospective AF12 topics:
  k-space h5, ESPIRiT maps, masks); the old T1+FLAIR bundle paths are in `configs/winner_v2*.yaml`.
- Results archive: `/data0/congcong/code/GS/Results_20260908/` — one topic per folder, README +
  scripts + logs + figures; the repo mirrors each topic (without volumes) as `<topic>/`.
- Figures are served at `http://localhost:8765/<dir>/<file>.png` from `/home/congcongliu/figs/`.

## Running things
- One MC-GS experiment: `./autoresearch_t1t2.sh EXP GPU ITERS [cycle.py args | --set key=value]`;
  `CONFIG=` picks the topic config, `SAVE_VOLUMES=1` dumps volumes, `WARMUP=20` for short smokes.
- Launch anything longer than minutes detached, from a **bash** script file:
  `setsid nohup ./autoresearch_t1t2.sh … > /dev/null 2>&1 < /dev/null &` — the session's memory
  watchdog kills background tasks on MemFree, and the tool shell (zsh) does not word-split `$VAR`.
- GPUs are shared with other users and sessions: check `nvidia-smi --query-compute-apps` first.

## Commits
- Author = the repo identity (`Congcong Liu <liuc23539@gmail.com>`), plain human messages, no
  AI trailers. Stage named files only: the working tree carries old untracked material.
- Branch topology: this framework's lineage is `master` (d0a47c0) → `af10-data-audit` →
  `af12-to-af10` → `t1t2-af12-retro` → `autoresearch-t1t2` (current). `main` / `origin/main`
  are an unrelated earlier snapshot plus the CE work; never merge across the two lineages.

## Pediatric TSC MTP line (branch `16yo-mtp`, this worktree) — stages 1-5 done (2026-09-17); the proposed arm is `gs_joint_p5`
Read `STATUS.md` first. Stage 1 built the reference k-space (spec `docs/superpowers/specs/2026-09-12-peds-tsc-mtp-grappa-design.md`,
topic `Results_20260908/Peds_TSC_MTP_3D/`, dataset `/data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT/`).
Stage 2 is the retrospective AF-12 ten-arm comparison (spec `...2026-09-12-16yo-mtp-stage2-af12-design.md`,
plan `docs/superpowers/plans/2026-09-12-16yo-mtp-stage2-af12.md`, topic
`Results_20260908/Peds_TSC_MTP_AF12_retro/` — its README is the authoritative numbers, mirrored here as
`peds_tsc_mtp_af12_retro/`, record `docs/superpowers/records/2026-09-13-16yo-mtp-af12.md`).
Result: **MC-GS does not lead this exam and joint is not better than single** (T1w head PSNR: L1-ESPIRiT 38.830 >
GS-single 38.086 > MC-GS 37.855; T2s: GS-single 34.175 ≈ MC-GS 34.160 > L1-ESPIRiT 33.860). Joint SENSE reproduces
the same pattern, so it is the contrast pair (r = 0.57, the least similar of the ten), not the implementation.
MC-INR (Niessen 2025) was added as a tenth arm on 2026-09-14 and lands eighth on both contrasts; its reproduction
lives in the worktree `/data0/congcong/code/GS/mcinr_peds_wt` (branch `peds-mtp`, no remote).
Stage 3 (closed 2026-09-15, spec `...2026-09-14-peds-af12-mcgs-autoresearch-design.md`, program `autoresearch_peds.md`,
record `docs/superpowers/records/2026-09-14-16yo-mtp-stage3.md`) tuned on this subject against the six-cell target (MC-GS
first in every head cell against every arm, GS-single counted). Not met: at matched 9.6M points the joint arm leads all
three T2s cells (34.227 / 0.9056 / 0.1836, union guide) and trails the single on T1w by 0.057 dB / 0.0002 SSIM / 0.0004
HFEN (38.936 vs 38.993); most of stage 2's deficit was cloud capacity (-0.332 at 3.6M, -0.057 at 9.6M). Stage 4 (parallel
session claude-eb, log `autoresearch_peds_p4.md`, record `...2026-09-15-16yo-mtp-stage4-lesion-slice.md`) looked for one
lesion slice 1 dB ahead of every arm and plateaued at +0.15 dB (T2s sagittal 172). **User decisions 2026-09-15 (STATUS §2):
+0.15 accepted and stage 4 closed; the six-cell target stands — MC-GS must end up first against every arm including
GS-single, so the T1w cells are still to be closed by real runs; no new training until the user has reviewed the figures;
L1-SPIRiT replaces L1-ESPIRiT (registered 2026-09-15 23:06: primary T1w 38.310 / 0.9607 / 0.0679, T2s 32.017 — it takes
the T1w head HFEN bar from GS-single, so MC-GS is 0.0089 short there); the 23 GB of stage-1 output for the other eight
contrasts stays.** Stage 5 (2026-09-16/17, record `...2026-09-16-16yo-mtp-stage5-levers.md`, README section
"Stage 5"): phase-only TV (cherry-pick 0ba2d3b; per-contrast weights `loss.lambda_ptv_plain / _contrast` added here) + 8000
iterations → `gs_joint_p5`, first on five of the six head cells against every arm incl. every GS-single (T1w 39.003 / 0.9693 /
0.0758, T2s 34.323 / 0.9111 / 0.1765; T1w HFEN stays with L1-SPIRiT 0.0679) and both-contrast first on 227 slices; figures of
record `*_53_158_150_*_allfirst_p5_zf_caps` and `*_46_149_171_*_lesionB_allfirst_p5_zf_caps` (ZF column, upper-case names).
GS-single is the published baseline: every registered version counts, none receives the joint arm's later levers (user).
Code: `uih_mtp/` (raw reader, 3D GRAPPA, `retro_mask`, `rescore`), configs `configs/peds_af12_*.yaml`, runner
`autoresearch_peds.sh`. GPU policy for this line (user, 2026-09-14): any card with no compute process and utilisation
at most 10 %, at most two of this line's trainings at once (two scheduler instances with disjoint `ALLOWED` lists);
stage 2's "card 2 only" rule is retired. **This topic's axes: axis 2 = transverse, axis 0 = coronal, axis 1 = sagittal** (adjudicated
against the vendor DICOM; the AF16 scripts' convention is the opposite and was corrected).
The two sections below describe the branch this worktree was cut from and are kept for reference.

## jiangyizhou AF16 autoresearch line (branch `autoresearch-jyz-af16`, this worktree)
Read `docs/handoff/2026-09-11-autoresearch-jyz-af16.md` first, then `autoresearch_jyz_af16.md`. Results summary:
`docs/superpowers/records/2026-09-11-autoresearch-jyz-af16.md`. Tests on this branch: 183 pass; `tests/test_score_edge.py`
needs the gitignored `results_arc/reg_aligned_4k_volumes.npz`, which only the main checkout has.

## Best 12x reconstruction so far (user decision 2026-09-11, provisional)
The user marked this line's result as the current best AF 12 reconstruction on jiangyizhou, to be used until superseded.
- Figure the user identified: `t1flair_af16_v3_jiangyizhou/figures/t1flair_v3_af16_{T1,FLAIR}_row3planes_compact_174_110_47_headerr.png`
  (served at `http://localhost:8765/t1flair_v3_af16/`), orange in-panel metrics = head-masked slice PSNR / SSIM / NMSE, error
  maps inside the head only; script `t1flair_af16_v3_jiangyizhou/scripts/fig_three_planes_compact.py`. That figure is at
  nominal AF 16 (actual T1 15.19x, FLAIR 12.49x of its acquired k-space), not AF 12.
- Recipe (P6, `autoresearch_jyz_af16.md`): repaired cold start (init_mode head, 3-NN scales, kappa from ZF), 3.6M points,
  wavelet T1 0.10 / FLAIR 0.07, edge-TV lambda_geo 0.10 / kappa_q 0.85, 4000 iterations. AF16 run `ar14_n36_v2tv_full4k`:
  head PSNR 35.552 / 34.807 (T1 / FLAIR), first on head PSNR, SSIM and HFEN over L1-ESPIRiT.
- The same recipe at AF 12 (the jyz AF12 topic's mask: one vd-Poisson mask, AF 11.71 on T1 / 9.74 on FLAIR against their
  own support): `results_arc/ar17_P6_tf_jyz_af12_full4k` — head PSNR 36.753 / 36.165, head SSIM 0.9480 / 0.9374, head HFEN
  0.1009 / 0.1002, whole-volume PSNR 40.071 / 39.539; first on every head metric of the jyz AF12 table except FLAIR head
  HFEN (level with L1-ESPIRiT). Volumes (magnitude only): `results_arc/ar17_P6_tf_jyz_af12_full4k_volumes.npz`, score json
  beside it. Not registered in `Results_20260908/T1FLAIR_AF12_v3_jiangyizhou`, whose proposed arm is the huangyuwei-selected
  v3.2 (36.299 / 36.039). No compact orange-metric figure exists at AF 12 yet.
- Caveats carried over: recipe and classical lambda were selected on this subject (fair, not held out); all numbers are on
  the magnitude gauge — on the strict complex gauge L1-ESPIRiT ranks first (record, section 4).
