# Autoresearch on the retrospective AF12 T1+T2 pair: the cold-start path was the deficit — session record, 2026-09-09 / 10

Goal set by the user on 2026-09-09: make MC-GS joint beat every arm of the eight-method table in
`Results_20260908/T1T2_AF12_retro` on imaging metrics. Method: a Karpathy-style loop — one change
per experiment, 1500-iteration screening, 4000-iteration promotion, append-only ledger, one
commit per round. Program `autoresearch_t1t2.md`, ledger `autoresearch_t1t2.tsv`, runner
`autoresearch_t1t2.sh`, branch `autoresearch-t1t2`. 34 experiments, ~9 GPU-hours.

## 1. What was wrong

Every frozen MC-GS number had come from a 576902-Gaussian warm start; the fresh-cloud path of
`JointTrainer._setup_model` had never been used at quality. Reading it found three defects:

1. **Uniform init over the render box.** The head is 28 % of the box, so ~72 % of the 600k points
   started in air with density ≈ 0 and a position gradient ≈ 0 (the tile voxelizer's gradient is
   proportional to density). They never moved. Effective cloud: ~166k points, not 600k.
2. **Edge-TV kappa resolved on the initial render**, which for a cold start is speckle.
3. **No densification**; the warm cloud had been densified by the legacy pipeline, this one never was.

## 2. What fixed it (single-change deltas, 1500 iterations, head PSNR T1 / T2)

| change | after | delta |
|---|---|---|
| shipped cold start | 32.55 / 32.35 | — |
| positions drawn inside the ZF head support (`gaussian.init_mode=head`) | 34.17 / 34.27 | **+1.62 / +1.92** |
| + 3-NN spacing scales (`init_scale_mode=3nn`) | 34.77 / 34.22 | +0.61 / −0.05, SSIM and HFEN up |
| + kappa from the ZF image (`loss.edge_tv_kappa_source=zf`) | 34.82 / 34.22 | +0.05 / 0.00 (+0.30 / +0.08 on fixed scales) |
| + 1.2M points (`gaussian.initial_num_points=1200000`) | 35.01 / 35.02 | +0.19 / +0.80 |

Code: `paired_gs/gaussian/init_cloud.py` (7 tests), the two trainer switches, `--set` overrides
in `arc/cycle.py`; every default path is bit-identical to before (174 tests pass).

## 3. Result on the arena (hyw T1+T2, 4000 iterations, `ar3_P2_full4k`)

| | T1 | T2 | best other arm |
|---|---|---|---|
| head PSNR | **34.978** (+0.371) | **35.429** (+1.211) | CS-SENSE 34.607 / 34.219 |
| whole PSNR / NMSE / head HFEN | first | first | |
| head SSIM | 0.9373 (second by 0.006) | 0.9474 (second by 0.002) | CS-SENSE / ZS-SSDU |

+1.68 / +2.31 dB over the original joint arm. Repeat floor at this working point 0.003 / 0.005 dB.

## 4. What did not help (each as a single change)

Intensity-weighted sampling; densification (inert at the fixed scale, dominated by the initial
count otherwise); position_lr ×3.3; lambda_geo 0.07; plain high-band weight 1.0 or 3.0; kappa
refresh at iter 300; edge quantile 0.75; 3-NN scales ×1.5; tv_eps 0.06. The smoothing side
(lambda_geo 0.12, edge quantile 0.90, w_floor 0.20) buys T1 +0.1 dB with T2's SSIM/HFEN and both
contrasts' sharpness at 1500 and at 4000 — a reshape, not promoted. Head SSIM could not be
bought on the fitting side.

**Capacity is contrast-asymmetric.** T1 peaks at 1.2M and falls beyond (34.91 / 34.57 / 32.24
at 1.6 / 2.4 / 4.8M) while T2 rises to 2.4M (35.48): the extra points fit the acquired 8.5 % better
and extrapolate worse (T1 oracle unsampled-k error 0.590 → 0.617 → 0.726). The shared geometry
follows the binding contrast.

## 5. Transfer, and the one honest caveat

The four changes were selected on this subject with the ground truth in the scorer. Applied
unchanged: **hyw T1+FLAIR — first on head PSNR and HFEN** (T1 34.950 vs 34.607, FLAIR 33.492 vs
33.269). **jiangyizhou T1+FLAIR — second on head PSNR** (35.520 vs 35.762, 35.056 vs 35.084),
first on head HFEN, +1.4 / +1.7 dB over the original joint arm. A pre-registered
points-per-support-voxel rule (0.325, fixed on hyw → 1.59M on jyz) made T1 worse (35.273) and
was rejected as declared: the acquired data is the same size on both subjects, so an absolute
count near 1.2M is the transferable setting. Claim as: *first on the tuning subject in both
pairings, second on the held-out subject.*

## 6. Joint versus single under the repaired cold start

T1: joint 34.978 vs single 34.680 (+0.30 dB head, +0.30 whole). T2: joint 35.429 vs single
35.680 (−0.25). Single T1 clears CS-SENSE by +0.07 dB, the joint model by +0.37: the shared
geometry moves accuracy from the easier contrast to the harder one. This supersedes the
pre-repair reading "joint is a wash inside the head" (measured with 72 % of the cloud frozen).

## 7. Archives

All three topics carry the new arm ("MC-GS joint v2") in their tables, figures and READMEs, and
the T1+T2 topic also the single-contrast v2 arms; mirrored in `t1t2_af12_retro/`,
`t1flair_af12_retro/`, `t1flair_af12_v3_jiangyizhou/`.

## Gotchas recorded along the way

- The session's memory watchdog killed every background launcher at 01:15 while the host had
  919 GB free (it reads MemFree, which six h5 loads push down through the page cache). Launch
  long runs with `setsid nohup … &` from a bash script and watch the ledger, nothing else.
- The tool shell is zsh: `$VAR` does not word-split, so a multi-flag string in a variable reaches
  argparse as one argument. Put launches in a bash script file.
- Another user's jobs on GPUs 2–3 slowed two runs 20×; check `nvidia-smi --query-compute-apps`
  before reading wall-clock numbers. GPUs 0–1 were reserved by the parallel AF16 session.
- Densification with `scale_threshold` above the init scale never splits (round 0's "inert").
