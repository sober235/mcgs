# Optimal plain-scan / enhanced result: data, code and parameters

User, 2026-11-24: "record the corresponding data, codes, and parameters here as the optimal values for both plain
scan and enhanced scan". Figure: ../ce_af12_prepost_clinical_verification_600dpi.pdf (MANIFEST.json, section
"provenance", lists the sha256 of every file below).

Optimal values = tuned MC-GS, AF12, seed 42. Trainer scores (head): plain scan PSNR 36.031 dB,
SSIM 0.9414; enhanced PSNR 38.473 dB, SSIM 0.9588.

| folder | file | content |
|---|---|---|
| data | mcgs_optimal_volumes.npz | tuned MC-GS output: recT1 = plain scan, recFL = enhanced (float32 magnitude, drawn in the figure), recT1_complex / recFL_complex, truth gt*, zero-filled zf* (complex64); 256 x 232 x 176 |
| data | mcgs_checkpoint_latest.pth | trained Gaussians (both scans) at the end of training; the volumes are rendered from this state |
| data | mcgs_checkpoint_best.pth | the trainer's best.pth of the same run |
| data | topic_sim_files.json | inputs of the AF12 topic, 9 files hashed (truth, coil maps, noisy k-space, mask) |
| code | trainer_source.tar.gz | source of the trainer that trained MC-GS (/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_framework_ablation_20261110/trainer; copy of source 54f8a2a) (62 files) |
| code | trainer_source_files.json | file list of trainer_source.tar.gz with sha256 |
| code | tune12.py | runner of the volumes run, job j-20261124T011133-8fbcb2a0 source snapshot |
| code | fig_af12_prepost.py | as run in figure job j-20261125T022030-76f94229 |
| code | fig_lesion_plane_prepost.py | as run in figure job j-20261125T022030-76f94229 |
| code | run_af12.py | as run in figure job j-20261125T022030-76f94229 |
| code | af12_cmp_support.py | as run in figure job j-20261125T022030-76f94229 |
| code | finalize_optimal_fig.py | as run in job j-20261125T024048-863d3d40 |
| code | comparator_scripts.tar.gz | scripts + configs of the AF12 topic and scripts of the 3D-DIP / 3D-INR support topic (30 files) |
| params | mcgs_resolved_config.yaml | the trainer config as resolved for this run |
| params | mcgs_run_info.json | run info (seed, determinism, config snapshot) (removed: timestamp_utc) |
| params | mcgs_train_score.json | trainer score file of the volumes run (removed: elapsed_s) |
| params | tune_experiments.jsonl | ledger of the tuning loop |
| params | gs_single_post_score.json | GS-single score file (September recipe) (removed: elapsed_s) |
| params | gs_single_pre_score.json | GS-single score file (September recipe) (removed: elapsed_s) |
| params | topic_summary.json | whole-head scores of the 11 methods |
| params | topic_setup.json | topic setup (script port, hashes) |
| params | params.json | all settings in one file |

MC-GS = registered recipe P3 plus `--lambda_edge 0 --set loss.edge_tv_guide_alpha=0.75 --set loss.ptv_guide_alpha=0.5
--set gaussian.rotation_lr=0.008 --set gaussian.scale_lr=0.004`, seed 42 (full argv in params/params.json).
Load the volumes: `np.load('data/mcgs_optimal_volumes.npz')['recT1']` (plain scan), `['recFL']` (enhanced).

Caveat: the MC-GS hyperparameters were selected on this scan against its truth; no other subject was tested.
No file here carries a clock value or a measured duration (project rule); cleaned files record the sha256 of the
untouched source.
