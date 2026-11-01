# MC-GS CE AF16 long-budget / LR-schedule autoresearch (2026-10-31)

Adapted Karpathy autoresearch (upstream `karpathy/autoresearch@228791f`). Continues the CE AF16
plain (pre) + contrast-enhanced (post) campaign; see
`/home/congcongliu/codex_artifacts/mcgs_ce_karpathy_20261022_relocation/AGENTS.md` for the full
campaign invariants, which still apply (fixed data/mask/CSM/normalization/ten opponents/evaluator,
no GT/ROI in training, no major architecture change, disclose single-case adaptive development).

## Settings agreed with the user (2026-10-31)

1. Workspace: this directory only. AR / K / densify workspaces are read-only imports.
2. Editable surface: `recipe.json` only (steps, milestones, gamma, lr_scale, geo_lr_scale, profile, alpha).
   `long_chain.py` is the harness; change it only for infrastructure, never during a comparison.
3. Fixed evaluator: `mcgs_ce_karpathy_20261022_relocation/evaluate.py`, unchanged, on the relax export.
4. Primary metric: lesion-plane S6 (best identical plane, min of six normalized margins over pre+post
   PSNR/SSIM/NMSE vs all ten opponents), higher is better. Success = S6 > 0 (strict six-way win).
   Whole-head score is reported as supplemental.
5. Direction for this round: longer training budget and learning-rate schedule.
6. Noise: every candidate runs on both seed bases (42 = i1_ctrl, 7 = b1s7_ctrl stage-1 checkpoints).
   Keep only if the two-seed mean relax S6 at some milestone beats the current keeper mean by more
   than 0.0001 (`CONTROLS` in long_chain.py points at the keeper's two objectives). Seed spread is about 0.0012.
   Keeper history: control chain 500 steps -0.023379; L1 milestone 1000 -0.021997;
   L4 (lr_scale 0.5) milestone 250 -0.020846; L6 (lr_scale 0.25) milestone 625 -0.020476.
7. GPUs: `pick_gpu.py` rule. Use empty GPUs but always leave one empty GPU free; when only one empty
   GPU remains, use a lightly loaded GPU (util <= 60%, >= 55 GB free) instead. Never touch other users' jobs.

## Loop

1. Write one idea as `recipes/<exp>.json`, `git commit`. `recipe.json` holds the current keeper recipe.
   Parallel candidates are separate recipe files run at the same time (each uses two GPUs).
2. `python long_chain.py run --name <exp> --recipe recipes/<exp>.json > logs/<exp>.log 2>&1` (GSTMI python; launches both seeds).
3. `python long_chain.py ledger --name <exp> --description "..."`.
4. keep: advance; discard: `git reset --hard` to the last kept commit (this repo only).

`results.tsv` columns: commit, exp, mean_s6, delta_s6_vs_control, mean_wh, delta_wh, per_seed_delta_s6, status, description.
