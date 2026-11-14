# CE AF16: why MC-GS does not beat every opponent on the fixed three views, and the next step (2026-11-06)

Target (unchanged): fixed views axial a0/201, coronal a1/130, sagittal a2/48; pre and post; PSNR,
SSIM and NMSE strictly better than all ten opponents (18 margins), dev seeds 42 and 7. The frozen
evaluator is `mcgs_ce_losstune_20261101/fixed_views_score.py`. Keeper FV37_lambda_wav_0: score
−0.1835, 6 of 18 positive.

All numbers below come from read-only CPU analysis of saved volumes on the server.

## Diagnosis

1. **ZS-SSDU is the strongest opponent in all 18 conditions.** Every other opponent, including
   GS-single, is already beaten. ZS-SSDU's lead over GS-single is largest on the axial view
   (+1.4 dB), which is the PE plane where its 2-D network runs.
2. **The deficit is volume-wide, not specific to one view.** Head-masked volume PSNR for FV37
   vs ZS-SSDU: pre 34.95 vs 35.32, post 37.33 vs 37.53. MC-GS wins only 0–33 % of slices along
   each axis.
   - Coronal 130 is a favourable outlier: it ranks in the top 1–4 % of coronal slices for MC-GS.
   - Axial 201 lies in a poor band (z≈189–213, −0.45 to −0.74 dB) and ranks in the bottom 2–6 %
     of axial slices.
   - Strict success therefore needs roughly +0.7 dB volume-wide.
3. **Cause: missing mid/high-frequency content in unsampled k-space.** In hybrid (ky,kz) space:
   - MC-GS has lower error than ZS-SSDU on acquired samples (257 vs 296) and at low frequency
     (82 vs 124).
   - MC-GS has higher error at radius > 0.5 (2644 vs 2451), and its edge SSE is 27 % higher.
   - MC-GS keeps 56–61 % of the GT high-frequency energy; ZS-SSDU keeps 92–93 %.
4. **The levers that have been tried do not move high-frequency error.** Across 25 saved MC-GS
   candidates, high-frequency error stays in 2584–2740.
   - Two weeks of tuning gained +0.32 dB volume PSNR (gs_joint 34.69 → 35.01).
   - Longer stage-2 training does nothing: C8 m625→m1000 changes PSNR by 0.000 dB, and C57
     m625→m1250 by +0.003 dB.
   - Stronger regularisers lower the high-frequency energy ratio (0.65 → 0.56) but barely lower
     the high-frequency error.
   - Gaussian count and adaptive density (densify study `mcgs_ce_densify_20261029`, 1.2–4.0 M
     points; split/clone with evidence, gradient or gradient-accumulation rules) stay within
     ±0.02 dB of each other.
5. **Ruled out.**
   - Intensity bias: a best-fit scale gains only +0.03–0.07 dB.
   - Seed variation: seeds 42 and 7 start from different stage-1 checkpoints but end within
     0.001 dB, so the two dev seeds are nearly redundant evidence.
6. **The errors are complementary.** MC-GS and ZS-SSDU errors correlate at 0.68–0.73. Averaging
   the two volumes, purely as a diagnostic and not as a method, beats ZS-SSDU on all six views by
   0.3–0.6 dB.

Conclusion: more Gaussians or more regulariser tuning will not close the gap. The missing piece
is a learned prior for the unacquired high frequencies.

## Next step implemented here: `mcgs_zsref.py`

Joint MC-GS + zero-shot self-supervised unrolled refinement. It uses the registered ZS-SSDU
opponent protocol (15 blocks, K=10, CG 10, mu .05, lr 5e-4, patience 40, Gamma 10 %, Theta 80 %)
with three changes:

- one joint network takes pre and post together;
- the unroll starts from the MC-GS joint volumes, and those volumes are also fed as prior
  channels;
- both scans are trained in one loop.

Selection uses only the Gamma k-space loss: no GT and no ROI.

Ablations: `--separate` (not joint) and `--no-prior-input`.

Disclosure: this is a hybrid (GS + CNN) and is not a pure-GS method. It changes the method
identity and falls outside the round-3 rule "no trainer change / no post-hoc denoise". The user
authorised this direction on 2026-11-06. Expected cost: the opponent took ~7.9 GPU-h per scan;
this run does both scans in one loop.

Run (after GPUs are assigned and added to `gpu-tools/config.json`):

```sh
/usr/bin/python3 ../gpu-tools/gpu.py run --gpu ID -- @python autoresearch/ce_af16_hybrid_20261106/mcgs_zsref.py \
  --device cuda --gpu ID --init-dir <WS>/runs/FV37_lambda_wav_0/s42/m0625/relax_a1 --out <server dir outside WS>
# then score with the frozen evaluator (read-only):
@python <WS>/fixed_views_score.py --candidate <out dir>
```

The candidate volumes are 83 MB each, so `--out` should be a server directory: fetch is limited
to 50 MiB.

## CPU smoke tests (2026-11-06, gpu.py --cpu, 8 readout slices 197–211, 2 epochs x 2 steps)

These are functional checks only, not an accuracy result.

| job | mode | Gamma val before | after ep1 | after ep2 |
|---|---|---|---|---|
| j-20261106T123518-8361ca5f | direct output | failed: empty batch in smoke step count (fixed) | | |
| j-20261106T124004-00f81e25 | direct output (opponent style) | 1.4106 | 1.2325 | 1.1099 |
| j-20261106T130016-8e4c05a2 | identity-initialised residual (default) | 0.9672 | 0.9654 | 0.9224 |

The residual start keeps the MC-GS prior: its Gamma loss before training is lower than the
direct-output run reaches after training. It becomes the default; `--no-residual` is the ablation.

## Full run 1 (2026-11-06 22:16 CST)

- Round-3 controller `fixed_views_round3.py` was stopped by SIGTERM at the user's request, to free
  GPU 2. FV55 was abandoned mid-run. The keeper (FV37), results.tsv and freeze.json are unchanged.
- `gpu-tools/config.json`: `allowed_gpu_ids = [2]`.
- Job `j-20261106T141635-92d00bfa`: physical GPU 2 (A800, uuid GPU-d7cbc50e…), default settings,
  init = FV37 s42. Gamma val 0.9975 before training, 0.8929 after epoch 1. About 28 GB and
  ~120 s per epoch; at most 400 epochs with early-stop patience 40.

### Result: negative (2026-11-07)

- The job early-stopped at epoch 220 (best Gamma val 0.8694, down from 0.9975) and finished in
  7.4 h (exit 0). Output: `/home/congcongliu/.cache/mcgs-vscode/jobs/j-20261106T141635-92d00bfa/artifacts/`.
- Frozen evaluator (read-only, CPU): score **−0.1972, 3/18 positive**. FV37 s42 scores −0.1831
  with 6/18 positive. The run is slightly worse than FV37 on 17 of the 18 margins.
  - Axial 201 NMSE: pre −0.1759 (FV37 −0.1737); post −0.1972 (FV37 −0.1831).
  - Coronal 130 pre no longer passes: PSNR −0.0002, SSIM −0.0044, NMSE −0.0018.
- Head-masked volume PSNR is essentially unchanged: pre 34.99 (FV37 34.95), post 37.32 (FV37
  37.33). ZS-SSDU scores 35.32 and 37.53.

Diagnosis (scratchpad `diag_zsref.py` / `diag_zsref2.py`, read-only):

- **Most of the change is at low frequency.** 79–87 % of the update's energy is at radius < 0.1
  and only 9–14 % at radius > 0.5. The missing content is high-frequency (79–80 % of the residual
  energy is at radius > 0.5).
- **The added high frequencies are mostly wrong.** The correlation between the update and the true
  residual (GT − FV37) is +0.03 to +0.08 in the high band. The high-frequency energy ratio rose
  (0.56 → 0.67 pre), but the high-frequency error rose too (2644 → 2826 pre, 2586 → 2700 post).
- **Inside the head, low-frequency error went down** (pre 116.5 → 76.8, post 67.2 → 49.5), and so
  did the intensity bias.
- **The background got worse.** Mean |v| outside the head rose 4× (0.0053 → 0.0219; GT 0.0045).
  The head-masked metrics ignore this.
- **Rescaling the update does not help.** The best step along the update direction is ~0.5, and
  even then the gain is ~corr² ≈ 1–2 % of the residual energy, which is negligible.

Cause: the prior leaks the held-out data. MC-GS was fitted to all acquired samples, including the
Lambda (loss) and Gamma (validation) splits. Starting the unroll from MC-GS, and also feeding it as
input, means the self-supervised loss can fall by pulling the densely sampled low-frequency centre
toward the measured, noisy data. This raises the background and improves in-head low frequency,
without learning the unacquired high frequencies, which are what the target needs. For the same
reason, the Gamma loss is not a clean selection signal here. The `--separate` and
`--no-prior-input` ablations would keep the leak (the unroll still starts from MC-GS), so they
were not run.

## Leak-free run 2 (2026-11-07): `theta_prior.py` + `mcgs_zsref.py --split-file`

The user approved this on 2026-11-07.

- **Split** (`theta_prior.py split`). One fixed partition of the 2552 acquired phase points:
  - Theta 1837 (72 %), Lambda 460 (18 %), Gamma 255 (10 %).
  - Theta alone is 22.2× undersampled.
  - The seeds match the step-0 split of `mcgs_zsref.py` (Gamma seed 0, Theta/Lambda seed 1234).
  - Root on the server: `codex_artifacts/mcgs_ce_thetaprior_20261107/` (split.npz, split.json,
    mask_theta_3d.h5, g2_theta/).
- **Prior.** The full FV37 seed-42 chain is replayed with only the mask path changed to Theta:
  1. base `ar_i1_k30p90` config, 0 → 4000;
  2. densify `stack_chain.stage1`, 4000 → 4500;
  3. losstune `long_chain.stage2` with the FV37 recipe, 4500 → 5125, relax α = 1.

  The frozen modules are imported read-only. Their G2 config is redirected through the module
  global `G2` to `g2_theta/`, a copy that differs only in `mask_path`. After every stage, the mask
  the trainer actually held (from the `frequency_weights/*_acquired_mask.npy` evidence, or from the
  trainer object for the base stage) is checked to equal Theta.
  - Normalisation stays the frozen gauge. In simulation mode it comes from the fully sampled
    image, as in every earlier chain run, so it does not depend on the mask, and stage 1 still
    asserts it.
  - Gaussian initialisation and the edge-TV kappa use the zero-filled Theta image.
- **Refinement.** `mcgs_zsref.py --split-file`: DC on Theta, loss on Lambda, early stop on Gamma
  (DC on Theta ∪ Lambda). The prior is the Theta-fit MC-GS. No sample used for the loss or for
  selection was seen by the prior.
- **Inference** uses the full acquired mask and writes two candidates from the same network:
  - `zsref/`: prior = Theta-fit MC-GS. This is consistent with training and is the primary result.
  - `zsref/infer_alt/`: prior = FV37 (full-data MC-GS). This is secondary, and its train/test
    shift is disclosed.
- **Disclosure.** The frozen evaluator scores both candidates against GT, so choosing between them
  would be GT-based selection. Both will be reported.

Runs:

- **GPU smoke** `j-20261107T023710-cd6da8a7` (GPU 2). Plumbing only: base 20 iterations, stage 1
  for 2 steps from the official 4000 checkpoint, stage 2 for 2 steps from the official 4500
  checkpoint, and a 1-epoch refinement smoke.
  - Every stage reported acceleration 22.23× and passed the Theta mask check; the gauges matched
    (183.532318 / 243.540939).
  - Refinement smoke: Gamma 0.9672 before training, the same value as the earlier smoke, as
    expected (same Gamma, same DC set, same prior). `SMOKE OK`.
- **Full chain** `j-20261107T040406-e3678e01` (GPU 2), started 2026-11-07 12:04 CST.

## Diagnosis scripts

`diagnosis/` holds the read-only CPU scripts behind the numbers above. Each one prints only and
writes nothing. Run each by piping it to the server's GSTMI Python:
`ssh -p 2222 alias 'cd /tmp && OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= python -I -' < diagnosis/X.py`.

- `diag_fv.py`: per-slice PSNR along each axis, hybrid-k error split (acquired vs unacquired,
  radial bands), and edge/flat SSE, bias and high-frequency energy on axial 201.
- `diag_fv2.py`: best-scale gain, the MC-GS/ZS-SSDU average and error correlation on the six
  views.
- `diag_hf.py`: high-frequency energy and error across 25 saved MC-GS candidates and milestones.
  - Stages: base 4043 s, stage 1 564 s, stage 2 984 s. Every stage passed the Theta mask check.
    The refinement started at about 13:38.
  - **Theta-fit prior alone** (frozen evaluator): score −0.7231, 0/18 positive. View PSNR is
    1.0–1.6 dB below FV37 (for example axial 201: pre 35.07 vs 36.38, post 37.02 vs 38.65). This
    is expected, because Theta alone is 22× undersampled.
  - **Gamma-loss reference** (scratchpad `diag_gamma.py`, diagnostic only, reads GT). Each volume
    predicts Gamma directly, with no unroll; the value is the sum over pre and post:

    | volume | Gamma | Theta |
    |---|---|---|
    | GT | 0.8983 | 0.8288 |
    | ZS-SSDU | 0.8688 | 0.8009 |
    | GS-single | 0.8693 | 0.8021 |
    | FV37 | 0.8770 | 0.8089 |
    | MC-GS-Theta | 0.9985 | 0.8021 |

    - GT sits at about 0.90 because of measurement noise. Every method that saw Gamma scores
      below GT, so it fits noise.
    - Full run 1 reached 0.8694, also below the GT floor. This independently confirms the leak.
    - The honest leak-free refinement went from 1.0475 to 0.968 by epoch 28, against a floor of
      about 0.90.

### Run 2 result (2026-11-07 15:15 CST): better worst case, still 0/18

The refinement early-stopped at epoch 61, so the best Gamma loss of 0.9685 came at epoch 21.
After that the training loss kept falling (0.77 → 0.74) while the Gamma loss rose, i.e. it
overfits the single fixed Lambda set. The chain took 1.6 h for the prior plus 1.6 h for the
refinement.

Frozen evaluator (read-only, CPU):

| candidate | score | positive |
|---|---|---|
| Theta-fit prior alone | −0.7231 | 0/18 |
| run 2, prior = Theta-fit MC-GS (primary) | −0.1921 | 0/18 |
| run 2, prior = FV37 at inference (`infer_alt`) | **−0.1535** | 0/18 |
| FV37 keeper | −0.1831 | 6/18 |

Axial 201 PSNR:

| candidate | pre | post |
|---|---|---|
| Theta prior | 35.07 | 37.02 |
| run 2, Theta prior | 36.32 | 38.64 |
| run 2, FV37 prior | 36.46 | 38.85 |
| FV37 | 36.38 | 38.65 |
| ZS-SSDU | 37.08 | 39.38 |

- The FV37-prior candidate beats FV37 on axial and on sagittal post.
- It loses on coronal (pre 36.54 vs 37.01) and on SSIM everywhere, so the six coronal wins are
  gone.

Volume level (scratchpad `diag_run2.py`, head mask):

| candidate | vol PSNR pre / post | HF-energy ratio pre / post | HF error pre / post |
|---|---|---|---|
| ZS-SSDU | 35.32 / 37.53 | 0.93 / 0.94 | 2264 / 2282 |
| FV37 | 34.95 / 37.33 | 0.60 / 0.65 | 2423 / 2400 |
| Theta prior | 33.77 / 35.91 | 0.55 / 0.59 | 3017 / 3123 |
| run 2, Theta prior | 34.80 / 37.17 | 0.87 / 0.87 | 2583 / 2505 |
| run 2, FV37 prior | 34.97 / 37.37 | 0.92 / 0.92 | 2506 / 2418 |

Reading:

- With the leak removed, the network now learns high frequencies. It adds 1.0–1.3 dB of volume
  PSNR over its own prior and brings the HF-energy ratio up to ZS-SSDU's level. Background stays
  near ZS-SSDU's (0.008, against 0.022 in run 1).
- The added high frequencies are only partly right. HF error does not fall below FV37's and stays
  well above ZS-SSDU's. Net volume PSNR is FV37-level, still 0.35 dB (pre) and 0.16 dB (post)
  below ZS-SSDU.
- Likely limits, in order:
  1. The single fixed Theta/Lambda split overfits by epoch ~21. The opponent resamples Theta and
     Lambda every step.
  2. The prior is weaker (22× undersampled).
  3. There is a train/inference shift when the FV37 prior is used.

## Run 3 (2026-11-07 21:22 CST): resample Lambda every step (`--lam-dc-frac 0.5`)

This run targets limit 1. The prior and the split stay as in run 2 (Theta prior from
`chain/stage2/m0625/relax_a1`, same `split.npz`). At every step a fresh random half of Lambda joins
Theta for data consistency, and the other half is the loss:

- DC: Theta ∪ Lambda_a, 2067 phase points; loss: Lambda_b, 230 points.
- The draw is seeded by the step number. Two consecutive steps share about half their loss points
  (115 in the startup check).
- It stays leak-free: the prior was fitted on Theta only, and the loss never contains a Theta
  point. Gamma is still used only for early stopping. Validation and inference are unchanged:
  validation runs DC on Theta ∪ Lambda, and inference runs DC on every acquired point.
- A startup check asserts disjointness and the partition of Theta ∪ Lambda. It also asserts that
  no loss point is in Theta and none of DC or loss is in Gamma.

GPU smoke j-20261107T131333-83794105 (8 slices, 2 epochs) passed. Full run
j-20261107T132230-34dffca1 on GPU 2, output `ROOT/chain/zsref_lamrs50` (+ `infer_alt/` with the
FV37 prior). Same budget as run 2: 400 epochs at most, patience 40.
