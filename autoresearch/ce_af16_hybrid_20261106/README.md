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

## Diagnosis scripts

`diagnosis/` holds the read-only CPU scripts behind the numbers above. Each one prints only and
writes nothing. Run each by piping it to the server's GSTMI Python:
`ssh -p 2222 alias 'cd /tmp && OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= python -I -' < diagnosis/X.py`.

- `diag_fv.py`: per-slice PSNR along each axis, hybrid-k error split (acquired vs unacquired,
  radial bands), and edge/flat SSE, bias and high-frequency energy on axial 201.
- `diag_fv2.py`: best-scale gain, the MC-GS/ZS-SSDU average and error correlation on the six
  views.
- `diag_hf.py`: high-frequency energy and error across 25 saved MC-GS candidates and milestones.
