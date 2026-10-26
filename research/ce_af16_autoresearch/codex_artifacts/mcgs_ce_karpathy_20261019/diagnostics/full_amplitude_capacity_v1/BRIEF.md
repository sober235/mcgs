# Frozen G2 coefficient-modulus approximation probe

The frozen Gaussian log scales improve this small network's fit to existing G2 coefficient moduli. The fit remains substantially approximate. These measurements support testing geometry-conditioned amplitude functions; they do not establish MRI quality, a capacity limit, or a replacement model ready to train.

This diagnostic read only the SHA-verified G2 iteration4500 checkpoint. It did not read k-space, CSM, masks, GT, comparators, ROIs or reconstructed images, and did not use a GPU. It adds **zero formal MRI experiments**. It does not change any earlier caveat about the benchmark's reference-derived normalization.

## Observed results

All values below are on the same 20,000 held-out Gaussian indices, after training on 60,000 other indices. Each pair is **plain / enhanced**. The holdout tests coefficient approximation within one existing G2 model. It is not an unseen-measurement or unseen-case test: G2 already used all acquired measurements before this partition was made.

| Function | Held-out log-modulus R² | Held-out relative modulus L2 | Parameters | Actual updates |
|---|---:|---:|---:|---:|
| Constant training median modulus | −0.0009 / −0.0028 | 0.6626 / 0.6454 | 2 | 0 |
| Coordinates, Fourier bands4 | 0.2150 / 0.1902 | 0.5857 / 0.5821 | 6,082 | 1,000 |
| Coordinates, Fourier bands8 | 0.1994 / 0.1814 | 0.5912 / 0.5862 | 7,618 | 1,000 |
| Coordinates, bands8 + three frozen log scales | 0.2852 / 0.3224 | 0.5261 / 0.4980 | 7,810 | 1,000 |

At this fixed budget, adding log scales to bands8 increased held-out log R² by 0.0857 / 0.1410 and reduced relative modulus L2 by 11.0% / 15.0%. Increasing bands alone did not help at the tested budget. This result is specific to one initialization, a random point partition, this network and 1,000 optimization updates. It does not show that coordinate functions cannot fit G2 coefficients with a different architecture or optimization budget.

The full run took 8.18 seconds on one CPU thread, including checkpoint verification/loading, feature construction and metric evaluation. Arm times were 2.40, 1.67 and 1.75 seconds. The first includes optimizer startup overhead, so these timings are not a controlled speed comparison. Updates and minibatch size were matched; input dimensions and parameter counts differ.

## Reproducible definition

- Checkpoint: `/home/congcongliu/codex_artifacts/mcgs_ce_autoresearch_20261017/frequency_adam/outputs/adamflat_g2_v1/training/ar_i1_k30p90/checkpoints/latest.pth`. SHA256: `282e0303c46a6c18775d6224c032611072bd465b4ef975ba7a7e2ea636230fd0`.
- G2 stores positions, log Gaussian scales, rotations and two independent complex coefficient pairs. This probe uses frozen positions and optionally the three stored log scales as inputs. Rotations and coefficient magnitudes are not input features. Targets are the logarithms of the two complex coefficient moduli; the fixed `1e-12` floor was inactive for every sampled coefficient.
- Features retain original normalized G2 coordinates: `xyz`, `sin(pi*2**k*xyz)`, `cos(pi*2**k*xyz)`, for `k=0..bands-1`. Log-scale features use means and population standard deviations computed on training indices only. Targets likewise use training-only log means and population standard deviations.
- A two-hidden-layer, width64 SiLU MLP predicts two unconstrained standardized log moduli. Adam uses LR0.001, betas(0.9,0.999), 1,000 updates, minibatches1024. The point-split and initialization seed is20261019; every arm uses the identical minibatch index sequence seeded20261020. Initialization uses each architecture's ordinary PyTorch linear layers; differing input sizes mean initial functions are not matched.
- Relative modulus L2 is `||exp(predicted_log)-teacher_modulus||2 / ||teacher_modulus||2`, independently per contrast. Log R² uses the held-out log-target mean in its denominator. The constant baseline predicts the training median modulus, not a value fitted to the holdout.
- Sample position ranges were approximately [−1.0094,0.9240], [−1.0188,1.0123], [−1.0161,1.0070] on the three axes. Median stored log scales were −4.8668, −4.8487, −4.8343. Training coefficient-median moduli were0.078915 /0.093787. Complete source hashes, split hash, feature statistics, loss samples and exact results are in `results.json`; fitted weights and normalizers are in `probe_weights.pt`.

Run from a **fresh directory containing a copy of `probe.py`**; the script refuses to overwrite an existing `results.json`:

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -u probe.py
```

The observed command exited0. A separate acceptance check parsed the script and JSON, verified the script and weight hashes, the three actual update counts, the input dimensions(27,51,54), and the zero formal-MRI count.

## Warm-start implications

A full amplitude replacement would change the coefficient moduli at initialization. The residual route's zero-output identity argument would no longer apply. A future route needs a separately labeled G2-only representation warm fit, with its actual budget and coefficient errors recorded, and must keep phase and geometry treatment explicit. The teacher fit cannot be relabeled AF16 training or evidence of physical reconstruction quality.

Before a full-replacement MRI trial, evaluate the resulting **raw Gaussian render** and the canonical export against G2 using the same geometry, phases, renderer and normalization. Coefficient error is not image error: Gaussian overlap, phase cancellation and the multi-coil operator can change their relationship. No such render was performed here. A stricter representation fit, a larger function, or a residual parameterization may be needed; these measurements alone do not select among them.
