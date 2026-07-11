# Paired ARC-GS — dual-contrast 3D MRI reconstruction by joint Gaussian Splatting

Joint reconstruction of **two co-registered MRI contrasts** from highly-undersampled
k-space, with **one shared 3D-Gaussian geometry** and a **per-primitive dual density**.
The high-SNR *plain* channel and the *contrast* channel are reconstructed together so
that the geometry they share compresses the reconstruction null-space of each — the
mechanism that lets both come back cleanly at accelerations (~AF 12) where a
single-contrast reconstruction would alias.

Nothing in the code is specific to any contrast pair. A pair is defined entirely by a
YAML config (`data.plain` / `data.contrast`, each with an optional display `name`). The
shipped baseline targets a **T1 (plain) + FLAIR (contrast)** brain pair and reproduces
the validated *winner-v2* result to within **< 0.001 dB PSNR** (see [Validation](#validation)).

```
Paired_ARC_GS_general_framework/
├── paired_gs/        # validated reconstruction core (contrast-agnostic)
│   ├── gaussian/     #   shared-geometry Gaussian model + dual density + CUDA voxelizer
│   ├── trainers/     #   JointTrainer (the joint plain+contrast training loop)
│   ├── data/         #   dataset loader, k-space transforms, mask conforming
│   ├── losses/       #   k-space DC, TV/geometry, cross-contrast coupling, ELBO
│   ├── metrics/      #   PSNR / SSIM / edge / lesion metrics
│   └── seed.py       #   deterministic seeding
├── arc/              # AutoReSearch harness (auto-tunes anti-artifact levers)
│   ├── cycle.py      #   one propose→train→score→keep/revert cycle
│   ├── program.md    #   the research spec / metric contract (TEMPLATE)
│   └── score_*.py    #   ESPIRiT-CSM scorers (frozen metric口径)
├── configs/
│   ├── winner_v2.yaml     # shipped baseline (T1+FLAIR, TV=0.06, kspace_high=2.2)
│   └── pair_template.yaml # documented template — copy this for a NEW pair
├── scripts/
│   ├── train_paired.py    # training entry point
│   ├── make_mask.py       # vdPoisson undersampling-mask recipe + verify
│   └── estimate_csm.py    # ESPIRiT CSM recipe + verify
├── tests/
│   ├── smoke_test.py                  # reload winner-v2 ckpt → reproduce its scores
│   └── smoke_result_reference.json    # committed PASS record
└── docs/
    └── method.md          # the model, forward operator, and losses in detail
```

## Method (short version)

Full derivation in [docs/method.md](docs/method.md). In brief:

1. **Shared geometry, dual density.** A single cloud of N anisotropic 3D Gaussians
   (`positions`, `scales`, `rotations`) is shared by both contrasts. Each Gaussian
   carries **two** complex densities — one for the plain channel, one for the contrast
   channel (`density_{real,imag}` and `density_{real,imag}_contrast`). The geometry is
   common; only the densities differ. This encodes the physical prior that two
   co-registered contrasts of the same anatomy share structure (edges, boundaries) but
   differ in tissue-dependent intensity.

2. **Forward model.** Each channel's Gaussian field is voxelized (JIT CUDA tile
   voxelizer) into a complex volume `x`, multiplied by the coil sensitivities `S`,
   Fourier-transformed, and masked:  `y ≈ M · F · (S ⊙ x)`. The training loss is
   per-channel k-space data consistency (frequency-weighted) + a geometric/TV
   smoothness term, with the shared geometry coupling the two channels.

3. **Reconstruction (scored path).** Render each channel, enforce **hard data
   consistency** on the sampled k-space lines, then **SENSE coil-combine**:
   `x_dc = Σ_c conj(S_c) · F⁻¹( M·y + (1−M)·F(S·x) )`.

4. **Coupling levers.** At high AF the primary anti-aliasing mechanism is the shared
   geometry (null-space compression), tuned via `shared_warmup_iters` and the optional
   plain-guided directional-TV coupling (`lambda_dtv`). TV (`tv_weight`→`lambda_geo`)
   and `kspace_high_weight` are secondary edge-preserving-denoise levers that oppose
   each other (TV suppresses grain, high-k weight amplifies it). See `arc/program.md`.

## Install

Runtime needs a CUDA GPU (the voxelizer JIT-compiles on first render; no prebuilt
`.so` is shipped) and:

```bash
pip install -r requirements.txt   # torch>=2.0, numpy, scipy, scikit-image, tqdm, pyyaml
```

Validated on the amax `GSTMI` env (torch 2.7.1+cu118). Any torch≥2.0 with a matching
CUDA toolchain (`nvcc` on `PATH`) should build the voxelizer.

## Run the shipped baseline (T1 + FLAIR)

```bash
python scripts/train_paired.py --config configs/winner_v2.yaml --gpu 0
```

`--gpu N` pins `CUDA_VISIBLE_DEVICES` before torch imports (the training cluster GPUs
are shared — always target an idle card). Outputs land in `results_winner_v2/`.

> **Data is not bundled.** k-space, CSM, mask, and the warm-start checkpoint are large
> (GB) and are `.gitignore`d. `configs/winner_v2.yaml` points at the canonical paths on
> the amax shared disks. On another machine, either copy those files over and edit the
> paths, or regenerate them (below).

## Bring your own dual-contrast pair

1. **Prepare data.** Save each contrast as a MATLAB `.mat` with multi-coil k-space.
   `paired_gs/data/dataset.py` auto-detects the coil axis and accepts k-space keys
   `ksp`/`kspace`/`k` and (optional bundled) sensitivity keys
   `map`/`csm`/`sensitivity_maps`/`maps`/`S`. **The two contrasts must be spatially
   co-registered** — shared geometry assumes it. Put the higher-SNR / more-complete
   contrast as `plain`; it anchors the geometry.

2. **Coil sensitivities (CSM).** If you don't have a bundled map, estimate one with
   ESPIRiT (the validated recipe):
   ```bash
   python scripts/estimate_csm.py --recipe                 # prints the BART ecalib recipe
   python scripts/estimate_csm.py --verify csm.mat --data kspace.mat   # SoS check
   ```
   Recipe: `bart ecalib -t 0.001 -c 0.0 -r 24 -m 1` from an ACS block (readout fully
   sampled + central 32×32 in the phase-encode plane). Save as key `map`, complex,
   dim-order `(axis0, axis1, axis2, coil)`.

3. **Undersampling mask.** Generate a house-convention variable-density Poisson mask
   (2D on the two phase-encode axes; readout fully sampled):
   ```bash
   python scripts/make_mask.py --af 12 --out mask_af12.mat   # prints the MATLAB vdPoisMex cmd
   python scripts/make_mask.py --verify mask_af12.mat        # samp / AF / central-ACS check
   ```
   A 2D `(PE1,PE2)` mask is auto-stacked along the readout axis by the dataset loader.

4. **Write a config.** Copy `configs/pair_template.yaml`, fill in the two `data.*`
   blocks (paths, `name`, `phase_axes`, `readout_axis`), and tune from the shipped
   priors (TV 0.05–0.06, kspace_high 2.0–2.2). Then:
   ```bash
   python scripts/train_paired.py --config configs/my_pair.yaml --gpu 0
   ```

5. **Auto-tune (optional).** Point the ARC harness at your config and let it sweep the
   anti-artifact levers:
   ```bash
   python arc/cycle.py --exp my_pair_sweep --gpu 0 --lambda_geo 0.06 --iters 1500
   ```
   The harness reads the contrast names from your config; its logs and scores are
   channel-neutral. Read `arc/program.md` for the metric contract and search plan.

## Validation

`tests/smoke_test.py` reloads the winner-v2 checkpoint through the generalized code and
re-scores it against the ESPIRiT-CSM ground truth. It reproduces the validated numbers to
**within < 0.001 dB PSNR** (`tests/smoke_result_reference.json`) — the residual is run
nondeterminism, far inside the 0.25 dB / 0.01 SSIM pass tolerance:

| channel        | name  | PSNR (dB) | SSIM    | reference        | Δ PSNR / Δ SSIM   |
|----------------|-------|-----------|---------|------------------|-------------------|
| plain          | T1    | 37.73835  | 0.87555 | 37.738 / 0.8755  | +0.00035 / +0.00005 |
| contrast       | FLAIR | 38.14307  | 0.89400 | 38.143 / 0.894   | +0.00007 / −0.00000 |

Scoring uses the reference **alpha-fit** convention (global intensity-scale fit before
PSNR/SSIM), matching `arc/score_winner_espirit_final.py`. The run is not bit-reproducible
(`deterministic: false`), so exact-equality is not expected or asserted; the deltas above
confirm the generalized code is numerically equivalent to the validated original.

```bash
# on a CUDA host (GSTMI env), pinned to an idle GPU:
PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  python tests/smoke_test.py --gpu 0 \
    --ckpt /path/to/winner_v2/checkpoints/best.pth
```

## Notes / conventions

- **Import root** is `paired_gs/`. Entry scripts insert it on `sys.path` and the core
  uses flat imports (`from trainers.joint_trainer import JointTrainer`). The validated
  core was copied byte-for-byte; numerics are untouched.
- **Axis convention** (shipped brain data, volume `(256,220,256)`): `phase_axes=[0,1]`
  (undersampled plane), `readout_axis=2` (fully sampled). Set these per your data.
- **Checkpoint formats.** A training output `best.pth` is
  `{"iter","model_state","optim_state"}` where `model_state` holds all 7 model params
  (both channels' densities). To *reload a trained winner*, size a fresh model to the
  checkpoint and `load_state_dict(model_state)` — do **not** route it through
  `gaussian.warm_start_checkpoint` (that path reads only the plain params and re-clones
  contrast=plain, discarding the trained contrast densities). See `tests/smoke_test.py`.
