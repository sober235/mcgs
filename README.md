# Paired-ARC-GS — Paired Anatomy–Response–Change Gaussian reconstruction

Joint reconstruction of **two co-registered MRI contrasts** (e.g. T1+FLAIR, or pre/post
contrast-enhanced) from highly-undersampled k-space. The model factorizes each contrast into
a **shared anatomy** it holds in common, a **response** that maps that shared anatomy to each
contrast's intensity, and a **measurement-audited change** branch for what is genuinely
different between the scans — so that the shared structure compresses each contrast's
reconstruction null-space (the anti-aliasing mechanism) while real contrast-specific signal
(a lesion, an enhancement) is admitted only when independent measurements support it.

This repository implements the **Paired-ARC-GS** specification
([`MS_GS/Paired_ARC_GS_general_framework.md`](MS_GS/Paired_ARC_GS_general_framework.md)). The
earlier shared-geometry / dual-density model ("MC-GS") is retained unchanged as the §17
ablation baseline (`paired_gs/`); the new architecture lives in `paired_arc_gs/`.

```
Paired_ARC_GS_general_framework/
├── paired_arc_gs/          # Paired-ARC-GS architecture (the spec)
│   ├── model/
│   │   └── paired_arc_model.py   # shared anatomy G_A: μ,Σ + per-Gaussian latent z_A + π (§5)
│   ├── modules/                  # the four ARC modules + hybrid representation
│   │   ├── response_decoder.py   #   §6 α_m = D_ψ(z_A, e_m): shared-trunk MLP, per-contrast heads
│   │   ├── change_branch.py      #   §7 per-contrast change Gaussians G_{Δ,m} (starts empty)
│   │   ├── deformation.py        #   §8 cross-scan T_m(r)=R_m r+t_m+u_m, u capped (no lesion absorption)
│   │   ├── phase_field.py        #   §9 low-freq phase e^{jφ_m} (coarse grid → phasor)
│   │   ├── gabor_branch.py       #   §11 Gabor primitives B_l=G_l·e^{j2πξ^T(r-μ)} (edges/high-freq)
│   │   └── interfaces.py         #   module ABCs + NoOp defaults (disabled = identity)
│   ├── train/
│   │   ├── stages.py             #   §13 10-stage schedule (StagePlan: opt groups + loss terms + actions)
│   │   ├── config_schema.py      #   config defaults, merge, validation (rejects §12.1/§12.4 deprecations)
│   │   ├── trainer.py            #   PairedARCTrainer: stage-driven loop, births, eval
│   │   ├── acs_init.py           #   §13-S1 ACS low-res → joint structural map → importance init
│   │   ├── routing.py            #   §10.2 fit-residual evidence + proposal; §10.3 audit gain
│   │   └── cg_project.py         #   §13-S10 CG data-consistency projection (replaces hard replacement)
│   └── losses/
│       └── arc_losses.py         #   §12 noise-normalized DC + geometry reg + change sparsity + exclusivity
├── paired_gs/              # MC-GS baseline (§17 ablation row 2) — retained UNCHANGED
├── arc/                    # AutoReSearch harness (auto-tunes the MC-GS baseline levers)
├── configs/                # winner_v2.yaml (MC-GS baseline) + pair_template.yaml
├── scripts/                # train_paired.py, make_mask.py, estimate_csm.py
├── tests/                  # smoke_test.py + smoke_result_reference.json (MC-GS regression)
└── docs/
    ├── method.md               # model, forward operator, losses in detail
    └── validation_layer{1,2,3}.md   # per-layer validation reports + figures
```

## Method (short version)

Full derivation in [docs/method.md](docs/method.md) and the spec. Each contrast `m` is
modeled (§4) as

```
x_m = e^{jφ_m} · [ V( T_m(G_A), D_m(z_A) )  +  V(G_{Δ,m}) ]
```

built from four factored modules:

1. **Shared anatomy `G_A` (§5).** One cloud of N anisotropic 3D Gaussians (`μ`=positions,
   `Σ`=scales+rotations) plus a **per-Gaussian latent `z_A`** and confidence `π`. Geometry
   and latent are shared by both contrasts — the structural prior that co-registered
   contrasts share anatomy.

2. **Response decoder `D_ψ` (§6).** Per-contrast density is *decoded* from the shared latent
   and a contrast code, `α_m = D_ψ(z_A, e_m)` — a small shared-trunk MLP with per-contrast
   heads (deliberately too small to memorize). This replaces the two independent density
   tables of MC-GS (deprecated §3.2): the contrasts must now express their difference through
   a shared function of shared anatomy, not two free lookup tables.

3. **Change branch `G_{Δ,m}` (§7).** A per-contrast set of change Gaussians for signal the
   shared anatomy+response cannot explain (a FLAIR hyperintensity, a post-contrast
   enhancement). It **starts empty** and grows only through the audit (below).

4. **Deformation `T_m` (§8) and phase `φ_m` (§9).** `T_m(r)=R_m r+t_m+u_m(r)` aligns each
   contrast to the reference scan (rigid + amplitude-capped nonrigid — the cap is load-bearing
   so deformation cannot absorb a genuine lesion); `φ_m` is a low-frequency phase field applied
   as a complex phasor.

5. **Measurement-audited routing (§10).** The k-space is split **fit ∪ audit ∪ held-out-test**.
   Change candidates are proposed from the fit-set residual `|A^H r_fit|` (§10.2); a candidate
   is accepted only if it reduces error on the *disjoint* audit set, audit gain
   `ΔE = ‖A_audit x−y‖² − ‖A_audit x⁺−y‖² > τ` (§10.3). This is the anti-aliasing guard: a
   branch that merely fits undersampling aliasing shows no gain on measurements that did not
   generate it, so it is rejected.

6. **Gaussian–Gabor hybrid (§11)** adds a strictly-limited set of Gabor primitives
   `B_l=G_l·e^{j2πξ^T(r−μ)}` for edges/high-frequency detail (a plain Gaussian's spectrum is
   low-frequency-biased); **CG data-consistency projection (§13-S10)** replaces per-coil hard
   k-space replacement with a regularized conjugate-gradient solve.

**Losses (§12):** noise-normalized k-space data consistency + a Gaussian-parameter geometry
regularizer + change sparsity + shared/change exclusivity. The old MC-GS 0.3/2.2 k-space
frequency weighting (§12.1) and image-space TV as a geometry prior (§12.4) are **deprecated**
and rejected by `config_schema.validate_config`.

**Training (§13):** a 10-stage schedule (`train/stages.py`) — ACS init → response fit →
deformation → low-LR geometry → change birth → Gabor birth → prune → stabilize → refine → CG
project. Each stage gates which parameter groups are optimized, which loss terms are active,
and which birth/prune actions fire; disabled modules are NoOps (identity), so the schedule
degrades gracefully to any subset.

## Install

Runtime needs a CUDA GPU (the voxelizer JIT-compiles on first render; no prebuilt
`.so` is shipped) and:

```bash
pip install -r requirements.txt   # torch>=2.0, numpy, scipy, scikit-image, tqdm, pyyaml
```

Validated on the amax `GSTMI` env (torch 2.7.1+cu118). Any torch≥2.0 with a matching
CUDA toolchain (`nvcc` on `PATH`) should build the voxelizer.

## Two worked examples

The framework is validated on two dual-contrast pairs that exercise different parts of the
model (§13 in the spec). A pair is defined entirely by config — nothing in the code is
contrast-specific.

**Example 1 — T1 + FLAIR (healthy brain).** The plain (T1) and contrast (FLAIR) channels of a
co-registered brain, undersampled at AF≈12. Fully-sampled ground truth exists, so PSNR/SSIM
are valid. Used to validate the shared anatomy + response decoder (Layer 1) and the
zero-regret behaviour of the change audit on a pair with *no* genuine lesion (Layer 2).

**Example 2 — pre/post contrast-enhanced (CE).** Two acquisitions of the same anatomy before
and after gadolinium, sharing one `StudyInstanceUID` (co-registered), 42 coils, presampled at
≈5.3×. There is **no fully-sampled ground truth**, so evaluation uses the GT-free held-out
k-space NRMSE (§18). This is the §7.2 scenario the change branch is designed for — genuine
localized enhancement between the two scans.

```bash
# Paired-ARC-GS on either pair (see paired_arc_gs/train/_l2final.py / _ablation17.py for the
# exact per-pair config dicts used in validation; each toggles module enable flags + stages):
python -m paired_arc_gs.train.trainer --config <your_pair>.yaml --gpu 0
```

## Run the MC-GS baseline (§17 ablation row 2)

The shared-geometry / dual-density MC-GS model is retained unchanged as the ablation baseline:

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

Each layer of the architecture is validated incrementally (T1/FLAIR + CE), with a controlled
test wherever ground truth can be known. Full reports + figures in `docs/validation_layer*.md`.

- **Layer 1 — shared anatomy + response decoder (§5, §6):** the response decoder matches or
  beats two independent density tables on 4/4 contrasts by held-out k-space NRMSE, with far
  fewer parameters (`docs/validation_layer1.md`).
- **Layer 2 — change branch + measurement-audited routing (§7, §10):** in a controlled
  synthetic scene the audit gain separates true change from undersampling aliasing by **8.3×**;
  on the healthy T1/FLAIR pair the change branch is a **zero-regret** gate (disabling, gating,
  or forcing change all give the same held-out error within 0.0013). Real-data birth gains are
  at the noise floor and reported honestly as such (`docs/validation_layer2.md`).
- **Layer 3 — deformation + phase (§8, §9):** controlled recovery — a known rigid shift is
  recovered to < 0.003 units; the §8 amplitude cap holds under a 10× over-drive (so deformation
  cannot absorb a genuine lesion); a known low-frequency phase ramp is recovered exactly
  (`docs/validation_layer3.md`).
- **Layer 4 — Gabor + CG (§11, §13-S10):** unit tests verify the §11 spectral property (a Gabor
  carrier shifts energy off DC where a plain Gaussian keeps 41% of its energy) and the CG solve
  (positive-definite normal equations, beats hard k-space replacement on data consistency).

### §17 core ablation (held-out k-space NRMSE, T1/FLAIR)

The five spec rows, run under one config on the healthy T1/FLAIR pair, differing only by the
four module flags (held-out k-space NRMSE, lower = better):

| model | shared anat | decoder | change | audit | T1 NRMSE | FLAIR NRMSE | change admitted |
|---|:--:|:--:|:--:|:--:|--:|--:|:--:|
| Independent GS | – | – | – | – | 0.8152 | 0.8941 | – |
| MC-GS | ✓ | – | – | – | 0.5866 | 0.8709 | – |
| Paired-ARC-GS w/o change | ✓ | ✓ | – | – | 0.5861 | 0.8667 | – |
| Paired-ARC-GS w/o audit | ✓ | ✓ | ✓ | – | 0.5870 | 0.8671 | 4000 |
| Paired-ARC-GS (full) | ✓ | ✓ | ✓ | ✓ | 0.5871 | 0.8672 | **0** |

What the table shows, matching the spec's five predictions:

1. **Shared anatomy** cuts held-out error sharply vs independent reconstructions (T1
   0.815→0.587, FLAIR 0.894→0.871) — the null-space compression that is the whole point.
2. **Response decoder** improves further over two independent density tables (FLAIR
   0.871→0.867), with far fewer parameters.
3–5. On this **healthy** pair there is no genuine change to recover, so change/audit are
   **zero-regret**: rows 3–5 agree within 0.0005. The decisive column is *change admitted* —
   bypass (w/o audit) forces **4000** change Gaussians onto a pair with no lesion, while the
   **audit admits 0** at identical held-out error. That is prediction 4: the audit prevents the
   change branch from fitting undersampling aliasing. (On a pair with real change — CE pre/post,
   or the synthetic scene in Layer 2 — the audit's 8.3× true-vs-aliasing separation is what lets
   it admit genuine change while still rejecting aliasing.)

![§17 ablation]({{artifact:art_8a72d665-5288-4580-9f64-148d6734f59b}})

Runner: `python -m paired_arc_gs.train._ablation17` (writes
`results_arc_ablation17/ablation17.json`). Runs are not bit-reproducible
(`deterministic: false`); the ordering, not the third decimal, is the result.

### MC-GS baseline regression

`tests/smoke_test.py` reloads the winner-v2 checkpoint through the MC-GS baseline code and
re-scores it. The baseline (`paired_gs/`) is **byte-identical** to the commit that produced
`tests/smoke_result_reference.json` (verifiable with `git diff <ref-commit> HEAD -- paired_gs/`),
so it cannot regress; the smoke reproduces T1 37.738 / FLAIR 38.143 to within < 0.001 dB PSNR.

```bash
PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  python tests/smoke_test.py --gpu 0 --ckpt /path/to/winner_v2/checkpoints/best.pth
```

## Notes / conventions

- **Import roots.** `paired_arc_gs/` is imported as a package (framework root on `sys.path`);
  it reuses the `paired_gs/` voxelizer and dataset (also on `sys.path` for its flat imports).
  `paired_gs/` is the MC-GS baseline, copied byte-for-byte; its numerics are untouched.
- **Axis convention** (shipped brain data, volume `(256,220,256)`): `phase_axes=[0,1]`
  (undersampled plane), `readout_axis=2` (fully sampled). Set these per your data.
- **Checkpoint formats.** A training output `best.pth` is
  `{"iter","model_state","optim_state"}` where `model_state` holds all 7 model params
  (both channels' densities). To *reload a trained winner*, size a fresh model to the
  checkpoint and `load_state_dict(model_state)` — do **not** route it through
  `gaussian.warm_start_checkpoint` (that path reads only the plain params and re-clones
  contrast=plain, discarding the trained contrast densities). See `tests/smoke_test.py`.
