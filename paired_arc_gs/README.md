# paired_arc_gs — Paired-ARC-GS architecture

Implements the spec `MS_GS/Paired_ARC_GS_general_framework.md` (Paired Anatomy–Response–Change
Gaussian Reconstruction). Built incrementally in layers; each layer is validated on both
T1/FLAIR and pre/post-contrast before the next.

This package **reuses** the validated low-level pieces from the sibling `paired_gs/` package
(the CUDA tile voxelizer `gaussian/tile_voxelizer.py`, the fallback `gaussian/voxelizer.py`,
and the general data loader `data/dataset.py`). `paired_gs/` itself is retained unchanged as
the **original MC-GS baseline** for the §17 ablation table (row "原 MC-GS").

## Package layout
```
paired_arc_gs/
  model/        PairedARCModel: shared anatomy G_A + latent z_A (the container)
  modules/      the 4 pluggable spec modules (§4 decomposition)
    interfaces.py   ABCs + NoOp defaults (ResponseDecoder §6, ChangeBranch §7,
                    PhaseField §9, Deformation §8)
  losses/       §12 loss terms (noise-normalized DC, geometry reg, change sparsity,
                shared-change exclusivity)
  train/
    stages.py         §13 10-stage scheduler (StagePlan / StageState)
    config_schema.py  DEFAULT_CONFIG + validate_config + merge_config
  utils/
```

## Build layers (each ends in T1/FLAIR + CE validation)
- **Layer 0** (this): package scaffold, config schema, §13 stage scheduler, §12 loss module.
- **Layer 1**: shared anatomy latent z_A + response decoders D1/D2 (§5-§6); Stage 1-2.
- **Layer 2**: measurement-audited change routing (§7, §10) — the spec's core contribution (§20).
- **Layer 3**: low-freq phase field φ_m (§9) + cross-scan deformation T_m (§8).
- **Layer 4**: Gaussian-Gabor hybrid (§11) + CG data-consistency projection (§13-S10).

## Config-driven
Every module has an `enable` flag; the default config = Layer-1 minimal (only Stage 2 active,
geometry frozen). Turning a module on and giving its stage `iters>0` activates it — the
scheduler and trainer need no code change between layers. Deprecated MC-GS knobs
(0.3/2.2 k-space frequency weighting, image-space TV as geometry reg) are **rejected** by
`validate_config` per spec §12.1/§12.4.
