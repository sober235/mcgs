"""Voxelizer config resolution shared by train.py and train_joint.py."""
from __future__ import annotations

from typing import Any, Dict, Optional


def apply_deterministic_voxelizer_reference(config: Dict[str, Any]) -> Optional[str]:
    """Use a non-atomic reference voxelizer when deterministic validation is on.

    ``tile_cuda`` remains the default fast backend. When
    ``training.deterministic`` is true, the custom CUDA backward path is not a
    valid same-GPU bit-identity backend because it uses floating-point atomics.
    In that regime, configs may name a deterministic reference backend through
    ``voxelizer.reference_type``; the default is the lower-memory ``chunk``
    backend.
    """
    train_cfg = config.get("training", {})
    if not bool(train_cfg.get("deterministic", False)):
        return None

    voxel_cfg = config.setdefault("voxelizer", {})
    requested_type = voxel_cfg.get("type", "chunk")
    if requested_type != "tile_cuda":
        voxel_cfg.setdefault("effective_type", requested_type)
        return None

    if not bool(voxel_cfg.get("deterministic_reference", True)):
        voxel_cfg["effective_type"] = requested_type
        return None

    reference_type = voxel_cfg.get("reference_type", "chunk")
    if reference_type not in {"tile", "chunk"}:
        raise ValueError(
            "voxelizer.reference_type must be 'tile' or 'chunk' when "
            "training.deterministic=true and voxelizer.type='tile_cuda'; "
            f"got {reference_type!r}."
        )

    voxel_cfg["requested_type"] = requested_type
    voxel_cfg["type"] = reference_type
    voxel_cfg["effective_type"] = reference_type
    voxel_cfg["deterministic_reference_active"] = True
    return (
        "training.deterministic=true: using "
        f"voxelizer.reference_type='{reference_type}' instead of tile_cuda "
        "because tile_cuda backward uses floating-point atomicAdd."
    )
