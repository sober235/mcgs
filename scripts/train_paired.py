#!/usr/bin/env python3
"""Paired dual-contrast 3D-GS training entry point.

Reconstructs two co-registered contrasts jointly with one shared-geometry Gaussian
cloud + dual per-primitive density. Contrast identity lives entirely in the config
(data.plain / data.contrast, with optional `name` labels) — this script is
contrast-agnostic.

Usage:
    python scripts/train_paired.py --config configs/winner_v2.yaml --gpu 2
    python scripts/train_paired.py --config configs/my_pair.yaml   --gpu 3

The --gpu flag pins CUDA_VISIBLE_DEVICES *before* importing torch (the amax GPUs
are shared; always pin an idle card). This mirrors the validated MC-GS train_joint.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default=None, help="override output.output_dir")
    ap.add_argument("--exp_name", default=None, help="override output.exp_name")
    args = ap.parse_args()

    # Pin the GPU BEFORE torch is imported (shared cluster; --gpu wins over any
    # external CUDA_VISIBLE_DEVICES).
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    print(f"CUDA_VISIBLE_DEVICES={args.gpu}")

    # Make paired_gs/ the import root so the validated core resolves with its
    # original flat imports (from trainers.joint_trainer import JointTrainer, ...).
    root = Path(__file__).resolve().parents[1]        # framework root
    pgs = root / "paired_gs"
    sys.path.insert(0, str(pgs))

    import torch
    from seed import set_seed
    from trainers.joint_trainer import JointTrainer
    from data.private_paths import resolve_subject_paths

    cfg = load_config(args.config)
    if args.output_dir:
        cfg.setdefault("output", {})["output_dir"] = args.output_dir
    if args.exp_name:
        cfg.setdefault("output", {})["exp_name"] = args.exp_name

    cfg.setdefault("training", {})["seed"] = args.seed
    deterministic = bool(cfg.get("training", {}).get("deterministic", False))
    set_seed(args.seed, deterministic=deterministic)

    # Resolve anonymized subject tokens (S01_T1.mat -> real path); real absolute
    # paths pass through verbatim.
    data_cfg = cfg.get("data", {})
    for sub in ("plain", "contrast"):
        sub_cfg = data_cfg.get(sub, {})
        for key in ("data_path", "csm_path", "mask_path"):
            if key in sub_cfg and sub_cfg[key]:
                sub_cfg[key] = resolve_subject_paths(sub_cfg[key])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = JointTrainer(cfg, device=device)
    trainer.train()
    trainer.export_results()


if __name__ == "__main__":
    main()
