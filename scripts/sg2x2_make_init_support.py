#!/usr/bin/env python3
"""Write the frozen init support for the sg2x2 ablation: the union-ZF head support of a config's pair (the rule of
gaussian.init_mode=head: union_magnitude > 0.05 * P99.9), binary-dilated, as a compressed .npz (key 'support').
Run it ONCE on the development subject; every cell and every later subject reads the same file.

    env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/sg2x2_make_init_support.py \
        --config configs/sg2x2/af16_jyz_joint.yaml --out configs/sg2x2/init_support_af16_jyz.npz --dilate 3
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paired_gs"))
from data.transforms import ifft3c                                   # noqa: E402
from gaussian.init_cloud import head_support, union_magnitude       # noqa: E402


def _zf(block: dict, mask: torch.Tensor) -> torch.Tensor:
    with h5py.File(block["data_path"], "r") as f:
        k = torch.from_numpy(f["kspace"][...]).to(torch.complex64)
    with h5py.File(block["csm_path"], "r") as f:
        s = torch.from_numpy(f["map"][...]).to(torch.complex64)
    return torch.sum(torch.conj(s) * ifft3c(k * mask.unsqueeze(0)), dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dilate", type=int, default=3, help="binary dilation iterations (voxels)")
    ap.add_argument("--head_frac", type=float, default=0.05)
    a = ap.parse_args()
    out = Path(a.out)
    if out.suffix != ".npz":
        raise SystemExit("--out must end in .npz")
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    cfg = yaml.safe_load(open(ROOT / a.config))
    plain, contrast = cfg["data"]["plain"], cfg["data"]["contrast"]
    with h5py.File(plain["mask_path"], "r") as f:
        mask = torch.from_numpy(f["mask"][...]).float()
    mag = union_magnitude(_zf(plain, mask), _zf(contrast, mask))
    sup = head_support(mag, a.head_frac).numpy()
    if a.dilate > 0:
        from scipy.ndimage import binary_dilation
        sup = binary_dilation(sup, iterations=a.dilate)
    np.savez_compressed(out, support=sup.astype(bool))
    print(f"wrote {out} shape={sup.shape} support_frac={sup.mean():.4f} dilate={a.dilate} head_frac={a.head_frac} "
          f"md5={hashlib.md5(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
