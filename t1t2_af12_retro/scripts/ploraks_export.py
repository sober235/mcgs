#!/usr/bin/env python3
"""ploraks_export.py -- write the readout-decoupled multi-coil k-space of one contrast for P-LORAKS.

The readout (axis 0) is fully sampled, so an inverse FFT along it turns the 3-D problem into 256
independent 2-D (ky, kz) problems, one per readout position x. For each x the toolbox gets the
zero-filled (ky, kz, coil) k-space and the 2-D mask. Data are normalised by the gauge's P99.9
scale (irrelevant for LORAKS, kept so the assembled volume lands in gauge units).

usage: ploraks_export.py --topic <dir> --contrast T1 --out <file.mat> [--slices 100,120,...]
"""
import argparse, sys
from pathlib import Path
import h5py, numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--topic", required=True); ap.add_argument("--contrast", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--slices", default=None)
args = ap.parse_args()
sys.path.insert(0, str(Path(args.topic) / "scripts"))
import t1t2_common as C
P = C.Pair(args.contrast, torch.device("cpu"))
hyb = C.ifft1c(P.y, 1)                                  # (coil, x, ky, kz): k-space along ky,kz; image along x
idx = list(range(P.G[0])) if args.slices is None else [int(s) for s in args.slices.split(",")]
hyb = hyb[:, idx].numpy().astype(np.complex64)          # (coil, nx, ky, kz)
kdata = np.ascontiguousarray(hyb.transpose(1, 2, 3, 0))  # (nx, ky, kz, coil) for MATLAB
mask = P.m2.cpu().numpy().astype(np.float32)             # (ky, kz)
with h5py.File(args.out, "w") as f:                      # MATLAB reads v7.3/HDF5 with h5read; complex as re/im
    f.create_dataset("kre", data=kdata.real); f.create_dataset("kim", data=kdata.imag)
    f.create_dataset("mask", data=mask); f.create_dataset("slices", data=np.array(idx, dtype=np.int32))
    f.create_dataset("scale", data=np.array(P.scale, dtype=np.float64))
print("wrote", args.out, kdata.shape, "slices", len(idx))
