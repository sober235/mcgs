#!/usr/bin/env python3
"""make_mask.py — generate a house-convention variable-density Poisson-disc
undersampling mask for a dual-contrast pair.

House convention (this group's 3D MRI): the mask is a 2D variable-density
Poisson pattern on the two PHASE-ENCODE axes; the READOUT axis is fully sampled.
The reconstruction dataset (paired_gs/data/dataset.py::_conform_mask_shape)
auto-stacks a 2D mask along the missing readout axis and transposes it to match
the volume layout, so you save/pass the 2D pattern.

Geometry used by the shipped winner_v2 baseline (T1/FLAIR brain, vol
(256,220,256)=(axis0,axis1,axis2)):
    phase_axes  = [0, 1]   -> the undersampled PE plane
    readout_axis = 2       -> fully sampled (S-I)
The 2D vdPoisson pattern is generated at (PE1,PE2) and broadcast along readout.

The generator itself is ESPIRiT-utils vdPoisMex, which needs MATLAB +
vdPoisMex.mexa64 on the host (present on amax at /data0/congcong/code/ESPIRiT/utils).
This script only BUILDS the command and verifies the result; run the MATLAB
command on the host (see README). Ready-made masks also live in that utils dir:
    mask_vdpois_220x256_af{08,10,12,14,16}.mat

Usage (verify an existing mask):
    python scripts/make_mask.py --verify /path/to/mask.mat
Usage (print the MATLAB command to generate AF=12):
    python scripts/make_mask.py --af 12 --out /path/to/mask_af12.mat
"""
from __future__ import annotations
import argparse


def verify(path: str) -> None:
    from scipy.io import loadmat
    import numpy as np
    m = loadmat(path)["mask"].astype(float)
    cy, cx = m.shape[0] // 2, m.shape[1] // 2
    acs = m[cy - 12:cy + 12, cx - 12:cx + 12].mean() if m.ndim == 2 else float("nan")
    print(f"mask {path}")
    print(f"  shape={m.shape}  samp={m.mean():.4f}  AF={m.size / m.sum():.2f}  centralACS24={acs:.3f}")


def matlab_cmd(af: int, out: str, shape=(220, 256), accely: int = 4, acs: int = 24,
               pp: int = 1, ellipse: int = 0) -> str:
    """Build the vdPoisMex command (house convention). accelx=AF/accely."""
    accelx = af / accely
    p1, p2 = shape
    return (
        f"cd /data0/congcong/code/ESPIRiT/utils && matlab -nodisplay -nosplash -r "
        f"\"mask=vdPoisMex({p1},{p2},{p1},{p2},{accelx},{accely},{acs},{ellipse},{pp}); "
        f"save('{out}','mask'); "
        f"fprintf('nnz=%d AF=%.2f samp=%.4f\\n',nnz(mask),numel(mask)/nnz(mask),nnz(mask)/numel(mask)); exit\""
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", metavar="MAT", help="verify an existing mask .mat and exit")
    ap.add_argument("--af", type=int, default=12, help="acceleration factor")
    ap.add_argument("--out", default="mask_vdpois_af12.mat")
    ap.add_argument("--shape", default="220,256", help="PE1,PE2")
    args = ap.parse_args()

    if args.verify:
        verify(args.verify)
        return
    shape = tuple(int(x) for x in args.shape.split(","))
    print("# Run this on a host with MATLAB + vdPoisMex.mexa64 (e.g. amax):")
    print(matlab_cmd(args.af, args.out, shape=shape))
    print("# Then verify with:  python scripts/make_mask.py --verify", args.out)


if __name__ == "__main__":
    main()
