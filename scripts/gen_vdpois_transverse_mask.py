#!/usr/bin/env python3
"""Generate the canonical 3D vdPois undersampling mask for MC-GS joint reconstruction.

House convention (verified against reconstructed anatomy):
  volume shape (256, 220, 256) = (axis0, axis1, axis2)
    axis0 = coronal    direction
    axis1 = sagittal   direction
    axis2 = transverse direction  <-- readout (fully-sampled) axis (S-I)

  The variable-density Poisson-disc (vdPois) undersampling acts on the
  TRANSVERSE plane, i.e. the phase-encoding plane = axis0 x axis1.
  The readout axis (axis2) is fully sampled.

Source of the 2D pattern: the authoritative ESPIRiT-utils generator
(mask_vdpois_220x256_af12.mat), a 2D (220, 256) vdPois pattern. We orient it
to (256, 220) = (axis0, axis1) and broadcast along axis2.

Result: mask3d_vdpois_af12_transverse.mat, AF ~= 11.83, transverse-plane
sampling ~8.45%, variance along axis2 = 0 (fully-sampled readout).

Usage:
    python gen_vdpois_transverse_mask.py \
        --src /path/to/mask_vdpois_220x256_af12.mat \
        --out mask3d_vdpois_af12_transverse.mat
"""
import argparse
import numpy as np
from scipy.io import loadmat, savemat


def build_transverse_mask(pattern_2d, vol_shape=(256, 220, 256)):
    """Broadcast a 2D PE-plane pattern along the readout axis (axis2)."""
    p = np.asarray(pattern_2d).astype(np.float32)
    if p.shape == (vol_shape[1], vol_shape[0]):      # (220, 256) -> transpose
        p = p.T
    assert p.shape == (vol_shape[0], vol_shape[1]), (
        f"pattern must map to (axis0={vol_shape[0]}, axis1={vol_shape[1]}), got {p.shape}"
    )
    return np.broadcast_to(p[:, :, None], vol_shape).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="ESPIRiT-utils 2D vdPois .mat (key 'mask'), e.g. mask_vdpois_220x256_af12.mat")
    ap.add_argument("--out", default="mask3d_vdpois_af12_transverse.mat")
    ap.add_argument("--vol", type=int, nargs=3, default=[256, 220, 256],
                    help="volume shape (axis0 axis1 axis2)")
    args = ap.parse_args()

    m2d = loadmat(args.src)["mask"].astype(np.float32)
    mask3d = build_transverse_mask(m2d, tuple(args.vol))

    frac = float(mask3d.mean())
    af = 1.0 / frac
    var_readout = float(np.mean(np.var(mask3d, axis=2)))
    transverse_samp = float(mask3d[:, :, args.vol[2] // 2].mean())

    savemat(args.out, {"mask": mask3d, "net_af": np.float32(af)})
    print(f"saved {args.out}")
    print(f"  shape            = {mask3d.shape}")
    print(f"  sampled fraction = {frac:.4f}  (AF = {af:.2f})")
    print(f"  transverse-plane sampling (fix axis2) = {transverse_samp:.4f}")
    print(f"  variance along readout axis2 (should be 0) = {var_readout:g}")


if __name__ == "__main__":
    main()
