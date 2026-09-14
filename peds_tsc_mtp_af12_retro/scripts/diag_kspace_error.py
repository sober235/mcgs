#!/usr/bin/env python3
"""diag_kspace_error.py -- where does each arm's error live in k-space? (stage-3 lever choice)

  ./scripts/py scripts/diag_kspace_error.py T1w

Splits the error energy of (recon - GT) over the phase-encode plane into the sampled positions Omega,
the measured zeros Z, and the rest of the acquired support, and into radial bands of the (PE, SPE)
plane. Error on Omega means the arm is not fitting the data it was given; error off Omega is what the
prior had to invent. Whole volume, not head-masked: k-space is global.
"""
import sys
import h5py
import numpy as np

R = "/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro"
c = sys.argv[1] if len(sys.argv) > 1 else "T1w"
ARMS = {"MC-GS": f"{R}/recon/gs_joint_ar_{c}_complex.npy",
        "GS-single": f"{R}/recon/gs_single_ar_{c}_complex.npy",
        "L1-ESPIRiT": f"{R}/recon/l1espirit_lam{'0.001' if c == 'T1w' else '0.003'}_{c}_complex.npy"}

gt = np.load(f"{R}/recon/gt_{c}_complex.npy")
with h5py.File(f"{R}/data/mask_af12_3d.h5", "r") as f:
    omega = np.asarray(f["mask"][:])[0] > 0                      # (PE, SPE), constant along readout
with h5py.File(f"{R}/data/support_2d.h5", "r") as f:
    support = np.asarray(f["support"][:]) > 0
zeros = support & ~omega if support.sum() < omega.size else ~omega   # printed below, then checked


def k(v):
    """centred orthonormal 3-D FFT of an image volume"""
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(v), norm="ortho"))


print(f"{c}: Omega {omega.sum()}, support map True {support.sum()}, plane {omega.size}")
Z = support & ~omega
rest = ~support & ~omega
print(f"  Omega(sampled) {omega.sum()}  unsampled-in-support {Z.sum()}  known zeros {rest.sum()}")

ky = np.arange(omega.shape[0]) - omega.shape[0] // 2
kz = np.arange(omega.shape[1]) - omega.shape[1] // 2
rad = np.sqrt((ky[:, None] / (omega.shape[0] / 2)) ** 2 + (kz[None, :] / (omega.shape[1] / 2)) ** 2)
BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 9.0)]

K_gt = k(gt)
gt_energy = float((np.abs(K_gt) ** 2).sum())
print(f"\n{'arm':11s} {'NMSE':>8s} | {'sampled':>9s} {'unsamp':>9s} {'knownzero':>9s} | "
      + " ".join(f"{a:.2f}-{b:.2f}".rjust(9) for a, b in BANDS))
print("(all numbers after NMSE are energies relative to the GT's, x1e3)")
for name, path in ARMS.items():
    err = np.load(path) - gt
    E = np.abs(k(err)) ** 2                                        # (RO, PE, SPE)
    tot = float(E.sum())
    plane = E.sum(axis=0)                                          # energy per (PE, SPE) position
    parts = [1e3 * float(plane[m].sum()) / gt_energy for m in (omega, Z, rest)]
    assert abs(sum(parts) - 1e3 * tot / gt_energy) < 1e-6 * 1e3, "the three regions must tile the plane"
    bands = [1e3 * float(plane[(rad >= a) & (rad < b)].sum()) / gt_energy for a, b in BANDS]
    print(f"{name:11s} {tot / gt_energy:8.5f} | " + " ".join(f"{p:9.3f}" for p in parts) + " | "
          + " ".join(f"{x:9.3f}" for x in bands))
