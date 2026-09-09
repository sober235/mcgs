#!/usr/bin/env python3
"""prep_04_export.py -- CFL -> h5 k-space (coil-first) + the 3D AF12 retrospective mask.

Geometry established by prep_01_inspect.py:
  volume (256, 256, 176) = (readout x, ky, kz);  readout_axis = 0, phase_axes = [1, 2]
  k-space DC is at the array centre on both PE axes (128, 88) -- no mask-registration roll needed.
  T1 support = the full PE plane; T2 support = an elliptical vendor shutter (77.71 %).

The SAME vd-Poisson mask is used for both contrasts (house convention). Each contrast then only
ever sees mask AND its own acquired support, which is what a scanner would deliver.
"""
import json, os
import h5py, numpy as np, scipy.io

SRC = "/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL/vol1_huangyuwei"
R = "/data0/congcong/code/GS/Results_20260908/T1T2_AF12_retro"
FILES = {"T1": "t1_fsp_train_huangyuwei_cc", "T2": "t2_train_huangyuwei_cc"}
MASK2D = f"{R}/data/mask_vdpois_256x176_af12.mat"


def cfl(base):
    with open(base + ".hdr") as f:
        dims = [int(x) for x in f.read().split("\n")[1].split()]
    n = int(np.prod(dims))
    return np.memmap(base + ".cfl", dtype=np.complex64, mode="r", shape=(n,)).reshape(dims, order="F")[..., 0]


def main():
    m2 = scipy.io.loadmat(MASK2D)["mask"].astype(np.float32)      # (256,176) = (ky,kz)
    assert m2.shape == (256, 176), m2.shape
    rep = {"mask2d_shape": list(m2.shape), "mask2d_samples": int(m2.sum()),
           "mask2d_frac": float(m2.mean()), "mask2d_AF_rect": float(1.0 / m2.mean()),
           "contrasts": {}}

    mask3d = np.broadcast_to(m2[None, :, :], (256, 256, 176)).astype(np.float32)
    with h5py.File(f"{R}/data/mask_af12_3d.h5", "w") as h:
        h.create_dataset("mask", data=mask3d.astype(np.uint8), compression="gzip", compression_opts=1)
    rep["mask3d_path"] = f"{R}/data/mask_af12_3d.h5"

    for tag, f in FILES.items():
        k = cfl(os.path.join(SRC, f))                              # (X,Y,Z,C)
        nx, ny, nz, nc = k.shape
        # acquired support of this contrast on the PE plane, from the readout-line energy
        sup = np.zeros((ny, nz), dtype=bool)
        for x in range(0, nx, 8):
            sup |= (np.abs(np.asarray(k[x])).max(axis=-1) > 0)
        eff = m2.astype(bool) & sup
        rep["contrasts"][tag] = {
            "src": os.path.join(SRC, f), "dims": [nx, ny, nz, nc],
            "support_frac": float(sup.mean()),
            "sampled_frac_of_plane": float(eff.mean()),
            "AF_vs_full_rect": float(1.0 / eff.mean()),
            "AF_vs_own_support": float(sup.mean() / eff.mean()),
            "acs24_kept": float(eff[ny // 2 - 12:ny // 2 + 12, nz // 2 - 12:nz // 2 + 12].mean()),
        }
        out = f"{R}/data/ksp_{tag}.h5"
        arr = np.ascontiguousarray(np.moveaxis(np.asarray(k), 3, 0))   # (C,X,Y,Z)
        with h5py.File(out, "w") as h:
            h.create_dataset("kspace", data=arr, chunks=(1, 32, 256, 176))
        rep["contrasts"][tag]["ksp_h5"] = out
        print(f"[{tag}] wrote {out}  shape={arr.shape}  "
              f"support={sup.mean():.4f} sampled={eff.mean():.4f} "
              f"AF_rect={1/eff.mean():.2f} AF_own={sup.mean()/eff.mean():.2f}", flush=True)
        del arr, k

    json.dump(rep, open(f"{R}/logs/prep_04_export.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
