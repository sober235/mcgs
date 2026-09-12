#!/usr/bin/env python3
"""prep_05_csm_to_h5.py -- BART ESPIRiT CSM (.cfl) -> h5 key 'map', coil-first (C,X,Y,Z).

Also verifies the map the way scripts/estimate_csm.py does: with `ecalib -c 0.0` the single
eigenvector map has unit modulus over the WHOLE FOV, so sqrt(sum_c |S_c|^2) == 1 everywhere
(air included). That is the expected behaviour, not a defect.
"""
import json, os, sys
import h5py, numpy as np

R = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v4_tangbing"


def cfl(base):
    with open(base + ".hdr") as f:
        dims = [int(x) for x in f.read().split("\n")[1].split()]
    n = int(np.prod(dims))
    return np.memmap(base + ".cfl", dtype=np.complex64, mode="r", shape=(n,)).reshape(dims, order="F")


def main(tags):
    rep = {}
    for tag in tags:
        base = f"{R}/data/csm_espirit_{tag}"
        s = np.squeeze(cfl(base))                       # (X,Y,Z,C)
        assert s.ndim == 4, s.shape
        arr = np.ascontiguousarray(np.moveaxis(np.asarray(s), 3, 0))     # (C,X,Y,Z)
        sos = np.sqrt((np.abs(arr) ** 2).sum(axis=0))
        out = f"{R}/data/csm_espirit_{tag}.h5"
        with h5py.File(out, "w") as h:
            h.create_dataset("map", data=arr, chunks=(1, 32, 256, 176))
        rep[tag] = {"cfl_dims": list(s.shape), "h5": out, "shape": list(arr.shape),
                    "sos_min": float(sos.min()), "sos_max": float(sos.max()),
                    "sos_mean": float(sos.mean()),
                    "sos_frac_within_5pct_of_1": float((np.abs(sos - 1) < 0.05).mean())}
        print(f"[{tag}] {out} shape={arr.shape} SoS min/mean/max="
              f"{sos.min():.4f}/{sos.mean():.4f}/{sos.max():.4f}", flush=True)
        del arr, s, sos
    p = f"{R}/logs/prep_05_csm.json"
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(rep); json.dump(old, open(p, "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1:] or ["T1", "FLAIR"])
