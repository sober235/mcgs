#!/usr/bin/env python3
"""prep_04b_export_draws.py -- 3-D masks and the AF read-out for the extra AF20 mask draws b, c, d.

Same broadcast as prep_04_export.py (the 2-D (ky,kz) pattern along the readout), written to
data/mask_af20_d<tag>_3d.h5; the registered mask (draw a) and the k-space symlinks are not touched.
Each contrast's acquired support is read from its own coil-first k-space h5 (readout-line energy),
so the AF against FLAIR's shutter is measured per draw. Report: logs/prep_04b_export_draws.json.
"""
import json
import h5py, numpy as np, scipy.io

R = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou"


def support(tag):
    with h5py.File(f"{R}/data/ksp_{tag}.h5", "r") as h:
        k = h["kspace"]                                   # (C, X, Y, Z)
        sup = np.zeros(k.shape[2:], dtype=bool)
        for x in range(0, k.shape[1], 8):
            sup |= np.abs(k[0, x]) > 0
    return sup


sups = {c: support(c) for c in ("T1", "FLAIR")}
rep = {}
for d in ("a", "b", "c", "d"):
    src = f"{R}/data/mask_vdpois_256x176_af20.mat" if d == "a" else f"{R}/data/mask_vdpois_256x176_af20_d{d}.mat"
    m2 = scipy.io.loadmat(src)["mask"].astype(bool)
    assert m2.shape == (256, 176), m2.shape
    if d != "a":
        out = f"{R}/data/mask_af20_d{d}_3d.h5"
        with h5py.File(out, "w") as h:
            h.create_dataset("mask", data=np.broadcast_to(m2[None], (256, 256, 176)).astype(np.uint8),
                             compression="gzip", compression_opts=1)
    ref = scipy.io.loadmat(f"{R}/data/mask_vdpois_256x176_af20.mat")["mask"].astype(bool)
    r = {"samples": int(m2.sum()), "AF_rect": float(m2.size / m2.sum()),
         "shared_with_draw_a": int((m2 & ref).sum()),
         "acs24_kept": float(m2[128 - 12:128 + 12, 88 - 12:88 + 12].mean())}
    for c, sup in sups.items():
        eff = m2 & sup
        r[f"{c}_AF_own"] = float(sup.mean() / eff.mean())
        r[f"{c}_off_support"] = int((m2 & ~sup).sum())
    rep[d] = r
    print(d, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
json.dump(rep, open(f"{R}/logs/prep_04b_export_draws.json", "w"), indent=2)
