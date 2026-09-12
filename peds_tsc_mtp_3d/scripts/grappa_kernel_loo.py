#!/usr/bin/env python
"""Choose the GRAPPA kernel by leave-one-line-out on the ACS block of one contrast.

    ./py grappa_kernel_loo.py --contrast FA16_TE3.25 --device cuda:0

Reads data/<contrast>/kspace_grappa.h5 and masks.npz (the acquired lines are untouched by GRAPPA, so
kspace * m_acq is the measured data), runs uih_mtp.grappa.acs_leave_one_out for a list of kernel settings
and writes logs/grappa_kernel_loo_<contrast>.json plus a table on stdout.
"""
import argparse, json, os, sys, time

import h5py
import numpy as np
import torch

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt")
from uih_mtp.grappa import acs_leave_one_out   # noqa: E402

TOPIC = "/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_3D"
SETTINGS = [  # (n_pe_sources, dx, dz, lam_rel)
    (4, 2, 2, 1e-2), (4, 2, 2, 1e-3), (4, 2, 2, 1e-4),
    (2, 2, 2, 1e-3), (4, 1, 1, 1e-3), (4, 3, 2, 1e-3), (4, 2, 3, 1e-3), (6, 2, 2, 1e-3),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    d = os.path.join(TOPIC, "data", args.contrast)
    masks = np.load(os.path.join(d, "masks.npz"))
    m_acq, acs = masks["m_acq"], tuple(int(v) for v in masks["acs"])
    with h5py.File(os.path.join(d, "kspace_grappa.h5"), "r") as f:
        k = torch.from_numpy(f["kspace"][()])
    k = k * torch.from_numpy(m_acq)[None, None]
    device = torch.device(args.device)
    rows = []
    print(f"{'n_pe':>4s} {'dx':>3s} {'dz':>3s} {'lam_rel':>8s} {'mean err':>9s} {'max err':>8s} {'s':>5s}", flush=True)
    for n_pe, dx, dz, lam in SETTINGS:
        t0 = time.time()
        loo = acs_leave_one_out(k, m_acq, acs_lines=acs, R=2, n_pe_sources=n_pe, dx=dx, dz=dz, lam_rel=lam, device=device)
        dt = time.time() - t0
        rows.append(dict(n_pe_sources=n_pe, dx=dx, dz=dz, lam_rel=lam, mean=loo["mean"], per_line=loo["per_line"], lines=loo["lines"], seconds=dt))
        print(f"{n_pe:4d} {dx:3d} {dz:3d} {lam:8.0e} {loo['mean']:9.4f} {max(loo['per_line']):8.4f} {dt:5.0f}", flush=True)
    best = min(rows, key=lambda r: r["mean"])
    print("best:", {k_: best[k_] for k_ in ("n_pe_sources", "dx", "dz", "lam_rel", "mean")})
    os.makedirs(os.path.join(TOPIC, "logs"), exist_ok=True)
    json.dump(dict(contrast=args.contrast, acs=list(acs), rows=rows, best=best),
              open(os.path.join(TOPIC, "logs", f"grappa_kernel_loo_{args.contrast}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
