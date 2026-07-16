#!/usr/bin/env python
"""fig_dealias.py — 3-view (axial/coronal/sagittal) de-aliasing panels for the AF=12 dual-contrast
recon. Columns: ZF (aliased zero-filled input) | GT | MC-GS recon | MC-GS |err| | ARC recon | ARC |err|
for each contrast (T1, FLAIR). Clean recons + incoherent (non-fringe) error maps = "no aliasing".

Usage:
  GSTMI python scripts/fig_dealias.py \
     --arc_npz  results_arc_frontier/a_dec24_full_volumes.npz \
     --mcgs_npz results_arc/b1A_vol_volumes.npz \
     --out results_arc_frontier/fig_dealias_af12.png
arc_npz must carry zfT1/zfFL (cycle_arc --save_volumes). Volumes (256,220,256)=(coronal,sagittal,transverse).
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--arc_npz", required=True)
ap.add_argument("--mcgs_npz", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--err_scale", type=float, default=0.10, help="error colormap max (jet, [0, err_scale] of GT max)")
ap.add_argument("--recon_label", default="Paired-ARC-GS", help="column title for the arc_npz recon")
args = ap.parse_args()

VIEWS = [("axial", lambda v: v[:, :, 128]),
         ("coronal", lambda v: v[128, :, :]),
         ("sagittal", lambda v: v[:, 110, :])]


def nz(a):
    a = np.asarray(a, np.float32)
    return a / (a.max() + 1e-9)


A = np.load(args.arc_npz)
M = np.load(args.mcgs_npz) if args.mcgs_npz else None
has_zf = "zfT1" in A
has_mcgs = M is not None

# per-contrast volume bundles
def bundle(con):
    k = "T1" if con == "T1" else "FL"
    d = {"gt": A[f"gt{k}"], "arc": A[f"rec{k}"]}
    if has_zf: d["zf"] = A[f"zf{k}"]
    if has_mcgs: d["mcgs"] = M[f"rec{k}"]
    return d

contrasts = ["T1", "FLAIR"]
# column plan
plan = []
if has_zf: plan.append(("ZF (aliased)", "zf", "gray", None))
plan.append(("GT", "gt", "gray", None))
if has_mcgs:
    plan.append(("MC-GS", "mcgs", "gray", None))
    plan.append(("MC-GS |err|", ("err", "mcgs"), "jet", args.err_scale))
plan.append((args.recon_label, "arc", "gray", None))
plan.append((f"{args.recon_label} |err|", ("err", "arc"), "jet", args.err_scale))

ncol = len(plan)
nrow = len(contrasts) * len(VIEWS)
fig, ax = plt.subplots(nrow, ncol, figsize=(2.15 * ncol, 2.15 * nrow))

for cidx, con in enumerate(contrasts):
    b = bundle(con)
    gtmax = float(np.abs(b["gt"]).max()) + 1e-9
    for vidx, (vname, slc) in enumerate(VIEWS):
        r = cidx * len(VIEWS) + vidx
        gslice = np.abs(slc(b["gt"]))
        for cc, (title, spec, cmap, vmax) in enumerate(plan):
            a = ax[r, cc]
            if isinstance(spec, tuple) and spec[0] == "err":
                img = np.abs(np.abs(slc(b[spec[1]])) - gslice) / gtmax
                a.imshow(np.rot90(img), cmap=cmap, vmin=0, vmax=vmax)
            else:
                a.imshow(np.rot90(nz(np.abs(slc(b[spec])))), cmap=cmap, vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(title, fontsize=10)
            if cc == 0:
                a.set_ylabel(f"{con}\n{vname}", fontsize=9)

fig.suptitle(f"AF=12 dual-contrast joint recon — de-aliasing  (|err| scaled to {args.err_scale:g}×GT max)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"saved -> {args.out}  ({ncol} cols x {nrow} rows; zf={has_zf} mcgs={has_mcgs})")
