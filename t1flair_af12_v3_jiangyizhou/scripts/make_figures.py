#!/usr/bin/env python3
"""make_figures.py -- three-plane comparison figures for the retrospective AF12 T1+FLAIR study.

Error panels follow the project figure convention: jet colormap, FIXED range [0, 0.1] of the
GT peak, never autoscaled per panel, colorbar tight to the right of each panel.

Volume (256,256,176) = (readout x = L-R for a sagittal 3D slab, ky, kz). The three orthogonal
mid-slices are taken as v[nx//2], v[:,ny//2], v[:,:,nz//2] and labelled by axis index only --
this data's anatomical axis assignment is NOT the (256,220,256) house convention and has not
been independently adjudicated, so the panels say "axis0/axis1/axis2", not "sagittal/coronal".
"""
import argparse, json, os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1", "FLAIR"])
ap.add_argument("--methods", default="zf,cgsense,cssense,zsssdu,dip,inr,gs_single,gs_joint")
ap.add_argument("--out", default=None)
args = ap.parse_args()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou")
FIGS = Path("/home/congcongliu/figs/t1flair_v3_af12")
FIGS.mkdir(parents=True, exist_ok=True)

ANAT = {0: "transverse (axial)", 1: "coronal", 2: "sagittal"}   # adjudicated 2026-09-09, see README
LABEL = {"zf": "Zero-filled", "cgsense": "CG-SENSE", "cssense": "CS-SENSE (TV)",
         "zsssdu": "ZS-SSDU", "dip": "3D-DIP", "inr": "3D-INR",
         "gs_single": "MC-GS single", "gs_joint": "MC-GS joint"}


def load(method, c):
    """Classical methods save a complex volume; the MC-GS runs save a magnitude one."""
    for suf in ("_complex.npy", "_mag.npy"):
        p = R / f"recon/{method}_{c}{suf}"
        if p.exists():
            return np.abs(np.load(p)).astype(np.float32)
    return None


def planes(v):
    nx, ny, nz = v.shape
    return [v[nx // 2], v[:, ny // 2], v[:, :, nz // 2]]


def main():
    c = args.contrast
    gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
    gmax = float(gt.max())
    methods = [m for m in args.methods.split(",")
               if (R / f"recon/{m}_{c}_complex.npy").exists() or (R / f"recon/{m}_{c}_mag.npy").exists()]
    if not methods:
        raise SystemExit(f"no reconstructions found for {c}")
    # canonical table written by scripts/summarize.py; fall back to the per-method logs
    scores = {}
    sp = R / "logs/summary.json"
    if sp.exists():
        for row in json.load(open(sp)):
            if row.get("contrast") == c:
                scores[row["method"]] = row
    else:
        for f in R.glob("logs/*.json"):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            for row in (d.get("rows") or []):
                if row.get("contrast") == c:
                    scores[row["method"]] = row

    ncol = len(methods) + 1
    for ip, pname in enumerate([f"axis{a} mid-slice = {ANAT[a]}" for a in range(3)]):
        fig, axes = plt.subplots(2, ncol, figsize=(3.4 * ncol, 7.6))
        gp = planes(gt)[ip]
        vmax = float(np.quantile(gp, 0.999))
        axes[0, 0].imshow(gp.T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
        axes[0, 0].set_title(f"Ground truth ({c})", fontsize=12, fontweight="bold")
        # the otherwise-empty cell carries the sampling pattern the whole row is conditioned on
        import scipy.io
        m2 = scipy.io.loadmat(R / "data/mask_vdpois_256x176_af12.mat")["mask"].astype(float)
        axes[1, 0].imshow(m2.T, cmap="gray", origin="lower", interpolation="nearest", aspect="auto")
        axes[1, 0].set_xlabel("AF12 vd-Poisson mask (ky,kz)\n8.54 % of the PE plane", fontsize=9)
        for j, m in enumerate(methods, start=1):
            v = load(m, c)
            vp = planes(v)[ip]
            axes[0, j].imshow(vp.T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
            s = scores.get(m)
            ttl = LABEL.get(m, m)
            if s:
                ttl += (f"\nwhole  PSNR {s['psnr']:.2f}  SSIM {s['ssim']:.3f}"
                        f"\nhead   PSNR {s['head_psnr']:.2f}  SSIM {s['head_ssim']:.3f}")
            axes[0, j].set_title(ttl, fontsize=10)
            err = np.abs(vp - gp) / gmax
            im = axes[1, j].imshow(err.T, cmap="jet", vmin=0.0, vmax=0.1, origin="lower")
            axes[1, j].set_title("|recon - GT| / max(GT)", fontsize=9)
            cb = fig.colorbar(im, ax=axes[1, j], fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6)
        for a in axes.ravel():
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{c}  retrospective AF 12 (vd-Poisson, PE plane 256x176)  --  {pname}"
                     f"   error maps: jet, fixed [0, 0.1]", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = FIGS / f"t1flair_v3_af12_{c}_plane{ip}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        fig.savefig(R / f"figures/{out.name}", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
