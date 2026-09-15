#!/usr/bin/env python3
"""fig_butterfly_final.py -- the visual-gate figure: one transverse slice, every arm of the final table,
the classical arms at their best lambda on this subject.

Default: the butterfly (lateral-ventricle) region of the slice, cropped around the head centroid.
--full: the whole slice (the complete brain in that plane) instead of the crop.

Panels (two blocks, magnitude row + error row each):
  GT | ZF | CG-SENSE | CS-SENSE (TV, converged) | joint TV | L1-ESPIRiT | ZS-SSDU
  3D-DIP | 3D-INR | MC-GS single (cold) | MC-GS joint (cold) | MC-GS single (autoresearch) | MC-GS joint (autoresearch)
The swept classical families use the lambda summarize.py kept (logs/summary.json); the autoresearch arm
is the first entry of logs/extra_gs_arms.json unless --gs_key is given. Titles carry the arm's head-only
PSNR over the whole volume (the table number) and its head-masked PSNR over the shown region (crop or
slice). Error maps: jet, fixed [0, 0.1] of the GT peak, never autoscaled.
"""
import argparse, json, re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1", "FLAIR"])
ap.add_argument("--axis", type=int, default=0)
ap.add_argument("--index", type=int, default=168)
ap.add_argument("--half", type=int, nargs=2, default=[52, 38])
ap.add_argument("--full", action="store_true", help="show the whole slice instead of the butterfly crop")
ap.add_argument("--gs_key", default=None)
ap.add_argument("--tag", default=None)
args = ap.parse_args()
tag = args.tag or ("fullslice" if args.full else "butterflyfinal")
reg = "slice" if args.full else "crop"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

R = Path(__file__).resolve().parents[1]
F = Path("/home/congcongliu/figs/t1flair_v3_af24"); F.mkdir(parents=True, exist_ok=True)
c = args.contrast
rows = {r["method"]: r for r in json.load(open(R / "logs/summary.json")) if r["contrast"] == c}
extra = json.load(open(R / "logs/extra_gs_arms.json")) if (R / "logs/extra_gs_arms.json").exists() else []
gs_key = args.gs_key or next((x["key"] for x in extra if x["key"] != "gs_single_ar"), None)
labels = {x["key"]: x["label"] for x in extra}


def lam_file(row_key, prefix):
    """the saved volume of a swept family (files recon/<prefix>_lam<L>_<c>_complex.npy) at the lambda
    summarize.py kept for that family's row"""
    lam = rows.get(row_key, {}).get("lam")
    if lam is None:
        return None
    for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
        m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
        if m and abs(float(m.group(1)) - float(lam)) < 1e-12:
            return p
    return None


ARMS = [("GT", "Ground truth", R / f"recon/gt_{c}_complex.npy"),
        ("zf", "Zero-filled", R / f"recon/zf_{c}_complex.npy"),
        ("cgsense", "CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
        ("cssense_o200", "CS-SENSE (TV, conv.)", lam_file("cssense_o200", "cssense_o200")),
        ("jcssense", "joint TV", lam_file("jcssense", "jcssense_o200")),
        ("l1espirit", "L1-ESPIRiT", lam_file("l1espirit", "l1espirit")),
        ("zsssdu", "ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
        ("dip", "3D-DIP", R / f"recon/dip_{c}_complex.npy"),
        ("inr", "3D-INR", R / f"recon/inr_{c}_complex.npy"),
        ("gs_single", "MC-GS single (cold)", R / f"recon/gs_single_{c}_mag.npy"),
        ("gs_joint", "MC-GS joint (cold)", R / f"recon/gs_joint_{c}_mag.npy"),
        (gs_key or "gs_ar", labels.get(gs_key, "MC-GS joint (autoresearch)"),
         (R / f"recon/{gs_key}_{c}_mag.npy") if gs_key else None)]
if (R / f"recon/gs_single_ar_{c}_mag.npy").exists():
    ARMS.insert(len(ARMS) - 1, ("gs_single_ar", "MC-GS single (autoresearch)", R / f"recon/gs_single_ar_{c}_mag.npy"))
NCOL = (len(ARMS) + 1) // 2

gt = np.abs(np.load(ARMS[0][2])).astype(np.float32)
gmax = float(gt.max())
head = gt > 0.05 * float(np.quantile(gt, 0.999))
sl = [slice(None)] * 3; sl[args.axis] = args.index; sl = tuple(sl)
gp, hp = gt[sl], head[sl]
if args.full:
    win = (slice(0, gp.shape[0]), slice(0, gp.shape[1]))
    h0, h1 = gp.shape[0] // 2, gp.shape[1] // 2
else:
    yy, xx = np.nonzero(hp); cy, cx = int(yy.mean()), int(xx.mean()); h0, h1 = args.half
    win = (slice(max(cy - h0, 0), cy + h0), slice(max(cx - h1, 0), cx + h1))
gcrop, hcrop = gp[win], hp[win]
g = gcrop[hcrop].astype(np.float64)
vmax = float(np.quantile(gcrop, 0.999))

crops, psnr = {}, {}
for key, lab, path in ARMS[1:]:
    if path is None or not Path(path).exists():
        continue
    v = np.abs(np.load(path)).astype(np.float32)[sl][win]
    crops[key] = v
    e = v[hcrop].astype(np.float64) - g
    psnr[key] = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
rank = {k: i + 1 for i, k in enumerate(sorted(psnr, key=lambda k: -psnr[k]))}

fig, axes = plt.subplots(4, NCOL, figsize=(4.0 * NCOL, 4.0 * 4 * (2 * h1) / (2 * h0) * 1.22))
for i, (key, lab, path) in enumerate(ARMS):
    b, j = divmod(i, NCOL)
    a_img, a_err = axes[2 * b, j], axes[2 * b + 1, j]
    if key == "GT":
        a_img.imshow(gcrop.T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
        a_img.set_title(f"Ground truth ({c})", fontsize=11, fontweight="bold")
        if args.full:
            a_err.imshow(hp.T, cmap="gray", origin="lower")
            a_err.set_title(f"head mask on slice {args.index}: {int(hp.sum())} voxels", fontsize=9)
        else:
            a_err.imshow(gp.T, cmap="gray", vmin=0, vmax=float(np.quantile(gp, 0.999)), origin="lower")
            a_err.add_patch(patches.Rectangle((win[0].start, win[1].start), win[0].stop - win[0].start,
                                              win[1].stop - win[1].start, fill=False, edgecolor="red", linewidth=1.5))
            a_err.set_title(f"crop on axis{args.axis} slice {args.index}", fontsize=9)
    elif key in crops:
        v = crops[key]
        a_img.imshow(v.T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
        vol = rows.get(key, {}).get("head_psnr")
        lamtxt = f" (λ {rows[key]['lam']:g})" if rows.get(key, {}).get("lam") else ""
        best = rank[key] == 1
        a_img.set_title(f"{lab}{lamtxt}{'  <-- best ' + reg if best else ''}\n"
                        f"volume head PSNR {vol:.2f}" if vol is not None else f"{lab}", fontsize=9.5,
                        fontweight="bold" if best else "normal")
        a_img.set_xlabel(f"{reg} head PSNR {psnr[key]:.2f} dB (rank {rank[key]}/{len(rank)})", fontsize=8.5)
        im = a_err.imshow((np.abs(v - gcrop) / gmax).T, cmap="jet", vmin=0.0, vmax=0.1, origin="lower")
        a_err.set_title("|recon - GT| / max(GT)", fontsize=8.5)
        cb = fig.colorbar(im, ax=a_err, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6)
    else:
        a_img.set_title(f"{lab}\n(not available)", fontsize=9.5); a_img.axis("off"); a_err.axis("off")
for k in range(len(ARMS), 2 * NCOL):
    b, j = divmod(k, NCOL); axes[2 * b, j].axis("off"); axes[2 * b + 1, j].axis("off")
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
what = "full transverse slice" if args.full else "butterfly region"
fig.suptitle(f"{c}  jiangyizhou  retrospective AF 24 -- {what}, axis{args.axis} slice {args.index}"
             f"   error maps jet, fixed [0, 0.1]", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = F / f"t1flair_v3_af16g_{c}_{tag}_axis{args.axis}_{args.index}.png"
fig.savefig(out, dpi=120, bbox_inches="tight"); fig.savefig(R / f"figures/{out.name}", dpi=120, bbox_inches="tight")
print("wrote", out)
print(json.dumps({k: round(psnr[k], 3) for k in sorted(psnr, key=lambda k: -psnr[k])}))
