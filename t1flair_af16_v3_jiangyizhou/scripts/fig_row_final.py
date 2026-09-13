#!/usr/bin/env python3
"""fig_row_final.py -- one slice (transverse; --axis 1 coronal, --axis 2 sagittal), the compared methods in ONE row, error maps underneath.

Layout (2 rows x 10 columns):
  row 1: GT | CG-SENSE | CS-SENSE | Joint SENSE | L1-ESPIRiT | ZS-SSDU | 3D-DIP | 3D-INR | GS-single | MC-GS
  row 2: head mask (bottom left) | |recon - GT| / max(GT) for each method, jet, FIXED [0, 0.1], one shared colorbar
Naming (user decision 2026-09-11): "GS-single" = the single-contrast MC-GS arm of the autoresearch recipe,
"MC-GS" = the autoresearch MC-GS joint. CS-SENSE and Joint SENSE are the converged, lambda-swept arms (the
lambda summarize.py kept). Titles: method + head-masked PSNR of the shown region; the best is bold.
Default: the whole slice; --crop: the butterfly region around the head centroid.
"""
import argparse, json, re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1", "FLAIR"])
ap.add_argument("--axis", type=int, default=0)
ap.add_argument("--index", type=int, default=168)
ap.add_argument("--crop", action="store_true", help="butterfly region instead of the whole slice")
ap.add_argument("--half", type=int, nargs=2, default=[52, 38])
ap.add_argument("--tag", default=None)
ap.add_argument("--mcgs_key", default="gs_joint_ar", help="which registered GS joint arm fills the MC-GS column")
ap.add_argument("--mcgs_label", default="MC-GS")
ap.add_argument("--err_complex", action="store_true",
                help="error map and panel PSNR from the COMPLEX difference |recon - GT| instead of the magnitude one")
args = ap.parse_args()
tag = args.tag or ("row_crop" if args.crop else "row") + ("_cplxerr" if args.err_complex else "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[1]
F = Path("/home/congcongliu/figs/t1flair_v3_af16"); F.mkdir(parents=True, exist_ok=True)
c = args.contrast
rows = {r["method"]: r for r in json.load(open(R / "logs/summary.json")) if r["contrast"] == c}
af_eff = json.load(open(R / "logs/prep_04_export.json"))["contrasts"][c]["AF_vs_own_support"]   # user 2026-09-11: label AF 16 as nominal


def lam_file(row_key, prefix):
    """saved volume of a swept family (recon/<prefix>_lam<L>_<c>_complex.npy) at the lambda kept for its row"""
    lam = rows.get(row_key, {}).get("lam")
    if lam is None:
        return None
    for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
        m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
        if m and abs(float(m.group(1)) - float(lam)) < 1e-12:
            return p
    return None


ARMS = [("cgsense", "CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
        ("cssense_o200", "CS-SENSE", lam_file("cssense_o200", "cssense_o200")),
        ("jcssense", "Joint SENSE", lam_file("jcssense", "jcssense_o200")),
        ("l1espirit", "L1-ESPIRiT", lam_file("l1espirit", "l1espirit")),
        ("zsssdu", "ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
        ("dip", "3D-DIP", R / f"recon/dip_{c}_complex.npy"),
        ("inr", "3D-INR", R / f"recon/inr_{c}_complex.npy"),
        ("mcinr", "MC-INR", R / f"recon/mcinr_{c}_complex.npy"),
        ("gs_single_ar", "GS-single", R / f"recon/gs_single_ar_{c}_{'complex' if args.err_complex else 'mag'}.npy"),
        (args.mcgs_key, args.mcgs_label,
         R / f"recon/{args.mcgs_key}_{c}_{'complex' if args.err_complex else 'mag'}.npy")]
missing = [lab for _, lab, p in ARMS if p is None or not Path(p).exists()]
assert not missing, f"missing volumes for {c}: {missing}"

gtc = np.load(R / f"recon/gt_{c}_complex.npy")
gt = np.abs(gtc).astype(np.float32)
gmax = float(gt.max())
head = gt > 0.05 * float(np.quantile(gt, 0.999))
sl = [slice(None)] * 3; sl[args.axis] = args.index; sl = tuple(sl)
gp, hp = gt[sl], head[sl]
if args.crop:
    yy, xx = np.nonzero(hp); cy, cx = int(yy.mean()), int(xx.mean()); h0, h1 = args.half
    win = (slice(max(cy - h0, 0), cy + h0), slice(max(cx - h1, 0), cx + h1))
else:
    win = (slice(0, gp.shape[0]), slice(0, gp.shape[1]))
g2, m2 = gp[win], hp[win]
g2c = gtc[sl][win] if args.err_complex else None          # complex GT of the shown region
g = g2[m2].astype(np.float64)
vmax = float(np.quantile(g2, 0.999))

imgs, errs, psnr = {}, {}, {}
for key, lab, path in ARMS:
    raw = np.load(path)[sl][win]
    v = np.abs(raw).astype(np.float32)
    imgs[key] = v
    if args.err_complex:
        assert np.iscomplexobj(raw), f"{path.name} is not complex; --err_complex needs complex volumes"
        d = raw.astype(np.complex128) - g2c
        errs[key] = (np.abs(d) / gmax).astype(np.float32)
        e = d[m2]
    else:
        errs[key] = (np.abs(v - g2) / gmax).astype(np.float32)
        e = v[m2].astype(np.float64) - g
    psnr[key] = 20 * np.log10(gmax) - 10 * np.log10(float((np.abs(e) ** 2).mean()))
best = max(psnr, key=psnr.get)

ncol = len(ARMS) + 1
D = (lambda x: x.T) if args.axis == 0 else (lambda x: x)  # axis 0 as before; axes 1, 2: axis 0 (inferior-superior) upright
H, W = D(g2).shape
pw, ws, cbw = 2.6, 0.03, 0.9                         # panel width (in), column gap (fraction of a panel), colorbar room (in)
ph = pw * H / W                                      # panel height: every grid cell has the image's aspect, no slack
top_in, gap_in, bot_in = 0.85, 0.30, 0.05            # room above row 1 (suptitle + titles), between the rows, below
fw, fh = pw * (ncol + (ncol - 1) * ws) + cbw, top_in + 2 * ph + gap_in + bot_in
fig, axes = plt.subplots(2, ncol, figsize=(fw, fh))
fig.subplots_adjust(left=0, right=1 - cbw / fw, bottom=bot_in / fh, top=1 - top_in / fh, wspace=ws, hspace=gap_in / ph)
axes[0, 0].imshow(D(g2), cmap="gray", vmin=0, vmax=vmax, origin="lower")
axes[0, 0].set_title(f"GT ({c})", fontsize=13, fontweight="bold")
axes[1, 0].imshow(D(m2), cmap="gray", origin="lower")
axes[1, 0].set_title("head mask", fontsize=11)
for j, (key, lab, _) in enumerate(ARMS, start=1):
    v = imgs[key]
    axes[0, j].imshow(D(v), cmap="gray", vmin=0, vmax=vmax, origin="lower")
    axes[0, j].set_title(f"{lab}\n{psnr[key]:.2f} dB", fontsize=12,
                         fontweight="bold" if key == best else "normal")
    im = axes[1, j].imshow(D(errs[key]), cmap="jet", vmin=0.0, vmax=0.1, origin="lower")
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
cax = fig.add_axes([axes[1, -1].get_position().x1 + 0.005, axes[1, -1].get_position().y0,
                    0.008, axes[1, -1].get_position().height])
cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=9)
region = "butterfly region" if args.crop else {0: "transverse", 1: "coronal", 2: "sagittal"}[args.axis] + " slice"
af_txt = f"nominal AF 16, actual {af_eff:.2f}x" + (" of its acquired k-space" if c == "FLAIR" else "")
fig.suptitle(f"{c}  jiangyizhou  {af_txt} -- {region} {args.index}   "
             f"(PSNR = head-masked PSNR of this {'region' if args.crop else 'slice'};  error "
             f"{'|recon - GT| on the COMPLEX volumes' if args.err_complex else '| |recon| - |GT| |'} / max(GT), jet [0, 0.1])",
             fontsize=13, fontweight="bold", y=0.99)
out = F / f"t1flair_v3_af16_{c}_{tag}_axis{args.axis}_{args.index}.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); fig.savefig(R / f"figures/{out.name}", dpi=130, bbox_inches="tight")
print("wrote", out)
print(json.dumps({lab: round(psnr[k], 3) for k, lab, _ in ARMS}))
