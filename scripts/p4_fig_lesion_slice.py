#!/usr/bin/env python3
"""Stage-4 deliverable: one lesion slice, all arms in one row, MC-GS taken from a results_arc run.

Layout and orientation as the topic's fig_row_final.py (vertex up, true 0.7 x 0.7 x 1.35 mm proportions, error
| |recon| - |GT| | / max GT inside the head, jet [0, 0.1]). Panel metrics as scripts/p4_lesion_margin.py and
fig_row_final.py: head = |GT| > 0.05 * P99.9 of the volume, PSNR at the volume max |GT|; SSIM = 7x7 map mean over the
head; NMSE over the head. The GS-single column is the strongest GS-single volume on this slice (stage 2, tuned, or a
results_arc single run), so the column shown is the bar the target was judged against. A red circle marks the lesion.

  $PY scripts/p4_fig_lesion_slice.py --exp p4_sgtv010 --axis 2 --index 69 --mark 166,196 --note "..."
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True, help="results_arc run that fills the MC-GS column")
ap.add_argument("--contrast", default="T2s", choices=["T1w", "T2s"])
ap.add_argument("--axis", type=int, required=True, help="2 transverse, 0 coronal, 1 sagittal")
ap.add_argument("--index", type=int, required=True)
ap.add_argument("--mark", default=None, help="lesion position(s) in the slice's own (row, col) of the volume axes, e.g. RO,PE for "
                                            "transverse or RO,SPE for sagittal; several separated by ';'")
ap.add_argument("--note", default="")
ap.add_argument("--out", default="/home/congcongliu/figs/peds_tsc_af12_lesion_1db")
a = ap.parse_args()
R = Path("/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro")
ARC = Path(__file__).resolve().parents[1] / "results_arc"
F = Path(a.out); F.mkdir(parents=True, exist_ok=True)
c = a.contrast
rows = {r["method"]: r for r in json.load(open(R / "logs/summary.json")) if r["contrast"] == c}
PLANE = {0: "coronal", 1: "sagittal", 2: "transverse"}


def lam_file(key, prefix):
    lam = rows[key]["lam"]
    for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
        m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
        if m and abs(float(m.group(1)) - float(lam)) < 1e-12:
            return p
    raise FileNotFoundError(key)


sl = [slice(None)] * 3; sl[a.axis] = a.index; sl = tuple(sl)
gt_vol = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float64)
gmax = float(gt_vol.max())
head_vol = gt_vol > 0.05 * float(np.quantile(gt_vol, 0.999))
g2, m2 = gt_vol[sl], head_vol[sl]
g = g2[m2]
vmax = float(np.quantile(g2, 0.999))


def load_slice(src):
    if isinstance(src, tuple):
        with np.load(src[0]) as z:
            return np.abs(z[src[1]][sl]).astype(np.float64)
    return np.abs(np.load(src, mmap_mode="r")[sl]).astype(np.float64)


def metrics(v):
    e = v[m2] - g
    _, smap = structural_similarity(g2, v, data_range=gmax, full=True, win_size=7)
    return dict(psnr=20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean())), ssim=float(smap[m2].mean()),
                nmse=float((e * e).sum() / (g * g).sum()))


singles = {"stage 2": R / f"recon/gs_single_ar_{c}_complex.npy"}
if (R / f"recon/gs_single_p3_{c}_complex.npy").exists():
    singles["tuned"] = R / f"recon/gs_single_p3_{c}_complex.npy"
for sj in ARC.glob(f"*single_{c}*_volumes.npz"):
    singles[sj.name[:-len("_volumes.npz")]] = (sj, "recT1_complex")
sing = {k: (load_slice(p), p) for k, p in singles.items()}
sing_met = {k: metrics(v) for k, (v, _) in sing.items()}
best_single = max(sing_met, key=lambda k: sing_met[k]["psnr"])

ARMS = [("CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"), ("CS-SENSE", lam_file("cssense_o200", "cssense_o200")),
        ("Joint SENSE", lam_file("jcssense", "jcssense_o200")), ("L1-ESPIRiT", lam_file("l1espirit", "l1espirit")),
        ("ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"), ("3D-DIP", R / f"recon/dip_{c}_complex.npy"),
        ("3D-INR", R / f"recon/inr_{c}_complex.npy"), ("MC-INR", R / f"recon/mcinr_{c}_complex.npy")]
imgs = {lab: load_slice(p) for lab, p in ARMS}
imgs["GS-single"] = sing[best_single][0]
imgs["MC-GS"] = load_slice((ARC / f"{a.exp}_volumes.npz", "recFL_complex" if c == "T2s" else "recT1_complex"))
met = {k: metrics(v) for k, v in imgs.items()}
order = [lab for lab, _ in ARMS] + ["GS-single", "MC-GS"]
best = dict(psnr=max(met, key=lambda k: met[k]["psnr"]), ssim=max(met, key=lambda k: met[k]["ssim"]),
            nmse=min(met, key=lambda k: met[k]["nmse"]))
margin = met["MC-GS"]["psnr"] - max(met[k]["psnr"] for k in order if k != "MC-GS")


def D(x):
    return x if a.axis == 2 else x.T


ncol = len(order) + 1
H, Wd = D(g2).shape
VOXEL = (0.7, 0.7, 1.35)
ra, ca = {0: (2, 1), 1: (2, 0), 2: (0, 1)}[a.axis]
pw, ws, cbw = 2.6, 0.03, 0.9
ph = pw * (H * VOXEL[ra]) / (Wd * VOXEL[ca])
top_in, gap_in, bot_in = 1.25, 0.30, 0.05
fw, fh = pw * (ncol + (ncol - 1) * ws) + cbw, top_in + 2 * ph + gap_in + bot_in
fig, axes = plt.subplots(2, ncol, figsize=(fw, fh))
fig.subplots_adjust(left=0, right=1 - cbw / fw, bottom=bot_in / fh, top=1 - top_in / fh, wspace=ws, hspace=gap_in / ph)
marks = [tuple(int(x) for x in m.split(",")) for m in a.mark.split(";")] if a.mark else []


def circle(ax):
    for mark in marks:
        r_, c_ = mark if a.axis == 2 else (mark[1], mark[0])
        ax.add_patch(plt.Circle((c_, r_), 7, fill=False, color="red", lw=1.4))


axes[0, 0].imshow(D(g2), cmap="gray", vmin=0, vmax=vmax, origin="upper"); circle(axes[0, 0])
axes[0, 0].set_title(f"GT ({c})", fontsize=13, fontweight="bold")
axes[1, 0].imshow(D(m2), cmap="gray", origin="upper"); axes[1, 0].set_title("head mask", fontsize=11)
for j, k in enumerate(order, start=1):
    axes[0, j].imshow(D(imgs[k]), cmap="gray", vmin=0, vmax=vmax, origin="upper"); circle(axes[0, j])
    axes[0, j].set_title(f"{k}\n{met[k]['psnr']:.2f} dB | {met[k]['ssim']:.4f} | {met[k]['nmse']:.5f}", fontsize=10.5,
                         fontweight="bold" if best["psnr"] == k else "normal")
    im = axes[1, j].imshow(D(np.abs(imgs[k] - g2) / gmax * m2), cmap="jet", vmin=0.0, vmax=0.1, origin="upper")
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
cax = fig.add_axes([axes[1, -1].get_position().x1 + 0.005, axes[1, -1].get_position().y0, 0.008, axes[1, -1].get_position().height])
fig.colorbar(im, cax=cax).ax.tick_params(labelsize=9)
fig.suptitle(f"{c}  16yo MTP  AF 12.00 (31616 / 2635 sampled points)  --  {PLANE[a.axis]} slice {a.index}, lesion circled\n"
             f"panel: head PSNR | SSIM (7x7) | NMSE of this slice, bold = best PSNR;  error | |recon| - |GT| | / max GT inside the head, jet [0, 0.1]"
             + (f"\n{a.note}" if a.note else ""), fontsize=12, fontweight="bold", y=0.995)
out = F / f"peds_af12_{c}_{PLANE[a.axis]}_{a.index}_{a.exp}.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
print(f"http://localhost:8765/{F.name}/{out.name}\n{out}")
print(json.dumps(dict(exp=a.exp, slice=f"{PLANE[a.axis]} {a.index}", margin_psnr=round(margin, 3), gs_single_shown=best_single,
                      gs_single_all={k: round(v["psnr"], 3) for k, v in sing_met.items()},
                      metrics={k: {m: round(x, 4) for m, x in met[k].items()} for k in order})))
