"""One-row comparison of all ten Peds AF12 arms on one slice, layout and orientation copied from the topic's
scripts/fig_row_final.py (vertex up, true 0.7 x 0.7 x 1.35 mm proportions, jet error [0, 0.1] of max GT).
Panel metrics as the slice ranking: head = |GT| > 0.05 * P99.9, SPE 12..91; PSNR at max GT; SSIM 7x7 map mean over head;
NMSE over head. Read-only on the topic; writes to /home/congcongliu/figs/peds_tsc_af12_mcgs_best/."""
import argparse, json, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", default="T2s", choices=["T1w", "T2s"])
ap.add_argument("--axis", type=int, required=True, help="2 transverse, 0 coronal, 1 sagittal")
ap.add_argument("--index", type=int, required=True)
ap.add_argument("--note", default="")
args = ap.parse_args()
R = Path("/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro")
F = Path("/home/congcongliu/figs/peds_tsc_af12_mcgs_best"); F.mkdir(parents=True, exist_ok=True)
c = args.contrast
rows = {r["method"]: r for r in json.load(open(R / "logs/summary.json")) if r["contrast"] == c}
PLANE = {0: "coronal", 1: "sagittal", 2: "transverse"}

def lam_file(key, prefix):
    lam = rows[key]["lam"]
    for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
        m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
        if m and abs(float(m.group(1)) - float(lam)) < 1e-12:
            return p
    raise FileNotFoundError(key)

ARMS = [("cgsense", "CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
        ("cssense_o200", "CS-SENSE", lam_file("cssense_o200", "cssense_o200")),
        ("jcssense", "Joint SENSE", lam_file("jcssense", "jcssense_o200")),
        ("l1espirit", "L1-ESPIRiT", lam_file("l1espirit", "l1espirit")),
        ("zsssdu", "ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
        ("dip", "3D-DIP", R / f"recon/dip_{c}_complex.npy"),
        ("inr", "3D-INR", R / f"recon/inr_{c}_complex.npy"),
        ("mcinr", "MC-INR", R / f"recon/mcinr_{c}_complex.npy"),
        ("gs_single_ar", "GS-single", R / f"recon/gs_single_ar_{c}_mag.npy"),
        ("gs_joint_ar", "MC-GS", R / f"recon/gs_joint_ar_{c}_mag.npy")]

gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float64)
gmax = float(gt.max())
head = gt > 0.05 * np.quantile(gt, 0.999)
head[:, :, :12] = False; head[:, :, 92:] = False
sl = [slice(None)] * 3; sl[args.axis] = args.index; sl = tuple(sl)
g2, m2 = gt[sl], head[sl]
g = g2[m2]
vmax = float(np.quantile(g2, 0.999))
imgs, errs, met = {}, {}, {}
for key, lab, path in ARMS:
    v = np.abs(np.load(path, mmap_mode="r")[sl]).astype(np.float64)
    e = v[m2] - g
    _, smap = structural_similarity(g2, v, data_range=gmax, full=True, win_size=7)
    imgs[key] = v; errs[key] = np.abs(v - g2) / gmax
    met[key] = dict(psnr=20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean())), ssim=float(smap[m2].mean()),
                    nmse=float((e * e).sum() / (g * g).sum()))
best = dict(psnr=max(met, key=lambda k: met[k]["psnr"]), ssim=max(met, key=lambda k: met[k]["ssim"]),
            nmse=min(met, key=lambda k: met[k]["nmse"]))

def D(x):
    return x if args.axis == 2 else x.T
ncol = len(ARMS) + 1
H, W = D(g2).shape
VOXEL = (0.7, 0.7, 1.35)
ra, ca = {0: (2, 1), 1: (2, 0), 2: (0, 1)}[args.axis]
pw, ws, cbw = 2.6, 0.03, 0.9
ph = pw * (H * VOXEL[ra]) / (W * VOXEL[ca])
top_in, gap_in, bot_in = 1.25, 0.30, 0.05
fw, fh = pw * (ncol + (ncol - 1) * ws) + cbw, top_in + 2 * ph + gap_in + bot_in
fig, axes = plt.subplots(2, ncol, figsize=(fw, fh))
fig.subplots_adjust(left=0, right=1 - cbw / fw, bottom=bot_in / fh, top=1 - top_in / fh, wspace=ws, hspace=gap_in / ph)
axes[0, 0].imshow(D(g2), cmap="gray", vmin=0, vmax=vmax, origin="upper")
axes[0, 0].set_title(f"GT ({c})", fontsize=13, fontweight="bold")
axes[1, 0].imshow(D(m2), cmap="gray", origin="upper"); axes[1, 0].set_title("head mask", fontsize=11)
for j, (key, lab, _) in enumerate(ARMS, start=1):
    axes[0, j].imshow(D(imgs[key]), cmap="gray", vmin=0, vmax=vmax, origin="upper")
    t = axes[0, j].set_title(f"{lab}\n{met[key]['psnr']:.2f} dB | {met[key]['ssim']:.4f} | {met[key]['nmse']:.5f}", fontsize=10.5,
                             fontweight="bold" if all(best[m] == key for m in best) else "normal")
    im = axes[1, j].imshow(D(errs[key] * m2), cmap="jet", vmin=0.0, vmax=0.1, origin="upper")
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
cax = fig.add_axes([axes[1, -1].get_position().x1 + 0.005, axes[1, -1].get_position().y0, 0.008, axes[1, -1].get_position().height])
fig.colorbar(im, cax=cax).ax.tick_params(labelsize=9)
fig.suptitle(f"{c}  16yo MTP  AF 12.00 (31616 / 2635 sampled points)  --  {PLANE[args.axis]} slice {args.index}\n"
             f"panel: head PSNR | SSIM (7x7) | NMSE of this slice, bold = best on all three;  error | |recon| - |GT| | / max GT inside the head, jet [0, 0.1]"
             + (f"\n{args.note}" if args.note else ""), fontsize=12, fontweight="bold", y=0.995)
out = F / f"peds_af12_{c}_mcgs_best_{PLANE[args.axis]}_{args.index}.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
print(f"http://localhost:8765/peds_tsc_af12_mcgs_best/{out.name}\n{out}")
print(json.dumps({lab: {k: round(v, 4) for k, v in met[key].items()} for key, lab, _ in ARMS}))
