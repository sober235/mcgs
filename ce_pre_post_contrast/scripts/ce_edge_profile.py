#!/usr/bin/env python3
"""ce_edge_profile.py -- how sharply each arm renders the lesion boundary, and whether it puts that boundary in the
same place on both scans (2026-09-16).

No ground truth exists for this pair, so one boundary -- the outer margin of the T1-dark lesion, built once from
CS-SENSE inside --box and anchored on the core mask of ce_lesion_candidates.py enhance -- fixes a frame, and every
arm is measured against it (the core's own boundary is useless here: it sits inside a gradual ramp, and every arm
then measures the same 8 mm):

  signed distance d  : + outside the lesion, - inside, in mm
  profile I(d)       : mean intensity in 0.5 mm bins, each scan in its own white-matter units
  edge width         : distance from the 20% to the 80% level of the step (small = sharp)
  50% shift          : the 50% crossing of post minus that of pre (small = the two scans agree on where the edge is)
  WM high frequency  : std of (I - gaussian(I, 2)) in normal white matter, a stand-in for noise, since a method can
                       buy a narrow edge width with noise

  python scripts/ce_edge_profile.py [--cache figures/cache/paper_mag.npz] [--rois ...] [--band 8]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

T = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="figures/cache/paper_mag.npz")
ap.add_argument("--rois", default="lesion_candidates/enhance_rois_cssense.npz")
ap.add_argument("--band", type=float, default=6.0, help="mm on each side of the boundary")
ap.add_argument("--lesion_thr", type=float, default=0.70,
                help="the lesion against white matter: smoothed CS-SENSE pre below this, component holding the core")
ap.add_argument("--box", default="180:224,85:150,25:75",
                help="search box 'z0:z1,y0:y1,x0:x1' around the lesion; without it the threshold takes the whole head")
ap.add_argument("--out", default="/home/congcongliu/figs/ce_pair")
a = ap.parse_args()

ORDER = ["ZF", "CG-SENSE", "CS-SENSE", "Joint SENSE", "L1-SPIRiT", "ZS-SSDU", "3D-DIP", "3D-INR", "MC-INR",
         "GS-single", "MC-GS"]
with np.load(T / a.rois) as r:
    ROI = {n: r[f"roi{i}"] for i, n in enumerate(r["names"])}
    CORE = r["core"]
WM = ROI["正常白质"]
c = np.load(T / a.cache)
have = [l for l in ORDER if f"pre|{l}" in c.files]

# the boundary to measure: the outer margin of the T1-dark lesion, not the boundary of the dark core, which sits
# inside a gradual ramp. Built once from CS-SENSE and used for every arm.
ref = c["pre|CS-SENSE"].astype(np.float32)
ref = ref / np.median(ref[WM])
box = np.zeros(ref.shape, bool)
box[tuple(slice(*map(int, s.split(":"))) for s in a.box.split(","))] = True    # around the lesion, brain only
BRAIN = ndi.binary_erosion(ndi.binary_fill_holes(WM | (ref > 0.25)), iterations=5)
les = box & BRAIN & (ndi.gaussian_filter(ref, 1.0) < a.lesion_thr)
les = ndi.binary_opening(les, structure=ndi.generate_binary_structure(3, 1), iterations=2)   # cut thin sulci
cc, _ = ndi.label(les, structure=np.ones((3, 3, 3)))
hit = np.bincount(cc[CORE], minlength=2)
hit[0] = 0                                                                    # label 0 is background, never the lesion
if hit.max() == 0:
    raise SystemExit("no component holds the lesion core; loosen --lesion_thr or widen --box")
LES = ndi.binary_fill_holes(cc == int(hit.argmax()))
# + outside, - inside, with the interface at 0: the nearest voxels sit at -0.5 and +0.5, so no bin is empty
d = np.where(LES, -(ndi.distance_transform_edt(LES) - 0.5), ndi.distance_transform_edt(~LES) - 0.5)
band = (np.abs(d) <= a.band) & BRAIN          # inside the brain: restricting the outside to white matter empties
rim = (np.abs(d) <= 1.5) & BRAIN               # the bins next to the boundary, which are cortex-labelled there
print(f"lesion mask {int(LES.sum())} mm3 (thr {a.lesion_thr}); boundary band {int(band.sum())} voxels")
bins = np.arange(-a.band, a.band + 0.5, 0.5)
idx = np.clip(np.digitize(d[band], bins) - 1, 0, len(bins) - 2)
cnt = np.bincount(idx, minlength=len(bins) - 1).astype(float)
ctr = bins[:-1] + 0.25


def profile(v):
    s = np.bincount(idx, weights=v[band].astype(np.float64), minlength=len(bins) - 1)
    return np.where(cnt >= 100, s / np.maximum(cnt, 1), np.nan)              # a bin with too few voxels is not a mean


def cross(p, lo, hi, frac):
    """d where the profile crosses lo + frac * (hi - lo), taking the crossing closest to the boundary."""
    t = lo + frac * (hi - lo)
    f = np.isfinite(p)                                                # a bin with too few voxels carries no mean
    xs, ys = ctr[f], p[f]
    sg = np.sign(ys - t)
    k = [i for i in range(1, len(ys)) if sg[i] != sg[i - 1]]
    if not k:
        return float("nan")
    i = min(k, key=lambda j: abs(xs[j]))
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return float(x0 + (t - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)


rows, prof = [], {}
for label in have:
    row = dict(method=label)
    for scan in ("pre", "post"):
        v = c[f"{scan}|{label}"].astype(np.float32)
        v = v / np.median(v[WM])
        p = profile(v)
        prof[(scan, label)] = p
        inner = float(np.nanmean(p[ctr <= -4])); outer = float(np.nanmean(p[ctr >= 4]))
        row[f"{scan}_in"], row[f"{scan}_out"] = round(inner, 3), round(outer, 3)
        row[f"{scan}_width"] = round(cross(p, inner, outer, 0.8) - cross(p, inner, outer, 0.2), 2)
        row[f"{scan}_mid"] = round(cross(p, inner, outer, 0.5), 2)
        slope = np.gradient(p, ctr)                                   # equivalent-rectangle width: robust to the
        row[f"{scan}_weq"] = round(float((outer - inner) / np.nanmax(slope[np.abs(ctr) <= 2])), 2)  # wiggles of noise
        hf = v - ndi.gaussian_filter(v, 2.0)
        row[f"{scan}_wm_hf"] = round(float(hf[WM].std()), 4)
        gx, gy, gz = np.gradient(v)                                   # sharpness without a profile model
        row[f"{scan}_grad"] = round(float(np.sqrt(gx[rim] ** 2 + gy[rim] ** 2 + gz[rim] ** 2).mean()), 4)
    row["mid_shift"] = round(abs(row["post_mid"] - row["pre_mid"]), 2)
    row["contrast_pre"] = round(row["pre_out"] - row["pre_in"], 3)
    rows.append(row)
    print(f"{label:<12} w_eq pre {row['pre_weq']:>4.2f} post {row['post_weq']:>4.2f} mm | 20-80% pre "
          f"{row['pre_width']:>4.2f} | 50% shift {row['mid_shift']:>4.2f} mm | rim grad {row['pre_grad']:.4f} | "
          f"WM hf {row['pre_wm_hf']:.4f} | grad/hf {row['pre_grad'] / row['pre_wm_hf']:.2f}")
json.dump(dict(band=a.band, bins=list(ctr), rows=rows), open(T / "lesion_edge_profile.json", "w"), indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
try:
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), layout="compressed")
cmap = plt.get_cmap("tab20")
for k, label in enumerate(have):
    for j, scan in enumerate(("pre", "post")):
        axs[j].plot(ctr, prof[(scan, label)], lw=1.6 if label in ("MC-GS", "GS-single") else 0.9,
                    color=cmap(k % 20), label=label if j == 0 else None)
        axs[j].set_title(f"{'平扫' if j == 0 else '增强'}：跨病灶外缘的强度剖面", fontsize=11)
        axs[j].set_xlabel("到边界的带符号距离 (mm)，负 = 病灶内"); axs[j].set_ylabel("强度（白质 = 1）")
        axs[j].grid(alpha=0.25)
axs[0].legend(fontsize=7, ncol=2, loc="upper left")
for k, r in enumerate(rows):
    axs[2].scatter(r["pre_wm_hf"], r["pre_grad"], s=38, color=cmap(k % 20))
    axs[2].annotate(r["method"], (r["pre_wm_hf"], r["pre_grad"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
lim = [0.055, 0.135]
axs[2].plot(lim, lim, "k--", lw=0.8, label="边界梯度 = 白质高频")
axs[2].set_xlim(*lim); axs[2].set_xlabel("白质高频 std（噪声代理）"); axs[2].set_ylabel("病灶边界带的平均梯度")
axs[2].set_title("边界“更锐”其实跟着噪声走", fontsize=11); axs[2].grid(alpha=0.25); axs[2].legend(fontsize=8)
fig.suptitle("病灶外缘的保持（边界由 CS-SENSE 平扫在病灶框内阈值得到，对所有方法固定；无全采样真值）", fontsize=12)
p = Path(a.out) / "ce_pair_lesion_edge_profile.png"
fig.savefig(p, dpi=130); plt.close(fig)
print("saved", p)
