#!/usr/bin/env python3
"""retro_figure.py -- the retrospective comparison at the slice where the enhancing tumour is largest (2026-09-17).

Same shape as the prospective enhancement figure, with a truth column added: pre / post / post - pre for the ground
truth and every arm, the enhancing label outlined in cyan and the non-enhancing necrotic core in orange, and under
each column the three numbers that matter -- PSNR, how much of the true uptake is kept, and how much uptake the arm
invents inside the necrotic core.

  python scripts/retro_figure.py [--case UCSF-PDGM-0053] [--out /home/congcongliu/figs/ce_retro]
"""
import argparse
import json
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--case", default="UCSF-PDGM-0053")
ap.add_argument("--arms", default="zf,cgsense,cssense,dip,inr,gs_single,gs_joint")
ap.add_argument("--out", default="/home/congcongliu/figs/ce_retro")
ap.add_argument("--vmax", type=float, default=1.6, help="grey scale of the images, in white-matter units")
ap.add_argument("--dvmax", type=float, default=0.8, help="top of the post - pre scale")
a = ap.parse_args()
LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense": "CS-SENSE", "dip": "3D-DIP", "inr": "3D-INR",
         "gs_single": "GS-single", "gs_joint": "MC-GS"}

g = np.load(T / "sim" / a.case / "gt.npz")
gt = {"pre": np.abs(g["pre"]), "post": np.abs(g["post"])}
BR, tum = g["brain"], g["tumor"]
enh, nec = tum == 4, tum == 1
z = int(np.argmax(enh.sum(axis=(1, 2))))                       # the slice with the most enhancing tumour
rows = {r["arm"]: r for r in json.load(open(T / "retro_scores.json"))["rows"] if r["case"] == a.case}
lam = json.load(open(T / "recon" / a.case / "cs_lambda.json")) if (T / "recon" / a.case / "cs_lambda.json").exists() else None


def load(arm, scan):
    R = T / "recon" / a.case
    if arm == "cssense":
        l = lam[scan]["lam"] if lam else 2e-3
        p, gs = R / "tv" / f"cssense_o200_lam{l:g}_full_{scan}_complex.npy", False
    elif arm == "gs_single":
        p, gs = R / f"gs_single_{scan}" / f"gs_single_{scan}_raw_complex.npy", True
    elif arm == "gs_joint":
        p, gs = R / "gs_joint" / f"gs_joint_{scan}_raw_complex.npy", True
    else:
        p, gs = R / f"{arm}_{scan}_complex.npy", False
    if not p.exists():
        return None
    v = np.load(p)
    if gs:
        v = np.transpose(v, (0, 2, 1))
    al = np.vdot(v[BR], g[scan][BR]) / np.vdot(v[BR], v[BR])
    return np.abs(al * v)


cols = [("真值 GT", gt["pre"], gt["post"])]
for arm in a.arms.split(","):
    xp, xq = load(arm, "pre"), load(arm, "post")
    if xp is not None and xq is not None:
        cols.append((LABEL.get(arm, arm), xp, xq))

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
ys, xs = np.where(BR[z])
y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
crop = lambda v: v[z, y0:y1, x0:x1]
fig, axs = plt.subplots(3, len(cols), figsize=(2.0 * len(cols), 6.6), layout="compressed")
for c, (name, xp, xq) in enumerate(cols):
    for r, (im, cm, vmx) in enumerate([(crop(xp), "gray", a.vmax), (crop(xq), "gray", a.vmax),
                                       (crop(xq - xp), "magma", a.dvmax)]):
        ax = axs[r, c]
        ax.imshow(im, cmap=cm, vmin=0, vmax=vmx)
        ax.contour(crop(enh).astype(float), levels=[0.5], colors="#3BE8FF", linewidths=0.5)
        ax.contour(crop(nec).astype(float), levels=[0.5], colors="#FFC23D", linewidths=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(name, fontsize=10)
        if c == 0: ax.set_ylabel(["平扫", "增强", "增强 − 平扫"][r], fontsize=11)
    if name in rows:
        q = rows[name]
        axs[2, c].set_xlabel(f"PSNR {(q['psnr_pre'] + q['psnr_post']) / 2:.1f}\n保留 {100 * q['kept']:.0f}%  "
                             f"假强化 {q['false_necrotic']:+.3f}", fontsize=7.5)
    elif c == 0:
        axs[2, c].set_xlabel("真值", fontsize=7.5)
t = rows.get("GS-single", {}).get("uptake_true", float("nan"))
fig.suptitle(f"回顾性验证（{a.case}，UCSF-PDGM，仿真 24 线圈 AF 5.34，白质 = 1）：横断位 z={z}，强化区最大层\n"
             f"青线 = 强化区（真值强化 {t:+.3f}），橙线 = 坏死核心（真值不强化）；下行色标 0–{a.dvmax:g}", fontsize=11)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
p = out / f"retro_{a.case}_z{z}.png"
fig.savefig(p, dpi=120)
plt.close(fig)
print("saved", p)
