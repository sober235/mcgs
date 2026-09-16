#!/usr/bin/env python3
"""ce_seed_null.py -- the floor of each randomly initialised arm (2026-09-16).

The enhancement table says 3D-DIP reports +0.057 white-matter units in a lesion that does not enhance, 3D-INR +0.029,
MC-GS +0.024, against +0.021 for CS-SENSE. Those numbers are differences between two independently reconstructed
scans, so part of them is the method's own randomness rather than contrast uptake. This scores the null: the same
scan, the same data, the same code, only the seed differs (plan `seednull` of ce_orchestrate.py), so whatever is left
is what the arm invents on its own.

  python scripts/ce_seed_null.py [--seed 1] [--rois lesion_candidates/enhance_rois_cssense.npz]
"""
import argparse
import json
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1, help="the second seed, i.e. the run in seed_null_s<seed>/")
ap.add_argument("--rois", default="lesion_candidates/enhance_rois_cssense.npz")
ap.add_argument("--cache", default="figures/cache/paper_mag.npz", help="for the observed post - pre of the same arms")
ap.add_argument("--fig", default="/home/congcongliu/figs/ce_pair/ce_pair_seed_null.png",
                help="side by side at the lesion slice: the two seeds on the same scan, and the two scans")
ap.add_argument("--slice", type=int, default=201)
ap.add_argument("--vmax", type=float, default=0.6, help="top of the colour scale, in white-matter units")
a = ap.parse_args()
B = T / f"seed_null_s{a.seed}"

# (label, run A, run B, GS layout?)
ARMS = [("3D-DIP", T / "dipinr/dip_pre_complex.npy", B / "dip_pre_complex.npy", False),
        ("3D-INR", T / "dipinr/inr_pre_complex.npy", B / "inr_pre_complex.npy", False),
        ("GS-single", T / "gs_single_pre/gs_single_pre_raw_complex.npy",
         B / "gs_single_pre/gs_single_pre_raw_complex.npy", True),
        ("MC-GS", T / "gs_joint/gs_joint_pre_raw_complex.npy", B / "gs_joint/gs_joint_pre_raw_complex.npy", True)]
with np.load(T / a.rois) as r:
    ROI = {n: r[f"roi{i}"] for i, n in enumerate(r["names"])}
WM = ROI["正常白质"]
cache = np.load(T / a.cache)


def wm_units(path, gs):
    v = np.abs(np.load(path)).astype(np.float32)
    if gs:                                            # GS renders in (S-I, L-R, A-P)
        v = np.transpose(v, (0, 2, 1))
    return v / np.median(v[WM])


def observed(label):
    """post - pre of the archived pair, the number the enhancement table reports."""
    p = np.abs(cache[f"pre|{label}"].astype(np.float32)); q = np.abs(cache[f"post|{label}"].astype(np.float32))
    return q / np.median(q[WM]) - p / np.median(p[WM])


rows, panels = [], []
print(f"{'arm':<11}{'core':>9}{'|core|':>9}{'vessels':>9}{'plexus':>9}{'WM':>8}   vs observed post-pre (core, vessels)")
for label, pa, pb, gs in ARMS:
    if not pb.exists():
        print(f"{label:<11}   waiting for {pb}")
        continue
    d = wm_units(pb, gs) - wm_units(pa, gs)
    o = observed(label)
    panels.append((label, d[a.slice], o[a.slice]))
    row = dict(arm=label, seed=a.seed, run_a=str(pa), run_b=str(pb),
               null_core=float(d[ROI["病灶核心"]].mean()), null_core_abs=float(np.abs(d)[ROI["病灶核心"]].mean()),
               null_vessels=float(d[ROI["血管/静脉窦"]].mean()), null_plexus=float(d[ROI["脉络丛"]].mean()),
               null_wm=float(d[WM].mean()), obs_core=float(o[ROI["病灶核心"]].mean()),
               obs_vessels=float(o[ROI["血管/静脉窦"]].mean()))
    rows.append(row)
    print(f"{label:<11}{row['null_core']:>+9.3f}{row['null_core_abs']:>9.3f}{row['null_vessels']:>+9.3f}"
          f"{row['null_plexus']:>+9.3f}{row['null_wm']:>+8.3f}   {row['obs_core']:>+7.3f}  {row['obs_vessels']:>+7.2f}")
if rows:
    json.dump(dict(seed=a.seed, note="null = second seed minus first seed on the pre scan; observed = post minus pre",
                   rows=rows), open(T / f"seed_null_s{a.seed}.json", "w"), indent=1)
    print(f"\nwrote {T / f'seed_null_s{a.seed}.json'}")
    print("read as: the lesion core is not enhancing, so obs_core above null_core is the part that is not explained "
          "by the arm's own randomness.")

if panels:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    try:
        font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    except Exception:
        pass
    with np.load(T / a.rois) as r:
        core2d = r["core"][a.slice].astype(float)
    y0, y1, x0, x1 = 25, 230, 0, 176
    fig, axs = plt.subplots(2, len(panels), figsize=(3.0 * len(panels), 6.6), layout="compressed")
    axs = np.atleast_2d(axs)
    for c, (label, dn, do) in enumerate(panels):
        for r_, (im, ttl) in enumerate([(dn, "同一扫描，换种子"), (do, "增强 − 平扫")]):
            ax = axs[r_, c]
            ax.imshow(np.abs(im[y0:y1, x0:x1]), cmap="magma", vmin=0, vmax=a.vmax)
            ax.contour(core2d[y0:y1, x0:x1], levels=[0.5], colors="#FFC23D", linewidths=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            if r_ == 0: ax.set_title(label, fontsize=11)
            if c == 0: ax.set_ylabel(ttl, fontsize=11)
    n = {r["arm"]: r for r in rows}
    for c, (label, _, _) in enumerate(panels):
        axs[0, c].set_xlabel(f"病灶 {n[label]['null_core']:+.3f}", fontsize=9)
        axs[1, c].set_xlabel(f"病灶 {n[label]['obs_core']:+.3f}", fontsize=9)
    fig.suptitle(f"自身随机性 vs 跨扫描差（z={a.slice}，|差值|，白质=1，色标 0–{a.vmax:g}，橙线=不强化的病灶）\n"
                 "上行：同一份平扫、两个种子之差（真强化为零）；下行：增强 − 平扫", fontsize=11)
    Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig, dpi=120); plt.close(fig)
    print("saved", a.fig)
