#!/usr/bin/env python3
"""zoom_presentation_test.py -- which way of showing the zoom-box error makes the methods' difference most visible?
(user, 2026-11-21, on the net-18.56x figure with selected boxes: "the error doesn't seem as prominent ... identify
the most obvious presentation method")

On the six zoom boxes of the figure, for the candidates below, (1) a number: the mean position on the colour bar of
each method's box (mean of clip(error, 0, vmax) / vmax) and its gap to MC-GS, (2) a test sheet with CS-SENSE,
ZS-SSDU, GS-single and MC-GS side by side.
  jet [0, vmax] for several vmax (0.1 = the record format)
  the same after a 5x5 local mean of the error (removes the pixel speckle, keeps the level)
  advantage map: (other method's error - MC-GS's error), 5x5 local mean, red = MC-GS lower
Volumes, arms, head mask as fig_t1flair_af_record_format.py (registered GS-single). Read-only, CPU; display study
against GT, nothing enters a reconstruction.
"""
import argparse, json, os, re
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--af", type=int, default=20)
ap.add_argument("--slices", type=int, nargs=3, default=[180, 124, 76])
ap.add_argument("--box_centres", type=int, nargs=6, default=[181, 85, 162, 33, 198, 175])
ap.add_argument("--box", type=int, default=48)
args = ap.parse_args()
R = Path(f"/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF{args.af}_v3_jiangyizhou")
K = "p14" if args.af == 12 else "p3"
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "zoom_out")); art.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)
MC, SHOW = "MC-GS", ["CS-SENSE", "ZS-SSDU", "GS-single", "MC-GS"]


def paths(c):
    rows = {r["method"]: r for r in json.load(open(R / "logs/summary.json")) if r["contrast"] == c}

    def lam(key, prefix):
        hit = [p for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy")
               if abs(float(re.search(r"_lam([0-9.]+)_", p.name).group(1)) - float(rows[key]["lam"])) < 1e-12]
        assert len(hit) == 1
        return hit[0]
    sp = rows["spirit"]
    return {"CG-SENSE": R / f"recon/cgsense_{c}_complex.npy", "CS-SENSE": lam("cssense_o200", "cssense_o200"),
            "Joint SENSE": lam("jcssense", "jcssense_o200"),
            "L1-SPIRiT": R / f"recon/spirit_ks{sp['ksize']}_lam{sp['lam']:g}_{c}_complex.npy",
            "ZS-SSDU": R / f"recon/zsssdu_{c}_complex.npy", "3D-DIP": R / f"recon/dip_{c}_complex.npy",
            "3D-INR": R / f"recon/inr_{c}_complex.npy", "MC-INR": R / f"recon/mcinr_{c}_complex.npy",
            "GS-single": R / f"recon/gs_single_{K}_{c}_mag.npy", MC: R / f"recon/gs_joint_{K}_{c}_mag.npy"}


D = lambda x, axis: np.rot90(x.T, 1) if axis == 0 else x
hw = args.box // 2
names = ("Transverse", "Coronal", "Sagittal")
rows = []                                   # (title, {method: error crop (display orientation)}, {method: smoothed crop})
for c in ("T1", "FLAIR"):
    gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
    gmax, head = float(gt.max()), gt > 0.05 * float(np.quantile(gt, 0.999))
    for l, p in paths(c).items():
        v = np.abs(np.load(p)).astype(np.float32)
        for axis, idx in enumerate(args.slices):
            sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
            ca, cb = args.box_centres[2 * axis:2 * axis + 2]
            e = np.abs(v[sl] - gt[sl]) / gmax * head[sl]
            es = uniform_filter(e, size=5, mode="nearest")
            cr = lambda x: D(x, axis)[::-1][:0:-1][::-1] if False else x
            box = (slice(ca - hw, ca + hw), slice(cb - hw, cb + hw))
            key = (axis, c)
            row = next((r for r in rows if r[0] == key), None)
            if row is None:
                row = (key, f"{names[axis]} {idx} {c}", {}, {}); rows.append(row)
            row[2][l] = np.rot90(e[box].T, 1) if axis == 0 else e[box]
            row[3][l] = np.rot90(es[box].T, 1) if axis == 0 else es[box]
rows.sort(key=lambda r: (r[0][0], r[0][1] != "T1"))
LABS = list(rows[0][2])
OTH = [l for l in LABS if l != MC]

# ---- numbers: mean colour-bar position and gap to MC-GS
VM = [0.1, 0.075, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01]
out = dict(af=args.af, box_centres=args.box_centres, candidates=[])
log("candidate | mean colour-bar position MC-GS | gap GS-single - MC-GS | gap CS-SENSE - MC-GS | gap ZS-SSDU - MC-GS | "
    "gap best-other - MC-GS | MC-GS pixels at the top of the bar")
for smooth in (False, True):
    for vm in VM:
        pos = {l: float(np.mean([np.clip(r[3 if smooth else 2][l], 0, vm).mean() / vm for r in rows])) for l in LABS}
        sat = float(np.mean([(r[3 if smooth else 2][MC] >= vm).mean() for r in rows]))
        best = min(OTH, key=pos.get)
        name = f"jet [0, {vm:g}]" + (", 5x5 local mean" if smooth else "")
        out["candidates"].append(dict(name=name, vmax=vm, smooth=smooth, position=pos, mcgs_saturated=sat))
        log(f"  {name:<32} {pos[MC]:.3f} | {pos['GS-single'] - pos[MC]:+.4f} | {pos['CS-SENSE'] - pos[MC]:+.4f} | "
            f"{pos['ZS-SSDU'] - pos[MC]:+.4f} | {pos[best] - pos[MC]:+.4f} ({best}) | {100 * sat:.1f}%")
adv = {l: float(np.mean([(r[3][l] - r[3][MC]).mean() for r in rows])) for l in OTH}
frac = {l: float(np.mean([((r[3][l] - r[3][MC]) > 0).mean() for r in rows])) for l in OTH}
out["advantage_map"] = dict(mean_error_gap=adv, fraction_of_box_where_mcgs_lower=frac)
log("advantage map (other - MC-GS, 5x5 local mean): fraction of the box where MC-GS is lower: "
    + ", ".join(f"{l} {100 * frac[l]:.0f}%" for l in OTH))
(art / f"zoom_presentation_af{args.af}.json").write_text(json.dumps(out, indent=1))

# ---- test sheet
groups = [("jet [0, 0.1] (record)", 2, 0.1), ("jet [0, 0.05]", 2, 0.05), ("jet [0, 0.03]", 2, 0.03),
          ("5x5 local mean, jet [0, 0.03]", 3, 0.03)]
ncol = len(groups) * len(SHOW) + 3 + len(groups)
fig, axs = plt.subplots(len(rows), ncol, figsize=(1.15 * ncol, 1.25 * len(rows) + 0.9),
                        gridspec_kw=dict(width_ratios=sum(([1] * len(SHOW) + [0.12] for _ in groups), []) + [1, 1, 1]))
for i, r in enumerate(rows):
    j = 0
    for gname, src, vm in groups:
        for l in SHOW:
            a = axs[i, j]; a.imshow(r[src][l], cmap="jet", vmin=0, vmax=vm, origin="lower", interpolation="nearest")
            if i == 0:
                a.set_title(f"{l}", fontsize=7, fontweight="bold" if l == MC else "normal")
            if j == 0:
                a.set_ylabel(r[1], fontsize=7)
            j += 1
        axs[i, j].axis("off"); j += 1
    for l in ("CS-SENSE", "ZS-SSDU", "GS-single"):
        a = axs[i, j]; a.imshow(r[3][l] - r[3][MC], cmap="RdBu_r", vmin=-0.01, vmax=0.01, origin="lower", interpolation="nearest")
        if i == 0:
            a.set_title(f"{l}\nminus MC-GS", fontsize=7)
        j += 1
for a in axs.ravel():
    a.set_xticks([]); a.set_yticks([])
x = 0.03
fig.suptitle("   |   ".join(g[0] for g in groups) + "   |   advantage map: other - MC-GS, 5x5 mean, [-0.01, 0.01], red = MC-GS lower\n"
             f"net 18.56x (nominal AF{args.af}), the six zoom boxes; four methods per group: " + ", ".join(SHOW), fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(art / f"zoom_presentation_af{args.af}.png", dpi=150)
log("DONE")
