#!/usr/bin/env python3
"""fig_masks_af16_diffmask.py -- the two alternative masks beside the AF16 house mask, for inspection before any
reconstruction (2026-09-15).
  t1flair_v3_af16_diffmask_masks.png        row 1: the full 256 x 176 PE plane per mask (sampled white; red: the edge
                                            of FLAIR's acquired region); row 2: the central 64 x 48 with the 24 x 16
                                            ACS outlined in cyan. Titles: points, T1 and FLAIR acceleration (user's
                                            definition of 2026-09-14), the 24x24 centre count, the pattern parameters.
  t1flair_v3_af16_diffmask_gauss_width.png  sampled fraction per ring of the normalised radius (20 rings, log scale)
                                            for the three masks, and the Gaussian mask at 0.7 s, s (registered) and
                                            1.4 s, so the width choice can be overruled by eye.
Written to figures/ of both topics and to /home/congcongliu/figs/t1flair_v3_af16_diffmask/ (served at
http://localhost:8765/t1flair_v3_af16_diffmask/).
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_masks_af16_diffmask import ACS, CY, CZ, REF_MASK, RN, TOPICS, draw_gauss, flair_shutter  # noqa: E402

NAME = "t1flair_v3_af16_diffmask"
SERVED = Path("/home/congcongliu/figs") / NAME
sup = flair_shutter()
supf = sup.astype(float)
ref = sio.loadmat(REF_MASK)["mask"].astype(bool)
masks = [("vd-Poisson disc (house mask, AF16 topic)", ref, None)]
info = {}
for kind in ("gauss", "radial"):
    T = TOPICS[kind]
    j = json.load(open(T / "logs/make_masks_af16_diffmask.json"))
    info[kind] = j
    masks.append((("Gaussian random" if kind == "gauss" else "pseudo-radial spokes") + f" ({kind} topic, new)",
                  sio.loadmat(T / f"data/mask_{kind}_256x176_af16.mat")["mask"].astype(bool), j))


def title(name, m, j):
    n, n_fl = int(m.sum()), int((m & sup).sum())
    c24 = int(m[CY - 12:CY + 12, CZ - 12:CZ + 12].sum())
    extra = ""
    if j is not None and j["kind"] == "gauss":
        g = j["gauss"]
        extra = f"\ns = {g['s']:.3f} (sigma {g['sigma_ky_lines']:.1f} x {g['sigma_kz_lines']:.1f} lines), seed {j['seed']}"
    elif j is not None and j["kind"] == "radial":
        r = j["radial"]
        extra = f"\n{r['spokes']} spokes every {r['angle_step_deg']:.2f} deg, {r['trimmed']} outermost points trimmed"
    return (f"{name}\n{n} of {m.size} points: T1 {m.size / n:.2f}x, FLAIR {m.size / n_fl:.2f}x "
            f"({n - n_fl} points outside the shutter)\n24x24 centre {c24}/576, median rn {np.median(RN[m]):.3f}{extra}")


fig, axes = plt.subplots(2, 3, figsize=(16.5, 11.5), gridspec_kw=dict(height_ratios=[1.5, 1]))
for col, (name, m, j) in enumerate(masks):
    a = axes[0, col]
    a.imshow(m, cmap="gray", interpolation="nearest")
    a.contour(supf, levels=[0.5], colors="r", linewidths=1.0)
    a.set_title(title(name, m, j), fontsize=9.5)
    a.set_xlabel("kz (176)"); a.set_ylabel("ky (256)")
    z = axes[1, col]
    z.imshow(m[CY - 32:CY + 32, CZ - 24:CZ + 24], cmap="gray", interpolation="nearest")
    z.add_patch(Rectangle((24 - ACS[1] // 2 - 0.5, 32 - ACS[0] // 2 - 0.5), ACS[1], ACS[0],
                          fill=False, edgecolor="c", linewidth=1.5))
    z.set_title(f"centre 64 x 48 (ky x kz); cyan: the {ACS[0]} x {ACS[1]} ACS block", fontsize=9.5)
    z.set_xticks([]); z.set_yticks([])
fig.suptitle("jiangyizhou T1+FLAIR, nominal AF 16: the house mask and the two diff-mask candidates "
             "(same 2964-point budget, T1 15.20x; red = FLAIR shutter edge)", fontsize=11)
fig.tight_layout()

fig2, ax = plt.subplots(1, 4, figsize=(20, 6.4), gridspec_kw=dict(width_ratios=[1.5, 1, 1, 1]))
edges = np.linspace(0.0, 1.0, 21)
mid = 0.5 * (edges[:-1] + edges[1:])
for name, m, j in masks:
    d = [m[(RN >= lo) & (RN < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
    ax[0].semilogy(mid, d, marker="o", ms=3, label=name.split(" (")[0])
ax[0].set_xlabel("normalised elliptical radius rn"); ax[0].set_ylabel("sampled fraction per ring")
ax[0].set_title("density profile, 20 rings on rn in [0, 1]", fontsize=10); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
s = info["gauss"]["gauss"]["s"]
for a, f in zip(ax[1:], (0.7, 1.0, 1.4)):
    sf = round(s * f, 3)
    m = draw_gauss(sf)
    a.imshow(m, cmap="gray", interpolation="nearest")
    a.contour(supf, levels=[0.5], colors="r", linewidths=0.8)
    a.set_title(f"Gaussian s = {sf:.3f} ({f:g} x s){'  [registered]' if f == 1.0 else ''}\n"
                f"median rn {np.median(RN[m]):.3f} (house mask {np.median(RN[ref]):.3f})\n"
                f"FLAIR {m.size / (m & sup).sum():.2f}x, {int((m & ~sup).sum())} outside the shutter", fontsize=9.5)
    a.set_xticks([]); a.set_yticks([])
fig2.suptitle("Gaussian width: the registered s matches the house mask's median sampled radius; its neighbours are shown for the eye",
              fontsize=11)
fig2.tight_layout()

SERVED.mkdir(parents=True, exist_ok=True)
for fg, tag in ((fig, "masks"), (fig2, "gauss_width")):
    fn = f"{NAME}_{tag}.png"
    first = None
    for T in TOPICS.values():
        (T / "figures").mkdir(exist_ok=True)
        p = T / "figures" / fn
        if first is None:
            fg.savefig(p, dpi=130); first = p
        else:
            shutil.copy2(first, p)
    shutil.copy2(first, SERVED / fn)
    print(f"http://localhost:8765/{NAME}/{fn}\n{SERVED / fn}\n" + "\n".join(str(T / 'figures' / fn) for T in TOPICS.values()))
