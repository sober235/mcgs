"""Spec 5.4 criterion (b), second correction 2026-09-11 (user-approved option 1): the respiratory third
whose anatomy sits more superior is end-expiration, measured where the anatomy moves. Both thirds are
reconstructed at echo 3. Inside a liver-dome box and a heart box (per subject, polarity_boxes.json, placed
on that subject's R0 image) the SI shift is the sub-voxel peak of the normalised cross-correlation of the
two magnitude volumes. Both boxes must agree in sign with |shift| >= 0.5 voxel, else the result is
ambiguous. (A whole-volume block is dominated by spine, costal cartilage and chest wall, which do not move
with breathing.) The result must agree with criterion (a) (physio.json); only then is ref_ids.npy written."""
import json
import sys

import numpy as np

from common import DERIVED_ROOT, LOGS, TOPIC, recon_dir, save_fig, set_ids

ECHO = 3
LAGS = np.arange(-8, 9)
MIN_SHIFT = 0.5

s = sys.argv[1]
ori = json.loads((TOPIC / "orientation.json").read_text())
si, sup_high = int(ori["si_axis"]), bool(ori["superior_is_high_index"])
boxes = json.loads((TOPIC / "polarity_boxes.json").read_text()).get(s)
if boxes is None:
    sys.exit(f"no polarity boxes for {s} in polarity_boxes.json: place them on its R0 image first")
low = np.abs(np.load(recon_dir(s) / f"low_e{ECHO}.npy"))
high = np.abs(np.load(recon_dir(s) / f"high_e{ECHO}.npy"))


def ncc(u, v):
    u = u - u.mean()
    v = v - v.mean()
    return float(np.sum(u * v) / np.sqrt(np.sum(u * u) * np.sum(v * v)))


def shift_in(box):
    sl = tuple(slice(a, b) for a, b in box)
    a = np.moveaxis(low[sl], si, -1)
    b = np.moveaxis(high[sl], si, -1)
    n = a.shape[-1]
    cc = np.array([ncc(a[..., max(0, -L):n - max(0, L)], b[..., max(0, L):n - max(0, -L)]) for L in LAGS])
    i = int(np.argmax(cc))
    f = 0.5 * (cc[i - 1] - cc[i + 1]) / (cc[i - 1] - 2 * cc[i] + cc[i + 1]) if 0 < i < len(cc) - 1 else 0.0
    return float(LAGS[i] + f), cc                    # high(z) = low(z - shift) along the SI axis


res = {name: shift_in(boxes[name]) for name in ("dome", "heart")}
d_dome, d_heart = res["dome"][0], res["heart"][0]
if min(abs(d_dome), abs(d_heart)) < MIN_SHIFT or np.sign(d_dome) != np.sign(d_heart):
    image_end = "ambiguous"
else:
    image_end = "high" if (d_dome > 0) == sup_high else "low"   # the more superior third is end-expiration
hist_end = json.loads((DERIVED_ROOT / s / "physio.json").read_text())["expiration_end_by_histogram"]
agreed = image_end == hist_end
rep = {"subject": s, "echo": ECHO, "boxes": boxes,
       "shift_vox_high_vs_low": {k: round(v[0], 3) for k, v in res.items()},
       "ncc_peak": {k: round(float(v[1].max()), 3) for k, v in res.items()},
       "image_end": image_end, "histogram_end": hist_end, "agreed": agreed}
if agreed:
    np.save(DERIVED_ROOT / s / "ref_ids.npy", set_ids(s, hist_end))
    rep["n_ref"] = int(set_ids(s, hist_end).size)
(LOGS / f"polarity_{s}.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1), flush=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
from matplotlib.patches import Rectangle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

cut = [k for k in range(3) if k != si][1]          # coronal-like plane that keeps the SI axis
keep = [k for k in range(3) if k != cut]
c = (max(boxes["dome"][cut][0], boxes["heart"][cut][0]) + min(boxes["dome"][cut][1], boxes["heart"][cut][1])) // 2
vmax = float(np.quantile(low[16:96, 16:96, 16:96], 0.995))
fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
for name, col in (("dome", "tab:red"), ("heart", "tab:blue")):
    axs[0].plot(LAGS, res[name][1], "o-", color=col, label=f"{name}: {res[name][0]:+.2f} vox")
axs[0].legend()
axs[0].set_xlabel("shift of the high third along the SI axis (voxels)")
axs[0].set_ylabel("normalised cross-correlation")
for k, (v, t) in enumerate(((low, "low third"), (high, "high third"))):
    sl = np.take(v, c, axis=cut)
    axs[k + 1].imshow(sl if keep.index(si) == 0 else sl.T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
    x_ax = keep[0] if keep.index(si) == 1 else keep[1]
    for name, col in (("dome", "tab:red"), ("heart", "tab:blue")):
        (x0, x1), (y0, y1) = boxes[name][x_ax], boxes[name][si]
        axs[k + 1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=1))
    axs[k + 1].set_title(f"{t}, echo {ECHO}, axis{cut}={c}")
fig.suptitle(f"{s}: dome {d_dome:+.2f} / heart {d_heart:+.2f} vox (superior = {'high' if sup_high else 'low'} index); "
             f"image says end-expiration = {image_end}, histogram says {hist_end}")
fig.tight_layout()
save_fig(fig, f"{s}_polarity.png")
sys.exit(0 if agreed else 1)
