"""Spec 5.4 criterion (b), as corrected on 2026-09-11 (approved by the user): the respiratory third whose
anatomy sits more superior is end-expiration. Both thirds are reconstructed at echo 3; their shift along
the SI axis is the sub-voxel peak of the normalised cross-correlation of the two magnitude volumes inside
the central block 16:96 on every axis (the bright band at the FOV edge does not move with breathing and
made the first version blind). |shift| < 0.5 voxel counts as ambiguous. The result must agree with
criterion (a) (physio.json); only then is ref_ids.npy written."""
import json
import sys

import numpy as np

from common import DERIVED_ROOT, LOGS, TOPIC, recon_dir, save_fig, set_ids

ECHO = 3
BLOCK = slice(16, 96)
LAGS = np.arange(-8, 9)

s = sys.argv[1]
ori = json.loads((TOPIC / "orientation.json").read_text())
si, sup_high = int(ori["si_axis"]), bool(ori["superior_is_high_index"])
low = np.abs(np.load(recon_dir(s) / f"low_e{ECHO}.npy"))
high = np.abs(np.load(recon_dir(s) / f"high_e{ECHO}.npy"))
a = np.moveaxis(low[BLOCK, BLOCK, BLOCK], si, -1)
b = np.moveaxis(high[BLOCK, BLOCK, BLOCK], si, -1)
n = a.shape[-1]


def ncc(u, v):
    u = u - u.mean()
    v = v - v.mean()
    return float(np.sum(u * v) / np.sqrt(np.sum(u * u) * np.sum(v * v)))


cc = np.array([ncc(a[..., max(0, -L):n - max(0, L)], b[..., max(0, L):n - max(0, -L)]) for L in LAGS])
i = int(np.argmax(cc))
frac = 0.5 * (cc[i - 1] - cc[i + 1]) / (cc[i - 1] - 2 * cc[i] + cc[i + 1]) if 0 < i < len(cc) - 1 else 0.0
shift = float(LAGS[i] + frac)                     # high(z) = low(z - shift) along the SI axis
if abs(shift) < 0.5:
    image_end = "ambiguous"
else:
    image_end = "high" if (shift > 0) == sup_high else "low"   # the more superior third is end-expiration
hist_end = json.loads((DERIVED_ROOT / s / "physio.json").read_text())["expiration_end_by_histogram"]
agreed = image_end == hist_end
rep = {"subject": s, "echo": ECHO, "block": [BLOCK.start, BLOCK.stop], "ncc_peak": float(cc[i]),
       "shift_vox_high_vs_low": round(shift, 3), "image_end": image_end, "histogram_end": hist_end,
       "agreed": agreed}
if agreed:
    np.save(DERIVED_ROOT / s / "ref_ids.npy", set_ids(s, hist_end))
    rep["n_ref"] = int(set_ids(s, hist_end).size)
(LOGS / f"polarity_{s}.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1), flush=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

other = [k for k in range(3) if k != si]
cut = other[0]
rest = [k for k in range(3) if k != cut]           # this plane keeps the SI axis
vmax = float(np.quantile(low[BLOCK, BLOCK, BLOCK], 0.995))
fig, axs = plt.subplots(1, 3, figsize=(13, 4.5))
axs[0].plot(LAGS, cc, "o-")
axs[0].axvline(shift, color="r", lw=0.8)
axs[0].set_xlabel("shift of the high third along the SI axis (voxels)")
axs[0].set_ylabel("normalised cross-correlation")
for k, (v, t) in enumerate(((low, "low third"), (high, "high third"))):
    sl = np.take(v, v.shape[cut] // 2, axis=cut)
    img = sl if rest.index(si) == 0 else sl.T         # SI vertical
    axs[k + 1].imshow(img, cmap="gray", origin="lower", vmin=0, vmax=vmax)
    axs[k + 1].axhline(BLOCK.start, color="y", lw=0.6)
    axs[k + 1].axhline(BLOCK.stop, color="y", lw=0.6)
    axs[k + 1].set_title(f"{t}, echo {ECHO}")
fig.suptitle(f"{s}: shift {shift:+.2f} vox (superior = {'high' if sup_high else 'low'} index); "
             f"image says end-expiration = {image_end}, histogram says {hist_end}")
fig.tight_layout()
save_fig(fig, f"{s}_polarity.png")
sys.exit(0 if agreed else 1)
