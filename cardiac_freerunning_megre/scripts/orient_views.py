"""Helper for Task 9 Step 6: coronal and sagittal slices of R0 (echo 3), the tissue fraction along
axis 2 and the central mean intensity per echo; used to decide which end of axis 2 is superior.
Writes <subject>_r0_orientation.png."""
import sys

import numpy as np

from common import recon_dir, save_fig

s = sys.argv[1] if len(sys.argv) > 1 else "V07"
x = np.abs(np.load(recon_dir(s) / "r0_e3.npy"))
core = x[16:96, 16:96, 16:96]                       # ignore the bright FOV-edge rim
vmax = float(np.quantile(core, 0.995))
frac = (core > 0.15 * vmax).mean(axis=(0, 1))
print("tissue fraction of transverse slices (|x| > 0.15 x P99.5 of the central 80^3):")
print("  " + "  ".join(f"axis2={16 + k}: {frac[k]:.2f}" for k in range(0, 80, 8)))
means = [float(np.abs(np.load(recon_dir(s) / f"r0_e{e}.npy"))[16:96, 16:96, 16:96].mean()) for e in range(8)]
print("central mean |x| per echo / echo 0:", [round(m / means[0], 3) for m in means])

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(2, 4, figsize=(16, 8.5))
for j, c1 in enumerate((40, 50, 60, 70)):
    ax[0, j].imshow(x[:, c1, :].T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
    ax[0, j].set_title(f"coronal: axis1={c1}  (x axis0, y axis2 up)", fontsize=9)
for j, c0 in enumerate((36, 48, 60, 72)):
    ax[1, j].imshow(x[c0, :, :].T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
    ax[1, j].set_title(f"sagittal: axis0={c0}  (x axis1, y axis2 up)", fontsize=9)
for a in ax.ravel():
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle(f"{s} R0 echo 3 (TE 7.40 ms), window = P99.5 of the central 80^3")
fig.tight_layout()
save_fig(fig, f"{s}_r0_orientation.png")
