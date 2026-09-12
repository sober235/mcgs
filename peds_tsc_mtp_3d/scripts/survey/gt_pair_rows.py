import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
GT = "/data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT"; FIG = "/home/congcongliu/figs/peds_tsc_age16"
def to_vendor(vol):
    v = vol[:, 2:302, 12:92][:, :, ::-1]; return np.transpose(v, (2, 0, 1))
t1 = to_vendor(np.abs(np.load(f"{GT}/gt_sense_FA16_TE3.25.npy"))); t2 = to_vendor(np.abs(np.load(f"{GT}/gt_sense_FA4_TE27.41.npy")))
zs = [18, 26, 34, 42, 46, 54, 62, 70]
fig, ax = plt.subplots(2, 8, figsize=(24, 6.6))
for r, (lab, v) in enumerate((("T1w  (FA16 / TE 3.25 ms)", t1), ("T2*w  (FA4 / TE 27.41 ms)", t2))):
    hi = np.percentile(v, 99.7)
    for c, z in enumerate(zs):
        ax[r, c].imshow(v[z], cmap="gray", vmin=0, vmax=hi); ax[r, c].axis("off")
        ax[r, c].set_title(f"partition {z}" if r == 0 else "", fontsize=10)
    ax[r, 0].text(-0.08, 0.5, lab, transform=ax[r, 0].transAxes, rotation=90, va="center", ha="center", fontsize=12)
fig.suptitle("GT pair: top row T1-weighted, bottom row T2*-weighted, same partitions"); fig.tight_layout()
fig.savefig(f"{FIG}/gt_pair_rows.png", dpi=100); print("ok")
