import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
GT = "/data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT"; FIG = "/home/congcongliu/figs/peds_tsc_age16"; VOX = (0.7, 0.7, 1.35)
def to_vendor(vol):
    v = vol[:, 2:302, 12:92][:, :, ::-1]; return np.transpose(v, (2, 0, 1))          # (Z=80, rows=RO, cols=PE)
t1 = to_vendor(np.abs(np.load(f"{GT}/gt_sense_FA16_TE3.25.npy"))); t2 = to_vendor(np.abs(np.load(f"{GT}/gt_sense_FA4_TE27.41.npy")))
win = lambda v: (0, np.percentile(v, 99.7))
# 1) three planes, both contrasts
iz, ir, ic = 46, 160, 150
fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
for r, (lab, v) in enumerate((("GT T1-weighted (FA16 / TE 3.25 ms)", t1), ("GT T2*-weighted (FA4 / TE 27.41 ms)", t2))):
    lo, hi = win(v)
    ax[r, 0].imshow(v[iz], cmap="gray", vmin=lo, vmax=hi); ax[r, 0].set_title(f"{lab}\ntransverse, partition {iz}/80", fontsize=10)
    ax[r, 1].imshow(v[:, ir, :], cmap="gray", vmin=lo, vmax=hi, aspect=VOX[2] / VOX[1]); ax[r, 1].set_title(f"coronal, row {ir}", fontsize=10)
    ax[r, 2].imshow(v[:, :, ic], cmap="gray", vmin=lo, vmax=hi, aspect=VOX[2] / VOX[0]); ax[r, 2].set_title(f"sagittal, column {ic}", fontsize=10)
    for a in ax[r]: a.axis("off")
fig.suptitle("Peds_TSC_age16_MTP_GT: the two reference images (ESPIRiT-SENSE of the GRAPPA k-space), 0.7 x 0.7 x 1.35 mm")
fig.tight_layout(); fig.savefig(f"{FIG}/gt_pair_three_planes.png", dpi=110); plt.close(fig)
# 2) transverse montage, T1 | T2* side by side for 8 partitions
zs = [18, 26, 34, 42, 46, 54, 62, 70]
fig, ax = plt.subplots(4, 4, figsize=(15, 16))
for k, z in enumerate(zs):
    r, c = divmod(k, 2)
    for j, (lab, v) in enumerate((("T1w", t1), ("T2*w", t2))):
        lo, hi = win(v); a = ax[r, 2 * c + j]; a.imshow(v[z], cmap="gray", vmin=lo, vmax=hi); a.set_title(f"{lab}, partition {z}", fontsize=10); a.axis("off")
fig.suptitle("GT pair, transverse partitions (T1-weighted left, T2*-weighted right of each pair)")
fig.tight_layout(); fig.savefig(f"{FIG}/gt_pair_transverse_montage.png", dpi=100); plt.close(fig)
print("ok")
