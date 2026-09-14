"""Lesion check at vendor slice 22 (SPE 69, z = 6.2 mm): vendor SWI, GT T2*w, GT T1w, nearest vendor 2D T2-FLAIR slices,
MC-GS T2*w. Pixel data and slice positions only. Display only, not a radiological reading."""
import glob, os
import numpy as np, pydicom
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = "/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/dicom"
GT = "/data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT"
R = "/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro"
OUT = "/home/congcongliu/figs/peds_tsc_af12_mcgs_best/selection/lesion_check_vendor22.png"
def series(name):
    lst = []
    for f in sorted(glob.glob(os.path.join(ROOT, name, "*.dcm"))):
        d = pydicom.dcmread(f)
        lst.append((float(d.ImagePositionPatient[2]), d.pixel_array.astype(np.float32)))
    lst.sort(key=lambda t: t[0])
    return [z for z, _ in lst], np.stack([a for _, a in lst], 0)
zs, swi = series("mtp_cartesian_.7x.7x1_SWI_801")
zf, flair = series("t2_fse_flair_tra_fs_ACS_301")
def vend(vol, i):                      # SPE = 91 - vendor index, PE crop 2:302, rows RO / cols PE (as the survey)
    return np.abs(vol[:, 2:302, 91 - i])
t2 = vend(np.load(os.path.join(GT, "gt_sense_FA4_TE27.41.npy"), mmap_mode="r"), 22)
t1 = vend(np.load(os.path.join(GT, "gt_sense_FA16_TE3.25.npy"), mmap_mode="r"), 22)
mc = vend(np.load(os.path.join(R, "recon/gs_joint_ar_T2s_mag.npy"), mmap_mode="r"), 22)
k = int(np.argmin([abs(z - zs[22]) for z in zf]))
panels = [(f"vendor SWI, slice 22 (z {zs[22]:.1f} mm)", swi[22]), ("GT T2*w FA4/TE27.41, slice 22", t2),
          ("MC-GS T2*w (AF 12), slice 22", mc), ("GT T1w FA16/TE3.25, slice 22", t1),
          (f"vendor 2D T2-FLAIR slice {k} (z {zf[k]:.1f} mm, 5 mm thick)", flair[k]),
          (f"vendor 2D T2-FLAIR slice {k-1} (z {zf[k-1]:.1f} mm)", flair[k - 1])]
fig, axes = plt.subplots(1, 6, figsize=(27, 5.2))
for ax, (lab, im) in zip(axes, panels):
    ax.imshow(im, cmap="gray", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.5)); ax.set_title(lab, fontsize=10); ax.axis("off")
fig.suptitle("Vendor slice 22 = SPE 69, the MC-GS best-slice figure. Stage-1 survey: nodules at vendor slices 38 / 46 (z 27.6 / 38.3 mm), "
             "tubers on FLAIR slices 10 / 13 / 14 (z 40.1 / 59.1 / 65.5 mm). Visual check only.", fontsize=11)
fig.tight_layout(); fig.savefig(OUT, dpi=110); print(OUT)
