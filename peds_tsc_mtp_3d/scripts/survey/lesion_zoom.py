import os, glob, numpy as np, pydicom
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/dicom"
GT = "/data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT"; FIG = "/home/congcongliu/figs/peds_tsc_age16"
def load_series(name):
    lst = [(float(pydicom.dcmread(f).ImagePositionPatient[2]), pydicom.dcmread(f).pixel_array.astype(np.float32)) for f in sorted(glob.glob(os.path.join(ROOT, name, "*.dcm")))]
    lst.sort(key=lambda t: t[0]); return np.stack([a for _, a in lst], 0)
def to_vendor(vol):
    v = vol[:, 2:302, 12:92][:, :, ::-1]; return np.transpose(v, (2, 0, 1))
swi = load_series("mtp_cartesian_.7x.7x1_SWI_801"); flair = load_series("t2_fse_flair_tra_fs_ACS_301")
t2s = to_vendor(np.abs(np.load(os.path.join(GT, "gt_sense_FA4_TE27.41.npy")))); t1 = to_vendor(np.abs(np.load(os.path.join(GT, "gt_sense_FA16_TE3.25.npy"))))
crop = (slice(90, 250), slice(70, 230))            # central region around the lateral ventricles (rows, cols) in the 320x300 frame
panels = [("vendor SWI, slice 38 (centre crop)", swi[38][crop]), ("vendor SWI, slice 46 (centre crop)", swi[46][crop]),
          ("GT T2*w FA4/TE27.41, slice 38", t2s[38][crop]), ("GT T2*w FA4/TE27.41, slice 46", t2s[46][crop]),
          ("vendor T2-FLAIR 2D, slice 10", flair[10]), ("vendor T2-FLAIR 2D, slice 13", flair[13]),
          ("vendor T2-FLAIR 2D, slice 14", flair[14]), ("GT T1w FA16/TE3.25, slice 46", t1[46][crop])]
fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))
for ax, (lab, im) in zip(axes.ravel(), panels):
    ax.imshow(im, cmap="gray", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.5)); ax.set_title(lab, fontsize=10); ax.axis("off")
fig.suptitle("Visual survey only (not a radiological reading): dark foci at the lateral-ventricle walls on SWI / T2*w, bright cortical-subcortical patches on FLAIR")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "lesion_survey_zoom.png"), dpi=100); print("ok")
