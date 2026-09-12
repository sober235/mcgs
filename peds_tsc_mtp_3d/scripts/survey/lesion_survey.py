"""Montages for a visual lesion survey: vendor 2D T2-FLAIR (tubers), vendor SWI and QSM from the MTP scan
(calcified nodules, veins), and our GT T2*-weighted and T1-weighted MTP images. Display only; no diagnosis."""
import os, glob, collections, numpy as np, pydicom
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/dicom"
GT = "/data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT"
FIG = "/home/congcongliu/figs/peds_tsc_age16"

def load_series(name):
    lst = []
    for f in sorted(glob.glob(os.path.join(ROOT, name, "*.dcm"))):
        d = pydicom.dcmread(f); lst.append((float(d.ImagePositionPatient[2]), d.pixel_array.astype(np.float32)))
    lst.sort(key=lambda t: t[0]); return np.stack([a for _, a in lst], 0)

def montage(vol, idxs, title, path, ncol=6, pct=(1, 99.5), cmap="gray", invert_rows=False):
    nrow = int(np.ceil(len(idxs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.4 * nrow))
    lo, hi = np.percentile(vol[idxs], pct)
    for ax, i in zip(axes.ravel(), idxs):
        ax.imshow(vol[i], cmap=cmap, vmin=lo, vmax=hi); ax.set_title(f"slice {i}", fontsize=9); ax.axis("off")
    for ax in axes.ravel()[len(idxs):]: ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig); print("wrote", path)

flair = load_series("t2_fse_flair_tra_fs_ACS_301")                      # (19, 456, 396)
montage(flair, list(range(19)), "vendor 2D T2-FLAIR transverse (fat-sat), all 19 slices", os.path.join(FIG, "lesion_survey_T2FLAIR_tra.png"), ncol=5)
swi = load_series("mtp_cartesian_.7x.7x1_SWI_801")                      # (80, 320, 300)
qsm = load_series("mtp_cartesian_.7x.7x1_QSM_814")
sel = list(range(14, 72, 4))
montage(swi, sel, "vendor SWI from the MTP scan (minIP-like; calcification and veins dark)", os.path.join(FIG, "lesion_survey_SWI.png"), ncol=5)
montage(qsm, sel, "vendor QSM from the MTP scan (paramagnetic bright, diamagnetic/calcium dark)", os.path.join(FIG, "lesion_survey_QSM.png"), ncol=5, pct=(0.5, 99.5))
def to_vendor(vol):
    v = vol[:, 2:302, 12:92][:, :, ::-1]; return np.transpose(v, (2, 0, 1))
for c, lab in (("FA4_TE27.41", "T2*-weighted"), ("FA16_TE3.25", "T1-weighted")):
    g = np.abs(np.load(os.path.join(GT, f"gt_sense_{c}.npy")))
    montage(to_vendor(g), sel, f"our GT {c} ({lab}, SENSE of the GRAPPA k-space)", os.path.join(FIG, f"lesion_survey_GT_{c}.png"), ncol=5, pct=(1, 99.7))
