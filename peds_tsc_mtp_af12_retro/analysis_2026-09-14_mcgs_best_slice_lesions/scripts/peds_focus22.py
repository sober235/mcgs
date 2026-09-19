import glob, os
import numpy as np, pydicom
from scipy.ndimage import gaussian_filter, binary_erosion
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = "/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/dicom"
GT = "/data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT"
R = "/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro"
def series(name):
    lst = []
    for f in sorted(glob.glob(os.path.join(ROOT, name, "*.dcm"))):
        d = pydicom.dcmread(f)
        a = d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0))
        lst.append((float(d.ImagePositionPatient[2]), a))
    lst.sort(key=lambda t: t[0])
    return np.stack([a for _, a in lst], 0)
swi = series("mtp_cartesian_.7x.7x1_SWI_801"); qsm = series("mtp_cartesian_.7x.7x1_QSM_814")
t2v = np.abs(np.load(os.path.join(GT, "gt_sense_FA4_TE27.41.npy"), mmap_mode="r"))
def vend(vol, i): return np.asarray(vol[:, 2:302, 91 - i])
t2 = vend(t2v, 22)
# darkest compact blob in the brain, in the right half of the image around the midbrain level
sm = gaussian_filter(t2, 1.0)
brain = gaussian_filter(t2, 4) > 0.35 * np.percentile(t2, 99)
brain = binary_erosion(brain, iterations=12)
win = np.zeros_like(brain); win[110:210, 160:240] = True
cand = np.where(brain & win, sm, np.inf)
r0, c0 = np.unravel_index(np.argmin(cand), cand.shape)
print("focus (row, col) in the vendor frame:", r0, c0)
def ringstat(img, r, c, rin=2, rout=7):
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]; d = np.hypot(yy - r, xx - c)
    core, ring = img[d <= rin], img[(d > rin + 1) & (d <= rout)]
    return float(core.mean()), float(ring.mean()), float(ring.std())
for lab, img in (("T2*w GT", t2), ("SWI", swi[22]), ("QSM", qsm[22])):
    cm, rm, rs = ringstat(img, r0, c0)
    print(f"{lab:8s} core mean {cm:10.3f}  ring mean {rm:10.3f}  ring sd {rs:8.3f}  (core - ring) / sd = {(cm - rm) / (rs + 1e-9):+.2f}")
print("QSM slice value range (P1, P50, P99):", [round(float(np.percentile(qsm[22], q)), 3) for q in (1, 50, 99)])
# continuity across neighbouring vendor slices: darkness of the T2*w core relative to its ring
for i in range(18, 27):
    img = vend(t2v, i); cm, rm, rs = ringstat(img, r0, c0)
    qm, qr, qs = ringstat(qsm[i], r0, c0)
    print(f"vendor slice {i:2d}: T2*w (core-ring)/sd {(cm - rm) / (rs + 1e-9):+6.2f}   QSM core {qm:+8.3f} ring {qr:+8.3f}")
mc = vend(np.load(os.path.join(R, "recon/gs_joint_ar_T2s_mag.npy"), mmap_mode="r"), 22)
t1 = vend(np.abs(np.load(os.path.join(GT, "gt_sense_FA16_TE3.25.npy"), mmap_mode="r")), 22)
h = 32; crop = (slice(max(r0 - h, 0), r0 + h), slice(max(c0 - h, 0), c0 + h))
panels = [("vendor SWI", swi[22]), ("vendor QSM (paramagnetic bright, calcium dark)", qsm[22]), ("GT T2*w", t2), ("MC-GS T2*w, AF 12", mc), ("GT T1w", t1)]
fig, axes = plt.subplots(2, 5, figsize=(20, 8.4))
for j, (lab, im) in enumerate(panels):
    for i, (view, sl) in enumerate((("whole slice 22", (slice(None), slice(None))), ("zoom", crop))):
        a = im[sl]; ax = axes[i, j]
        lo, hi = (np.percentile(im, 1), np.percentile(im, 99.5))
        ax.imshow(a, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        rr, cc = (r0, c0) if i == 0 else (r0 - crop[0].start, c0 - crop[1].start)
        ax.add_patch(plt.Circle((cc, rr), 9 if i == 0 else 6, fill=False, color="red", lw=1.5))
        ax.set_title(f"{lab}, {view}", fontsize=10); ax.axis("off")
fig.suptitle(f"Vendor slice 22 (SPE 69): the dark focus marked in red, row {r0} col {c0} of the 320 x 300 frame. Visual check only, not a radiological reading.", fontsize=11)
fig.tight_layout(); out = "/home/congcongliu/figs/peds_tsc_af12_mcgs_best/selection/lesion_focus_vendor22.png"; fig.savefig(out, dpi=110); print(out)
