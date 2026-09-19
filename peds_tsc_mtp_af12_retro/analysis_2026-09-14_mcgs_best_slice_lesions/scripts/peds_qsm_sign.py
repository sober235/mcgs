import glob, os
import numpy as np, pydicom
from scipy.ndimage import label
ROOT = "/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/dicom"
def series(name):
    lst = []
    for f in sorted(glob.glob(os.path.join(ROOT, name, "*.dcm"))):
        d = pydicom.dcmread(f)
        lst.append((float(d.ImagePositionPatient[2]), d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0))))
    lst.sort(key=lambda t: t[0]); return np.stack([a for _, a in lst], 0)
swi = series("mtp_cartesian_.7x.7x1_SWI_801"); qsm = series("mtp_cartesian_.7x.7x1_QSM_814")
r0, c0 = 166, 194
yy, xx = np.mgrid[:320, :300]; d = np.hypot(yy - r0, xx - c0)
s, q = swi[22], qsm[22]
# 1) dark vessel-like pixels near the focus but outside it
ring = (d > 5) & (d < 30) & (q != 0)
thr = np.percentile(s[ring], 5)
dark = ring & (s <= thr)
print(f"near the focus, SWI-darkest 5 % outside the focus disk: {dark.sum()} px, QSM mean {q[dark].mean():+.1f}, median {np.median(q[dark]):+.1f}; all ring px QSM mean {q[ring].mean():+.1f}")
# 2) slice-wide: elongated SWI-dark components (veins) versus the focus
brain = q != 0
thr_all = np.percentile(s[brain], 3)
lab, n = label(brain & (s <= thr_all))
el_q, el_n = [], 0
for k in range(1, n + 1):
    ys, xs = np.nonzero(lab == k)
    if ys.size < 8 or np.hypot(ys.mean() - r0, xs.mean() - c0) < 8:
        continue
    cov = np.cov(np.vstack([ys, xs])); ev = np.sort(np.linalg.eigvalsh(cov))
    if ev[0] > 0 and ev[1] / ev[0] > 6:          # clearly elongated -> vessel-like
        el_q.append(q[lab == k].mean()); el_n += 1
print(f"slice 22: {el_n} elongated SWI-dark components (vessel-like), their QSM means: median {np.median(el_q):+.1f}, fraction positive {np.mean(np.array(el_q) > 0):.2f}")
core = d <= 2
print(f"the focus core: QSM mean {q[core].mean():+.1f}")
