"""T1 vs PD vs T2 tissue-contrast check, UIH 3D knee set, subjects S01, S05, S12 (read-only, CPU only).

Usage: python contrast_check.py /data2/congcong/data/3D/3D_data20261112/kspace_csm/knee
The body below is the code that was run read-only over SSH on 2026-11-25; its printed output is in
contrast_check_output.txt. Subjects are numbered S01..S16 in sorted path order; no names are printed.

Per subject: RSS images (centred inverse 3D FFT, root-sum-of-squares over coils) of T1, PD and T2; T1 and T2
moved onto PD by the integer shift that maximises the cross-correlation of gradient magnitudes; mask = PD above
10 % of its 99.5th percentile, eroded by 2 voxels; 3x3x3 box smoothing; each contrast divided by its median in
the mask. Reported: Pearson r and Spearman rho of the log intensities, the IQR of the log ratios, and a 5-class
k-means (k-means++ seeding, seed 0) on the three log intensities with each class's intensity per contrast.
"""
import sys, glob, os, numpy as np, h5py
from scipy import fft as F, ndimage as nd
K = sys.argv[1]
subs = sorted({os.path.dirname(p) for p in glob.glob(K + "/**/*.mat", recursive=True)})
def rss(si, tag):
    p = glob.glob(subs[si - 1] + "/*_" + tag + "_mx3d*.mat")[0]
    with h5py.File(p, "r") as f:
        k = f["kspace"][()]
    ax = (0, 2, 3)
    x = F.fftshift(F.ifftn(F.ifftshift(k, axes=ax), axes=ax, norm="ortho", workers=4), axes=ax)
    return np.sqrt((np.abs(x) ** 2).sum(axis=1)).astype(np.float32)
def shift_to(a, b):
    ga = np.sqrt(sum(g ** 2 for g in np.gradient(a))); gb = np.sqrt(sum(g ** 2 for g in np.gradient(b)))
    ga -= ga.mean(); gb -= gb.mean()
    xc = np.real(F.ifftn(F.fftn(ga) * np.conj(F.fftn(gb))))
    pk = np.unravel_index(np.argmax(xc), xc.shape)
    s = [int(p if p <= n // 2 else p - n) for p, n in zip(pk, xc.shape)]
    return np.roll(a, [-v for v in s], axis=(0, 1, 2)), s
def kmeans(X, k, rng):
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1), axis=1)
        C.append(X[rng.choice(len(X), p=d / d.sum())])
    C = np.array(C)
    for _ in range(100):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        Cn = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(Cn, C, atol=1e-6): break
        C = Cn
    return C, lab
rng = np.random.default_rng(0)
for si in (1, 5, 12):
    im = {t: rss(si, t) for t in ("t1", "pd", "t2")}
    if im["t1"].shape != im["pd"].shape or im["t2"].shape != im["pd"].shape:
        print("S%02d grid mismatch, skipped" % si); continue
    sh = {}
    for t in ("t1", "t2"):
        im[t], sh[t] = shift_to(im[t], im["pd"])
    mask = im["pd"] > 0.1 * np.percentile(im["pd"], 99.5)
    mask = nd.binary_erosion(mask, iterations=2)
    sm = {t: nd.uniform_filter(im[t], 3) for t in im}
    nv = {t: sm[t] / np.median(sm[t][mask]) for t in sm}
    L = {t: np.log(np.maximum(nv[t][mask], 1e-6)) for t in nv}
    def r(a, b): return float(np.corrcoef(L[a], L[b])[0, 1])
    sub = rng.choice(mask.sum(), 200000, replace=False)
    def rho(a, b):
        ra = np.argsort(np.argsort(L[a][sub])); rb = np.argsort(np.argsort(L[b][sub]))
        return float(np.corrcoef(ra, rb)[0, 1])
    def iqr(a, b):
        q = np.percentile(L[a] - L[b], [25, 75]); return float(q[1] - q[0])
    print("S%02d shifts (voxels) T1->PD %s, T2->PD %s; mask %d voxels" % (si, sh["t1"], sh["t2"], mask.sum()))
    print("  log-intensity correlation (3x3x3 smoothed): T1-PD r %.3f rho %.3f | T2-PD r %.3f rho %.3f | T1-T2 r %.3f rho %.3f" % (
        r("t1", "pd"), rho("t1", "pd"), r("t2", "pd"), rho("t2", "pd"), r("t1", "t2"), rho("t1", "t2")))
    print("  log-ratio spread (IQR): log(T1/PD) %.3f | log(T2/PD) %.3f" % (iqr("t1", "pd"), iqr("t2", "pd")))
    X = np.stack([L["t1"][sub], L["pd"][sub], L["t2"][sub]], axis=1)
    C, lab = kmeans(X, 5, rng)
    order = np.argsort(C[:, 1])
    print("  tissue clusters (k-means on log T1/PD/T2), intensities relative to each contrast median:")
    print("    share   T1     PD     T2   | T1/PD  T2/PD")
    for j in order:
        e = np.exp(C[j]); print("    %.3f  %.2f   %.2f   %.2f  |  %.2f   %.2f" % ((lab == j).mean(), e[0], e[1], e[2], e[0] / e[1], e[2] / e[1]))
