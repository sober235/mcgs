"""Same contrast-difference metrics as mtp_contrast_diff.py but inside a brain-only mask (scalp/skull peeled)."""
import os, glob, json, itertools, collections
import numpy as np, pydicom
ROOT = "/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/dicom"
SCR = "/tmp/claude-1002/-home-congcongliu--claude/fd861f29-d86a-49cf-be4f-a26d183fe383/scratchpad"
FIG = "/home/congcongliu/figs/peds_tsc_age16"
def load_series(name):
    groups = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ROOT, name, "*.dcm"))):
        d = pydicom.dcmread(f); key = (float(d.FlipAngle), float(d.EchoTime))
        groups[key].append((float(d.ImagePositionPatient[2]), d.pixel_array.astype(np.float32)))
    return {k: np.stack([a for _, a in sorted(lst, key=lambda t: t[0])], 0) for k, lst in groups.items()}
acq = load_series("mtp_cartesian_.7x.7x1_M_802"); keys = sorted(acq); names = [f"FA{int(fa)}_TE{te:.2f}" for fa, te in keys]
V = np.stack([acq[k] for k in keys], 0)                                   # (10, Z, Y, X)
mean = V.mean(0); head = mean > 0.12 * np.percentile(mean, 99.9)
# fill holes per slice and peel 2D by a disk of radius R px (0.7 mm/px) -> brain-ish interior
from numpy.lib.stride_tricks import sliding_window_view as swv
def erode2d(m, r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; disk = (yy ** 2 + xx ** 2) <= r * r
    out = np.zeros_like(m)
    for z in range(m.shape[0]):
        p = np.pad(m[z], r, mode="constant"); w = swv(p, (2 * r + 1, 2 * r + 1))
        out[z] = (w | ~disk).all(axis=(-1, -2))
    return out
def fill2d(m):   # fill enclosed holes per slice by flood fill from the border
    out = m.copy()
    for z in range(m.shape[0]):
        bg = ~m[z]; reach = np.zeros_like(bg); reach[0, :] = bg[0, :]; reach[-1, :] = bg[-1, :]; reach[:, 0] = bg[:, 0]; reach[:, -1] = bg[:, -1]
        for _ in range(400):
            p = np.pad(reach, 1); grown = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | reach) & bg
            if np.array_equal(grown, reach): break
            reach = grown
        out[z] = m[z] | (bg & ~reach)
    return out
brain = erode2d(fill2d(head), 18)
z0, z1 = 12, V.shape[1] - 12; brain[:z0] = False; brain[z1:] = False
print("brain-ish mask voxels", int(brain.sum()), "(head", int(head.sum()), ")")
X = np.stack([v[brain] for v in V], 0); X = X / np.percentile(X, 99.9, axis=1, keepdims=True)
pearson = lambda a, b: float(np.corrcoef(a, b)[0, 1])
def nmi(a, b, bins=64):
    ha, _ = np.histogram(a, bins, (0, 1.2)); hb, _ = np.histogram(b, bins, (0, 1.2)); hab, _, _ = np.histogram2d(a, b, bins, [[0, 1.2], [0, 1.2]])
    ent = lambda p: -np.sum(p[p > 0] * np.log(p[p > 0])); pa, pb, pab = ha / ha.sum(), hb / hb.sum(), hab / hab.sum()
    return float((ent(pa) + ent(pb)) / ent(pab))
rng = np.random.default_rng(0); sub = rng.choice(X.shape[1], min(200000, X.shape[1]), replace=False); Xs = X[:, sub].T
k = 3; cent = Xs[rng.choice(len(Xs), k, replace=False)]
for _ in range(40):
    lab = np.argmin(((Xs[:, None, :] - cent[None]) ** 2).sum(-1), 1); cent = np.stack([Xs[lab == j].mean(0) for j in range(k)])
# label clusters: CSF = brightest in FA4 short TE relative to FA16 short TE (PD-like/T1-like ratio), WM = brightest in FA16 short TE
i_pd, i_t1 = names.index("FA4_TE3.25"), names.index("FA16_TE3.25")
ratio = cent[:, i_pd] / (cent[:, i_t1] + 1e-9); csf = int(np.argmax(ratio)); rest = [j for j in range(k) if j != csf]
wm = rest[int(np.argmax(cent[rest, i_t1]))]; gm = [j for j in rest if j != wm][0]; order = [wm, gm, csf]
tissue = cent[order].T; frac = [float((lab == j).mean()) for j in order]
print("cluster fractions WM/GM/CSF-like:", np.round(frac, 3))
print("tissue means (rows=contrast, cols=WM GM CSF; each contrast normalised to its P99.9):")
for n, t in zip(names, tissue): print(f"  {n:14s} {t[0]:.3f} {t[1]:.3f} {t[2]:.3f}   WM/GM {t[0]/t[1]:.2f}  CSF/WM {t[2]/t[0]:.2f}")
cv = lambda t: (t - t.mean()) / (np.linalg.norm(t - t.mean()) + 1e-12)
rows = []
for i, j in itertools.combinations(range(len(keys)), 2):
    rows.append(dict(a=names[i], b=names[j], pearson=pearson(X[i], X[j]), nmi=nmi(X[i], X[j]),
                     tissue_angle_deg=float(np.degrees(np.arccos(np.clip(np.dot(cv(tissue[i]), cv(tissue[j])), -1, 1))))))
print("\nmost different by Pearson (brain mask), top 10:")
for d in sorted(rows, key=lambda d: d["pearson"])[:10]: print(f"  {d['a']+' vs '+d['b']:32s} r={d['pearson']:7.3f}  NMI={d['nmi']:.3f}  tissue angle={d['tissue_angle_deg']:6.1f}")
print("most similar by Pearson, top 5:")
for d in sorted(rows, key=lambda d: -d["pearson"])[:5]: print(f"  {d['a']+' vs '+d['b']:32s} r={d['pearson']:7.3f}  NMI={d['nmi']:.3f}  tissue angle={d['tissue_angle_deg']:6.1f}")
print("largest tissue-contrast angle, top 6:")
for d in sorted(rows, key=lambda d: -d["tissue_angle_deg"])[:6]: print(f"  {d['a']+' vs '+d['b']:32s} r={d['pearson']:7.3f}  NMI={d['nmi']:.3f}  tissue angle={d['tissue_angle_deg']:6.1f}")
R = np.corrcoef(X); json.dump(dict(names=names, pearson=R.tolist(), tissue=tissue.tolist(), frac=frac, pairs=rows), open(os.path.join(SCR, "mtp_contrast_metrics_brain.json"), "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
z = V.shape[1] // 2
best = sorted(rows, key=lambda d: d["pearson"])[0]; ia, ib = names.index(best["a"]), names.index(best["b"])
fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
for ax, i in zip(axes[:2], (ia, ib)):
    sl = V[i][z]; ax.imshow(sl, cmap="gray", vmin=0, vmax=np.percentile(sl[brain[z]], 99.5)); ax.contour(brain[z], levels=[0.5], colors="y", linewidths=0.6); ax.set_title(names[i]); ax.axis("off")
axes[2].hist2d(X[ia][sub], X[ib][sub], bins=80, range=[[0, 1.1], [0, 1.1]], cmap="magma"); axes[2].set_xlabel(names[ia]); axes[2].set_ylabel(names[ib]); axes[2].set_title(f"joint histogram, r={best['pearson']:.2f}, NMI={best['nmi']:.3f}")
fig.suptitle("Most different acquired MTP pair inside the brain mask (yellow), slice %d" % z); fig.tight_layout(); fig.savefig(os.path.join(FIG, "mtp_most_different_pair.png"), dpi=110)
fig, ax = plt.subplots(figsize=(7.5, 6.5)); im = ax.imshow(R, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(10)); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8); ax.set_yticks(range(10)); ax.set_yticklabels(names, fontsize=8)
for i in range(10):
    for j in range(10): ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="Pearson r inside brain mask"); ax.set_title("MTP acquired contrasts: correlation (brain only)"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "mtp_contrast_correlation_brain.png"), dpi=120)
print("figures written")
