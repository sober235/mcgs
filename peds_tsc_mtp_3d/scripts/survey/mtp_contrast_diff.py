"""Quantify contrast differences between the MTP Cartesian acquired contrasts (5 TE x 2 FA) from the
vendor magnitude DICOMs (M_802), plus the derived series. Read-only on the data; writes npz + figures
to the scratchpad and ~/figs/peds_tsc_age16."""
import os, glob, sys, json, itertools, collections
import numpy as np, pydicom
ROOT = "/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/dicom"
SCR = "/tmp/claude-1002/-home-congcongliu--claude/fd861f29-d86a-49cf-be4f-a26d183fe383/scratchpad"
FIG = "/home/congcongliu/figs/peds_tsc_age16"

def load_series(name):
    fs = sorted(glob.glob(os.path.join(ROOT, name, "*.dcm")))
    groups = collections.defaultdict(list)
    for f in fs:
        d = pydicom.dcmread(f)
        key = (float(getattr(d, "FlipAngle", -1) or -1), float(getattr(d, "EchoTime", -1) or -1))
        z = float(d.ImagePositionPatient[2]) if hasattr(d, "ImagePositionPatient") else len(groups[key])
        groups[key].append((z, d.pixel_array.astype(np.float32)))
    vols = {}
    for k, lst in groups.items():
        lst.sort(key=lambda t: t[0])
        vols[k] = np.stack([a for _, a in lst], 0)
    return vols

acq = load_series("mtp_cartesian_.7x.7x1_M_802")
keys = sorted(acq.keys())                      # (FA, TE)
print("acquired contrasts:", keys, "vol shape", acq[keys[0]].shape)
names = [f"FA{int(fa)}_TE{te:.2f}" for fa, te in keys]
V = np.stack([acq[k] for k in keys], 0)        # (10, Z, Y, X)

# head mask from the mean magnitude: threshold at 0.15 of the 99.9 percentile, keep largest region, erode
mean = V.mean(0)
thr = 0.12 * np.percentile(mean, 99.9)
mask = mean > thr
# crude erosion (3x3x3 min filter) to avoid edges
from numpy.lib.stride_tricks import sliding_window_view as swv
def erode(m):
    p = np.pad(m, 1, mode="constant")
    return swv(p, (3, 3, 3)).all(axis=(-1, -2, -3))
mask = erode(erode(mask))
print("mask voxels", int(mask.sum()), "of", mask.size)
X = np.stack([v[mask] for v in V], 0)          # (10, N)
X = X / np.percentile(X, 99.9, axis=1, keepdims=True)

def pearson(a, b): return float(np.corrcoef(a, b)[0, 1])
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return pearson(ra, rb)
def nmi(a, b, bins=64):
    ha, _ = np.histogram(a, bins=bins, range=(0, 1.2)); hb, _ = np.histogram(b, bins=bins, range=(0, 1.2))
    hab, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 1.2], [0, 1.2]])
    pa = ha / ha.sum(); pb = hb / hb.sum(); pab = hab / hab.sum()
    ent = lambda p: -np.sum(p[p > 0] * np.log(p[p > 0]))
    return float((ent(pa) + ent(pb)) / ent(pab))          # 1 = independent, 2 = identical

# tissue clustering: k-means (k=3) on the 10-D vectors (subsample for speed), then tissue mean vectors
rng = np.random.default_rng(0)
sub = rng.choice(X.shape[1], size=min(200000, X.shape[1]), replace=False)
Xs = X[:, sub].T                                 # (n, 10)
k = 3
cent = Xs[rng.choice(len(Xs), k, replace=False)]
for it in range(30):
    lab = np.argmin(((Xs[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
    cent = np.stack([Xs[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(k)])
# order clusters by FA16_TE1 (T1w-like) brightness so labels are stable
t1_idx = names.index([n for n in names if n.startswith("FA16_TE3")][0])
order = np.argsort(-cent[:, t1_idx]); cent = cent[order]
frac = [float((lab == order[j]).mean()) for j in range(k)]
print("cluster fractions (ordered by FA16 short-TE brightness, i.e. ~WM, ~GM, ~CSF):", np.round(frac, 3))
tissue = cent.T                                  # (10 contrasts, 3 tissues)

def contrast_vec(t):                             # normalised tissue-mean vector (mean removed) -> direction of contrast
    v = t - t.mean(); return v / (np.linalg.norm(v) + 1e-12)
rows = []
for i, j in itertools.combinations(range(len(keys)), 2):
    r = pearson(X[i], X[j]); s = spearman(X[i][sub], X[j][sub]); m = nmi(X[i], X[j])
    ang = float(np.degrees(np.arccos(np.clip(np.dot(contrast_vec(tissue[i]), contrast_vec(tissue[j])), -1, 1))))
    rows.append(dict(a=names[i], b=names[j], pearson=r, spearman=s, nmi=m, tissue_angle_deg=ang))
rows.sort(key=lambda d: d["pearson"])
print("\nlowest Pearson correlation (most different contrast), top 12:")
print(f"{'pair':34s} {'pearson':>8s} {'spearman':>9s} {'NMI':>6s} {'tissue angle':>13s}")
for d in rows[:12]:
    print(f"{d['a']+' vs '+d['b']:34s} {d['pearson']:8.3f} {d['spearman']:9.3f} {d['nmi']:6.3f} {d['tissue_angle_deg']:13.1f}")
print("\nlargest tissue-contrast angle, top 8:")
for d in sorted(rows, key=lambda d: -d["tissue_angle_deg"])[:8]:
    print(f"{d['a']+' vs '+d['b']:34s} {d['pearson']:8.3f} {d['spearman']:9.3f} {d['nmi']:6.3f} {d['tissue_angle_deg']:13.1f}")
print("\ntissue means (rows = contrast; cols = ~WM, ~GM, ~CSF; normalised to each contrast's P99.9):")
for n, t in zip(names, tissue): print(f"{n:14s} {t[0]:.3f} {t[1]:.3f} {t[2]:.3f}")
R = np.corrcoef(X)
np.savez(os.path.join(SCR, "mtp_contrast_metrics.npz"), names=np.array(names), R=R, tissue=tissue, rows=json.dumps(rows), mask=mask)
json.dump(dict(names=names, pearson_matrix=R.tolist(), tissue_means=tissue.tolist(), pairs=rows), open(os.path.join(SCR, "mtp_contrast_metrics.json"), "w"), indent=1)

# derived series, same mask (each is a different quantity; we only report correlation with the acquired ones)
derived = {}
for s in ["PDW_811", "T1W_812", "aT1W_807", "SWI_801", "T2Star_810", "R2Star_809", "T1Map_805", "PDMap_808", "QSM_814"]:
    try:
        v = list(load_series("mtp_cartesian_.7x.7x1_" + s).values())[0]; derived[s] = v
    except Exception as e:
        print("skip", s, e)
print("\nderived series vs acquired (Pearson within head mask):")
hdr = "series".ljust(12) + "".join(n.rjust(14) for n in names); print(hdr)
for s, v in derived.items():
    x = v[mask].astype(np.float64); x = (x - np.median(x)) / (np.percentile(x, 99) - np.percentile(x, 1) + 1e-9)
    print(s.ljust(12) + "".join(f"{pearson(x, X[i]):14.3f}" for i in range(len(names))))

# figures
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    z = V.shape[1] // 2
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.2))
    for ax, n, v in zip(axes.ravel(), names, V):
        sl = v[z]; ax.imshow(sl, cmap="gray", vmin=0, vmax=np.percentile(sl[mask[z]] if mask[z].any() else sl, 99.5)); ax.set_title(n, fontsize=11); ax.axis("off")
    fig.suptitle("MTP Cartesian 0.7x0.7x1.35 mm, acquired contrasts (vendor magnitude), slice %d of %d" % (z, V.shape[1])); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mtp_acquired_contrasts_slice%d.png" % z), dpi=110); plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.2))
    for ax, (s, v) in zip(axes.ravel(), derived.items()):
        sl = v[z].astype(np.float32); lo, hi = np.percentile(sl[mask[z]], [1, 99]) if mask[z].any() else (sl.min(), sl.max())
        ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi); ax.set_title(s, fontsize=11); ax.axis("off")
    for ax in axes.ravel()[len(derived):]: ax.axis("off")
    fig.suptitle("MTP Cartesian derived series (vendor), slice %d" % z); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mtp_derived_series_slice%d.png" % z), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 6.5)); im = ax.imshow(R, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)): ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson r inside head mask"); ax.set_title("MTP acquired contrasts: intensity correlation"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mtp_contrast_correlation.png"), dpi=120); plt.close(fig)
    print("\nfigures written to", FIG)
except Exception as e:
    print("plotting skipped:", repr(e))
