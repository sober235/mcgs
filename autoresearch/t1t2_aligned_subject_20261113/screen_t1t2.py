"""screen_t1t2.py -- read-only data QC: which MC_Jia subject has a T1 / T2 pair without a voxel offset?

For every subject: coil-RSS of the fully sampled T1, T2 and FLAIR k-space (same centred 3D IFFT as the
T1T2_AF12_retro preparation), then the rigid sub-voxel offset of T2 (and FLAIR, as the known-aligned control)
relative to T1 on each axis, by two contrast-insensitive metrics (normalised mutual information, NGF).
Also: is the "t2" file a distinct acquisition from "flair" (byte check + image correlation), T2 support fraction.
Prints JSON lines (ROW ...) and, under gpu.py, writes them to $MCGS_ARTIFACT_DIR/screen_t1t2.json; reads the
source data only.

    gpu.py run --cpu -- @python autoresearch/t1t2_aligned_subject_20261113/screen_t1t2.py [vol...]
"""
import glob
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import scipy.fft as sfft

D = "/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL"
SUBJ = ["vol1_huangyuwei", "vol2_jiangdian", "vol3_jiangyizhou", "vol4_tangbing", "vol5_sunxinyi",
        "vol6_jiangdian_790", "vol7_tangbin_790"]


def cfl(base):
    dims = [int(x) for x in open(base + ".hdr").read().split("\n")[1].split()]
    n = int(np.prod(dims))
    return np.memmap(base + ".cfl", dtype=np.complex64, mode="r", shape=(n,)).reshape(dims, order="F")[..., 0]


def rss(base):
    k = cfl(base)
    acc = np.zeros(k.shape[:3], dtype=np.float64)
    for c in range(k.shape[3]):
        x = np.asarray(k[..., c])
        img = sfft.fftshift(sfft.ifftn(sfft.ifftshift(x), norm="ortho", workers=4))
        acc += np.abs(img) ** 2
    return np.sqrt(acc)


def head_md5(path, nbytes=100 * 2 ** 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def fshift(v, s, axis):
    n = v.shape[axis]
    ph = np.exp(-2j * np.pi * np.fft.fftfreq(n) * s)
    shp = [1] * v.ndim
    shp[axis] = n
    return np.real(sfft.ifft(sfft.fft(v, axis=axis, workers=4) * ph.reshape(shp), axis=axis, workers=4))


def ent(q):
    q = q[q > 0]
    return -np.sum(q * np.log(q))


def search(a, b, axis, grid, bins=64):
    m = (a > 0.1 * np.quantile(a, 0.999)) | (b > 0.1 * np.quantile(b, 0.999))
    qa, qb = np.quantile(a[m], 0.999), np.quantile(b[m], 0.999)
    x = np.clip(a[m] / qa, 0, 1)
    ga = [g[m] for g in np.gradient(a)]
    na = sum(g ** 2 for g in ga)
    na = na + (1e-2 * np.median(np.sqrt(na))) ** 2
    res = []
    for s in grid:
        bs = fshift(b, s, axis)
        y = np.clip(bs[m] / qb, 0, 1)
        h, _, _ = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])
        p = h / h.sum()
        gb = [g[m] for g in np.gradient(bs)]
        nb = sum(g ** 2 for g in gb)
        nb = nb + (1e-2 * np.median(np.sqrt(nb))) ** 2
        dot = sum(u * w for u, w in zip(ga, gb))
        res.append((float(s), (ent(p.sum(1)) + ent(p.sum(0))) / ent(p), float(np.mean(dot ** 2 / (na * nb)))))
    r = np.array(res)
    return {name: float(r[np.argmax(r[:, j]), 0]) for j, name in ((1, "nmi"), (2, "ngf"))}


def offset(a, b, axis, lo, hi):
    coarse = search(a, b, axis, np.arange(lo, hi + 1e-9, 0.25))
    out = {}
    for name in ("nmi", "ngf"):
        c = coarse[name]
        out[name] = round(search(a, b, axis, np.arange(c - 0.25, c + 0.25 + 1e-9, 0.05))[name], 2)
    return out


def one(s):
    files = {}
    for t in ("t1_fsp", "t2", "flair"):
        cands = sorted(glob.glob(f"{D}/{s}/{t}_train_*_cc*.cfl"))
        good = [c for c in cands if c.endswith("_good.cfl")]   # vol7 has t2 ..._cc (256x232) and ..._cc_good (256x256)
        files[t] = (good or cands)[0][:-4]
    v = {t: rss(p) for t, p in files.items()}
    t1, t2, fl = v["t1_fsp"], v["t2"], v["flair"]
    m = (t2 > 0.1 * np.quantile(t2, 0.999)) | (fl > 0.1 * np.quantile(fl, 0.999))
    r = {"subject": s,
         "files": {t: os.path.basename(p) for t, p in files.items()},
         "t2_equals_flair_bytes": head_md5(files["t2"] + ".cfl") == head_md5(files["flair"] + ".cfl"),
         "corr_t2_flair": float(np.corrcoef(t2[m], fl[m])[0, 1]),
         "max_over_p999": {t: float(x.max() / np.quantile(x, 0.999)) for t, x in v.items()}}
    for tag, b in (("T2", t2), ("FLAIR", fl)):
        r[f"T1-{tag}"] = {"ax0": offset(t1, b, 0, -2, 2), "ax1": offset(t1, b, 1, -1, 1), "ax2": offset(t1, b, 2, -1, 1)}
    print("ROW " + json.dumps(r), flush=True)
    return r


if __name__ == "__main__":
    subs = sys.argv[1:] or SUBJ
    with Pool(min(len(subs), 7)) as pool:
        rows = pool.map(one, subs)
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if art:
        os.makedirs(art, exist_ok=True)
        json.dump(rows, open(os.path.join(art, "screen_t1t2.json"), "w"), indent=1)
    print("SCREEN DONE", len(rows), "subjects", flush=True)
