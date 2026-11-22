#!/usr/bin/env python3
"""screen_cases.py -- which volumes of the MC_Jia fully sampled set can serve as cases of the T1 + FLAIR
acceleration study (the study of record ran on vol3_jiangyizhou only; user 2026-11-21: filter the test set,
count the usable cases, then run all of them for statistics).

CPU only, read only on the source data. For every volume and for T1 and FLAIR it measures what the topic scripts
prep_00 / prep_01 / prep_06 measured on the subjects already used, with the same definitions:
  header dims, acquired support of the PE plane, readout all-or-nothing test, DC position, a fingerprint of the
  file (duplicate detection), the root-sum-of-squares image of the FULL k-space, its peak / P99.9, the air noise
  floor, and for the pair: brain-mask Dice and the integer shift of peak mask cross-correlation.
Across volumes: peak normalised cross-correlation of the T1 images (same person scanned twice shows up here).
No reconstruction, no mask, no score: nothing here can select a result.

  gpu.py run --cpu -- @python autoresearch/t1flair_af_testset_20261121/screen_cases.py
"""
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.fft as tfft

B = Path("/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL")
PROSP = Path("/data1/data1_congcong/data/MC_Jia/CFL格式的前瞻性测试数据/spirit_diffusion_prospectivetest_cfl")
EXPECT = [256, 256, 176, 26]
SLICES = (180, 124, 76)                      # the record slices (transverse, coronal, sagittal)
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "screen_out")).resolve()
art.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)
torch.set_num_threads(16)


def hdr_dims(base):
    with open(str(base) + ".hdr") as f:
        return [int(x) for x in f.read().split("\n")[1].split()]


def cfl(base):
    dims = hdr_dims(base)
    n = int(np.prod(dims))
    return np.memmap(str(base) + ".cfl", dtype=np.complex64, mode="r", shape=(n,)).reshape(dims, order="F")[..., 0]


def fingerprint(base, n=64 << 20):
    """md5 of the first and the last 64 MiB: identical files (a copy under another name) share it"""
    h = hashlib.md5()
    p = str(base) + ".cfl"
    with open(p, "rb") as f:
        h.update(f.read(n))
        f.seek(max(0, os.path.getsize(p) - n))
        h.update(f.read(n))
    return h.hexdigest()


def pick(vol_dir, prefix):
    """the one k-space file of a contrast; vol7 has two T2 files, T1 and FLAIR have one each everywhere"""
    c = sorted(glob.glob(str(vol_dir / f"{prefix}_train_*_cc.hdr")))
    assert len(c) == 1, (vol_dir, prefix, c)
    return Path(c[0][:-4])


def _c(fn, x, d):
    return tfft.fftshift(fn(tfft.ifftshift(x, dim=d), dim=d, norm="ortho"), dim=d)


def probe(base):
    k = cfl(base)
    nx, ny, nz, nc = k.shape
    r = dict(file=base.name, dims=[nx, ny, nz, nc], fingerprint=fingerprint(base))
    # per PE line: on how many of the probed readout positions is it nonzero (all-or-nothing on a readout axis)
    xs = list(range(0, nx, 8))
    cnt = np.zeros((ny, nz), dtype=np.int32)
    e = np.zeros((ny, nz))
    for x in xs:
        a = np.abs(np.asarray(k[x]))
        cnt += (a.max(axis=-1) > 0)
        e += (a ** 2).sum(axis=-1)
    sup = cnt > 0
    r["support_frac"] = float(sup.mean())
    r["partial_line_frac"] = float(((cnt > 0) & (cnt < len(xs))).mean())
    r["dc_yz"] = [int(v) for v in np.unravel_index(np.argmax(e), e.shape)]
    ex = [(np.abs(np.asarray(k[x, ny // 2 - 2:ny // 2 + 2, nz // 2 - 2:nz // 2 + 2, :])) ** 2).sum() for x in range(nx)]
    r["dc_x"] = int(np.argmax(ex))
    r["dc_on_centre"] = r["dc_yz"] == [ny // 2, nz // 2] and r["dc_x"] == nx // 2
    # root-sum-of-squares image of the full k-space
    acc = torch.zeros((nx, ny, nz), dtype=torch.float32)
    for i in range(0, nc, 2):
        blk = torch.from_numpy(np.ascontiguousarray(np.moveaxis(np.asarray(k[..., i:i + 2]), 3, 0)))
        acc += _c(tfft.ifftn, blk, (-3, -2, -1)).abs().square().sum(0)
    v = acc.sqrt().numpy()
    p999 = float(np.quantile(v.ravel()[::4], 0.999))
    r.update(max=float(v.max()), p999=p999, max_over_p999=float(v.max() / p999))
    c = 12                                                    # the eight 12-voxel corner cubes = air
    corners = np.concatenate([v[sx, sy, sz].ravel() for sx in (slice(0, c), slice(-c, None))
                              for sy in (slice(0, c), slice(-c, None)) for sz in (slice(0, c), slice(-c, None))])
    r["air_median_over_p999"] = float(np.median(corners) / p999)
    head = v > 0.05 * p999
    r["head_frac"] = float(head.mean())
    r["outside_head_mean_over_air"] = float(v[~head].mean() / np.median(corners))
    return r, v


def align(a, b):
    """brain-mask Dice and the integer shift of peak mask cross-correlation (prep_06 definitions)"""
    ma = torch.from_numpy(a > 0.10 * np.quantile(a.ravel()[::37], 0.999)).float()
    mb = torch.from_numpy(b > 0.10 * np.quantile(b.ravel()[::37], 0.999)).float()
    dice = float(2 * (ma * mb).sum() / (ma.sum() + mb.sum()))
    cc = tfft.ifftn(tfft.fftn(ma) * tfft.fftn(mb).conj()).real
    sh = np.unravel_index(int(torch.argmax(cc)), tuple(ma.shape))
    sh = [int(s) - int(n) if int(s) > int(n) // 2 else int(s) for s, n in zip(sh, ma.shape)]
    return dict(mask_dice=dice, peak_xcorr_shift_vox=sh, brain_frac_T1=float(ma.mean()), brain_frac_FLAIR=float(mb.mean()))


def peak_ncc(a, b):
    """peak over all integer circular shifts of the normalised cross-correlation of two images"""
    za = torch.from_numpy((a - a.mean()) / a.std())
    zb = torch.from_numpy((b - b.mean()) / b.std())
    cc = tfft.ifftn(tfft.fftn(za) * tfft.fftn(zb).conj()).real / za.numel()
    return float(cc.max())


def view(v, axis, i):
    x = np.take(v, i, axis=axis)
    return np.rot90(x.T, 1) if axis == 0 else x              # the record figure's orientation rule


vols = sorted(p for p in B.iterdir() if p.is_dir())
out = dict(source=str(B), expected_dims=EXPECT, volumes={}, pairs_t1_peak_ncc={}, prospective={})
imgs = {}
for vd in vols:
    rec = {}
    for tag, prefix in (("T1", "t1_fsp"), ("FLAIR", "flair")):
        base = pick(vd, prefix)
        dims = hdr_dims(base)[:4]
        if dims != EXPECT:
            rec[tag] = dict(file=base.name, dims=dims, usable=False)
            continue
        rec[tag], imgs[(vd.name, tag)] = probe(base)
        log(f"[{vd.name} {tag}] " + json.dumps({k: rec[tag][k] for k in rec[tag] if k != "fingerprint"}))
    if all((vd.name, t) in imgs for t in ("T1", "FLAIR")):
        rec["alignment"] = align(imgs[(vd.name, "T1")], imgs[(vd.name, "FLAIR")])
        log(f"[{vd.name}] alignment " + json.dumps(rec["alignment"]))
    out["volumes"][vd.name] = rec

names = [v.name for v in vols if (v.name, "T1") in imgs]
for i, a in enumerate(names):
    for b in names[i + 1:]:
        out["pairs_t1_peak_ncc"][f"{a}|{b}"] = peak_ncc(imgs[(a, "T1")], imgs[(b, "T1")])
log("T1 peak NCC between volumes: " + json.dumps(out["pairs_t1_peak_ncc"], indent=1))

fp = {}
for n, rec in out["volumes"].items():
    for t in ("T1", "FLAIR"):
        if "fingerprint" in rec[t]:
            fp.setdefault(rec[t]["fingerprint"], []).append(f"{n}:{t}")
out["duplicate_files"] = [v for v in fp.values() if len(v) > 1]
log("duplicate files: " + json.dumps(out["duplicate_files"]))

# the prospective set: listed only (undersampled at acquisition, so no ground truth for a retrospective study)
if PROSP.is_dir():
    for h in sorted(PROSP.glob("*.hdr")):
        k = cfl(Path(str(h)[:-4]))
        cnt = sum((np.abs(np.asarray(k[x])).max(axis=-1) > 0) for x in range(0, k.shape[0], 32))
        out["prospective"][h.stem] = dict(dims=list(k.shape), support_frac=float((cnt > 0).mean()))
    log("prospective: " + json.dumps(out["prospective"]))

(art / "screen_cases.json").write_text(json.dumps(out, indent=1))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(len(names), 6, figsize=(15, 2.55 * len(names)), squeeze=False)
for r, n in enumerate(names):
    for j, tag in enumerate(("T1", "FLAIR")):
        v = imgs[(n, tag)]
        vmax = np.quantile(v.ravel()[::4], 0.999)
        for a, (axis, s) in enumerate(zip((0, 1, 2), SLICES)):
            ax = axs[r, 3 * j + a]
            ax.imshow(view(v, axis, s), cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if a == 0:
                ax.set_ylabel(n if j == 0 else "", fontsize=9)
            if r == 0:
                ax.set_title(f"{tag} {('transverse', 'coronal', 'sagittal')[a]} {s}", fontsize=9)
fig.suptitle("Fully sampled root-sum-of-squares images of the seven volumes (slices 180 / 124 / 76)", fontsize=11)
fig.tight_layout()
fig.savefig(art / "screen_cases.png", dpi=110)
log("SCREEN_DONE")
