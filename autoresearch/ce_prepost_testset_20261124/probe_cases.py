#!/usr/bin/env python3
"""probe_cases.py -- read-only check of the plain-scan / enhanced test subjects in 3D_PE_20261119 (CPU).

User, 2026-11-24: use the plain-scan / enhanced subjects of /data2/congcong/data/3D/3D_PE_20261119 as test sets;
first a traditional reconstruction gives the complete image (truth), then 12-fold downsampling. The user chose to
start with the subjects whose multi-coil k-space is readable for both scans.

For every candidate array (BART cfl, dims [readout, ky, kz, coil]) this prints and writes to probe.json:
  shape, coils, the k-space peak (must be the grid centre for the centred FFT of the topic), whether the sampling is
  the same for every readout index (prospective 2-D mask), the mask's sampled points and AF (= ky x kz / points),
  the side of the largest fully sampled square around the centre (ACS), whether plain scan and enhanced share the
  mask, the plain / enhanced label from scanparam.xml (<IsInjection>1</IsInjection> = enhanced), and the record
  subject's arrays (recon_real work dirs) for comparison.
It also renders the zero-filled coil-combined (root-sum-of-squares) image of each scan, three central planes, to
probe_<subject>.png, to check orientation and data quality by eye. Nothing is written outside MCGS_ARTIFACT_DIR.

  gpu.py run --cpu -- @python autoresearch/ce_prepost_testset_20261124/probe_cases.py
"""
import json
import os
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = Path("/data2/congcong/data/3D/3D_PE_20261119")
REC = Path("/data0/congcong/code/NV-Generate-CTMR/recon_real")
ART = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
# subject -> scan -> (series folder, cfl stem); the 42- / 50-coil raw arrays where both exist
CASES = {
    "9000103445": {"pre": ("t1_3D_gre_fsp_sag_iso_ACS_16884476230000", ["rawdata_T1_turmor_9000103445", "Echo_1"]),
                   "post": ("t1_3D_gre_fsp_sag_iso_ACS_16884476230050", ["rawdata_CE_turmor_9000103445", "T1_CE"])},
    "9002944080": {"pre": ("t1_3D_gre_fsp_sag_iso_ACS_16886244540000", ["rawdata"]),
                   "post": ("t1_3D_gre_fsp_sag_iso_ACS_16886244540050", ["rawdata"])},
    "9003136585": {"pre": ("t1_3D_gre_fsp_sag_iso_ACS_16886147050000", ["T1"]),
                   "post": ("t1_3D_gre_fsp_sag_iso_ACS_16886147050050", ["T1_CE"])},
}
RECORD = {"pre": REC / "work_ce_t1_16887049370010", "post": REC / "work_ce_t1_16887049370060"}


def cfl(stem):
    lines = Path(str(stem) + ".hdr").read_text().split("\n")
    dims = [int(x) for x in lines[1].split()]
    while len(dims) > 4 and dims[-1] == 1:
        dims.pop()
    return np.memmap(str(stem) + ".cfl", dtype=np.complex64, mode="r", shape=tuple(dims), order="F")


def injection(series):
    x = next(series.glob("*scanparam.xml"), None)
    if x is None:
        return None
    m = re.search(r"<IsInjection>\s*(\d)\s*</IsInjection>", x.read_text(errors="ignore"))
    return int(m.group(1)) if m else 0


def acs_side(m2):
    ny, nz = m2.shape
    cy, cz = ny // 2, nz // 2
    s = 0
    for h in range(1, min(cy, cz)):
        if m2[cy - h:cy + h, cz - h:cz + h].all():
            s = 2 * h
        else:
            break
    return s


def probe(a, label):
    """a: (readout, ky, kz, coil) memmap or array"""
    nx, ny, nz, nc = a.shape
    c0 = np.asarray(a[:, :, :, 0])
    e = np.abs(c0)
    peak = [int(i) for i in np.unravel_index(int(np.argmax(e)), e.shape)]
    nzr = e > 0
    m2 = nzr.any(axis=0)
    uniform = bool(all((nzr[r] == m2).all() for r in range(0, nx, max(1, nx // 16)) if nzr[r].any()))
    pts = int(m2.sum())
    out = dict(label=label, shape=[nx, ny, nz, nc], peak_coil0=peak, grid_centre=[nx // 2, ny // 2, nz // 2],
               readout_uniform=uniform, sampled_points=pts, af=round(ny * nz / pts, 3), acs_side=acs_side(m2),
               max_abs=float(e.max()))
    print(f"  {label:44s} {out['shape']} peak {peak} centre {out['grid_centre']} uniform {uniform} "
          f"points {pts} AF {out['af']} ACS {out['acs_side']}")
    return out, m2


def rss_zf(a):
    nx, ny, nz, nc = a.shape
    acc = np.zeros((nx, ny, nz), np.float32)
    for c in range(nc):
        k = np.asarray(a[:, :, :, c])
        im = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k)))
        acc += (np.abs(im) ** 2).astype(np.float32)
    return np.sqrt(acc)


res = dict(cases={}, record={})
print("record subject 9003135093 (recon_real work dirs)")
for scan, w in RECORD.items():
    k = np.load(w / "ksp_full.npy", mmap_mode="r")                     # (coil, x, ky, kz)
    m = np.load(w / "mask2d.npy", mmap_mode="r")
    m2 = np.asarray(m[0]) > 0.5
    res["record"][scan] = dict(ksp_shape=list(k.shape), mask_points=int(m2.sum()),
                               af=round(m2.size / int(m2.sum()), 3), acs_side=acs_side(m2),
                               csm_shape=list(np.load(w / "csm.npy", mmap_mode="r").shape))
    print(f"  {scan}: {res['record'][scan]}")

for subj, scans in CASES.items():
    print(f"subject {subj}")
    r = res["cases"][subj] = {}
    masks, imgs = {}, {}
    for scan, (series, stems) in scans.items():
        sd = D / subj / series
        r[scan] = dict(series=series, injection=injection(sd), arrays={})
        first = None
        for st in stems:
            p = sd / st
            if not Path(str(p) + ".cfl").exists():
                r[scan]["arrays"][st] = "missing"
                continue
            a = cfl(p)
            info, m2 = probe(a, f"{scan}/{st}")
            r[scan]["arrays"][st] = info
            if first is None:
                first = st
                masks[scan] = m2
                imgs[scan] = rss_zf(a)
        r[scan]["use"] = first
        print(f"  {scan}: IsInjection {r[scan]['injection']}, use {first}")
    if "Echo_1" in r["pre"]["arrays"] and isinstance(r["pre"]["arrays"]["Echo_1"], dict):
        a1, a2 = cfl(D / subj / scans["pre"][0] / "rawdata_T1_turmor_9000103445"), cfl(D / subj / scans["pre"][0] / "Echo_1")
        same = bool(np.array_equal(np.asarray(a1[:, :, :, 0]), np.asarray(a2[:, :, :, 0])))
        r["pre"]["echo1_equals_rawdata_coil0"] = same
        print(f"  Echo_1 == rawdata_T1_turmor (coil 0): {same}")
    if len(masks) == 2:
        same = bool(masks["pre"].shape == masks["post"].shape and (masks["pre"] == masks["post"]).all())
        r["same_mask_pre_post"] = same
        print(f"  same prospective mask for plain scan and enhanced: {same}")
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for i, scan in enumerate(("pre", "post")):
        v = imgs[scan]
        v = v / np.percentile(v, 99.9)
        nx, ny, nz = v.shape
        for j, (sl, t) in enumerate(((v[:, :, nz // 2], "kz centre (sagittal)"), (v[nx // 2], "readout centre"),
                                     (v[:, ny // 2, :], "ky centre"))):
            ax[i, j].imshow(sl, cmap="gray", vmin=0, vmax=1)
            ax[i, j].set_title(f"{subj} {'plain' if scan == 'pre' else 'enhanced'}: {t}", fontsize=8)
            ax[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(ART / f"probe_{subj}.png", dpi=110)
    plt.close(fig)

(ART / "probe.json").write_text(json.dumps(res, indent=2))
print("PROBE OK")
