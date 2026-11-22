#!/usr/bin/env python3
"""prep_cases.py -- k-space and coil-sensitivity files of the six test scans (T1 + FLAIR acceleration study).

For each scan, OUT/data/<scan>/{ksp_T1.h5, ksp_FLAIR.h5, csm_espirit_T1.h5, csm_espirit_FLAIR.h5, prep.json}
with the layout of the jiangyizhou topics (h5 key `kspace` / `map`, shape (coil, x, ky, kz), complex64).
  - vol1 and vol4 already have prepared files: symlinks, after checking that the h5 k-space equals the source CFL
    on 3 sampled coils and that the maps have the k-space shape.
  - the others: export from the source CFL, `bart ecalib -t 0.001 -c 0.0 -r 24 -m 1` (house recipe, one run per
    contrast, both in parallel, 12 threads each), convert to h5.
Refuses to overwrite: a complete directory (prep.json present) is skipped, an incomplete one is an error.

    gpu.py run --cpu -- @python autoresearch/t1flair_af_testset_20261121/prep_cases.py [--scans vol2_jiangdian ...]
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ts_common import AFS, MASK, OUT, SCANS, SRC, claim, data_dir, sha256  # noqa: E402

BART = "/home/congcongliu/src/bart"
ECALIB = ["ecalib", "-t", "0.001", "-c", "0.0", "-r", "24", "-m", "1"]   # house recipe
THREADS = 12                                                              # per ecalib; two run at once -> 24
CONTRASTS = (("T1", "t1"), ("FLAIR", "flair"))
FILES = ("ksp_T1.h5", "ksp_FLAIR.h5", "csm_espirit_T1.h5", "csm_espirit_FLAIR.h5")
log = lambda *a: print(*a, flush=True)


def read_cfl(base):
    with open(str(base) + ".hdr") as f:
        dims = [int(x) for x in f.read().split("\n")[1].split()]
    n = int(np.prod(dims))
    return np.memmap(str(base) + ".cfl", dtype=np.complex64, mode="r", shape=(n,)).reshape(dims, order="F")


def stats_from_cfl(k):
    """support (ky, kz) from the whole volume, DC position as in screen_cases.py (readout x / ky,kz)"""
    nx, ny, nz, nc = k.shape
    sup = np.zeros((ny, nz), bool)
    e = np.zeros((ny, nz))
    ex = np.zeros(nx)
    for c in range(nc):
        a = np.abs(np.asarray(k[..., c]))
        sup |= a.max(axis=0) > 0
        e += (a ** 2).sum(axis=0)
        ex += (a[:, ny // 2 - 2:ny // 2 + 2, nz // 2 - 2:nz // 2 + 2] ** 2).sum(axis=(1, 2))
    return sup, [int(v) for v in np.unravel_index(np.argmax(e), e.shape)], int(np.argmax(ex))


def check_ksp(h5path, k, coils=(0, 13, 25)):
    with h5py.File(h5path, "r") as h:
        d = h["kspace"]
        assert d.shape == (k.shape[3], k.shape[0], k.shape[1], k.shape[2]), (h5path, d.shape, k.shape)
        for c in coils:
            assert np.array_equal(d[c], np.asarray(k[..., c])), f"{h5path}: coil {c} differs from the source CFL"


def sos_range(h5path):
    with h5py.File(h5path, "r") as h:
        d = h["map"]
        acc = np.zeros(d.shape[1:])
        for c in range(d.shape[0]):
            acc += np.abs(d[c]) ** 2
        s = np.sqrt(acc)
        return d.shape, float(s.min()), float(s.max())


def export_ksp(k, out):
    with h5py.File(out, "w") as h:
        d = h.create_dataset("kspace", shape=(k.shape[3], *k.shape[:3]), dtype=np.complex64, chunks=(1, 32, 256, 176))
        for c in range(k.shape[3]):
            d[c] = np.asarray(k[..., c])


def run_ecalib(src_base, out_base):
    env = dict(os.environ, TOOLBOX_PATH=BART, PATH=f"{BART}:{os.environ.get('PATH', '')}", OMP_NUM_THREADS=str(THREADS))
    r = subprocess.run([f"{BART}/bart", *ECALIB, str(src_base), str(out_base)], env=env, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-800:]


def csm_to_h5(base, out):
    s = np.squeeze(np.asarray(read_cfl(base)))                     # (x, ky, kz, coil)
    with h5py.File(out, "w") as h:
        d = h.create_dataset("map", shape=(s.shape[3], *s.shape[:3]), dtype=np.complex64, chunks=(1, 32, 256, 176))
        for c in range(s.shape[3]):
            d[c] = s[..., c]


def mask_rows():
    rows = {}
    for af in AFS:
        with h5py.File(MASK(af), "r") as h:
            m = h["mask"]
            m2 = np.asarray(m[0]).astype(bool)
            for x in (1, 128, 255):
                assert np.array_equal(np.asarray(m[x]).astype(bool), m2), f"mask AF{af} varies along x"
        rows[af] = (m2, str(MASK(af)), sha256(MASK(af)))
    return rows


def prep(scan, masks):
    spec = SCANS[scan]
    dd = data_dir(scan)
    if (dd / "prep.json").is_file():
        log(f"PREP_OK {scan} (already)")
        return
    if not claim(dd.parent, scan):
        raise SystemExit(f"ERROR {dd} exists but has no prep.json (incomplete); the controller must decide")
    rep = {"scan": scan, "person": spec["person"], "group": spec["group"], "reuse": str(spec["reuse"]) if spec["reuse"] else None,
           "ecalib": " ".join(ECALIB) if not spec["reuse"] else "reused from the existing topic", "contrasts": {}, "masks": {}, "sha256": {}}
    ks, sup = {}, {}
    for tag, key in CONTRASTS:
        base = SRC / scan / spec[key]
        k = read_cfl(base)
        k = k[..., 0] if k.ndim == 5 else k
        assert k.shape == (256, 256, 176, 26), (scan, tag, k.shape)
        s, dc_yz, dc_x = stats_from_cfl(k)
        ks[tag], sup[tag] = k, s
        rep["contrasts"][tag] = {"src": str(base), "dims": list(k.shape), "support_frac": float(s.mean()),
                                 "dc_yz": dc_yz, "dc_x": dc_x}
        log(f"[{scan} {tag}] support {s.mean():.4f} dc_yz {dc_yz} dc_x {dc_x}")
    if spec["reuse"]:
        for f in FILES:
            tgt = Path(spec["reuse"]) / f
            assert tgt.is_file(), tgt
            (dd / f).symlink_to(tgt)
        for tag, _ in CONTRASTS:
            check_ksp(dd / f"ksp_{tag}.h5", ks[tag])
            log(f"[{scan} {tag}] reused k-space equals the source CFL on 3 coils")
    else:
        for tag, _ in CONTRASTS:
            export_ksp(ks[tag], dd / f"ksp_{tag}.h5")
            check_ksp(dd / f"ksp_{tag}.h5", ks[tag])
        def job(tag_key):
            tag, key = tag_key
            return tag, run_ecalib(SRC / scan / spec[key], dd / f"csm_espirit_{tag}")
        with ThreadPoolExecutor(2) as ex:
            for tag, (rc, tail) in ex.map(job, CONTRASTS):
                log(f"[{scan} {tag}] ecalib rc {rc}\n{tail}")
                assert rc == 0, f"ecalib failed on {scan} {tag}"
        for tag, _ in CONTRASTS:
            b = dd / f"csm_espirit_{tag}"
            csm_to_h5(b, dd / f"csm_espirit_{tag}.h5")
            for ext in (".cfl", ".hdr"):                              # CFL copies were only for the conversion
                os.remove(str(b) + ext)
    for tag, _ in CONTRASTS:
        shape, lo, hi = sos_range(dd / f"csm_espirit_{tag}.h5")
        assert shape == (26, 256, 256, 176), (scan, tag, shape)      # same shape as the k-space
        rep["contrasts"][tag]["csm_shape"] = list(shape)
        rep["contrasts"][tag]["csm_sos_min"], rep["contrasts"][tag]["csm_sos_max"] = lo, hi
        log(f"[{scan} {tag}] csm SoS [{lo:.6f}, {hi:.6f}]")
        assert 0.99 <= lo and hi <= 1.01, f"SoS outside [0.99, 1.01]: {scan} {tag} {lo} {hi}"
    for af, (m2, path, msha) in masks.items():
        row = {"mask": path, "mask_sha256": msha, "mask_samples": int(m2.sum())}
        for tag, _ in CONTRASTS:
            n = int((m2 & sup[tag]).sum())
            row[tag] = {"effective_samples": n, "net_af": float(m2.size / n),
                        "af_vs_own_support": float(sup[tag].sum() / n)}
        rep["masks"][f"AF{af}"] = row
    for f in FILES:
        rep["sha256"][f] = sha256(dd / f)                             # symlinks are followed
    with open(dd / "prep.json", "w") as f:
        json.dump(rep, f, indent=2)
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if art:
        json.dump(rep, open(os.path.join(art, f"{scan}.json"), "w"), indent=2)
    log(f"PREP_OK {scan}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", nargs="*", default=list(SCANS))
    a = ap.parse_args()
    for s in a.scans:
        assert s in SCANS, s
    masks = mask_rows()
    for s in a.scans:
        prep(s, masks)


if __name__ == "__main__":
    main()
