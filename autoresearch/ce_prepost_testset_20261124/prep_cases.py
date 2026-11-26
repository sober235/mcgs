#!/usr/bin/env python3
"""prep_cases.py -- k-space, prospective mask and coil maps of the plain-scan / enhanced test subjects, in the layout
of the record subject (recon_real/work_ce_t1_*), CPU only.

For each subject and scan, OUT/subjects/<subject>/work_<pre|post>/
    ksp_full.npy   (coil, x, ky, kz) complex64, exact zeros off the acquired points (byte content of the source cfl)
    mask2d.npy     (x, ky, kz) float32, the prospective mask repeated along the readout (as the record's file)
    csm.npy        (coil, x, ky, kz) complex64, ESPIRiT: bart ecalib -t 0.001 -c 0.0 -r 24 -m 1 (house recipe)
    prep.json      source, shapes, mask points, AF, ACS side, k-space peak, map SoS range, sha256
A complete directory (prep.json present) is skipped; an incomplete one is an error; nothing is overwritten.

--record_check: the same ecalib on the record subject's plain-scan k-space, compared with the record's csm.npy
(written to OUT/record_check/; the record files are only read). Reported: per-voxel agreement
|sum_c conj(a_c) b_c| / (|a| |b|) inside the head (1 = the same maps up to a per-voxel phase), and the readout
position of the record's k-space peak. This says whether house-recipe maps stand in for the record's maps.

--echo_check: how 9000103445's two plain-scan arrays (rawdata_T1_turmor, Echo_1) differ.

  gpu.py run --cpu -- @python autoresearch/ce_prepost_testset_20261124/prep_cases.py --record_check --echo_check
  gpu.py run --cpu -- @python autoresearch/ce_prepost_testset_20261124/prep_cases.py --subjects 9002944080
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

D = Path("/data2/congcong/data/3D/3D_PE_20261119")
REC = Path("/data0/congcong/code/NV-Generate-CTMR/recon_real")
OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_prepost_testset_20261124")
ART = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
BART = "/home/congcongliu/src/bart"
ECALIB = ["ecalib", "-t", "0.001", "-c", "0.0", "-r", "24", "-m", "1"]
THREADS = 16
CASES = {
    "9000103445": {"pre": ("t1_3D_gre_fsp_sag_iso_ACS_16884476230000", "rawdata_T1_turmor_9000103445"),
                   "post": ("t1_3D_gre_fsp_sag_iso_ACS_16884476230050", "rawdata_CE_turmor_9000103445")},
    "9002944080": {"pre": ("t1_3D_gre_fsp_sag_iso_ACS_16886244540000", "rawdata"),
                   "post": ("t1_3D_gre_fsp_sag_iso_ACS_16886244540050", "rawdata")},
    "9003136585": {"pre": ("t1_3D_gre_fsp_sag_iso_ACS_16886147050000", "T1"),
                   "post": ("t1_3D_gre_fsp_sag_iso_ACS_16886147050050", "T1_CE")},
}
log = lambda *a: print(*a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_cfl(base):
    dims = [int(x) for x in Path(str(base) + ".hdr").read_text().split("\n")[1].split()]
    while len(dims) > 4 and dims[-1] == 1:
        dims.pop()
    return np.memmap(str(base) + ".cfl", dtype=np.complex64, mode="r", shape=tuple(dims), order="F")


def write_cfl(base, a):
    """a: (x, ky, kz, coil)"""
    Path(str(base) + ".hdr").write_text("# Dimensions\n" + " ".join(str(d) for d in a.shape) + " 1\n")
    m = np.memmap(str(base) + ".cfl", dtype=np.complex64, mode="w+", shape=a.shape, order="F")
    m[:] = a
    m.flush()


def ecalib(src_base, out_base, ev_base=None):
    """ev_base: also write the ESPIRiT eigenvalue map (x, ky, kz), the quantity bart's -c crop thresholds"""
    env = dict(os.environ, TOOLBOX_PATH=BART, PATH=f"{BART}:{os.environ.get('PATH', '')}", OMP_NUM_THREADS=str(THREADS))
    cmd = [f"{BART}/bart", *ECALIB, str(src_base), str(out_base)] + ([str(ev_base)] if ev_base else [])
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]


def acs_side(m2):
    cy, cz = m2.shape[0] // 2, m2.shape[1] // 2
    s = 0
    for h in range(1, min(cy, cz)):
        if not m2[cy - h:cy + h, cz - h:cz + h].all():
            break
        s = 2 * h
    return s


def peak_x(k_cxyz):
    """readout index of the largest energy on the central ky, kz lines, all coils"""
    _, nx, ny, nz = k_cxyz.shape
    e = (np.abs(np.asarray(k_cxyz[:, :, ny // 2 - 2:ny // 2 + 2, nz // 2 - 2:nz // 2 + 2])) ** 2).sum(axis=(0, 2, 3))
    return int(np.argmax(e))


def prep(subj, scan):
    series, stem = CASES[subj][scan]
    wd = OUT / "subjects" / subj / f"work_{scan}"
    if (wd / "prep.json").is_file():
        log(f"PREP_OK {subj} {scan} (already)")
        return
    assert not wd.exists(), f"{wd} exists without prep.json (incomplete); decide by hand"
    wd.mkdir(parents=True)
    src = D / subj / series / stem
    k = read_cfl(src)                                              # (x, ky, kz, coil)
    nx, ny, nz, nc = k.shape
    ksp = np.lib.format.open_memmap(wd / "ksp_full.npy", mode="w+", dtype=np.complex64, shape=(nc, nx, ny, nz))
    sup = np.zeros((nx, ny, nz), bool)
    for c in range(nc):
        a = np.asarray(k[..., c])
        ksp[c] = a
        sup |= a != 0
    ksp.flush()
    m2 = sup.any(axis=0)
    assert (sup == m2[None]).all(), "sampling differs along the readout"
    np.save(wd / "mask2d.npy", np.broadcast_to(m2[None].astype(np.float32), (nx, ny, nz)).copy())
    log(f"[{subj} {scan}] k-space {k.shape}, points {int(m2.sum())}, AF {m2.size / m2.sum():.3f}, ACS {acs_side(m2)}")
    tmp, tev = wd / "_csm_tmp", wd / "_ev_tmp"
    ecalib(src, tmp, tev)
    np.save(wd / "ev.npy", np.abs(np.asarray(read_cfl(tev))).reshape(nx, ny, nz).astype(np.float32))
    for ext in (".cfl", ".hdr"):
        os.remove(str(tev) + ext)
    s = read_cfl(tmp)
    assert s.shape == k.shape, (s.shape, k.shape)
    csm = np.lib.format.open_memmap(wd / "csm.npy", mode="w+", dtype=np.complex64, shape=(nc, nx, ny, nz))
    sos = np.zeros((nx, ny, nz), np.float32)
    for c in range(nc):
        a = np.asarray(s[..., c])
        csm[c] = a
        sos += np.abs(a) ** 2
    csm.flush()
    del s
    for ext in (".cfl", ".hdr"):
        os.remove(str(tmp) + ext)
    sos = np.sqrt(sos)
    rep = dict(subject=subj, scan=scan, source=str(src), shape_coil_x_ky_kz=[nc, nx, ny, nz],
               mask_points=int(m2.sum()), af=float(m2.size / m2.sum()), acs_side=acs_side(m2),
               peak_x=peak_x(ksp), grid_centre_x=nx // 2, ecalib=" ".join(ECALIB),
               csm_sos_min=float(sos.min()), csm_sos_max=float(sos.max()),
               sha256={f: sha(wd / f) for f in ("ksp_full.npy", "mask2d.npy", "csm.npy", "ev.npy")})
    (wd / "prep.json").write_text(json.dumps(rep, indent=2))
    (ART / f"prep_{subj}_{scan}.json").write_text(json.dumps(rep, indent=2))
    log(f"PREP_OK {subj} {scan}: peak_x {rep['peak_x']} (centre {nx // 2}), csm SoS [{rep['csm_sos_min']:.4f}, {rep['csm_sos_max']:.4f}]")


def record_check():
    w = REC / "work_ce_t1_16887049370010"
    od = OUT / "record_check"
    od.mkdir(parents=True, exist_ok=True)
    k = np.load(w / "ksp_full.npy", mmap_mode="r")                 # (coil, x, ky, kz)
    px = peak_x(k)
    if not (od / "csm_house.npy").is_file():
        write_cfl(od / "_ksp", np.moveaxis(np.asarray(k), 0, -1))
        ecalib(od / "_ksp", od / "_csm")
        s = np.moveaxis(np.asarray(read_cfl(od / "_csm")), -1, 0)
        np.save(od / "csm_house.npy", s.astype(np.complex64))
        for b in ("_ksp", "_csm"):
            for ext in (".cfl", ".hdr"):
                os.remove(str(od / b) + ext)
    a = np.load(od / "csm_house.npy", mmap_mode="r")
    b = np.load(w / "csm.npy", mmap_mode="r")
    num = np.zeros(a.shape[1:], np.complex64)
    na = np.zeros(a.shape[1:], np.float32)
    nb = np.zeros(a.shape[1:], np.float32)
    for c in range(a.shape[0]):
        x, y = np.asarray(a[c]), np.asarray(b[c])
        num += np.conj(x) * y
        na += np.abs(x) ** 2
        nb += np.abs(y) ** 2
    agree = np.abs(num) / np.sqrt(na * nb + 1e-20)
    zf = np.load(w / "zf_rss.npy")
    head = zf > 0.1 * np.percentile(zf, 99.9)
    rep = dict(record_peak_x=px, grid_centre_x=k.shape[1], house_sos=[float(np.sqrt(na).min()), float(np.sqrt(na).max())],
               record_sos=[float(np.sqrt(nb).min()), float(np.sqrt(nb).max())],
               record_sos_head_median=float(np.median(np.sqrt(nb)[head])),
               agreement_head=dict(mean=float(agree[head].mean()), p05=float(np.percentile(agree[head], 5)),
                                   median=float(np.median(agree[head]))),
               agreement_all_mean=float(agree.mean()), head_fraction=float(head.mean()))
    rep["grid_centre_x"] = k.shape[1] // 2
    (od / "record_check.json").write_text(json.dumps(rep, indent=2))
    (ART / "record_check.json").write_text(json.dumps(rep, indent=2))
    log("RECORD_CHECK", json.dumps(rep))


def backfill_ev(subj, scan):
    """ev.npy for a work dir prepared before the eigenvalue map was kept (csm.npy is checked to be reproduced)"""
    series, stem = CASES[subj][scan]
    wd = OUT / "subjects" / subj / f"work_{scan}"
    assert (wd / "prep.json").is_file() and not (wd / "ev.npy").exists(), wd
    tmp, tev = wd / "_csm_tmp", wd / "_ev_tmp"
    ecalib(D / subj / series / stem, tmp, tev)
    s = np.moveaxis(np.asarray(read_cfl(tmp)), -1, 0)
    same = bool(np.array_equal(s, np.load(wd / "csm.npy", mmap_mode="r")))
    ev = np.abs(np.asarray(read_cfl(tev))).reshape(s.shape[1:]).astype(np.float32)
    np.save(wd / "ev.npy", ev)
    for b in (tmp, tev):
        for ext in (".cfl", ".hdr"):
            os.remove(str(b) + ext)
    rep = dict(subject=subj, scan=scan, csm_reproduced=same, ev_range=[float(ev.min()), float(ev.max())],
               sha256_ev=sha(wd / "ev.npy"))
    (wd / "ev.json").write_text(json.dumps(rep, indent=2))
    log("EV_OK", json.dumps(rep))


def record_ev():
    """Which rule zeroes the record's maps outside the head: compare the record's support (SoS > 0) with
    eigenvalue-map thresholds (bart's -c crop) of the house ecalib on the same k-space."""
    w = REC / "work_ce_t1_16887049370010"
    od = OUT / "record_check"
    k = np.load(w / "ksp_full.npy", mmap_mode="r")
    if not (od / "ev_house.npy").is_file():
        write_cfl(od / "_ksp", np.moveaxis(np.asarray(k), 0, -1))
        ecalib(od / "_ksp", od / "_csm", od / "_ev")
        np.save(od / "ev_house.npy", np.abs(np.asarray(read_cfl(od / "_ev"))).reshape(k.shape[1:]).astype(np.float32))
        for b in ("_ksp", "_csm", "_ev"):
            for ext in (".cfl", ".hdr"):
                os.remove(str(od / b) + ext)
    ev = np.load(od / "ev_house.npy")
    b = np.load(w / "csm.npy", mmap_mode="r")
    sup = np.zeros(ev.shape, bool)
    for c in range(b.shape[0]):
        sup |= np.asarray(b[c]) != 0
    rows = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99):
        m = ev > t
        rows.append(dict(thr=t, dice=float(2 * (m & sup).sum() / (m.sum() + sup.sum())), frac=float(m.mean())))
    post = np.load(REC / "work_ce_t1_16887049370060" / "csm.npy", mmap_mode="r")
    same_post = bool(all(np.array_equal(np.asarray(b[c]), np.asarray(post[c])) for c in (0, b.shape[0] // 2, b.shape[0] - 1)))
    rep = dict(record_support_fraction=float(sup.mean()), ev_in_support=[float(np.percentile(ev[sup], q)) for q in (0, 1, 5, 50)],
               ev_outside=[float(np.percentile(ev[~sup], q)) for q in (50, 95, 99, 100)] if (~sup).any() else None,
               thresholds=rows, record_pre_post_same_maps=same_post)
    (od / "record_ev.json").write_text(json.dumps(rep, indent=2))
    (ART / "record_ev.json").write_text(json.dumps(rep, indent=2))
    log("RECORD_EV", json.dumps(rep))


def echo_check():
    sd = D / "9000103445" / "t1_3D_gre_fsp_sag_iso_ACS_16884476230000"
    a, b = read_cfl(sd / "rawdata_T1_turmor_9000103445"), read_cfl(sd / "Echo_1")
    out = []
    for c in (0, 20, 41):
        x, y = np.asarray(a[..., c]), np.asarray(b[..., c])
        best = max(range(b.shape[3]), key=lambda j: abs(np.vdot(x, np.asarray(b[..., j])))) if c == 0 else None
        al = np.vdot(y, x) / (np.vdot(y, y) + 1e-30)
        out.append(dict(coil=c, norm_raw=float(np.linalg.norm(x)), norm_echo=float(np.linalg.norm(y)),
                        rel_diff=float(np.linalg.norm(x - y) / np.linalg.norm(x)),
                        rel_diff_after_scale=float(np.linalg.norm(x - al * y) / np.linalg.norm(x)),
                        scale_abs=float(abs(al)), same_support=bool(((x != 0) == (y != 0)).all()),
                        best_matching_echo_coil_for_raw_coil0=best))
    (ART / "echo_check.json").write_text(json.dumps(out, indent=2))
    log("ECHO_CHECK", json.dumps(out))


ap = argparse.ArgumentParser()
ap.add_argument("--subjects", nargs="*", default=[])
ap.add_argument("--record_check", action="store_true")
ap.add_argument("--echo_check", action="store_true")
ap.add_argument("--record_ev", action="store_true")
ap.add_argument("--backfill_ev", nargs="*", default=[])
args = ap.parse_args()
if args.record_ev:
    record_ev()
for s in args.backfill_ev:
    for scan in ("pre", "post"):
        backfill_ev(s, scan)
if args.echo_check:
    echo_check()
if args.record_check:
    record_check()
for s in args.subjects:
    assert s in CASES, s
    for scan in ("pre", "post"):
        prep(s, scan)
log("DONE")
