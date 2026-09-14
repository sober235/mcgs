#!/usr/bin/env python3
"""export_phase3_bundle.py -- collect everything behind the AF16 phase-3 figure of record into one self-contained folder.

Copies (real copies, not links) the reconstruction volume of every arm the figure shows, the ground truth, the sampling
mask, the score/summary logs and the figures, then writes a manifest with shape, dtype, size and md5 of each file.

  python scripts/export_phase3_bundle.py [--out export/af16_phase3] [--dry_run]
"""
import argparse, hashlib, json, re, shutil, sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
MC = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="export/af16_phase3")
ap.add_argument("--mcgs_key", default="gs_joint_p3")
ap.add_argument("--single_key", default="gs_single_p3")
ap.add_argument("--with_raw", action="store_true",
                help="also copy the acquisition itself: the 26-coil k-space and the ESPIRiT maps of both "
                     "contrasts (symlinks into the AF12 topic, dereferenced here) and the FLAIR shutter "
                     "support. About 9 GiB on top.")
ap.add_argument("--dry_run", action="store_true")
a = ap.parse_args()
OUT = R / a.out
summary = json.load(open(R / "logs/summary.json"))


def lam_file(c, row_key, prefix):
    lam = {r["method"]: r for r in summary if r["contrast"] == c}.get(row_key, {}).get("lam")
    for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
        m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
        if lam is not None and m and abs(float(m.group(1)) - float(lam)) < 1e-12:
            return p
    raise FileNotFoundError(f"{prefix} {c} lam={lam}")


def arms(c):
    return [("ground truth", R / f"recon/gt_{c}_complex.npy"),
            ("CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
            ("CS-SENSE (TV, converged)", lam_file(c, "cssense_o200", "cssense_o200")),
            ("Joint SENSE (JTV)", lam_file(c, "jcssense", "jcssense_o200")),
            ("L1-ESPIRiT", lam_file(c, "l1espirit", "l1espirit")),
            ("ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
            ("3D-DIP", R / f"recon/dip_{c}_complex.npy"),
            ("3D-INR", R / f"recon/inr_{c}_complex.npy"),
            ("MC-INR (Niessen 2025), topic units", MC / f"recon/af16/mcinr_{c}_complex.npy"),
            ("GS-single (phase 3), magnitude", R / f"recon/{a.single_key}_{c}_mag.npy"),
            ("GS-single (phase 3), complex", R / f"recon/{a.single_key}_{c}_complex.npy"),
            ("MC-GS (phase 3), magnitude", R / f"recon/{a.mcgs_key}_{c}_mag.npy"),
            ("MC-GS (phase 3), complex", R / f"recon/{a.mcgs_key}_{c}_complex.npy")]


jobs = []                                                  # (dest_rel, src, what)
for c in ("T1", "FLAIR"):
    for what, src in arms(c):
        jobs.append((f"volumes/{c}/{src.name}", src, f"{c}: {what}"))
if a.with_raw:
    for c in ("T1", "FLAIR"):
        jobs.append((f"raw/ksp_{c}.h5", (R / f"data/ksp_{c}.h5").resolve(),
                     f"{c}: acquired k-space, 26 coils, complex64 (the full grid; the mask is applied retrospectively)"))
        jobs.append((f"raw/csm_espirit_{c}.h5", (R / f"data/csm_espirit_{c}.h5").resolve(),
                     f"{c}: ESPIRiT coil sensitivity maps used by every arm"))
    jobs.append(("raw/flair_support_2d.h5", R / "data/flair_support_2d.h5",
                 "FLAIR's acquired support on the PE plane (the vendor elliptical shutter)"))
jobs.append(("mask/mask_af16_3d.h5", R / "data/mask_af16_3d.h5", "3-D sampling mask (readout-broadcast)"))
jobs.append(("mask/mask_vdpois_256x176_af16.mat", R / "data/mask_vdpois_256x176_af16.mat", "the 2-D vd-Poisson mask"))
for n in ("summary.json", "summary.tsv", "bar_complex_gauge.json", "prep_04_export.json", "extra_gs_arms.json",
          "mcinr_T1.json", "mcinr_FLAIR.json"):
    p = R / "logs" / n
    if p.exists():
        jobs.append((f"logs/{n}", p, f"topic log: {n}"))
ARC = Path("/data0/congcong/code/GS/mcgs_af16_wt/results_arc")
for e in ("p3_r3_e0001_ptv003_full4k", "p3_r3_ptv003_s43_full4k", "p3_r3_singleT1_full4k", "p3_r3_singleFLAIR_full4k"):
    p = ARC / f"{e}_score.json"
    if p.exists():
        jobs.append((f"logs/{e}_score.json", p, f"scorer output of run {e}"))
# the MC-INR arm's own provenance, as in the AF12 bundle
for c in ("T1", "FLAIR"):
    jobs.append((f"mcinr/mcinr_norm_{c}_complex.npy", MC / f"recon/af16/mcinr_norm_{c}_complex.npy",
                 f"{c}: MC-INR network output in the paper's normalised units (before the linear rescale to topic units)"))
jobs.append(("mcinr/losses.npy", MC / "recon/af16/losses.npy", "MC-INR per-slice training loss curves"))
jobs.append(("mcinr/meta.json", MC / "recon/af16/meta.json", "MC-INR run metadata: mask, hyperparameters, per-slice timings"))
for n in ("score_af16.json", "gauge_check_af16.json", "check_real_data.json", "slice_jitter.json"):
    q = MC / "logs" / n
    if q.exists():
        jobs.append((f"mcinr/{n}", q, f"MC-INR reproduction log: {n}"))
for n, src in (("SUMMARY_zh.md", MC / "SUMMARY_zh.md"), ("README.md", MC / "README.md"),
               ("fidelity-review.md", Path("/data0/congcong/code/GS/mcinr_niessen2025/docs/2026-09-13-fidelity-review.md"))):
    if src.exists():
        jobs.append((f"mcinr/{n}", src, f"MC-INR reproduction doc: {n}"))
jobs.append(("README_topic.md", R / "README.md", "this topic's full README at the time of export"))
for p in sorted((R / "figures").glob("*180_124_7*_inset*_p3*.*")):
    what = ("figure of record, ten methods (MC-INR included)" if "_p3_mcinr" in p.name else
            "figure of record, nine methods (as delivered before MC-INR was registered)" if "inset48" in p.name else
            "alternative 64-box figure, nine methods")
    jobs.append((f"figures/{p.name}", p, what))

missing = [str(s) for _, s, _ in jobs if not s.exists()]
if missing:
    sys.exit("missing sources:\n  " + "\n  ".join(missing))
total = sum(s.stat().st_size for _, s, _ in jobs)
print(f"{len(jobs)} files, {total/2**30:.2f} GiB -> {OUT}")
if a.dry_run:
    for d, s, w in jobs:
        print(f"  {d:58s} {s.stat().st_size/2**20:8.1f} MiB  {w}")
    sys.exit(0)


def md5(p, buf=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


rows = []
for dest, src, what in jobs:
    d = OUT / dest
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, d)
    shape = dtype = ""
    if d.suffix == ".npy":
        with open(d, "rb") as f:
            v = np.lib.format.read_magic(f)
            sh, _, dt = np.lib.format._read_array_header(f, v)
        shape, dtype = "x".join(map(str, sh)), str(dt)
    rows.append((dest, what, shape, dtype, d.stat().st_size, md5(d)))
    print("  ", dest, flush=True)

with open(OUT / "MANIFEST.tsv", "w") as f:
    f.write("file\twhat\tshape\tdtype\tbytes\tmd5\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
with open(OUT / "checksums.md5", "w") as f:
    for dest, _, _, _, _, h in rows:
        f.write(f"{h}  {dest}\n")
print(f"wrote {OUT}/MANIFEST.tsv and checksums.md5 ({len(rows)} files, {total/2**30:.2f} GiB)")
