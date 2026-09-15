#!/usr/bin/env python3
"""export_af24_bundle.py -- everything behind the AF 24 figure set of record, in one self-contained folder.

Same layout and manifest format as ../T1FLAIR_AF16_v3_jiangyizhou/export/af16_phase3 and
../T1FLAIR_AF12_v3_jiangyizhou/export/af12_p14_mcinr (user request 2026-09-15: save the AF 24 raw data and the result
figures on this server). Real copies, symlinks dereferenced: the acquisition (26-coil k-space of both contrasts, the
ESPIRiT maps), the mask and its three variance draws, the reconstruction volume of every arm in the tables (the ten
shown, plus L1-ESPIRiT which is data although no longer shown), the ground truth, the score logs, the MC-INR
provenance, every figure of the topic, and the topic README at export time. MANIFEST.tsv carries shape / dtype / size /
md5 per file; `md5sum -c checksums.md5` verifies the copy.

  python scripts/export_af24_bundle.py [--out export/af24_p3_mcinr_spirit] [--no_raw] [--dry_run]
"""
import argparse, hashlib, json, re, shutil, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="export/af24_p3_mcinr_spirit")
ap.add_argument("--mcgs_key", default="gs_joint_p3")
ap.add_argument("--single_key", default="gs_single_p3")
ap.add_argument("--no_raw", action="store_true", help="skip the ~9 GiB acquisition (k-space + maps)")
ap.add_argument("--dry_run", action="store_true")
a = ap.parse_args()
OUT = R / a.out
summary = json.load(open(R / "logs/summary.json"))
by = {(r["contrast"], r["method"]): r for r in summary}


def lam_file(c, row_key, prefix):
    lam = by.get((c, row_key), {}).get("lam")
    for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
        m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
        if lam is not None and m and abs(float(m.group(1)) - float(lam)) < 1e-12:
            return p
    raise FileNotFoundError(f"{prefix} {c} lam={lam}")


def spirit_file(c):
    r = by[(c, "spirit")]
    return R / f"recon/spirit_ks{r['ksize']}_lam{r['lam']:g}_{c}_complex.npy"


def arms(c):
    return [("ground truth (conjugate-SENSE image of the full k-space)", R / f"recon/gt_{c}_complex.npy"),
            ("zero-filled", R / f"recon/zf_{c}_complex.npy"),
            ("CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
            ("CS-SENSE, 35 outer ADMM iterations", R / f"recon/cssense_{c}_complex.npy"),
            ("CS-SENSE (TV, converged), lambda kept by the table", lam_file(c, "cssense_o200", "cssense_o200")),
            ("Joint SENSE (JTV), lambda kept by the table", lam_file(c, "jcssense", "jcssense_o200")),
            ("L1-ESPIRiT, lambda kept by the table (data only: not shown since 2026-09-15)", lam_file(c, "l1espirit", "l1espirit")),
            ("L1-SPIRiT, kernel size and lambda kept by the table", spirit_file(c)),
            ("ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
            ("3D-DIP", R / f"recon/dip_{c}_complex.npy"),
            ("3D-INR", R / f"recon/inr_{c}_complex.npy"),
            ("MC-INR (Niessen 2025), topic units", (R / f"recon/mcinr_{c}_complex.npy").resolve()),
            ("GS-single (AF16 phase-3 recipe), magnitude", R / f"recon/{a.single_key}_{c}_mag.npy"),
            ("GS-single (AF16 phase-3 recipe), complex", R / f"recon/{a.single_key}_{c}_complex.npy"),
            ("MC-GS (AF16 phase-3 recipe), magnitude", R / f"recon/{a.mcgs_key}_{c}_mag.npy"),
            ("MC-GS (AF16 phase-3 recipe), complex", R / f"recon/{a.mcgs_key}_{c}_complex.npy")]


jobs = []
for c in ("T1", "FLAIR"):
    for what, src in arms(c):
        jobs.append((f"volumes/{c}/{src.name}", src, f"{c}: {what}"))
if not a.no_raw:
    for c in ("T1", "FLAIR"):
        jobs.append((f"raw/ksp_{c}.h5", (R / f"data/ksp_{c}.h5").resolve(),
                     f"{c}: acquired k-space, 26 coils, complex64, full grid (the mask is applied retrospectively)"))
        jobs.append((f"raw/csm_espirit_{c}.h5", (R / f"data/csm_espirit_{c}.h5").resolve(),
                     f"{c}: ESPIRiT coil sensitivity maps used by every arm and by the ground truth"))
jobs.append(("mask/mask_af24_3d.h5", R / "data/mask_af24_3d.h5", "3-D sampling mask of record (readout-broadcast), 1877 PE points"))
jobs.append(("mask/mask_vdpois_256x176_af24.mat", R / "data/mask_vdpois_256x176_af24.mat", "the 2-D vd-Poisson mask of record"))
for d in ("db", "dc", "dd"):
    jobs.append((f"mask/mask_af24_{d}_3d.h5", R / f"data/mask_af24_{d}_3d.h5", f"mask draw {d} of the mask-draw variance study (not the mask of record)"))
for n in ("summary.json", "summary.tsv", "readme_tables.md", "prep_04_export.json", "extra_gs_arms.json",
          "mcinr_T1.json", "mcinr_FLAIR.json", "post_summarize.log"):
    p = R / "logs" / n
    if p.exists():
        jobs.append((f"logs/{n}", p, f"topic log: {n}"))
for c in ("T1", "FLAIR"):
    r = by[(c, "spirit")]
    jobs.append((f"logs/spirit_{c}_ks{r['ksize']}_lam{r['lam']:g}.json", R / f"logs/spirit_{c}_ks{r['ksize']}_lam{r['lam']:g}.json",
                 f"{c}: L1-SPIRiT score at the kept kernel size / lambda"))
for p in sorted(R.glob("logs/af24_*_score.json")):
    jobs.append((f"logs/{p.name}", p, f"scorer output of the GS run {p.name[:-11]}"))
mc = (R / "recon/mcinr_T1_complex.npy").resolve().parent          # the reproduction's recon/af24 directory
for n, what in (("mcinr_norm_T1_complex.npy", "T1: MC-INR network output in the paper's normalised units"),
                ("mcinr_norm_FLAIR_complex.npy", "FLAIR: MC-INR network output in the paper's normalised units"),
                ("losses.npy", "MC-INR per-slice training loss curves"), ("meta.json", "MC-INR run metadata")):
    if (mc / n).exists():
        jobs.append((f"mcinr/{n}", mc / n, what))
for n in ("score_af24.json", "gauge_check_af24.json"):
    q = mc.parents[1] / "logs" / n
    if q.exists():
        jobs.append((f"mcinr/{n}", q, f"MC-INR reproduction log: {n}"))
jobs.append(("README_topic.md", R / "README.md", "this topic's full README at the time of export"))
for p in sorted((R / "figures").iterdir()):
    if p.suffix not in (".png", ".pdf"):
        continue
    what = ("figure of record, ten methods, L1-SPIRiT in the L1 column, MC-INR included" if "_p3_mcinr_spirit" in p.name else
            "archive: nine-method render of 2026-09-15 05:30 (L1-ESPIRiT column, old FLAIR AF label)" if "_p3." in p.name or "_p3.pdf" in p.name or p.name.endswith("_headerr_p3.png") else
            "mask / acquisition-support figure")
    jobs.append((f"figures/{p.name}", p, what))

missing = [str(s) for _, s, _ in jobs if not s.exists()]
if missing:
    sys.exit("missing sources:\n  " + "\n  ".join(missing))
total = sum(s.stat().st_size for _, s, _ in jobs)
print(f"{len(jobs)} files, {total/2**30:.2f} GiB -> {OUT}", flush=True)
if a.dry_run:
    for d, s, w in jobs:
        print(f"  {d:62s} {s.stat().st_size/2**20:8.1f} MiB  {w}")
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
print("EXPORT_DONE", flush=True)
