#!/usr/bin/env python3
"""export_ce_bundle.py -- collect everything behind the CE pre/post-contrast figure of record into one folder (2026-09-15).

Real copies (not links) of: the 100%-line reconstruction of every arm in the figure (plus L1-ESPIRiT, reconstructed but
dropped from the comparison on 2026-09-15), the acquisition itself (42-coil k-space, coil maps, sampling mask), the
frozen 95/5 held-out split, the per-run metas and selection records, the result tables and the figures. Writes
MANIFEST.tsv (file, what, scan, shape, dtype, bytes, md5, source, note) and checksums.md5.

  python scripts/export_ce_bundle.py [--out export/ce_l1spirit_20260915] [--no_raw] [--dry_run]
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parents[1]
RR = Path("/data0/congcong/code/NV-Generate-CTMR/recon_real")
WORK = {"pre": RR / "work_ce_t1_16887049370010", "post": RR / "work_ce_t1_16887049370060"}
FIGS = Path("/home/congcongliu/figs/ce_pair")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="export/ce_l1spirit_20260915")
ap.add_argument("--no_raw", action="store_true", help="skip the 42-coil k-space and coil maps (about 14 GiB)")
ap.add_argument("--dry_run", action="store_true")
a = ap.parse_args()
OUT = T / a.out
L = T / "logs/ext_20260914"
sel = {f: json.load(open(L / f"selection_{f}.json"))["best"] for f in ("cs_pre", "cs_post", "joint", "l1_pre", "l1_post")}
sp = {s: json.load(open(L / f"selection_spirit_{s}.json")) for s in ("pre", "post")}

ALIGNED = "layout (S-I 256, A-P 232, L-R 176); complex, alpha-scaled onto the measured k-space"
GSNOTE = ("layout (S-I 256, L-R 176, A-P 232) -> np.transpose(v, (0, 2, 1)); GS units: multiply by the P99.9 "
          "scale of the scan (pre 139.94, post 167.92) to reach the measured-data scale")


def arms(s):
    """(key, label, source, note, in_figure)"""
    return [
        ("zf", "ZF", T / f"classical/zf_{s}_complex.npy", ALIGNED, True),
        ("cgsense", "CG-SENSE (tik 0.01, 60 CG)", T / f"classical/cgsense_{s}_complex.npy", ALIGNED, True),
        ("cssense", f"CS-SENSE (TV, 200 outer ADMM, lambda {sel['cs_' + s]:g})",
         T / f"tv_sense/cssense_o200_lam{sel['cs_' + s]:g}_full_{s}_complex.npy", ALIGNED, True),
        ("jointsense", f"Joint SENSE (vectorial TV, lambda {sel['joint']:g})",
         T / f"tv_sense/jsense_o200_lam{sel['joint']:g}_full_{s}_complex.npy", ALIGNED, True),
        ("l1spirit", f"L1-SPIRiT (kernel {sp[s]['ksize']}, lambda {sp[s]['lam']:g}, 150 FISTA)",
         T / f"spirit/spirit_ks{sp[s]['ksize']}_i150_lam{sp[s]['lam']:g}_full_{s}_complex.npy", ALIGNED, True),
        ("zsssdu", "ZS-SSDU", T / f"classical/zsssdu_{s}_complex.npy", ALIGNED, True),
        ("dip3d", "3D-DIP (ConvDecoder, best held-out checkpoint)", T / f"dipinr/dip_{s}_complex.npy", ALIGNED, True),
        ("inr3d", "3D-INR (hash grid, best held-out checkpoint)", T / f"dipinr/inr_{s}_complex.npy", ALIGNED, True),
        ("mcinr", "MC-INR (Niessen 2025)", T / f"mcinr/mcinr_full_{s}_complex.npy", ALIGNED, True),
        ("gs_single", "GS-single (600k Gaussians, 09-07 recipe)",
         T / f"gs_single_{s}/gs_single_{s}_raw_complex.npy", GSNOTE, True),
        ("mcgs", "MC-GS (joint, shared geometry)", T / f"gs_joint/gs_joint_{s}_raw_complex.npy", GSNOTE, True),
        ("l1espirit", f"L1-ESPIRiT (BART pics -l1, lambda {sel['l1_' + s]:g}) -- not in the figure since 2026-09-15",
         T / f"l1espirit/l1espirit_i100_lam{sel['l1_' + s]:g}_full_{s}_complex.npy", ALIGNED, False),
    ]


plan = []                                        # (dest relative path, what, scan, source, note)
for s in ("pre", "post"):
    for key, label, src, note, shown in arms(s):
        plan.append((f"volumes/{s}/{key}_{s}_complex.npy", label, s, src, note))
    if not a.no_raw:
        plan.append((f"raw/{s}/ksp_full.npy", "42-coil k-space, zeros off the acquired mask", s, WORK[s] / "ksp_full.npy",
                     "(coil 42, S-I 256, A-P 232, L-R 176); readout = axis 1, fully sampled; prospective AF 5.33"))
        plan.append((f"raw/{s}/csm.npy", "coil sensitivity maps used by every arm", s, WORK[s] / "csm.npy",
                     "(coil, S-I, A-P, L-R); SoS == 1 inside the support"))
    plan.append((f"raw/{s}/mask2d.npy", "sampling mask", s, WORK[s] / "mask2d.npy", "(S-I, A-P, L-R), constant along S-I"))
plan.append(("heldout_split/split_95_5_seed0.npz", "95/5 held-out line split (fit / ho planes, A-P x L-R)", "both",
             T / "heldout_split/split_95_5_seed0.npz", "7663 lines: 7280 fit, 383 held out"))
for f in ("kres_summary_20260914.json", "heldout4_scores.json", "lambda_selection.json", "lambda_grids.json",
          "kres_summary_errormap.json", "kres_summary.json"):
    plan.append((f"tables/{f}", "result table", "both", T / f, ""))
for p in sorted(L.glob("selection_*.json")):
    plan.append((f"logs/{p.name}", "held-out selection record", "both", p, ""))
for s in ("pre", "post"):
    for key, label, src, note, shown in arms(s):
        m = src.with_name(src.name.replace("_complex.npy", "_meta.json").replace("_raw_meta.json", "_meta.json"))
        if m.exists():
            plan.append((f"logs/meta/{m.name}", f"run meta: {label}", s, m, ""))
    for m in [T / f"tv_sense/cssense_o200_lam{sel['cs_' + s]:g}_fit_{s}_meta.json",
              T / f"tv_sense/jsense_o200_lam{sel['joint']:g}_fit_{s}_meta.json",
              T / f"spirit/spirit_ks{sp[s]['ksize']}_i150_lam{sp[s]['lam']:g}_fit_{s}_meta.json",
              T / f"l1espirit/l1espirit_i100_lam{sel['l1_' + s]:g}_fit_{s}_meta.json",
              T / f"mcinr/mcinr_fit_{s}_meta.json"]:
        plan.append((f"logs/meta_heldout/{m.name}", "95%-line run meta (held-out kres)", s, m, ""))
for n in ("ce_pair_recon_3planes.png", "ce_pair_recon_3planes.pdf", "ce_pair_methods_errormap.png",
          "ce_pair_methods_errormap_zoom.png"):
    plan.append((f"figures/{n}", "figure of record", "both", FIGS / n, "axial anterior up; L1-SPIRiT in the L1 slot"))
plan.append(("README_topic.md", "topic README at export time", "both", T / "README.md", ""))

missing = [str(src) for _, _, _, src, _ in plan if not Path(src).exists()]
if missing:
    raise SystemExit("missing sources:\n" + "\n".join(missing))
total = sum(Path(src).stat().st_size for _, _, _, src, _ in plan)
print(f"{len(plan)} files, {total / 2**30:.2f} GiB -> {OUT}")
if a.dry_run:
    for d, w, s, src, n in plan:
        print(f"  {d:<70s} <- {src}")
    raise SystemExit(0)


def md5(path, bufsize=16 * 2**20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                return h.hexdigest()
            h.update(b)


rows = []
for d, what, s, src, note in plan:
    dst = OUT / d
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)                     # real copy, dereferences links
    shape = dtype = ""
    if dst.suffix == ".npy":
        v = np.load(dst, mmap_mode="r")
        shape, dtype = "x".join(map(str, v.shape)), str(v.dtype)
    digest = md5(dst)
    rows.append((d, what, s, shape, dtype, dst.stat().st_size, digest, str(src), note))
    print(f"  {d}  {dst.stat().st_size / 2**20:.1f} MiB  {digest}", flush=True)

with open(OUT / "MANIFEST.tsv", "w") as f:
    f.write("file\twhat\tscan\tshape\tdtype\tbytes\tmd5\tsource\tnote\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
with open(OUT / "checksums.md5", "w") as f:
    for r in rows:
        f.write(f"{r[6]}  {r[0]}\n")
print(f"DONE {len(rows)} files, {sum(r[5] for r in rows) / 2**30:.2f} GiB")
