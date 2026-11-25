#!/usr/bin/env python3
"""finalize_optimal_fig.py -- store the current optimal clinical-verification figure on the server under the short
names used for the 2026-11-22 figure, check the PDF, and write a manifest (user, 2026-11-24: "Save the corresponding
original data and the vector at 600 dpi in PDF format to the GPU server").

In DIR (default: the [0, 0.08], text-clear drawing of the tuned MC-GS):
  <long>.pdf       -> ce_af12_prepost_clinical_verification_600dpi.pdf
  <long>.png       -> ce_af12_prepost_clinical_verification.png
  <long>_data.npz  -> ce_af12_prepost_clinical_verification_data.npz
Copies are byte-checked (sha256). The PDF check uses poppler (pdffonts, pdfimages -list): every font must be embedded
(text stays vector) and every embedded raster must have at least --min_ppi in both directions. The data npz is listed
by key, and the source volumes (paths + sha256) are taken over from the record JSON. Writes MANIFEST.json into DIR and
a copy to MCGS_ARTIFACT_DIR. Existing short-name files are not overwritten unless they are byte-identical.

  gpu.py run --cpu -- @python autoresearch/ce_af12_tune_20261122/finalize_optimal_fig.py
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", type=Path, default=Path(
    "/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_af12_prepost_tuned_20261123/v3_err008_textclear"))
ap.add_argument("--stem", default="ce_af12_lesion_three_planes_201_131_48_prepost_tuned_MC_GS_AF12_autoresearch_seed_42_"
                                  "zoomerr_inferno_err0.08_arrowlesion_onzoom_txtauto")
ap.add_argument("--record", default="record_tuned_MC_GS_AF12_autoresearch_seed_42_three.json")
ap.add_argument("--min_ppi", type=int, default=600)
args = ap.parse_args()
d = args.dir
assert d.is_dir(), d


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


short = {".pdf": "ce_af12_prepost_clinical_verification_600dpi.pdf",
         ".png": "ce_af12_prepost_clinical_verification.png",
         "_data.npz": "ce_af12_prepost_clinical_verification_data.npz"}
copies = {}
for suf, name in short.items():
    src, dst = d / f"{args.stem}{suf}", d / name
    assert src.is_file(), src
    if dst.exists():
        assert sha(dst) == sha(src), f"{dst.name} exists with other content; not overwritten"
    else:
        shutil.copy2(src, dst)
    assert sha(dst) == sha(src)
    copies[name] = dict(source=src.name, sha256=sha(dst), bytes=dst.stat().st_size)
    print(f"{name}  =  {src.name}  ({dst.stat().st_size} bytes, sha256 {copies[name]['sha256'][:16]})")

pdf = d / short[".pdf"]
fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True, check=True).stdout.splitlines()[2:]
font_rows = []
for ln in fonts:
    parts = ln.split()
    # name type [type2] encoding emb sub uni object ID : emb / sub / uni are the yes/no columns before "object ID"
    yn = [p for p in parts if p in ("yes", "no")]
    font_rows.append(dict(name=parts[0], embedded=yn[0] == "yes" if yn else None, line=ln.strip()))
imgs = subprocess.run(["pdfimages", "-list", str(pdf)], capture_output=True, text=True, check=True).stdout.splitlines()[2:]
img_rows = []
for ln in imgs:
    p = ln.split()
    # page num type width height color comp bpc enc interp object ID x-ppi y-ppi size ratio
    img_rows.append(dict(type=p[2], width=int(p[3]), height=int(p[4]), x_ppi=int(p[12]), y_ppi=int(p[13])))
img_main = [r for r in img_rows if r["type"] == "image"]           # smask rows are the alpha planes of the same images
min_ppi = min(min(r["x_ppi"], r["y_ppi"]) for r in img_main)
pages = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
npages = next((int(l.split()[-1]) for l in pages.splitlines() if l.startswith("Pages:")), None)
check = dict(pages=npages, fonts=len(font_rows), all_fonts_embedded=all(r["embedded"] for r in font_rows),
             images=len(img_main), min_image_ppi=min_ppi, all_images_at_least=args.min_ppi,
             passed=bool(font_rows) and all(r["embedded"] for r in font_rows) and min_ppi >= args.min_ppi)
print("PDF check:", json.dumps(check))
assert check["passed"], "PDF check failed"

with np.load(d / short["_data.npz"], allow_pickle=False) as z:
    keys = sorted(z.files)
    meta = json.loads(str(z["meta_json"]))
rec = json.loads((d / args.record).read_text())
man = dict(
    request='user 2026-11-24: "Save the corresponding original data and the vector at 600 dpi in PDF format to the GPU server"',
    figure="optimal clinical-verification figure, CE AF12 plain scan (pre) + enhanced (post), planes 201 / 131 / 48, "
           "MC-GS = tuned (AF12 autoresearch, seed 42), error inferno [0, 0.08], text clear of the zoom box",
    dir=str(d), short_name_copies=copies, pdf_check=check, pdf_fonts=[r["line"] for r in font_rows],
    data_npz=dict(keys=len(keys), contents="for each of the six rows: GT slice, the eleven reconstructed slices, head mask, "
                  "error region and lesion core (volume orientation); the sampling-mask plane of each orientation; "
                  "meta_json (boxes, arrows, panel numbers, display settings, source volumes)",
                  error_display=meta.get("error_display")),
    source_volumes=rec["volumes"], gt_sha256=rec.get("gt_sha256"), mc_npz=rec.get("mc_npz"),
    files={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(d.iterdir())
           if p.is_file() and p.name != "MANIFEST.json"})
(d / "MANIFEST.json").write_text(json.dumps(man, indent=2, ensure_ascii=False))
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
shutil.copy2(d / "MANIFEST.json", art / "MANIFEST.json")
print(f"wrote {d / 'MANIFEST.json'} ({len(man['files'])} files listed)")
