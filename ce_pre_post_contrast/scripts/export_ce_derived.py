#!/usr/bin/env python3
"""export_ce_derived.py -- bundle everything derived on 2026-09-15/16 from the CE pre/post pair (2026-09-16).

The reconstructions themselves are already in export/ce_l1spirit_20260915 and are not copied again. This collects what
was built on top of them: the lesion work (SynthSeg labels, white-matter-normalised volumes, enhancement map, candidate
tables, reference-region masks), the enhancement and boundary measurements, every figure of this round, and the scripts
that produce them. Writes MANIFEST.tsv (file, what, shape, dtype, bytes, md5, source) and checksums.md5.

  python scripts/export_ce_derived.py [--out export/ce_enhancement_20260916] [--dry_run]

The data are not de-identified (one real patient, and the NIfTI volumes include the face): keep this on the server.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parents[1]
FIGS = Path("/home/congcongliu/figs/ce_pair")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="export/ce_enhancement_20260916")
ap.add_argument("--dry_run", action="store_true")
a = ap.parse_args()
OUT = T / a.out

plan = []                                            # (dest, what, source)
add = lambda d, w, s: plan.append((d, w, Path(s)))

for n, w in [("cssense_pre.nii.gz", "CS-SENSE pre as NIfTI (1 mm, P99.9-normalised) -- includes the face"),
             ("cssense_post.nii.gz", "CS-SENSE post as NIfTI -- includes the face"),
             ("cssense_pre_seg.nii.gz", "SynthSeg robust 2.0 labels on the pre scan"),
             ("cssense_pre_volumes.csv", "SynthSeg volume table"),
             ("cssense_labels.npy", "SynthSeg labels resampled onto the (S-I, A-P, L-R) grid"),
             ("cssense_pn.npy", "pre in white-matter units (median of white matter = 1)"),
             ("cssense_qn.npy", "post in white-matter units"),
             ("cssense_enhancement_D.npy", "smoothed post - pre enhancement map"),
             ("cssense_candidates.json", "focal-enhancement candidates (vessels, sinuses, plexus)"),
             ("cssense_blobs.json", "Hessian blob peaks of the enhancement map"),
             ("blob_contrast_rank.json", "blob peaks ranked by post-minus-pre local contrast"),
             ("post_only_candidates.json", "foci that look like normal tissue on pre and bright on post"),
             ("lesion_cores_cssense.json", "dark cores in the lesion box, slice choice z 201 / y 131 / x 48"),
             ("enhance_rois_cssense.json", "post - pre of the reference regions"),
             ("enhance_rois_cssense.npz", "the reference-region masks used by the figures"),
             ("lesion_mask_thr0.55.npy", "superseded threshold segmentation (leaks into CSF) -- kept for the record"),
             ("lesion_mask_thr0.6.npy", "superseded threshold segmentation -- kept for the record"),
             ("lesion_mask_cssense.npy", "superseded threshold segmentation -- kept for the record"),
             ("lesion_measure_cssense.json", "superseded measurement from the leaky mask -- kept for the record")]:
    add(f"lesion/{n}", w, T / "lesion_candidates" / n)
for n, w in [("lesion_edge_profile.json", "lesion boundary: profiles, widths, 50% shift, rim gradient, WM noise"),
             ("seed_null_s1.json", "seed null test: a second seed on the same scan against the observed post - pre"),
             ("kres_summary_errormap.json", "kres of every arm as drawn in the error-map figures"),
             ("kres_summary_20260914.json", "the full result table"), ("heldout4_scores.json", "held-out kres")]:
    add(f"tables/{n}", w, T / n)
for p in sorted((T / "figures").glob("*.png")) + sorted((T / "figures").glob("*.pdf")):
    add(f"figures/{p.name}", "figure", p)
for p in sorted((FIGS / "lesion_candidates").glob("*.png")):
    add(f"figures/lesion/{p.name}", "lesion figure", p)
for p_ in sorted((T / "seed_null_s1").rglob("*_meta.json")):
    add(f"seed_null_s1/{p_.relative_to(T / 'seed_null_s1')}", "seed null run meta (volumes stay in seed_null_s1/)", p_)
for n in ("ce_lesion_candidates.py", "ce_figure_paper.py", "ce_figure.py", "ce_edge_profile.py", "ce_seed_null.py",
          "ce_orchestrate.py", "export_ce_derived.py", "export_ce_bundle.py"):
    add(f"scripts/{n}", "script", T / "scripts" / n)
add("README_topic.md", "the topic README at export time", T / "README.md")

missing = [str(s) for _, _, s in plan if not s.exists()]
if missing:
    raise SystemExit("missing sources:\n" + "\n".join(missing))
total = sum(s.stat().st_size for _, _, s in plan)
print(f"{len(plan)} files, {total / 2**20:.1f} MiB -> {OUT}")
if a.dry_run:
    for d, _, s in plan:
        print(f"  {d:<58s} <- {s}")
    raise SystemExit(0)


def md5(path, bufsize=8 * 2**20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                return h.hexdigest()
            h.update(b)


rows = []
for d, what, src in plan:
    dst = OUT / d
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    shape = dtype = ""
    if dst.suffix == ".npy":
        v = np.load(dst, mmap_mode="r")
        shape, dtype = "x".join(map(str, v.shape)), str(v.dtype)
    rows.append((d, what, shape, dtype, dst.stat().st_size, md5(dst), str(src)))
    print(f"  {d}  {dst.stat().st_size / 2**20:.1f} MiB", flush=True)

with open(OUT / "MANIFEST.tsv", "w") as f:
    f.write("file\twhat\tshape\tdtype\tbytes\tmd5\tsource\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
with open(OUT / "checksums.md5", "w") as f:
    for r in rows:
        f.write(f"{r[5]}  {r[0]}\n")
print(f"DONE {len(rows)} files, {sum(r[4] for r in rows) / 2**20:.1f} MiB")
