#!/usr/bin/env python3
"""compare_old_new.py -- side-by-side view of the MC-GS column of the designated optimal clinical-verification figure
(September recipe, untuned) and of the same figure redrawn with the tuned MC-GS (x1combo3_vol).

Both PNGs come from fig_lesion_plane_prepost.py with identical options, so they share one layout. The script finds the
panel columns (runs of non-white pixel columns), takes GS-single (the runner-up, identical in both) and the MC-GS column
of each figure, and writes GS-single | MC-GS old | MC-GS new with a label band on top. Display only; no numbers are
recomputed (the panel numbers printed in each column are those of the source figures).

  gpu.py run --cpu -- @python autoresearch/ce_af12_tune_20261122/compare_old_new.py
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

A = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts")
OLD = A / ("mcgs_ce_af12_prepost_optimal_20261122/v4_clinical_verification_zoom_arrow/"
           "ce_af12_lesion_three_planes_201_131_48_prepost_Sept_recipe_untuned_BART_Poisson_disc_AF12_zoomerr_inferno_"
           "err0.08_arrowlesion_onzoom.png")
NEW = A / ("mcgs_ce_af12_prepost_tuned_20261123/v1_tuned/"
           "ce_af12_lesion_three_planes_201_131_48_prepost_tuned_MC_GS_AF12_autoresearch_seed_42_zoomerr_inferno_"
           "err0.08_arrowlesion_onzoom.png")
OUT = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
Image.MAX_IMAGE_PIXELS = None


def columns(im):
    """x ranges of the twelve method columns: runs of pixel columns that are not white over the panel area."""
    a = np.asarray(im.convert("L"), dtype=np.float32)
    h = a.shape[0]
    body = a[int(0.08 * h):int(0.98 * h)]
    dark = (body < 245).mean(axis=0) > 0.30
    runs, x = [], 0
    while x < len(dark):
        if dark[x]:
            x0 = x
            while x < len(dark) and dark[x]:
                x += 1
            runs.append((x0, x))
        x += 1
    big = [r for r in runs if r[1] - r[0] > 0.04 * a.shape[1]]
    assert len(big) >= 12, f"expected 12 panel columns, found {len(big)}: {big}"
    return big[:12]


old, new = Image.open(OLD).convert("RGB"), Image.open(NEW).convert("RGB")
assert old.size == new.size, (old.size, new.size)
co, cn = columns(old), columns(new)
assert co == cn, "the two figures do not share one column layout"
W, H = old.size
y0 = int(0.0375 * H)                                               # from the method-name header row (caption cut)
pad = 6
crops = [(new, co[10], "GS-single (as in new)"), (old, co[11], "OLD: September recipe"),
         (new, cn[11], "NEW: tuned (autoresearch)")]
band = 80
widths = [(x1 - x0) + 2 * pad for _, (x0, x1), _ in crops]
canvas = Image.new("RGB", (sum(widths), H - y0 + band), "white")
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
except OSError:
    font = ImageFont.load_default()
x = 0
for (im, (x0, x1), label), w in zip(crops, widths):
    canvas.paste(im.crop((x0 - pad, y0, x1 + pad, H)), (x, band))
    tw = draw.textlength(label, font=font)
    draw.text((x + (w - tw) / 2, band * 0.3), label, fill="black", font=font)
    x += w
dst = OUT / "ce_af12_clinical_mcgs_old_vs_new.png"
canvas.save(dst, optimize=True)
small = canvas.resize((canvas.width // 3, canvas.height // 3), Image.LANCZOS)
small.save(OUT / "ce_af12_clinical_mcgs_old_vs_new_preview.jpg", quality=90)
print("columns", co)
print("wrote", dst.name, canvas.size)
