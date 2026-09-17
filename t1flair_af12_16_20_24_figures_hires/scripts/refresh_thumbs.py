#!/usr/bin/env python
"""refresh_thumbs.py -- rebuild the browse thumbnails of the four topics' figure of record.

/home/congcongliu/figs/spirit_index.html browses the delivered set through JPEG thumbnails in
spirit_thumbs/. Those were made by hand on 2026-09-16 15:06 and went stale the moment the figures were
re-rendered at 200 dpi that evening, with no script to rebuild them -- the same trap that left
mask_check_af12_af11_af16.png without a generator. This is that script.

Thumbnails keep the hand-made specification: 1900 px wide, JPEG quality 82, aspect preserved.
"""
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
FIGS = Path("/home/congcongliu/figs")
THUMBS = FIGS / "spirit_thumbs"
WIDTH, QUALITY = 1900, 82
TOPICS = [("12", "p14"), ("16", "p3"), ("20", "p3"), ("24", "p3")]

THUMBS.mkdir(exist_ok=True)
n = 0
for af, tag in TOPICS:
    for c in ("T1", "FLAIR"):
        base = (f"t1flair_v3_af{af}_{c}_row3planes_compact_180_124_76_inset48_headerr_"
                f"{tag}_mcinr_spirit_zf_caps")
        src = FIGS / f"t1flair_v3_af{af}" / f"{base}.png"
        if not src.exists():
            print("MISSING", src)
            continue
        im = Image.open(src)
        h = round(im.height * WIDTH / im.width)
        im.convert("RGB").resize((WIDTH, h), Image.LANCZOS).save(THUMBS / f"{base}.jpg", quality=QUALITY)
        print(f"{base[:58]:58s} {im.width}x{im.height} -> {WIDTH}x{h}")
        n += 1
print(f"{n} thumbnails rebuilt in {THUMBS}")
