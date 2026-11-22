#!/usr/bin/env python3
"""save_recon_data.py -- copy the reconstructed volumes behind a diff-mask figure into its server folder.

User request 2026-11-22: "the reconstructed data with different masks should also be saved to the GPU server".
Reads the figure's record JSON (every column's file and SHA-256), copies each volume, the two GT volumes and each
pattern's sampling mask into <folder>/recon_data/<pattern>/, re-hashes every copy and writes DATA_MANIFEST.json.
Read-only on the topics; CPU only; refuses an existing recon_data folder.

  gpu.py run --cpu -- @python autoresearch/diffmask_af16_masks_20261111/save_recon_data.py --folder <figure folder>
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--folder", required=True)
args = ap.parse_args()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


folder = Path(args.folder)
rec = json.load(open(next(folder.glob("record_*.json"))))
out = folder / "recon_data"
assert not out.exists(), f"{out} exists: not overwriting"
man, total = {}, 0
for pattern, topic in rec["topics"].items():
    d = out / pattern.replace(" ", "_")
    d.mkdir(parents=True)
    items = [("sampling mask", Path(topic) / "data/mask_af16_3d.h5", None)]
    for c in ("T1", "FLAIR"):
        items.append((f"GT {c}", Path(topic) / f"recon/gt_{c}_complex.npy", rec["gt_sha256"][c]))
        for lab, v in rec["volumes"][pattern][c].items():
            items.append((f"{lab} {c}", Path(v["path"]), v["sha256"]))
    man[pattern] = {}
    for what, src, want in items:
        dst = d / src.name
        assert not dst.exists(), dst
        shutil.copyfile(src, dst)
        got = sha(dst)
        assert want is None or got == want, f"{what} ({pattern}): copy sha256 {got} != record {want}"
        assert got == sha(src), f"{what} ({pattern}): copy differs from the source"
        total += dst.stat().st_size
        man[pattern][what] = dict(file=f"recon_data/{d.name}/{dst.name}", bytes=dst.stat().st_size, sha256=got, source=str(src))
        print(f"  ok {pattern:<16} {what:<18} {dst.name}", flush=True)
(folder / "DATA_MANIFEST.json").write_text(json.dumps(dict(
    what="reconstructed volumes of the diff-mask study (T1 + FLAIR, acceleration factor 16): every column of the figure, "
         "the GT volumes and each pattern's 3-D sampling mask; complex or magnitude .npy as the topics registered them",
    slice_of_figure=rec["slice"], files=sum(len(v) for v in man.values()), bytes=total, patterns=man), indent=2))
print(f"DONE {sum(len(v) for v in man.values())} files, {total / 2**30:.2f} GiB, every copy re-hashed", flush=True)
