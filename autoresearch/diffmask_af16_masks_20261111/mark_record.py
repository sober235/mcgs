#!/usr/bin/env python3
"""mark_record.py -- put the user's record designation of the different-mask experiment (RECORD.md, 2026-11-13) next
to the record figure on the GPU server, and install the updated topic README, in one CPU job.

  gpu.py run --cpu -- @python autoresearch/diffmask_af16_masks_20261111/mark_record.py \
      --dest /data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_diffmask_masks_record_tra205_20261112 \
      [--topic_readme autoresearch/diffmask_af16_masks_20261111/uniform_recon/TOPIC_README.md]

Writes only RECORD.md into --dest, which must be one of this study's record folders and already hold MANIFEST.json;
an existing RECORD.md is replaced only if --replace is given. The topic README goes through run_uniform.py --readme
(it keeps the topic's TABLES block). Nothing else is touched; the files listed in MANIFEST.json are re-hashed and
must still match.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--dest", required=True)
ap.add_argument("--note", default=str(HERE / "RECORD.md"))
ap.add_argument("--topic_readme")
ap.add_argument("--replace", action="store_true")
a = ap.parse_args()

dest = Path(a.dest)
assert dest.parent == Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts") and \
    dest.name.startswith("mcgs_diffmask_masks_record_"), f"not a record folder of this study: {dest}"
man = json.loads((dest / "MANIFEST.json").read_text())
for name, v in man["files"].items():
    h = hashlib.sha256((dest / name).read_bytes()).hexdigest()
    assert h == v["sha256"], f"{name} no longer matches MANIFEST.json"
print(f"{dest}: the {len(man['files'])} files of MANIFEST.json still match")

src = Path(a.note).read_bytes()
out = dest / "RECORD.md"
if out.exists() and out.read_bytes() != src:
    assert a.replace, f"{out} exists with other content (use --replace)"
out.write_bytes(src)
print(f"wrote {out} ({len(src)} bytes, sha256 {hashlib.sha256(src).hexdigest()})")

if a.topic_readme:
    r = subprocess.run([sys.executable, str(HERE / "run_uniform.py"), "--readme", a.topic_readme], check=True)
print("DONE")
