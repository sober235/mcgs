#!/usr/bin/env python3
"""fig_af12_prepost.py -- the plain-scan / enhanced figure of record, drawn for the AF12 rerun (2026-11-13).

The record figure (user decision 2026-11-11) is made by ../ce_af16_lesion_figs_20261107/fig_lesion_plane_prepost.py
in topic mode (--topic, --three 201 131 48, --err_layout zoom, --cbar single). That script names the AF16 mask file,
the AF16 label and an "ce_af16" file stem in three places. This wrapper reads the script, replaces exactly those
texts (each must occur exactly once, otherwise it stops), and runs it with the given arguments, so the drawing,
the panel arithmetic and the checks are the record script's own. The script file itself is not edited.

Only --topic = the AF12 topic is accepted. After the drawing, the PNG / PDF / preview / record JSON are also copied
to MCGS_ARTIFACT_DIR so that gpu.py fetch brings them to the Mac.

  gpu.py run --cpu -- @python autoresearch/ce_af12_rerun_20261112/fig_af12_prepost.py \
      --topic /data0/code/.../CE_retro_cssense_af12 --label "..." --out DIR --three 201 131 48 \
      --err_layout zoom --cbar single [--pdf --pdf_dpi 600]
"""
import hashlib
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "ce_af16_lesion_figs_20261107" / "fig_lesion_plane_prepost.py"
TOPIC12 = "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af12"
PATCH = [
    ('with h5py.File(TOPIC / "sim/mask/mask_af16_3d.h5", "r") as f:',
     'with h5py.File(TOPIC / "sim/mask/mask_af12_3d.h5", "r") as f:'),
    ('AF_LABEL = f"nominal AF 16, actual {NPL / NSAMP:.2f}x"',
     'AF_LABEL = f"nominal AF 12, actual {NPL / NSAMP:.2f}x"'),
    ('f"ce_af16_lesion_three_planes_{t}_{c_}_{s_}_prepost_{SAFE}"',
     'f"ce_af12_lesion_three_planes_{t}_{c_}_{s_}_prepost_{SAFE}"'),
    ('f"ce_af16_lesion_{ANAT[axis].lower()}{idx}_prepost_{SAFE}"',
     'f"ce_af12_lesion_{ANAT[axis].lower()}{idx}_prepost_{SAFE}"'),
]

argv = sys.argv[1:]
assert "--topic" in argv and argv[argv.index("--topic") + 1].rstrip("/") == TOPIC12, f"--topic must be {TOPIC12}"
assert "--out" in argv, "--out is required"
out = Path(argv[argv.index("--out") + 1])
text = SRC.read_text()
print(f"fig_af12_prepost: {SRC.name} sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}", flush=True)
for old, new in PATCH:
    assert text.count(old) == 1, f"{SRC.name}: expected exactly one occurrence of\n{old}\nfound {text.count(old)}"
    text = text.replace(old, new)
assert "mask_af16" not in text and "nominal AF 16" not in text and "ce_af16_lesion" not in text
sys.argv = [str(SRC), *argv]
before = set(out.glob("*")) if out.exists() else set()
exec(compile(text, str(SRC), "exec"), {"__name__": "__main__", "__file__": str(SRC)})
if "--save_data" in argv:      # keep the code that drew the figure beside its data
    (out / "fig_lesion_plane_prepost_as_run_af12.py").write_text(text)
    shutil.copy2(__file__, out / Path(__file__).name)
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", ""))
if art.is_dir():
    for p in sorted(set(out.glob("*")) - before):
        if p.suffix in (".png", ".pdf", ".jpg", ".json"):
            shutil.copy2(p, art / p.name)
            print(f"artifact {p.name} ({p.stat().st_size} bytes)", flush=True)
