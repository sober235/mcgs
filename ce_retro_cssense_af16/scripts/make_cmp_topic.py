#!/usr/bin/env python3
"""make_cmp_topic.py -- build a confirmation topic for the comparison-arm improvement loop (2026-10-09), the way the
autoresearch _ar / _ar2 topics were built: configs / scripts / sim / trainer are symlinks to the source topic, recon/ holds
one symlink per file of the source recon/, logs/ holds copies of every per-arm score json -- except that the arms named
in --replace point at their improved runs (the registered volume name links to the improved volume; the registered json
row is rewritten from the improved run's row with the registered method key, so summarize.py / arms.py need no change).
Nothing in the source topic is touched; the output directory must not exist yet.

  python scripts/make_cmp_topic.py --out /data0/.../CE_retro_cssense_af16_vdpois_cmp \
      --replace dip:pre=dip_d2_b96_sup6_lr1e2_it8k --replace inr:post=inr_i1_n256_sup6 ...

--replace ARM:SCAN=KEY: ARM is a registered stem with its own json (cgsense -> logs/classical_<scan>.json row 'cgsense';
zsssdu, dip, inr, mcinr -> logs/<stem>_<scan>.json) and KEY the improved run's method key, i.e. recon/<KEY>_<scan>_complex.npy
plus the json whose row has method == KEY. The swept families (cssense_o200, jcssense, spirit) need no --replace: their
improved jsons are copied in with everyone else and summarize.py keeps the best head PSNR per family and scan -- but only
if the family glob in summarize.py matches the new file names; check logs/summary.json afterwards.
"""
import argparse, json, os, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois")
ap.add_argument("--out", required=True)
ap.add_argument("--replace", action="append", default=[], help="ARM:SCAN=KEY (see the docstring)")
ap.add_argument("--note", default="", help="one line for README_CMP.txt")
a = ap.parse_args()
src, out = Path(a.src), Path(a.out)
STEM_JSON = {"cgsense": "classical_{scan}.json", "zsssdu": "zsssdu_{scan}.json", "dip": "dip_{scan}.json",
             "inr": "inr_{scan}.json", "mcinr": "mcinr_{scan}.json"}


def find_row(key, scan):
    """the (json path, row) of the improved run: its row has method == key and contrast == scan."""
    for js in sorted((src / "logs").glob("*.json")):
        try:
            doc = json.load(open(js))
        except Exception:
            continue
        for r in doc.get("rows", []):
            if r.get("method") == key and r.get("contrast", scan) == scan:
                return js, r
    raise SystemExit(f"no score row with method {key} for {scan} under {src / 'logs'}")


repl = {}
for spec in a.replace:
    arm_scan, key = spec.split("=")
    arm, scan = arm_scan.split(":")
    assert arm in STEM_JSON, f"unknown arm {arm} (families are chosen by summarize.py, see the docstring)"
    vol = src / "recon" / f"{key}_{scan}_complex.npy"
    assert vol.exists(), vol
    repl[(arm, scan)] = (key, vol, find_row(key, scan))

out.mkdir(parents=True, exist_ok=False)
for d in ("configs", "scripts", "sim", "trainer"):
    os.symlink(src / d, out / d)
(out / "recon").mkdir(); (out / "logs").mkdir(); (out / "figures").mkdir()
replaced_names = {f"{arm}_{scan}_complex.npy" for (arm, scan) in repl}
for f in sorted((src / "recon").iterdir()):
    if (f.is_file() or f.is_symlink()) and f.name not in replaced_names:
        os.symlink(f.resolve(), out / "recon" / f.name)
for (arm, scan), (key, vol, _) in repl.items():
    os.symlink(vol.resolve(), out / "recon" / f"{arm}_{scan}_complex.npy")
for f in sorted((src / "logs").glob("*.json")):
    shutil.copy2(f, out / "logs" / f.name)
for d in ("orch", "ar"):
    if (src / "logs" / d).exists():
        os.symlink(src / "logs" / d, out / "logs" / d)
for (arm, scan), (key, vol, (js, row)) in repl.items():
    jpath = out / "logs" / STEM_JSON[arm].format(scan=scan)
    doc = json.load(open(jpath))
    new_rows = []
    for r in doc["rows"]:
        if r["method"] == arm:
            nr = dict(row); nr["method"] = arm; nr["improved_from"] = key; nr["improved_json"] = str(js); new_rows.append(nr)
        else:
            new_rows.append(r)
    assert any(r.get("improved_from") == key for r in new_rows), (jpath, arm)
    doc["rows"] = new_rows; doc.setdefault("improved", {})[arm] = key
    json.dump(doc, open(jpath, "w"), indent=2)
(out / "README_CMP.txt").write_text(
    f"{out.name}: the vdPoisMex line with improved comparison arms (2026-10-09 audit, CMP_ARMS_AUDIT.md). "
    f"Replacements: {', '.join(a.replace) or 'none'}. {a.note}\nEverything else is a symlink into {src}.\n")
print(f"built {out} with {len(repl)} replacement(s): {a.replace}")
