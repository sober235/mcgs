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
plus the json whose row has method == KEY. ARM may also be 'spirit': the improved run (e.g. a coil-compressed one,
spirit_vc24_ks5_lam0.001) is installed under the registered family name of its kernel and lambda
(recon/spirit_ks5_lam0.001_<scan>_complex.npy, logs/spirit_<scan>_ks5_lam0.001.json, method spirit_ks5_lam0.001), so
summarize.py's per-family maximum and arms.py's file rule see it unchanged. The cssense_o200 / jcssense families need no
--replace (their improved jsons already carry the family names).
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
             "inr": "inr_{scan}.json", "mcinr": "mcinr_{scan}.json",
             "gs_joint": "gs_joint_{scan}.json", "gs_single": "gs_single_{scan}.json"}   # the MC-GS / GS-single of record (the _ar2 winners)


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
    assert arm in STEM_JSON or arm == "spirit", f"unknown arm {arm} (families are chosen by summarize.py, see the docstring)"
    vol = src / "recon" / f"{key}_{scan}_complex.npy"
    assert vol.exists(), vol
    repl[(arm, scan)] = (key, vol, find_row(key, scan))

out.mkdir(parents=True, exist_ok=False)
for d in ("configs", "scripts", "sim", "trainer"):
    os.symlink(src / d, out / d)
(out / "recon").mkdir(); (out / "logs").mkdir(); (out / "figures").mkdir()
def reg_name(arm, scan, row):
    """the registered volume stem an improved run replaces: <arm>_<scan> for the single arms, spirit_ks<k>_lam<l>_<scan> for L1-SPIRiT"""
    return f"spirit_ks{row['ksize']}_lam{row['lam']:g}_{scan}" if arm == "spirit" else f"{arm}_{scan}"


replaced_names = {reg_name(arm, scan, row) + "_complex.npy" for (arm, scan), (key, vol, (js, row)) in repl.items()}
for f in sorted((src / "recon").iterdir()):
    if (f.is_file() or f.is_symlink()) and f.name not in replaced_names:
        os.symlink(f.resolve(), out / "recon" / f.name)
for (arm, scan), (key, vol, (js, row)) in repl.items():
    os.symlink(vol.resolve(), out / "recon" / (reg_name(arm, scan, row) + "_complex.npy"))
for f in sorted((src / "logs").glob("*.json")):
    shutil.copy2(f, out / "logs" / f.name)
for d in ("orch", "ar"):
    if (src / "logs" / d).exists():
        os.symlink(src / "logs" / d, out / "logs" / d)
for (arm, scan), (key, vol, (js, row)) in repl.items():
    if arm == "spirit":
        fam = f"spirit_ks{row['ksize']}_lam{row['lam']:g}"
        jpath = out / "logs" / f"spirit_{scan}_ks{row['ksize']}_lam{row['lam']:g}.json"
        doc = json.load(open(jpath)) if jpath.exists() else {"contrast": scan, "rows": [dict(row)]}
        nr = dict(row); nr["method"] = fam; nr["improved_from"] = key; nr["improved_json"] = str(js)
        doc["rows"] = [nr]; doc["improved"] = {fam: key}
        for k in ("lam", "ksize", "tyk", "iters", "levels", "solver", "calib", "vcoils"):
            if k in nr:
                doc[k] = nr[k]
        json.dump(doc, open(jpath, "w"), indent=2)
        continue
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
