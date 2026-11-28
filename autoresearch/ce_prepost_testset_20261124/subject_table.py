#!/usr/bin/env python3
"""subject_table.py -- (CPU) the eleven-method table of one test subject at AF12, from the finished result files.

Sources and rules (nothing is rescored here):
  topic_af12/logs          the files run_af12.py's table reads, with its rule: a swept arm (CS-SENSE, Joint SENSE,
                           L1-SPIRiT) is its best lambda / kernel by head PSNR; an optimum on a grid end is flagged.
  topic_af12/logs/gs_joint_tuned_<scan>.json   MC-GS, tuned final config (the MC-GS row of the table).
  topic_af12/logs/gs_joint_<scan>.json         MC-GS, registered recipe (kept as an extra row, not a comparator).
  cmp_support/logs/record_<dip|inr>_<scan>.json  3D-DIP / 3D-INR with output support (the optimal figure's versions).

  gpu.py run --cpu -- @python autoresearch/ce_prepost_testset_20261124/subject_table.py --subjects A,B,C
Writes table_<subject>.md / .json to <subject>/tables/ and to the artifact dir; --all_md adds table_all.md.
No clock value and no measured duration is read or written (the "seconds" fields of the result files are dropped).
"""
import argparse
import json
import os
from pathlib import Path

OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_prepost_testset_20261124")
ART = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
SCANS = ("pre", "post")
NAME = {"pre": "plain scan", "post": "enhanced"}
ORDER = ["zf", "cgsense", "cssense_o200", "jcssense", "spirit", "zsssdu", "dip", "inr", "mcinr", "gs_single", "gs_joint_reg",
         "gs_joint"]
LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense_o200": "CS-SENSE", "jcssense": "Joint SENSE", "spirit": "L1-SPIRiT",
         "zsssdu": "ZS-SSDU", "dip": "3D-DIP (support)", "inr": "3D-INR (support)", "mcinr": "MC-INR",
         "gs_single": "GS-single", "gs_joint_reg": "MC-GS, registered recipe (extra)", "gs_joint": "MC-GS (tuned final config)"}
KEYS = ("head_psnr", "head_ssim", "head_nmse", "head_hfen")
LOWER = {"head_psnr": False, "head_ssim": False, "head_nmse": True, "head_hfen": True}


def rows_of(p, c):
    return [dict(r, contrast=c) for r in json.load(open(p)).get("rows", []) if r.get("contrast", c) == c]


def collect(subj):
    L = OUT / "subjects" / subj / "topic_af12/logs"
    Ls = OUT / "subjects" / subj / "cmp_support/logs"
    rows = []
    for c in SCANS:
        for stem in ("classical", "zsssdu", "mcinr", "gs_single"):
            p = L / f"{stem}_{c}.json"
            if p.exists():
                rows += [r for r in rows_of(p, c) if r["method"] in ("zf", "cgsense", "zsssdu", "mcinr", "gs_single")]
        for key, pat in (("cssense_o200", f"classical_{c}_o200_lam*.json"), ("jcssense", "jcssense_o200_lam*.json"),
                         ("spirit", f"spirit_{c}_ks*_lam*.json")):
            cands = [r for f in sorted(L.glob(pat)) for r in rows_of(f, c) if r.get("head_psnr") is not None]
            if cands:
                b = max(cands, key=lambda r: r["head_psnr"])
                lams = sorted({float(r["lam"]) for r in cands if r.get("lam") is not None})
                rows.append(dict(b, method=key, lams=lams, edge=float(b.get("lam", -1)) in (lams[0], lams[-1])))
        for key, stem in (("gs_joint_reg", "gs_joint"), ("gs_joint", "gs_joint_tuned")):
            p = L / f"{stem}_{c}.json"
            if p.exists():
                rows += [dict(r, method=key) for r in rows_of(p, c)[:1]]
        for m in ("dip", "inr"):
            p = Ls / f"record_{m}_{c}.json"
            if p.exists():
                d = json.load(open(p))
                rows.append(dict(d["score"], contrast=c, method=m, variant=d["score"]["method"]))
    for r in rows:
        r.pop("seconds", None)
    return rows


def table(subj):
    rows = collect(subj)
    meta = json.load(open(OUT / "subjects" / subj / "topic_af12/logs/setup.json"))["summary"]
    out = [f"## {subj} — AF12 ({meta['points']} points on {meta['grid'][1]} x {meta['grid'][2]}, {meta['coils']} coils; "
           f"truth lambda {meta['lam']['pre']:g} / {meta['lam']['post']:g})"]
    summ = dict(subject=subj, setup=meta, rows=rows, standing={})
    for c in SCANS:
        sub = sorted((r for r in rows if r["contrast"] == c), key=lambda r: ORDER.index(r["method"]))
        out += [f"\n### {NAME[c]}\n", "| method | head PSNR (dB) | head SSIM | head NMSE | head HFEN | note |", "|---|---|---|---|---|---|"]
        comp = [r for r in sub if r["method"] not in ("gs_joint", "gs_joint_reg")]
        mc = next((r for r in sub if r["method"] == "gs_joint"), None)
        best = {k: (min if LOWER[k] else max)(sub, key=lambda r: r[k])["method"] for k in KEYS} if sub else {}
        for r in sub:
            note = ""
            if r.get("lams"):
                note = f"lam {float(r['lam']):g}" + (f", kernel {r['ksize']}" if r.get("ksize") else "") + (" (GRID EDGE)" if r.get("edge") else "")
            cell = lambda k, f: (f"**{r[k]:{f}}**" if best[k] == r["method"] else f"{r[k]:{f}}")
            out.append(f"| {LABEL[r['method']]} | {cell('head_psnr', '.3f')} | {cell('head_ssim', '.4f')} | {cell('head_nmse', '.5f')} | "
                       f"{cell('head_hfen', '.4f')} | {note} |")
        if mc and comp:
            st = {}
            for k in KEYS:
                b = (min if LOWER[k] else max)(comp, key=lambda r: r[k])
                d = (b[k] - mc[k]) if LOWER[k] else (mc[k] - b[k])
                st[k] = dict(mcgs=mc[k], best_other=b["method"], best_other_value=b[k], margin=d, first=bool(d > 0))
                out.append(f"\nMC-GS {k}: {mc[k]:.4f}; best of the ten others {LABEL[b['method']]} {b[k]:.4f}; margin {d:+.4f} "
                           f"({'first' if d > 0 else 'NOT first'})")
            summ["standing"][c] = st
        missing = [LABEL[m] for m in ORDER if not any(r["method"] == m for r in sub)]
        out.append(f"\nmethods without a result: {missing or 'none'}")
        summ.setdefault("missing", {})[c] = missing
    text = "\n".join(out) + "\n"
    for d in (OUT / "subjects" / subj / "tables", ART):
        d.mkdir(parents=True, exist_ok=True)
        (d / f"table_{subj}.md").write_text(text)
        json.dump(summ, open(d / f"table_{subj}.json", "w"), indent=1)
    print(text, flush=True)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", required=True)
    a = ap.parse_args()
    texts = [table(s) for s in a.subjects.split(",")]
    (ART / "table_all.md").write_text("\n".join(texts))
