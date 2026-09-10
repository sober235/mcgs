#!/usr/bin/env python3
"""summarize.py -- collect every method's scores for both contrasts into one table.

Sources:
  logs/classical_<c>.json   ZF / CG-SENSE / CS-SENSE      (scripts/recon_classical.py)
  logs/zsssdu_<c>.json      ZS-SSDU                        (scripts/recon_zsssdu.py)
  logs/dip_<c>.json, logs/inr_<c>.json                     (scripts/recon_dip_inr.py)
  results_arc/t1flair_v3_af16_{joint,single_T1,single_FLAIR}_score.json   (arc/cycle.py)

All of them are on the one gauge (scripts/check_gauge.py proves the shared module reproduces
recon_classical exactly), so the rows are directly comparable.
"""
import json
from pathlib import Path

R = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou")
ARC = Path("/data0/congcong/code/GS/mcgs_af16_wt/results_arc")

COLS = ["psnr", "ssim", "nmse", "hfen", "head_psnr", "head_ssim", "head_hfen",
        "sharp", "grain", "grain_ratio"]
ORDER = ["zf", "cgsense", "cssense", "zsssdu", "dip", "inr", "gs_single", "gs_joint"]
LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense": "CS-SENSE", "zsssdu": "ZS-SSDU",
         "dip": "3D-DIP", "inr": "3D-INR", "gs_single": "MC-GS single", "gs_joint": "MC-GS joint"}


def main():
    rows = []
    for c in ("T1", "FLAIR"):
        for stem in ("classical", "zsssdu", "dip", "inr"):
            p = R / f"logs/{stem}_{c}.json"
            if not p.exists():
                continue
            for r in json.load(open(p)).get("rows", []):
                rows.append({"contrast": c, "method": r["method"],
                             **{k: r.get(k) for k in COLS},
                             "seconds": r.get("seconds")})

    jp = ARC / "t1flair_v3_af16_joint_score.json"
    if jp.exists():
        d = json.load(open(jp))
        for ch, name in (("plain", d.get("plain_name", "T1")), ("contrast", d.get("contrast_name", "FLAIR"))):
            rows.append({"contrast": name, "method": "gs_joint",
                         **{k: d[ch].get(k) for k in COLS}, "seconds": None})
    for c in ("T1", "FLAIR"):
        sp = ARC / f"t1flair_v3_af16_single_{c}_score.json"
        if sp.exists():
            d = json.load(open(sp))
            rows.append({"contrast": c, "method": "gs_single",
                         **{k: d["plain"].get(k) for k in COLS}, "seconds": None})

    rows.sort(key=lambda r: (r["contrast"], ORDER.index(r["method"]) if r["method"] in ORDER else 99))
    hdr = ["contrast", "method"] + COLS
    lines = ["\t".join(hdr)]
    for r in rows:
        vals = [str(r["contrast"]), LABEL.get(r["method"], r["method"])]
        for k in COLS:
            v = r.get(k)
            vals.append("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
        lines.append("\t".join(vals))
    (R / "logs/summary.tsv").write_text("\n".join(lines) + "\n")
    json.dump(rows, open(R / "logs/summary.json", "w"), indent=2)

    for c in ("T1", "FLAIR"):
        sub = [r for r in rows if r["contrast"] == c]
        if not sub:
            continue
        print(f"\n=== {c}   (retrospective AF 16)")
        print(f"{'method':14s} {'PSNR':>7s} {'SSIM':>7s} {'NMSE':>8s} {'HFEN':>7s} "
              f"{'headPSNR':>9s} {'headSSIM':>9s} {'headHFEN':>9s} {'sharp':>6s} {'grain':>8s}")
        for r in sub:
            f = lambda k, w, p: (f"{r[k]:{w}.{p}f}" if isinstance(r.get(k), float) else " " * w)
            print(f"{LABEL.get(r['method'], r['method']):14s} {f('psnr',7,3)} {f('ssim',7,4)} "
                  f"{f('nmse',8,5)} {f('hfen',7,4)} {f('head_psnr',9,3)} {f('head_ssim',9,4)} "
                  f"{f('head_hfen',9,4)} {f('sharp',6,3)} {f('grain',8,5)}")
    print(f"\nwrote {R/'logs/summary.tsv'}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
