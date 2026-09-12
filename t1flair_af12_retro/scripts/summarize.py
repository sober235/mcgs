#!/usr/bin/env python3
"""summarize.py -- collect every method's scores for both contrasts into one table.

Sources:
  logs/classical_<c>.json   ZF / CG-SENSE / CS-SENSE      (scripts/recon_classical.py)
  logs/zsssdu_<c>.json      ZS-SSDU                        (scripts/recon_zsssdu.py)
  logs/dip_<c>.json, logs/inr_<c>.json                     (scripts/recon_dip_inr.py)
  results_arc/t1flair_af12_{joint,single_T1,single_FLAIR}_score.json   (arc/cycle.py)

All of them are on the one gauge (scripts/check_gauge.py proves the shared module reproduces
recon_classical exactly), so the rows are directly comparable.
"""
import json
from pathlib import Path

R = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_retro")
ARC = Path("/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_arc")

COLS = ["psnr", "ssim", "nmse", "hfen", "head_psnr", "head_ssim", "head_hfen",
        "sharp", "grain", "grain_ratio"]
ORDER = ["zf", "cgsense", "cssense", "cssense_o200", "jcssense", "l1espirit", "ploraks", "zsssdu", "dip", "inr", "gs_single", "gs_joint", "gs_single_v2", "gs_joint_v2", "gs_single_v3", "gs_joint_v3", "gs_joint_v3x2", "gs_joint_v32", "gs_joint_v32x2"]
LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense": "CS-SENSE", "zsssdu": "ZS-SSDU",
         "dip": "3D-DIP", "inr": "3D-INR", "gs_single": "MC-GS single", "gs_joint": "MC-GS joint",
         "gs_single_v2": "MC-GS single v2", "gs_joint_v2": "MC-GS joint v2",
         "cssense_o200": "CS-SENSE (TV, converged)", "jcssense": "joint CS-SENSE (JTV)",
         "l1espirit": "L1-ESPIRiT", "ploraks": "P-LORAKS",
         "gs_single_v3": "MC-GS single v3", "gs_joint_v3": "MC-GS joint v3", "gs_joint_v3x2": "MC-GS joint v3 x2 seeds", "gs_joint_v32": "MC-GS joint v3.2", "gs_joint_v32x2": "MC-GS joint v3.2 x2 seeds"}


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

    jp = ARC / "t1flair_af12_joint_score.json"
    if jp.exists():
        d = json.load(open(jp))
        for ch, name in (("plain", d.get("plain_name", "T1")), ("contrast", d.get("contrast_name", "FLAIR"))):
            rows.append({"contrast": name, "method": "gs_joint",
                         **{k: d[ch].get(k) for k in COLS}, "seconds": None})
    # v2 = autoresearch recipe (framework repo autoresearch_t1t2.md), transfer check, 4000 iters
    vp = ARC / "ar5_tf_hyw_t1flair_score.json"
    if vp.exists():
        d = json.load(open(vp))
        for ch, name in (("plain", d.get("plain_name", "T1")), ("contrast", d.get("contrast_name", "FLAIR"))):
            rows.append({"contrast": name, "method": "gs_joint_v2",
                         **{k: d[ch].get(k) for k in COLS}, "seconds": d.get("elapsed_s")})
    # single-contrast arms of the v2 recipe (ablation)
    for exp, c in (("ar6_single_T1_v2", "T1"), ("ar9_single_FLAIR_v2_hyw", "FLAIR")):
        sp = ARC / f"{exp}_score.json"
        if sp.exists():
            d = json.load(open(sp))
            rows.append({"contrast": c, "method": "gs_single_v2",
                         **{k: d["plain"].get(k) for k in COLS}, "seconds": d.get("elapsed_s")})
    for c in ("T1", "FLAIR"):
        sp = ARC / f"t1flair_af12_single_{c}_score.json"
        if sp.exists():
            d = json.load(open(sp))
            rows.append({"contrast": c, "method": "gs_single",
                         **{k: d["plain"].get(k) for k in COLS}, "seconds": None})

    # classical arms revisited (2026-09-10): converged TV (200 outer), joint vectorial TV (200 outer),
    # L1-wavelet ESPIRiT (BART, 100 FISTA iterations) at the lambdas selected on huangyuwei, P-LORAKS.
    LAM_L1 = {'T1': '0.0015', 'FLAIR': '0.005'}
    for c in ['T1', 'FLAIR']:
        for stem, method, key in ((f"classical_{c}_o200", "cssense_o200", None),
                                  (f"l1espirit_{c}_lam{LAM_L1[c]}", "l1espirit", None),
                                  (f"ploraks_{c}", "ploraks", None)):
            p = R / f"logs/{stem}.json"
            if p.exists():
                for r in json.load(open(p)).get("rows", []):
                    rows.append({"contrast": c, "method": method, **{k: r.get(k) for k in COLS}, "seconds": r.get("seconds")})
    p = R / "logs/jcssense_o200.json"
    if p.exists():
        for r in json.load(open(p)).get("rows", []):
            rows.append({"contrast": r["contrast"], "method": "jcssense", **{k: r.get(k) for k in COLS}, "seconds": r.get("seconds")})
    # v3 = v2 + 2.4M points + wavelet sparsity 0.05 (framework repo autoresearch_t1t2.md, rounds 11-14), 4000 iters
    p = ARC / "ar13_tf_hyw_t1flair_v3b_score.json"
    if p.exists():
        d = json.load(open(p))
        for ch, name in (("plain", d.get("plain_name", "T1")), ("contrast", d.get("contrast_name", "FLAIR"))):
            rows.append({"contrast": name, "method": "gs_joint_v3", **{k: d[ch].get(k) for k in COLS}, "seconds": d.get("elapsed_s")})
    for c, e in {'T1': 'ar15_single_T1_n24_wav005_full4k', 'FLAIR': 'ar15_single_FLAIR_hyw_v3'}.items():
        p = ARC / f"{e}_score.json"
        if p.exists():
            d = json.load(open(p))
            rows.append({"contrast": c, "method": "gs_single_v3", **{k: d["plain"].get(k) for k in COLS}, "seconds": d.get("elapsed_s")})
    # v3 x2 = voxelwise mean of the magnitudes of two v3 runs that differ only in the seed (42, 43); twice the compute (program Round 18)
    p = ARC / "ar13_tf_hyw_t1flair_v3b_x2_score.json"
    if p.exists():
        d = json.load(open(p))
        for ch, name in (("plain", d.get("plain_name", "T1")), ("contrast", d.get("contrast_name", "FLAIR"))):
            rows.append({"contrast": name, "method": "gs_joint_v3x2", **{k: d[ch].get(k) for k in COLS}, "seconds": d.get("elapsed_s")})
    # v3.2 = v3 + T1 wavelet 0.06 + edge-TV 0.07 (+ FLAIR wavelet 0.12 on T1+FLAIR); framework program "## v3.2 — adopted"
    for exp, meth in (("ar19_tf_hyw_full_C012", "gs_joint_v32"), ("ar19_tf_hyw_full_C012_x2", "gs_joint_v32x2")):
        p = ARC / f"{exp}_score.json"
        if p.exists():
            d = json.load(open(p))
            for ch, name in (("plain", d.get("plain_name", "T1")), ("contrast", d.get("contrast_name", "FLAIR"))):
                rows.append({"contrast": name, "method": meth, **{k: d[ch].get(k) for k in COLS}, "seconds": d.get("elapsed_s")})
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
        print(f"\n=== {c}   (retrospective AF 12)")
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
