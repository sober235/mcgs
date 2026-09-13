#!/usr/bin/env python3
"""summarize.py -- collect every method's scores for both contrasts into one table.

Sources:
  logs/classical_<c>.json   ZF / CG-SENSE / CS-SENSE      (scripts/recon_classical.py)
  logs/zsssdu_<c>.json      ZS-SSDU                        (scripts/recon_zsssdu.py)
  logs/dip_<c>.json, logs/inr_<c>.json                     (scripts/recon_dip_inr.py)
  results_arc/peds_af12_{joint,single_T1w,single_T2s}_score.json   (arc/cycle.py)

All of them are on the one gauge (scripts/check_gauge.py proves the shared module reproduces
recon_classical exactly), so the rows are directly comparable.
"""
import json
from pathlib import Path

R = Path("/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro")
ARC = Path("/data0/congcong/code/GS/mcgs_peds_wt/results_arc")

COLS = ["psnr", "ssim", "nmse", "hfen", "head_psnr", "head_ssim", "head_hfen",
        "sharp", "grain", "grain_ratio"]
ORDER = ["zf", "cgsense", "cssense", "cssense_o200", "jcssense", "l1espirit", "zsssdu", "dip", "inr", "gs_single", "gs_joint"]
LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense": "CS-SENSE", "zsssdu": "ZS-SSDU",
         "dip": "3D-DIP", "inr": "3D-INR", "gs_single": "MC-GS single", "gs_joint": "MC-GS joint",
         "cssense_o200": "CS-SENSE (TV, converged)", "jcssense": "joint CS-SENSE (JTV)", "l1espirit": "L1-ESPIRiT"}


def main():
    rows = []
    for c in ("T1w", "T2s"):
        for stem in ("classical", "zsssdu", "dip", "inr"):
            p = R / f"logs/{stem}_{c}.json"
            if not p.exists():
                continue
            for r in json.load(open(p)).get("rows", []):
                rows.append({"contrast": c, "method": r["method"],
                             **{k: r.get(k) for k in COLS},
                             "seconds": r.get("seconds")})

    # classical arms revisited (2026-09-10, ported from the AF12 topics' audit): converged TV (200 outer
    # ADMM), L1-wavelet ESPIRiT (BART pics, 100 FISTA) and joint vectorial TV (200 outer), each with
    # lambda SWEPT ON THIS SUBJECT against the ground truth -- the same privilege the MC-GS
    # autoresearch loop has -- and the best lambda per contrast kept (scripts/run_classical_revisited.sh).
    FAM = [("cssense_o200", "logs/classical_{c}_o200_lam*.json"),
           ("l1espirit", "logs/l1espirit_{c}_lam*.json"),
           ("jcssense", "logs/jcssense_o200_lam*.json")]
    for c in ("T1w", "T2s"):
        for key, pat in FAM:
            cands = []
            for f in sorted(R.glob(pat.format(c=c))):
                for r in json.load(open(f)).get("rows", []):
                    if r.get("contrast", c) == c and r.get("head_psnr") is not None:
                        cands.append(r)
            if cands:
                b = max(cands, key=lambda r: r["head_psnr"])
                rows.append({"contrast": c, "method": key, **{k: b.get(k) for k in COLS},
                             "seconds": b.get("seconds"), "lam": b.get("lam"), "n_lambda": len(cands),
                             "lams": sorted(float(r.get("lam")) for r in cands if r.get("lam") is not None)})

    jp = ARC / "peds_af12_joint_score.json"
    if jp.exists():
        d = json.load(open(jp))
        for ch, name in (("plain", d.get("plain_name", "T1w")), ("contrast", d.get("contrast_name", "T2s"))):
            rows.append({"contrast": name, "method": "gs_joint",
                         **{k: d[ch].get(k) for k in COLS}, "seconds": None})
    for c in ("T1w", "T2s"):
        sp = ARC / f"peds_af12_single_{c}_score.json"
        if sp.exists():
            d = json.load(open(sp))
            rows.append({"contrast": c, "method": "gs_single",
                         **{k: d["plain"].get(k) for k in COLS}, "seconds": None})

    # extra MC-GS arms from the autoresearch loop (scripts/add_gs_arm.py): metrics straight from the
    # run's own score json in the worktree, so the row is exactly what its scorer printed
    xp = R / "logs/extra_gs_arms.json"
    if xp.exists():
        for arm in json.load(open(xp)):
            for c, ch in arm["map"].items():
                exp_c = arm.get("exps", {}).get(c, arm["exp"])
                d = json.load(open(ARC / f"{exp_c}_score.json"))
                rows.append({"contrast": c, "method": arm["key"], **{k: d[ch].get(k) for k in COLS},
                             "seconds": d.get("elapsed_s"), "exp": exp_c, "iters": d.get("iters")})
            LABEL.setdefault(arm["key"], arm["label"])
            if arm["key"] not in ORDER:
                ORDER.append(arm["key"])

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

    for c in ("T1w", "T2s"):
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
