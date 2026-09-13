#!/usr/bin/env python3
"""Append one experiment row to autoresearch_peds.tsv from a cycle.py score json.

Margins are against the best NON-GS arm of this topic's nine-method table
(Results_20260908/Peds_TSC_MTP_AF12_retro/logs/summary.json), so the numbers cannot drift from the
archive. Before that table exists the margins are blank. The row is what the ledger reads; the score
json keeps everything else.

Usage: autoresearch_jyz_af16_row.py --exp NAME [--status keep|discard|crash|pending] [--note TEXT]
"""
import argparse, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "autoresearch_peds.tsv"
TOPIC = Path("/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro")
GS = {"gs_single", "gs_joint", "MC-GS single", "MC-GS joint"}
COLS = ["exp", "status", "iters", "npts", "vram_gb", "min", "T1w_head_psnr", "T2s_head_psnr", "dT1w", "dT2s",
        "min_margin", "T1w_head_ssim", "T2s_head_ssim", "T1w_head_hfen", "T2s_head_hfen", "T1w_psnr", "T2s_psnr",
        "T1w_sharp", "T2s_sharp", "T1w_N", "T2s_N", "commit", "args", "note"]


def best_competitors():
    if not (TOPIC / "logs/summary.json").exists():
        return {c: dict(head_psnr=float("nan"), head_ssim=float("nan"), head_hfen=float("nan"),
                        psnr=float("nan"), who="(no table yet)") for c in ("T1w", "T2s")}
    rows = json.load(open(TOPIC / "logs/summary.json"))
    best = {}
    for c in ("T1w", "T2s"):
        # every non-MC-GS arm counts as a competitor; any MC-GS variant (gs_*, "MC-GS ...") never does
        sub = [r for r in rows if r["contrast"] == c and r["method"] not in GS
               and not str(r["method"]).startswith(("gs_", "MC-GS"))]
        best[c] = dict(head_psnr=max(r["head_psnr"] for r in sub), head_ssim=max(r["head_ssim"] for r in sub),
                       head_hfen=min(r["head_hfen"] for r in sub), psnr=max(r["psnr"] for r in sub),
                       who=max(sub, key=lambda r: r["head_psnr"])["method"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--status", default="pending")
    ap.add_argument("--note", default="")
    ap.add_argument("--print_best", action="store_true")
    a = ap.parse_args()
    BEST = best_competitors()
    if a.print_best:
        print(json.dumps(BEST, indent=1)); return
    sj = ROOT / "results_arc" / f"{a.exp}_score.json"
    if not sj.exists():
        row = {c: "" for c in COLS}
        row.update(exp=a.exp, status="crash", note=a.note or "no score json")
    else:
        d = json.loads(sj.read_text())
        # map channels by NAME: a joint run is (T1w, T2s); a single-contrast run has both channels the
        # same contrast (plain is used) and leaves the other contrast blank
        ch = {}
        for c in ("plain", "contrast"):
            nm = d.get(f"{c}_name", {"plain": "T1w", "contrast": "T2s"}[c])
            ch.setdefault(nm, d[c])
        p1, c1 = ch.get("T1w"), ch.get("T2s")
        dT1 = p1["head_psnr"] - BEST["T1w"]["head_psnr"] if p1 else None
        dFL = c1["head_psnr"] - BEST["T2s"]["head_psnr"] if c1 else None
        mm = min(x for x in (dT1, dFL) if x is not None)
        fmt = lambda r, k, f: (f"{r[k]:{f}}" if r else "")
        sgn = lambda x: (f"{x:+.3f}" if x is not None else "")
        try:
            commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            commit = "?"
        row = dict(exp=a.exp, status=a.status, iters=d["iters"], npts=d["npts"], vram_gb=f"{d['vram_gb']:.1f}",
                   min=f"{(d.get('elapsed_s') or 0) / 60:.0f}",
                   T1w_head_psnr=fmt(p1, "head_psnr", ".3f"), T2s_head_psnr=fmt(c1, "head_psnr", ".3f"),
                   dT1w=sgn(dT1), dT2s=sgn(dFL), min_margin=f"{mm:+.3f}",
                   T1w_head_ssim=fmt(p1, "head_ssim", ".4f"), T2s_head_ssim=fmt(c1, "head_ssim", ".4f"),
                   T1w_head_hfen=fmt(p1, "head_hfen", ".4f"), T2s_head_hfen=fmt(c1, "head_hfen", ".4f"),
                   T1w_psnr=fmt(p1, "psnr", ".3f"), T2s_psnr=fmt(c1, "psnr", ".3f"),
                   T1w_sharp=fmt(p1, "sharp", ".3f"), T2s_sharp=fmt(c1, "sharp", ".3f"),
                   T1w_N=fmt(p1, "grain_ratio", ".3f"), T2s_N=fmt(c1, "grain_ratio", ".3f"),
                   commit=commit, args=" ".join(d.get("argv", [])),
                   note=(a.note + f" bar={BEST['T1w']['who']}:{BEST['T1w']['head_psnr']:.3f}/"
                         f"{BEST['T2s']['who']}:{BEST['T2s']['head_psnr']:.3f}").strip())
    if not TSV.exists():
        TSV.write_text("\t".join(COLS) + "\n")
    with open(TSV, "a") as f:
        f.write("\t".join(str(row[k]) for k in COLS) + "\n")
    print("\t".join(f"{k}={row[k]}" for k in COLS[:11]))


if __name__ == "__main__":
    main()
