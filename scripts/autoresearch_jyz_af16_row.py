#!/usr/bin/env python3
"""Append one experiment row to autoresearch_jyz_af16.tsv from a cycle.py score json.

Margins are against the best NON-MC-GS arm of the eight-method table of
Results_20260908/T1FLAIR_AF16_v3_jiangyizhou (head-only PSNR: CS-SENSE 34.409 on T1, ZS-SSDU
34.005 on FLAIR), read from that topic's logs/summary.json so the numbers cannot drift from the
archive. The row is what the loop reads; the score json keeps everything else.

Usage: autoresearch_jyz_af16_row.py --exp NAME [--status keep|discard|crash|pending] [--note TEXT]
"""
import argparse, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "autoresearch_jyz_af16.tsv"
TOPIC = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou")
GS = {"gs_single", "gs_joint", "MC-GS single", "MC-GS joint"}
COLS = ["exp", "status", "iters", "npts", "vram_gb", "min", "T1_head_psnr", "FLAIR_head_psnr", "dT1", "dFLAIR",
        "min_margin", "T1_head_ssim", "FLAIR_head_ssim", "T1_head_hfen", "FLAIR_head_hfen", "T1_psnr", "FLAIR_psnr",
        "T1_sharp", "FLAIR_sharp", "T1_N", "FLAIR_N", "commit", "args", "note"]


def best_competitors():
    rows = json.load(open(TOPIC / "logs/summary.json"))
    best = {}
    for c in ("T1", "FLAIR"):
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
        p, c = d["plain"], d["contrast"]
        assert d.get("plain_name", "T1") == "T1" and d.get("contrast_name", "FLAIR") == "FLAIR", (d.get("plain_name"), d.get("contrast_name"))
        dT1 = p["head_psnr"] - BEST["T1"]["head_psnr"]
        dFL = c["head_psnr"] - BEST["FLAIR"]["head_psnr"]
        try:
            commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            commit = "?"
        row = dict(exp=a.exp, status=a.status, iters=d["iters"], npts=d["npts"], vram_gb=f"{d['vram_gb']:.1f}",
                   min=f"{(d.get('elapsed_s') or 0) / 60:.0f}",
                   T1_head_psnr=f"{p['head_psnr']:.3f}", FLAIR_head_psnr=f"{c['head_psnr']:.3f}",
                   dT1=f"{dT1:+.3f}", dFLAIR=f"{dFL:+.3f}", min_margin=f"{min(dT1, dFL):+.3f}",
                   T1_head_ssim=f"{p['head_ssim']:.4f}", FLAIR_head_ssim=f"{c['head_ssim']:.4f}",
                   T1_head_hfen=f"{p['head_hfen']:.4f}", FLAIR_head_hfen=f"{c['head_hfen']:.4f}",
                   T1_psnr=f"{p['psnr']:.3f}", FLAIR_psnr=f"{c['psnr']:.3f}",
                   T1_sharp=f"{p['sharp']:.3f}", FLAIR_sharp=f"{c['sharp']:.3f}",
                   T1_N=f"{p['grain_ratio']:.3f}", FLAIR_N=f"{c['grain_ratio']:.3f}",
                   commit=commit, args=" ".join(d.get("argv", [])),
                   note=(a.note + f" bar={BEST['T1']['who']}:{BEST['T1']['head_psnr']:.3f}/"
                         f"{BEST['FLAIR']['who']}:{BEST['FLAIR']['head_psnr']:.3f}").strip())
    if not TSV.exists():
        TSV.write_text("\t".join(COLS) + "\n")
    with open(TSV, "a") as f:
        f.write("\t".join(str(row[k]) for k in COLS) + "\n")
    print("\t".join(f"{k}={row[k]}" for k in COLS[:11]))


if __name__ == "__main__":
    main()
