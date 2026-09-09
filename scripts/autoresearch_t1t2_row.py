#!/usr/bin/env python3
"""Append one experiment row to autoresearch_t1t2.tsv from a cycle.py score json.

Margins are against the best competitor in the eight-method table of
Results_20260908/T1T2_AF12_retro (head-only PSNR: CS-SENSE on both contrasts). The row is
what the loop reads; the score json keeps everything else.

Usage: autoresearch_t1t2_row.py --exp NAME [--status keep|discard|crash|pending] [--note TEXT]
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "autoresearch_t1t2.tsv"
# best competitor per metric (hyw T1+T2, head-only unless noted), from logs/summary.tsv of the topic
BEST = {
    "T1": dict(head_psnr=34.6069, head_ssim=0.9436, head_hfen=0.1071, psnr=39.2429),
    "T2": dict(head_psnr=34.2185, head_ssim=0.9493, head_hfen=0.0916, psnr=39.2292),
}
COLS = ["exp", "status", "iters", "npts", "vram_gb", "min", "T1_head_psnr", "T2_head_psnr", "dT1", "dT2",
        "min_margin", "T1_head_ssim", "T2_head_ssim", "T1_head_hfen", "T2_head_hfen", "T1_psnr", "T2_psnr",
        "T1_sharp", "T2_sharp", "T1_N", "T2_N", "commit", "args", "note"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--status", default="pending")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    sj = ROOT / "results_arc" / f"{a.exp}_score.json"
    if not sj.exists():
        row = {c: "" for c in COLS}
        row.update(exp=a.exp, status="crash", note=a.note or "no score json")
    else:
        d = json.loads(sj.read_text())
        p, c = d["plain"], d["contrast"]
        dT1 = p["head_psnr"] - BEST["T1"]["head_psnr"]
        dT2 = c["head_psnr"] - BEST["T2"]["head_psnr"]
        try:
            commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            commit = "?"
        row = dict(exp=a.exp, status=a.status, iters=d["iters"], npts=d["npts"], vram_gb=f"{d['vram_gb']:.1f}",
                   min=f"{d.get('elapsed_s', 0) / 60:.0f}",
                   T1_head_psnr=f"{p['head_psnr']:.3f}", T2_head_psnr=f"{c['head_psnr']:.3f}",
                   dT1=f"{dT1:+.3f}", dT2=f"{dT2:+.3f}", min_margin=f"{min(dT1, dT2):+.3f}",
                   T1_head_ssim=f"{p['head_ssim']:.4f}", T2_head_ssim=f"{c['head_ssim']:.4f}",
                   T1_head_hfen=f"{p['head_hfen']:.4f}", T2_head_hfen=f"{c['head_hfen']:.4f}",
                   T1_psnr=f"{p['psnr']:.3f}", T2_psnr=f"{c['psnr']:.3f}",
                   T1_sharp=f"{p['sharp']:.3f}", T2_sharp=f"{c['sharp']:.3f}",
                   T1_N=f"{p['grain_ratio']:.3f}", T2_N=f"{c['grain_ratio']:.3f}",
                   commit=commit, args=" ".join(d.get("argv", [])), note=a.note)
    if not TSV.exists():
        TSV.write_text("\t".join(COLS) + "\n")
    with open(TSV, "a") as f:
        f.write("\t".join(str(row[c]) for c in COLS) + "\n")
    print("\t".join(f"{c}={row[c]}" for c in COLS[:11]))


if __name__ == "__main__":
    main()
