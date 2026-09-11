#!/usr/bin/env python3
"""Append one experiment row to autoresearch_jyz_af12.tsv from a cycle.py score json.

Margins are against the best NON-MC-GS arm of Results_20260908/T1FLAIR_AF12_v3_jiangyizhou, read
from that topic's logs/summary.json so the numbers cannot drift from the archive. Six head cells:
per contrast, PSNR margin (dB, higher is better), SSIM margin (higher is better) and HFEN margin
(best_competitor - ours, so higher is better too). min_margin is the PSNR one, the number the loop
reads; the four other cells are the no-regression gates of autoresearch_jyz_af12.md.

Usage: autoresearch_jyz_af12_row.py --exp NAME [--status keep|discard|crash|pending] [--note TEXT]
       autoresearch_jyz_af12_row.py --print_best
"""
import argparse, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "autoresearch_jyz_af12.tsv"
TOPIC = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou")
COLS = ["exp", "status", "iters", "npts", "vram_gb", "min", "T1_head_psnr", "FLAIR_head_psnr", "dT1", "dFLAIR",
        "min_margin", "T1_head_ssim", "FLAIR_head_ssim", "dT1_ssim", "dFLAIR_ssim",
        "T1_head_hfen", "FLAIR_head_hfen", "dT1_hfen", "dFLAIR_hfen",
        "T1_psnr", "FLAIR_psnr", "T1_sharp", "FLAIR_sharp", "T1_N", "FLAIR_N", "commit", "args", "note"]


def is_gs(method):
    m = str(method)
    return m.startswith(("gs_", "MC-GS", "GS-")) or m in {"gs_single", "gs_joint"}


FAM = [("cssense_o200", "logs/classical_{c}_o200*.json"), ("l1espirit", "logs/l1espirit_{c}_lam*.json"),
       ("jcssense", "logs/jcssense_o200*.json")]


def best_competitors():
    """Per contrast and cell, the best non-MC-GS value. summary.json keeps one lambda per classical family
    (the head-PSNR-best one); the bar is stricter: every lambda of the three swept families counts for every
    cell, so a competitor tuned for SSIM or HFEN is also beaten (pre-registered 2026-09-11 20:10, before
    any candidate row existed)."""
    rows = json.load(open(TOPIC / "logs/summary.json"))
    best = {}
    for c in ("T1", "FLAIR"):
        sub = [r for r in rows if r["contrast"] == c and not is_gs(r["method"]) and r["method"] != "zf"]
        for key, pat in FAM:
            for f in sorted(TOPIC.glob(pat.format(c=c))):
                for r in json.load(open(f)).get("rows", []):
                    if r.get("contrast", c) == c and r.get("head_psnr") is not None:
                        sub.append({**r, "method": f"{key}@{r.get('lam')}"})
        best[c] = dict(head_psnr=max(r["head_psnr"] for r in sub), head_ssim=max(r["head_ssim"] for r in sub),
                       head_hfen=min(r["head_hfen"] for r in sub), psnr=max(r["psnr"] for r in sub),
                       who_psnr=max(sub, key=lambda r: r["head_psnr"])["method"],
                       who_ssim=max(sub, key=lambda r: r["head_ssim"])["method"],
                       who_hfen=min(sub, key=lambda r: r["head_hfen"])["method"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True if "--print_best" not in __import__("sys").argv else False)
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
        ch = {}
        for c in ("plain", "contrast"):
            nm = d.get(f"{c}_name", {"plain": "T1", "contrast": "FLAIR"}[c])
            ch.setdefault(nm, d[c])
        p1, c1 = ch.get("T1"), ch.get("FLAIR")
        m = lambda r, c, k, sign: (sign * (r[k] - BEST[c][k]) if r else None)
        dT1, dFL = m(p1, "T1", "head_psnr", 1), m(c1, "FLAIR", "head_psnr", 1)
        mm = min(x for x in (dT1, dFL) if x is not None)
        fmt = lambda r, k, f: (f"{r[k]:{f}}" if r else "")
        sgn = lambda x, f="+.3f": (f"{x:{f}}" if x is not None else "")
        try:
            commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            commit = "?"
        row = dict(exp=a.exp, status=a.status, iters=d["iters"], npts=d["npts"], vram_gb=f"{d['vram_gb']:.1f}",
                   min=f"{(d.get('elapsed_s') or 0) / 60:.0f}",
                   T1_head_psnr=fmt(p1, "head_psnr", ".3f"), FLAIR_head_psnr=fmt(c1, "head_psnr", ".3f"),
                   dT1=sgn(dT1), dFLAIR=sgn(dFL), min_margin=f"{mm:+.3f}",
                   T1_head_ssim=fmt(p1, "head_ssim", ".4f"), FLAIR_head_ssim=fmt(c1, "head_ssim", ".4f"),
                   dT1_ssim=sgn(m(p1, "T1", "head_ssim", 1), "+.4f"), dFLAIR_ssim=sgn(m(c1, "FLAIR", "head_ssim", 1), "+.4f"),
                   T1_head_hfen=fmt(p1, "head_hfen", ".4f"), FLAIR_head_hfen=fmt(c1, "head_hfen", ".4f"),
                   dT1_hfen=sgn(m(p1, "T1", "head_hfen", -1), "+.4f"), dFLAIR_hfen=sgn(m(c1, "FLAIR", "head_hfen", -1), "+.4f"),
                   T1_psnr=fmt(p1, "psnr", ".3f"), FLAIR_psnr=fmt(c1, "psnr", ".3f"),
                   T1_sharp=fmt(p1, "sharp", ".3f"), FLAIR_sharp=fmt(c1, "sharp", ".3f"),
                   T1_N=fmt(p1, "grain_ratio", ".3f"), FLAIR_N=fmt(c1, "grain_ratio", ".3f"),
                   commit=commit, args=" ".join(d.get("argv", [])),
                   note=(a.note + f" bar={BEST['T1']['who_psnr']}:{BEST['T1']['head_psnr']:.3f}/"
                         f"{BEST['FLAIR']['who_psnr']}:{BEST['FLAIR']['head_psnr']:.3f}").strip())
    if not TSV.exists():
        TSV.write_text("\t".join(COLS) + "\n")
    with open(TSV, "a") as f:
        f.write("\t".join(str(row[k]) for k in COLS) + "\n")
    print("\t".join(f"{k}={row[k]}" for k in COLS[:11]))


if __name__ == "__main__":
    main()
