#!/usr/bin/env python3
"""Append one experiment row to autoresearch_peds.tsv from a cycle.py score json.

Stage 3 (spec 2026-09-14-peds-af12-mcgs-autoresearch-design.md): the run is judged where the table
is judged -- the primary region, head inside the vendor's central 80 partitions -- against the best
value any other arm reaches there, GS-single counted as a competitor and every joint GS arm left
out. Those numbers come from results_arc/<exp>_primary.json (scripts/score_candidate.py of the
topic); dT1w / dT2s are its head-PSNR margins and min_margin is the smallest of all six cells, so a
positive min_margin is the target of this stage. Without that file the margins stay blank rather
than quietly fall back to a different region. The head columns themselves are the run's own
full-volume scorer output, kept as the cross-check.

Usage: autoresearch_jyz_af16_row.py --exp NAME [--status keep|discard|crash|pending] [--note TEXT]
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "autoresearch_peds.tsv"
TOPIC = Path("/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro")
sys.path.insert(0, str(ROOT_PKG := Path(__file__).resolve().parents[1]))
from uih_mtp.bars import competitor_bars, margins  # noqa: E402
COLS = ["exp", "status", "iters", "npts", "vram_gb", "min", "T1w_head_psnr", "T2s_head_psnr", "dT1w", "dT2s",
        "min_margin", "T1w_head_ssim", "T2s_head_ssim", "T1w_head_hfen", "T2s_head_hfen", "T1w_psnr", "T2s_psnr",
        "T1w_sharp", "T2s_sharp", "T1w_N", "T2s_N", "commit", "args", "note"]


def best_competitors():
    """The primary table's bars: {contrast: {metric: (bar, arm)}}, GS-single in, joint GS arms out."""
    p = TOPIC / "logs/rescore.json"
    if not p.exists():
        return None
    return competitor_bars(json.load(open(p))["rows"], ("T1w", "T2s"))


def bar_note(best):
    """Which arm sets each head-PSNR bar, for the ledger's note column."""
    return "bar=" + "/".join(f"{c}:{best[c]['head_psnr'][1]}:{best[c]['head_psnr'][0]:.3f}" for c in best)


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
        # the verdict comes from the primary region, never from these full-volume numbers
        pj = ROOT / "results_arc" / f"{a.exp}_primary.json"
        dT1 = dFL = mm = None
        if pj.exists() and BEST is not None:
            prim = json.load(open(pj))
            six = {(c, m): v for c, d in prim["margins"].items() for m, v in d.items()}
            dT1 = six.get(("T1w", "head_psnr"))
            dFL = six.get(("T2s", "head_psnr"))
            mm = min(six.values())
        fmt = lambda r, k, f: (f"{r[k]:{f}}" if r else "")
        sgn = lambda x: (f"{x:+.3f}" if x is not None else "")
        try:
            commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            commit = "?"
        row = dict(exp=a.exp, status=a.status, iters=d["iters"], npts=d["npts"], vram_gb=f"{d['vram_gb']:.1f}",
                   min=f"{(d.get('elapsed_s') or 0) / 60:.0f}",
                   T1w_head_psnr=fmt(p1, "head_psnr", ".3f"), T2s_head_psnr=fmt(c1, "head_psnr", ".3f"),
                   dT1w=sgn(dT1), dT2s=sgn(dFL), min_margin=(f"{mm:+.4f}" if mm is not None else ""),
                   T1w_head_ssim=fmt(p1, "head_ssim", ".4f"), T2s_head_ssim=fmt(c1, "head_ssim", ".4f"),
                   T1w_head_hfen=fmt(p1, "head_hfen", ".4f"), T2s_head_hfen=fmt(c1, "head_hfen", ".4f"),
                   T1w_psnr=fmt(p1, "psnr", ".3f"), T2s_psnr=fmt(c1, "psnr", ".3f"),
                   T1w_sharp=fmt(p1, "sharp", ".3f"), T2s_sharp=fmt(c1, "sharp", ".3f"),
                   T1w_N=fmt(p1, "grain_ratio", ".3f"), T2s_N=fmt(c1, "grain_ratio", ".3f"),
                   commit=commit, args=" ".join(d.get("argv", [])),
                   note=(a.note + (" " + bar_note(BEST) if BEST else " (no primary table yet)")).strip())
    if not TSV.exists():
        TSV.write_text("\t".join(COLS) + "\n")
    with open(TSV, "a") as f:
        f.write("\t".join(str(row[k]) for k in COLS) + "\n")
    print("\t".join(f"{k}={row[k]}" for k in COLS[:11]))


if __name__ == "__main__":
    main()
