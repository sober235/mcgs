#!/usr/bin/env python3
"""Stage-4 table: every scored run of the 16yo MTP line, primary region, against the +1 dB bar.

For each run with results_arc/<exp>_primary.json: head PSNR / SSIM / HFEN per contrast, the head-PSNR
margin over the best other arm of the published table (logs/rescore.json, GS-single counted, joint GS
arms excluded) and how far that margin is from +1.00 dB. A second margin uses the best GS-single run
of THIS ledger at the same points and iterations when one exists (the stage-3 fairness rule: the single gets every
generic lever), so a capacity or iteration gain is not counted twice.

  PYTHONNOUSERSITE=1 $PY scripts/p4_summary.py [--prefix p4_] [--also p3_c1_wav006_gp003,p3_n60]
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uih_mtp.bars import competitor_bars  # noqa: E402

TOPIC = Path("/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro")
ARC = ROOT / "results_arc"
ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="p4_")
ap.add_argument("--also", default="p3_c1_wav006_gp003,p3_i6000,p3_n60,p3_single_T1w_c1,p3_single_T1w_i6000")
a = ap.parse_args()

bars = competitor_bars(json.load(open(TOPIC / "logs/rescore.json"))["rows"], ("T1w", "T2s"))
names = sorted({p.name[:-len("_primary.json")] for p in ARC.glob(f"{a.prefix}*_primary.json")}
               | {n for n in a.also.split(",") if (ARC / f"{n}_primary.json").exists()})


def npts(exp):
    """(points, iterations) of a run: the generic settings a fair single comparison must share"""
    sj = ARC / f"{exp}_score.json"
    if not sj.exists():
        return None
    d = json.load(open(sj))
    return (d.get("npts"), d.get("iters"))


runs = {n: (json.load(open(ARC / f"{n}_primary.json")), npts(n)) for n in names}
# best single per (contrast, capacity) in this ledger
best_single = {}
for n, (d, pts) in runs.items():
    if "single" not in n:
        continue
    for c, r in d["rows"].items():
        k = (c, pts)
        best_single[k] = max(best_single.get(k, -1e9), r["head_psnr"])

print(f"bars (published table): T1w {bars['T1w']['head_psnr'][0]:.3f} [{bars['T1w']['head_psnr'][1]}], "
      f"T2s {bars['T2s']['head_psnr'][0]:.3f} [{bars['T2s']['head_psnr'][1]}]; target = bar + 1.00")
hdr = f"{'run':28s} {'pts/it':>8s} | {'T1w PSNR/SSIM/HFEN':>24s} {'m':>7s} {'m_same':>7s} | {'T2s PSNR/SSIM/HFEN':>24s} {'m':>7s} {'m_same':>7s} | {'to +1dB':>7s}"
print(hdr); print("-" * len(hdr))
for n, (d, pts) in runs.items():
    cells, gaps = [], []
    for c in ("T1w", "T2s"):
        r = d["rows"].get(c)
        if r is None:
            cells.append(f"{'':>24s} {'':>7s} {'':>7s}"); continue
        m = r["head_psnr"] - bars[c]["head_psnr"][0]
        bs = best_single.get((c, pts))
        ms = (r["head_psnr"] - max(bs, bars[c]["head_psnr"][0])) if (bs is not None and "single" not in n) else None
        cells.append(f"{r['head_psnr']:8.3f}/{r['head_ssim']:.4f}/{r['head_hfen']:.4f} {m:+7.3f} "
                     f"{(f'{ms:+7.3f}' if ms is not None else '      -')}")
        gaps.append((ms if ms is not None else m) - 1.0)
    pt = f"{pts[0] / 1e6:.1f}M/{pts[1] // 1000}k" if pts else "?"
    worst = f"{min(gaps):+7.3f}" if (len(gaps) == 2 and "single" not in n) else "      -"
    print(f"{n:28s} {pt:>8s} | {cells[0]} | {cells[1]} | {worst}")
