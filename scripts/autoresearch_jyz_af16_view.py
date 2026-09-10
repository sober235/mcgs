#!/usr/bin/env python3
"""autoresearch_jyz_af16_view.py -- every ledger row re-read against the CURRENT bar.

The ledger is append-only and each row's margins were computed against the bar of its day (the
row's note says which). When the competitor table changes (e.g. the classical arms were re-run
converged and lambda-swept on 2026-09-10), this prints all rows against the current best
non-MC-GS arm of Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/logs/summary.json, from each row's
own score json, without editing the ledger.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from autoresearch_jyz_af16_row import best_competitors   # noqa: E402

B = best_competitors()
print(f"bar: T1 {B['T1']['who']} {B['T1']['head_psnr']:.3f}   FLAIR {B['FLAIR']['who']} {B['FLAIR']['head_psnr']:.3f}")
hdr = f"{'exp':24s} {'status':14s} {'it':>5s} {'T1':>7s} {'FLAIR':>7s} {'dT1':>7s} {'dFL':>7s} {'min':>7s} {'sT1':>6s} {'sFL':>6s} {'hfT1':>6s} {'hfFL':>6s} {'shT1':>5s} {'shFL':>5s}"
print(hdr)
for line in (ROOT / "autoresearch_jyz_af16.tsv").read_text().splitlines()[1:]:
    f = line.split("\t")
    exp, status = f[0], f[1]
    sj = ROOT / "results_arc" / f"{exp}_score.json"
    if not sj.exists():
        print(f"{exp:24s} {status:14s} (no score json)"); continue
    d = json.loads(sj.read_text()); p, c = d["plain"], d["contrast"]
    dT1, dFL = p["head_psnr"] - B["T1"]["head_psnr"], c["head_psnr"] - B["FLAIR"]["head_psnr"]
    print(f"{exp:24s} {status:14s} {d['iters']:5d} {p['head_psnr']:7.3f} {c['head_psnr']:7.3f} {dT1:+7.3f} {dFL:+7.3f} {min(dT1, dFL):+7.3f} "
          f"{p['head_ssim']:6.4f} {c['head_ssim']:6.4f} {p['head_hfen']:6.4f} {c['head_hfen']:6.4f} {p['sharp']:5.3f} {c['sharp']:5.3f}")
