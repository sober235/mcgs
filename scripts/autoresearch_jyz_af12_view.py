#!/usr/bin/env python3
"""autoresearch_jyz_af12_view.py -- every ledger row re-read against the CURRENT bar, six head cells.

The ledger is append-only and each row's margins were computed against the bar of its day (the row's
note says which). When the competitor table changes (the classical lambda re-sweep on this subject),
this prints all rows against the current best non-MC-GS arm per cell of
Results_20260908/T1FLAIR_AF12_v3_jiangyizhou/logs/summary.json, from each row's own score json,
without editing the ledger. Margins: PSNR in dB, SSIM as is, HFEN as competitor - ours (higher is better).
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from autoresearch_jyz_af12_row import best_competitors   # noqa: E402

B = best_competitors()
for c in ("T1", "FLAIR"):
    print(f"bar {c}: PSNR {B[c]['head_psnr']:.3f} ({B[c]['who_psnr']})  SSIM {B[c]['head_ssim']:.4f} ({B[c]['who_ssim']})  "
          f"HFEN {B[c]['head_hfen']:.4f} ({B[c]['who_hfen']})")
hdr = (f"{'exp':26s} {'status':14s} {'it':>5s} {'T1':>7s} {'FL':>7s} {'dT1':>7s} {'dFL':>7s} {'min':>7s} "
       f"{'dsT1':>7s} {'dsFL':>7s} {'dhT1':>7s} {'dhFL':>7s} {'shT1':>5s} {'shFL':>5s}")
print(hdr)
for line in (ROOT / "autoresearch_jyz_af12.tsv").read_text().splitlines()[1:]:
    f = line.split("\t")
    exp, status = f[0], f[1]
    sj = ROOT / "results_arc" / f"{exp}_score.json"
    if not sj.exists():
        print(f"{exp:26s} {status:14s} (no score json)"); continue
    d = json.loads(sj.read_text())
    ch = {}
    for k in ("plain", "contrast"):
        ch.setdefault(d.get(f"{k}_name", {"plain": "T1", "contrast": "FLAIR"}[k]), d[k])
    p, c = ch.get("T1"), ch.get("FLAIR")
    if p is None or c is None:
        r = p or c; nm = "T1" if p else "FLAIR"
        print(f"{exp:26s} {status:14s} {d['iters']:5d} single {nm}: head {r['head_psnr']:.3f} "
              f"d {r['head_psnr'] - B[nm]['head_psnr']:+.3f} ssim {r['head_ssim']:.4f} hfen {r['head_hfen']:.4f}"); continue
    dT1, dFL = p["head_psnr"] - B["T1"]["head_psnr"], c["head_psnr"] - B["FLAIR"]["head_psnr"]
    print(f"{exp:26s} {status:14s} {d['iters']:5d} {p['head_psnr']:7.3f} {c['head_psnr']:7.3f} {dT1:+7.3f} {dFL:+7.3f} {min(dT1, dFL):+7.3f} "
          f"{p['head_ssim'] - B['T1']['head_ssim']:+7.4f} {c['head_ssim'] - B['FLAIR']['head_ssim']:+7.4f} "
          f"{B['T1']['head_hfen'] - p['head_hfen']:+7.4f} {B['FLAIR']['head_hfen'] - c['head_hfen']:+7.4f} {p['sharp']:5.3f} {c['sharp']:5.3f}")
