#!/usr/bin/env python3
"""rescore_topic.py AF CONTRAST -- re-score every arm of one jiangyizhou T1+FLAIR topic (AF 12, 16, 20 or 24) with ONE scorer.

Extension of T1FLAIR_AF12_16_20_trend_v3_jiangyizhou/scripts/rescore_topic.py (2026-09-14) to the AF 24 topic (2026-09-15);
the scoring is unchanged. Reads (never writes) Results_20260908/T1FLAIR_AF<AF>_v3_jiangyizhou: the saved volume of each arm
of record and the GT (bit-identical across the four topics). Scorer = diag_artefact/headonly.py of the framework (md5
32a4e4a9): head PSNR, head SSIM, head HFEN. The LoG of the GT is computed once and reused; the arithmetic is that of
headonly.head_hfen. Writes logs/rescore_af<AF>_<CONTRAST>.json with the recorded numbers of each arm beside the re-score.

Arms of record: the classical swept families at the lambda their topic's summary keeps; MC-GS / GS-single = the AF 12
line's final P14 at AF 12 and the AF 16 phase-3 recipe at AF 16, AF 20 and AF 24 (the recipes are close but not identical).
MC-INR was not run at AF 24, so its AF 24 row reads "no summary row".
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage
sys.path.insert(0, "/data0/congcong/code/GS/mcgs_af16_wt")
from diag_artefact.headonly import head_mask, head_psnr, head_ssim, log_kernel

T = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou")
ROOT = Path("/data0/congcong/code/GS/Results_20260908")
af, c = sys.argv[1], sys.argv[2]
assert af in ("12", "16", "20", "24") and c in ("T1", "FLAIR")
out = T / f"logs/rescore_af{af}_{c}.json"
topic = ROOT / f"T1FLAIR_AF{af}_v3_jiangyizhou"
summ = {(r["method"], r["contrast"]): r for r in json.load(open(topic / "logs/summary.json"))}
gs = {"12": ("gs_single_p14", "gs_joint_p14"), "16": ("gs_single_p3", "gs_joint_p3"), "20": ("gs_single_p3", "gs_joint_p3"),
      "24": ("gs_single_p3", "gs_joint_p3")}[af]
ARMS = [("ZF", "zf", False), ("CG-SENSE", "cgsense", False), ("CS-SENSE (35 it)", "cssense", False),
        ("CS-SENSE (converged TV)", "cssense_o200", True), ("Joint SENSE", "jcssense", True), ("L1-ESPIRiT", "l1espirit", True),
        ("ZS-SSDU", "zsssdu", False), ("3D-DIP", "dip", False), ("3D-INR", "inr", False), ("MC-INR", "mcinr", False),
        ("GS-single", gs[0], False), ("MC-GS", gs[1], False)]
SWEPT = {"cssense_o200": "cssense_o200_lam{L}", "jcssense": "jcssense_o200_lam{L}", "l1espirit": "l1espirit_lam{L}"}

gt = np.abs(np.load(topic / f"recon/gt_{c}_complex.npy")).astype(np.float32)
head = head_mask(gt); gmax = float(gt.max()); K = log_kernel()
t0 = time.time(); lg = ndimage.convolve(gt.astype(np.float64) / gmax, K, mode="reflect"); den = float(np.linalg.norm(lg[head]))
print(f"[AF{af} {c}] GT LoG {time.time()-t0:.0f}s", flush=True)
rows = []
for label, key, swept in ARMS:
    rec_row = summ.get((key, c))
    if rec_row is None:
        rows.append(dict(af=af, contrast=c, label=label, key=key, missing="no summary row")); continue
    lam = rec_row.get("lam")
    stem = SWEPT[key].format(L=lam) if swept else key
    cands = [f"{stem}_{c}_mag.npy", f"{stem}_{c}_complex.npy", f"{key}_{c}_complex.npy"]
    path = next((topic / "recon" / f for f in cands if (topic / "recon" / f).exists()), None)
    if path is None:
        rows.append(dict(af=af, contrast=c, label=label, key=key, lam=lam, missing="no volume")); continue
    t = time.time()
    rec = np.abs(np.load(path)).astype(np.float32)
    p = head_psnr(rec, gt, head, gmax); s = head_ssim(rec, gt, head, gmax)
    lr = ndimage.convolve(rec.astype(np.float64) / gmax, K, mode="reflect")
    h = float(np.linalg.norm((lr - lg)[head]) / den)
    r = dict(af=af, contrast=c, label=label, key=key, lam=lam, file=path.name,
             head_psnr=p, head_ssim=s, head_hfen=h,
             rec_head_psnr=rec_row["head_psnr"], rec_head_ssim=rec_row["head_ssim"], rec_head_hfen=rec_row["head_hfen"],
             rec_psnr=rec_row.get("psnr"), rec_ssim=rec_row.get("ssim"), rec_nmse=rec_row.get("nmse"), rec_sharp=rec_row.get("sharp"))
    rows.append(r)
    print(f"[AF{af} {c}] {label:24s} {path.name:40s} PSNR {p:.4f} (rec {r['rec_head_psnr']:.4f})  SSIM {s:.5f} (rec {r['rec_head_ssim']:.5f})  "
          f"HFEN {h:.5f} (rec {r['rec_head_hfen']:.5f})  {time.time()-t:.0f}s", flush=True)
    json.dump(rows, open(out, "w"), indent=1)
json.dump(rows, open(out, "w"), indent=1)
print(f"[AF{af} {c}] DONE", flush=True)
