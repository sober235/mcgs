#!/usr/bin/env python3
"""head_nmse.py -- head-only NMSE of every arm of record at AF 12 / 16 / 20 / 24, both contrasts (user 2026-09-16 evening:
the third metric of the trend figure is NMSE, as on the three-plane figures). The frozen gauge (diag_artefact/headonly.py)
has no NMSE, so it is computed here with the gauge's own head mask, |GT| > 0.05 * P99.9(|GT|), on magnitudes:
    head NMSE = sum_head |rec - gt|^2 / sum_head |gt|^2
which is the same error sum the gauge's head PSNR uses (its MSE over the head), only normalised by the head's energy instead
of the volume peak. Reads the volumes listed in logs/rescore_af<AF>_<c>.json (never writes to the source topics); writes
logs/head_nmse_af<AF>_<c>.json with one row per arm: key, label, file, head_nmse, plus the rescored head_psnr for a cross-check
(psnr_from_nmse must equal head_psnr, since both come from the same head MSE).
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/data0/congcong/code/GS/mcgs_af16_wt")
from diag_artefact.headonly import head_mask

T = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou")
ROOT = Path("/data0/congcong/code/GS/Results_20260908")
for af in ("12", "16", "20", "24"):
    for c in ("T1", "FLAIR"):
        topic = ROOT / f"T1FLAIR_AF{af}_v3_jiangyizhou"
        gt = np.abs(np.load(topic / f"recon/gt_{c}_complex.npy")).astype(np.float32)
        head = head_mask(gt); gmax = float(gt.max()); e_gt = float((gt[head] ** 2).sum()); n = int(head.sum())
        rows = []
        for r in json.load(open(T / f"logs/rescore_af{af}_{c}.json")):
            if "head_psnr" not in r:
                continue
            v = np.load(topic / "recon" / r["file"])
            v = (np.abs(v) if np.iscomplexobj(v) else v).astype(np.float32)
            assert v.shape == gt.shape, (r["file"], v.shape)
            d = float(((v[head] - gt[head]) ** 2).sum())
            nmse = d / e_gt
            psnr_chk = 20 * np.log10(gmax) - 10 * np.log10(d / n)       # the gauge's head PSNR from the same error sum
            rows.append(dict(key=r["key"], label=r["label"], file=r["file"], head_nmse=nmse,
                             head_psnr_rescored=r["head_psnr"], psnr_from_nmse=float(psnr_chk)))
        worst = max(abs(x["psnr_from_nmse"] - x["head_psnr_rescored"]) for x in rows)
        json.dump(dict(af=af, contrast=c, head_voxels=n, head_energy=e_gt, gmax=gmax, rows=rows),
                  open(T / f"logs/head_nmse_af{af}_{c}.json", "w"), indent=1)
        print(f"AF{af} {c:5s} {len(rows)} arms, head voxels {n}, max |PSNR(from NMSE) - head PSNR| = {worst:.1e}  "
              + "  ".join(f"{x['label']}={x['head_nmse']:.5f}" for x in rows if x['label'] in ('MC-GS', 'GS-single', 'ZF')), flush=True)
