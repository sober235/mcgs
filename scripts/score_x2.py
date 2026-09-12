#!/usr/bin/env python3
"""score_x2.py -- score the voxelwise mean of the magnitude volumes of two (or more) runs that differ only in the
seed, with the same functions arc/cycle.py uses (whole-volume evaluate_reconstruction, hfen, sharp_ratio, grain,
head-only scorer), and write results_arc/<out>_score.json in cycle.py's layout so the ledger row script and
summarize.py can read it. Self-check: --members A A reproduces A's own score json.

usage: score_x2.py --out ar5_P10_x2_full4k --members ar5_best_C10_T112_full4k ar5_P10_s43_full4k
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--members", nargs="+", required=True)
a = ap.parse_args()
sys.argv = ["cycle.py", "--exp", "_score_x2", "--gpu", "0", "--iters", "10"]   # cycle.py parses argv at import
sys.path.insert(0, str(ROOT / "arc"))
import cycle  # noqa: E402  (brings hfen, sharp_ratio, grain, evaluate_reconstruction, score_headonly)

RES = ROOT / "results_arc"
vols = [np.load(RES / f"{m}_volumes.npz") for m in a.members]
scores = [json.load(open(RES / f"{m}_score.json")) for m in a.members]
out = dict(exp=a.out, iters=scores[0]["iters"], npts=scores[0]["npts"], vram_gb=scores[0]["vram_gb"],
           elapsed_s=sum(s.get("elapsed_s", 0) for s in scores), members=a.members, argv=scores[0].get("argv", []),
           set=scores[0].get("set", []), plain_name=scores[0].get("plain_name", "T1"),
           contrast_name=scores[0].get("contrast_name", "FLAIR"))
for ch, key in (("plain", "T1"), ("contrast", "FL")):
    rec = np.mean([np.asarray(v[f"rec{key}"], dtype=np.float32) for v in vols], axis=0)
    gt = np.asarray(vols[0][f"gt{key}"], dtype=np.float32)
    zf = np.asarray(vols[0][f"zf{key}"], dtype=np.float32)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   # the 3-D SSIM is slow and memory-hungry on the host
    t_rec, t_gt, t_zf = (torch.from_numpy(x).to(dev) for x in (rec, gt, zf))
    met = cycle.evaluate_reconstruction(t_rec, t_gt, compute_extended=False)
    hf, sr, gr = (float(v) for v in (cycle.hfen(t_rec, t_gt), cycle.sharp_ratio(t_rec, t_gt), cycle.grain(t_rec, t_gt)))
    ho = cycle.score_headonly(rec, gt)
    mz = cycle.evaluate_reconstruction(t_zf, t_gt, compute_extended=False)
    zf_hf = cycle.hfen(t_zf, t_gt)
    psnr, ssim, nmse = float(met["psnr"]), float(met["ssim"]), float(met["nmse"])
    comp = psnr + 25.0 * ssim - 60.0 * nmse - 25.0 * hf - 40.0 * max(0.0, 1.0 - sr) - 200.0 * gr
    out[ch] = dict(psnr=psnr, ssim=ssim, nmse=nmse, hfen=float(hf), sharp=float(sr), grain=float(gr), comp=comp,
                   beat=bool(ssim >= float(mz["ssim"]) and nmse <= float(mz["nmse"]) and hf <= zf_hf),
                   zf=(float(mz["psnr"]), float(mz["ssim"]), float(mz["nmse"]), float(zf_hf)), **ho)
    ref = scores[0][ch]
    print(f"{ch}: psnr {psnr:.3f} ssim {ssim:.4f} hfen {hf:.4f} sharp {sr:.3f} head_psnr {ho['head_psnr']:.3f} "
          f"head_ssim {ho['head_ssim']:.4f} head_hfen {ho['head_hfen']:.4f}   (member0: {ref['head_psnr']:.3f} / {ref['head_ssim']:.4f} / {ref['head_hfen']:.4f})")
np.savez(RES / f"{a.out}_volumes.npz", **{f"rec{k}": np.mean([np.asarray(v[f"rec{k}"], dtype=np.float32) for v in vols], axis=0) for k in ("T1", "FL")},
         **{f"gt{k}": np.asarray(vols[0][f"gt{k}"], dtype=np.float32) for k in ("T1", "FL")})
json.dump(out, open(RES / f"{a.out}_score.json", "w"), indent=1)
print("wrote", RES / f"{a.out}_score.json")
