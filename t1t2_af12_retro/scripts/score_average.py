#!/usr/bin/env python3
"""Score the voxelwise mean of the magnitudes of two runs (program Round 18, "v3 x2") the way arc/cycle.py scores a run:
evaluate_reconstruction (psnr/ssim/nmse), the cycle's hfen/sharp_ratio/grain, and the frozen head-only scorer.
Re-scores run A from its saved volumes first and prints each field next to A's logged value.
usage: score_average.py <exp A> <exp B> <out name>   -> results_arc/<out>_score.json and <out>_volumes.npz"""
import json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path("/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paired_gs"))
from metrics import evaluate_reconstruction
from diag_artefact.headonly import score_headonly


# hfen, sharp_ratio and grain are copied verbatim from arc/cycle.py (importing it would run its argument parser)
def hfen(rec, gt):
    def lap(v):
        return (6.0 * v - torch.roll(v, 1, -3) - torch.roll(v, -1, -3)
                - torch.roll(v, 1, -2) - torch.roll(v, -1, -2)
                - torch.roll(v, 1, -1) - torch.roll(v, -1, -1))
    g = gt / (gt.max() + 1e-9)
    r = rec / (gt.max() + 1e-9)
    lg = lap(g)
    return float(torch.linalg.norm(lap(r) - lg) / (torch.linalg.norm(lg) + 1e-9))


def sharp_ratio(rec, gt):
    def gm(v):
        gx = torch.zeros_like(v); gy = torch.zeros_like(v); gz = torch.zeros_like(v)
        gx[:-1] = v[1:] - v[:-1]; gy[:, :-1] = v[:, 1:] - v[:, :-1]; gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
        return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
    brain = gt > 0.05 * gt.max()
    return float(gm(rec)[brain].mean() / (gm(gt)[brain].mean() + 1e-9))


def grain(rec, gt):
    import torch.nn.functional as F
    g = gt / (gt.max() + 1e-9)
    r = rec / (gt.max() + 1e-9)
    sm = F.avg_pool3d(r[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    hp = r - sm
    def gmag(v):
        gx = torch.zeros_like(v); gy = torch.zeros_like(v); gz = torch.zeros_like(v)
        gx[:-1] = v[1:] - v[:-1]; gy[:, :-1] = v[:, 1:] - v[:, :-1]; gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
        return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
    brain = g > 0.05 * g.max()
    gg = gmag(g)
    thr = torch.quantile(gg[brain], 0.30)
    flat = brain & (gg <= thr)
    return float(hp[flat].std())


def score(rec, gt):
    r, g = torch.from_numpy(rec), torch.from_numpy(gt)
    met = evaluate_reconstruction(r, g, compute_extended=False)
    d = dict(psnr=float(met["psnr"]), ssim=float(met["ssim"]), nmse=float(met["nmse"]),
             hfen=hfen(r, g), sharp=sharp_ratio(r, g), grain=grain(r, g))
    d.update({k: float(v) for k, v in score_headonly(rec, gt).items()})
    return d


a_name, b_name, out = sys.argv[1:4]
ARC = ROOT / "results_arc"
A, B = np.load(ARC / f"{a_name}_volumes.npz"), np.load(ARC / f"{b_name}_volumes.npz")
ja, jb = json.load(open(ARC / f"{a_name}_score.json")), json.load(open(ARC / f"{b_name}_score.json"))
res, vols = {}, {}
for rk, gk, ch in (("recT1", "gtT1", "plain"), ("recFL", "gtFL", "contrast")):
    gt = np.abs(A[gk]).astype(np.float32)
    ra, rb = np.abs(A[rk]).astype(np.float32), np.abs(B[rk]).astype(np.float32)
    chk = score(ra, gt)
    print(f"{ch} check vs logged {a_name}: " + " ".join(f"{k} {ja[ch][k]:.5f}/{chk[k]:.5f}" for k in
          ("psnr", "ssim", "nmse", "hfen", "sharp", "grain", "head_psnr", "head_ssim", "head_hfen", "grain_ratio")), flush=True)
    avg = 0.5 * (ra + rb)
    res[ch] = score(avg, gt)
    vols[rk], vols[gk] = avg, gt
    print(f"{ch} x2: " + " ".join(f"{k}={res[ch][k]:.4f}" for k in
          ("psnr", "ssim", "nmse", "head_psnr", "head_ssim", "head_hfen", "sharp", "grain_ratio")), flush=True)
row = dict(exp=out, iters=ja["iters"], npts=int(ja["npts"]) + int(jb["npts"]), vram_gb=max(ja["vram_gb"], jb["vram_gb"]),
           elapsed_s=ja.get("elapsed_s", 0) + jb.get("elapsed_s", 0), members=[a_name, b_name],
           argv=[f"voxelwise mean of the magnitudes of {a_name} and {b_name}"], set=[],
           plain=res["plain"], contrast=res["contrast"], plain_name=ja.get("plain_name"), contrast_name=ja.get("contrast_name"))
(ARC / f"{out}_score.json").write_text(json.dumps(row, indent=2))
np.savez(ARC / f"{out}_volumes.npz", **vols)
print("wrote", ARC / f"{out}_score.json", flush=True)
