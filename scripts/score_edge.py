#!/usr/bin/env python3
"""score_edge.py — edge-region / ventricle / gradient-quintile fidelity for a recon run,
from its *_volumes.npz. Implements the strict-bar metrics in
docs/superpowers/specs/2026-07-23-edge-preserving-tv-csf-design.md.

Usage:
  PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/score_edge.py \
    --npz results_arc/<exp>_volumes.npz --tag <exp> [--tsv results_arc/edge_scores.tsv]
"""
import argparse, os
import numpy as np
from scipy.ndimage import sobel, gaussian_laplace, binary_dilation, binary_opening
from skimage.metrics import structural_similarity as ssim


def grad_mag(v):
    return np.sqrt(sobel(v, 0) ** 2 + sobel(v, 1) ** 2 + sobel(v, 2) ** 2)


def nmse(a, g, m):
    return float(np.sum(((a - g) * m) ** 2) / np.sum((g * m) ** 2))


def hfen(a, g, m, sigma=1.5):
    la, lg = gaussian_laplace(a, sigma), gaussian_laplace(g, sigma)
    return float(np.sqrt(np.sum(((la - lg) * m) ** 2) / np.sum((lg * m) ** 2)))


def edge_set(gt, brain, q=0.80, dilate=2):
    gm = grad_mag(gt)
    thr = np.quantile(gm[brain], q)
    return binary_dilation((gm >= thr) & brain, iterations=dilate) & brain


def ventricle_mask(gt, zc=160, half=42, zrange=7, pct=24):
    peak = gt.max()
    head = gt[:, :, zc] > 0.02 * peak
    ys, xs = np.where(head); cy, cx = ys.mean(), xs.mean()
    y0, y1, x0, x1 = int(cy - half), int(cy + half), int(cx - half), int(cx + half)
    m = np.zeros_like(gt, dtype=bool)
    for k in range(zc - zrange, zc + zrange + 1):
        box = gt[y0:y1, x0:x1, k]
        m[y0:y1, x0:x1, k] = binary_opening(box < np.percentile(box, pct), iterations=1)
    return m


def quintile_curve(rc, gt, brain):
    ae = (np.abs(rc - gt) / gt.max())[brain]
    gm = grad_mag(gt)[brain]
    e = np.quantile(gm, [0, .2, .4, .6, .8, 1.0])
    return [float(ae[(gm >= e[i]) & (gm <= e[i + 1])].mean()) for i in range(5)]


def score_contrast(rc, gt, name):
    peak = gt.max(); brain = gt > 0.05 * peak
    es, vent = edge_set(gt, brain), ventricle_mask(gt)
    return dict(
        contrast=name,
        nmse_whole=nmse(rc, gt, brain),
        ssim=float(ssim(gt / peak, rc / peak, data_range=1.0)),
        hfen=hfen(rc, gt, brain),
        nmse_edge=nmse(rc, gt, es),
        hfen_edge=hfen(rc, gt, es),
        nmse_vent=nmse(rc, gt, vent),
        quintiles=quintile_curve(rc, gt, brain),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tsv", default="results_arc/edge_scores.tsv")
    a = ap.parse_args()
    d = np.load(a.npz)
    rows = [score_contrast(np.abs(d["recT1"]), np.abs(d["gtT1"]), "T1"),
            score_contrast(np.abs(d["recFL"]), np.abs(d["gtFL"]), "FLAIR")]
    for r in rows:
        print(f"[{a.tag}] {r['contrast']}: nmse_edge={r['nmse_edge']:.4f} hfen_edge={r['hfen_edge']:.4f} "
              f"ssim={r['ssim']:.4f} nmse_whole={r['nmse_whole']:.4f} nmse_vent={r['nmse_vent']:.4f} "
              f"quintiles={['%.4f' % q for q in r['quintiles']]}")
    new = not os.path.exists(a.tsv)
    with open(a.tsv, "a") as f:
        if new:
            f.write("tag\tcontrast\tnmse_edge\thfen_edge\tssim\tnmse_whole\thfen\tnmse_vent\tq1\tq2\tq3\tq4\tq5\n")
        for r in rows:
            f.write(f"{a.tag}\t{r['contrast']}\t{r['nmse_edge']:.5f}\t{r['hfen_edge']:.5f}\t{r['ssim']:.5f}\t"
                    f"{r['nmse_whole']:.5f}\t{r['hfen']:.5f}\t{r['nmse_vent']:.5f}\t"
                    + "\t".join('%.5f' % q for q in r['quintiles']) + "\n")
    print(f"appended -> {a.tsv}")


if __name__ == "__main__":
    main()
