#!/usr/bin/env python3
"""diag_gauss_width.py -- how wide are the Gaussians along each array axis, and how much high frequency can such
kernels pass at all? (CPU, read-only; reads one stage-2 checkpoint, writes nothing next to it.)

Conventions copied from the frozen tile_voxelizer: positions in [-1, 1] per axis, centre_vox = (p + 1) / 2 * N - 0.5,
scales = exp(model.scales) in normalised units, Sigma = R diag(s^2) R^T, voxel std along axis i =
sqrt(Sigma_ii) * N_i / 2. A Gaussian of std sigma voxels passes power exp(-4 pi^2 sigma^2 f^2) at f cycles/voxel,
so the |density|^2-weighted mean of that transfer is the ceiling of rendered power along that axis before
any density arrangement compensates.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

SHAPE = (256, 232, 176)
FREQS = (0.15, 0.25, 0.35, 0.45)


def quat_to_rot(q):
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w], -1),
        torch.stack([2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w], -1),
        torch.stack([2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y], -1)], -2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    a = ap.parse_args()
    st = torch.load(a.ckpt, map_location='cpu', weights_only=True)['model_state']
    s = torch.exp(st['scales'].double())
    R = quat_to_rot(st['rotations'].double())
    var = ((R ** 2) * (s ** 2).unsqueeze(1)).sum(-1)                     # Sigma_ii, normalised units
    half = torch.tensor(SHAPE, dtype=torch.float64) / 2
    std = (var.sqrt() * half).numpy()                                     # voxels, (N, 3)
    out = {'points': int(std.shape[0]), 'iter': None, 'axes': {}}
    for scan, (kr, ki) in {'pre': ('density_real', 'density_imag'),
                           'post': ('density_real_contrast', 'density_imag_contrast')}.items():
        w = (st[kr].double() ** 2 + st[ki].double() ** 2).numpy()
        w = w / w.sum()
        rows = {}
        for ax in range(3):
            x = std[:, ax]
            order = np.argsort(x)
            cw = np.cumsum(w[order])
            wq = {q: float(x[order][np.searchsorted(cw, q)]) for q in (.1, .5, .9)}
            rows[ax] = dict(q_unweighted={q: float(np.quantile(x, q)) for q in (.1, .5, .9)}, q_weighted=wq,
                            frac_below_half_voxel_weighted=float(w[x < .5].sum()),
                            transfer={f: float((w * np.exp(-4 * math.pi ** 2 * x ** 2 * f ** 2)).sum()) for f in FREQS})
        out['axes'][scan] = rows
        print(f'\n== {scan} (weights |density|^2)')
        print('axis  std q10/q50/q90 unweighted   weighted q10/q50/q90   w(<0.5vox)   power transfer at f=' + ' '.join(map(str, FREQS)))
        for ax in range(3):
            r = rows[ax]
            print(f"a{ax}    {r['q_unweighted'][.1]:.2f}/{r['q_unweighted'][.5]:.2f}/{r['q_unweighted'][.9]:.2f}"
                  f"               {r['q_weighted'][.1]:.2f}/{r['q_weighted'][.5]:.2f}/{r['q_weighted'][.9]:.2f}"
                  f"            {r['frac_below_half_voxel_weighted']:.3f}       "
                  + '  '.join(f"{r['transfer'][f]:.3f}" for f in FREQS))
    art = os.environ.get('MCGS_ARTIFACT_DIR')
    if art:
        Path(art, 'diag_gauss_width.json').write_text(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
