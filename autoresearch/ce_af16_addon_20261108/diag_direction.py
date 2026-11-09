#!/usr/bin/env python3
"""diag_direction.py -- is the missing high frequency direction-specific, and could the readout axis serve as a
self-reference? (CPU, read-only, diagnostic only; nothing here feeds training or the frozen evaluator.)

Readout (array axis 0) is fully sampled; the AF16 mask lives on (ky, kz) = axes 1, 2. For each method and each array
axis a, the 1-D spectrum along a (centred, orthonormal FFT along that axis only, power averaged over the other two
axes) is binned by |f| in cycles/voxel. Reported per band:
  keep[a]  = power(rec) / power(GT)            -- how much of the true energy along axis a the method keeps
  err[a]   = power(rec - GT) / power(GT)       -- error energy along axis a
  iso_gt   = power_GT[a] / power_GT[0]         -- is the anatomy isotropic enough for axis 0 to be a reference?
  iso_rec  = power_rec[a] / power_rec[0]       -- the same ratio measured on the reconstruction alone (no GT)
The add-on idea is admissible only if iso_gt is close to 1 at high |f| on axes 1/2 while iso_rec is clearly lower.
GT is used here only to judge that premise.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

CMP = Path('/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois_cmp/recon')
EDGES = [0, .05, .1, .15, .2, .3, .4, .5001]


def spectrum(vol, axis):
    """Mean power along `axis` per |f| bin (other axes averaged)."""
    F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(vol, axes=axis), norm='ortho', axis=axis), axes=axis)
    p = (np.abs(F) ** 2).mean(axis=tuple(i for i in range(3) if i != axis))
    f = np.abs(np.fft.fftshift(np.fft.fftfreq(vol.shape[axis])))
    b = np.digitize(f, EDGES) - 1
    return np.array([p[b == k].sum() / max((b == k).sum(), 1) for k in range(len(EDGES) - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', action='append', required=True, help='label=dir with candidate_{pre,post}_complex.npy')
    a = ap.parse_args()
    methods = [('ZS-SSDU', None)] + [tuple(m.split('=', 1)) for m in a.method]
    bands = [f'[{EDGES[k]:.2f},{min(EDGES[k + 1], .5):.2f})' for k in range(len(EDGES) - 1)]
    out = {'bands': bands, 'scans': {}}
    for scan in ('pre', 'post'):
        gt = np.load(CMP / f'gt_{scan}_complex.npy').astype(np.complex64)
        sg = {ax: spectrum(gt, ax) for ax in range(3)}
        res = {'GT': {'iso': {ax: (sg[ax] / sg[0]).tolist() for ax in (1, 2)}}}
        for label, d in methods:
            path = CMP / f'zsssdu_{scan}_complex.npy' if d is None else Path(d) / f'candidate_{scan}_complex.npy'
            rec = np.load(path).astype(np.complex64)
            sr = {ax: spectrum(rec, ax) for ax in range(3)}
            se = {ax: spectrum(rec - gt, ax) for ax in range(3)}
            res[label] = dict(keep={ax: (sr[ax] / sg[ax]).tolist() for ax in range(3)},
                              err={ax: (se[ax] / sg[ax]).tolist() for ax in range(3)},
                              iso={ax: (sr[ax] / sr[0]).tolist() for ax in (1, 2)})
            del rec
        out['scans'][scan] = res
        print(f'\n===== {scan}')
        print(f"{'band |f|':14s}" + ''.join(f'{b:>14s}' for b in bands))
        for ax in (1, 2):
            print(f'GT iso a{ax}/a0  ' + ''.join(f'{v:14.3f}' for v in res['GT']['iso'][ax]))
        for label, _ in methods:
            r = res[label]
            for ax in range(3):
                print(f'{label[:8]:8s} keep a{ax}' + ''.join(f'{v:14.3f}' for v in r['keep'][ax]))
            for ax in range(3):
                print(f'{label[:8]:8s} err  a{ax}' + ''.join(f'{v:14.4f}' for v in r['err'][ax]))
            for ax in (1, 2):
                print(f'{label[:8]:8s} iso a{ax}/a0' + ''.join(f'{v:14.3f}' for v in r['iso'][ax]))
    art = os.environ.get('MCGS_ARTIFACT_DIR')
    if art:
        Path(art, 'diag_direction.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
