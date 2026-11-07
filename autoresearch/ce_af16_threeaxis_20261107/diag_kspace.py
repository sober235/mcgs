#!/usr/bin/env python3
"""diag_kspace.py -- where in (ky, kz) does MC-GS lose to ZS-SSDU on the transverse lesion planes? (CPU, read-only)

A transverse plane (array axis 0 fixed) is the phase-encode plane: its 2-D FFT is exactly the (ky, kz) grid of the
AF16 mask (constant along readout). For every frozen transverse lesion plane, the complex error e = rec - GT and the
magnitude error |rec| - |GT| are transformed (centred, orthonormal) and their energy is binned by normalised radius
and by sampled / unsampled phase-encode point. Energies are summed over the planes and divided by the summed GT
energy of the same planes. Diagnostic only: nothing here feeds training or the frozen evaluator.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

CMP = Path('/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois_cmp/recon')
GTNPZ = Path('/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois/sim/gt.npz')
EDGES = [0, .03, .06, .1, .15, .2, .3, .5, 1.]


def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho', axes=(-2, -1)), axes=(-2, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--planes-from', required=True, help='a score dir whose lesion_planes.json lists the frozen planes')
    ap.add_argument('--method', action='append', required=True, help='label=dir with candidate_{pre,post}_complex.npy')
    a = ap.parse_args()
    planes = json.loads((Path(a.planes_from) / 'lesion_planes.json').read_text())['planes']
    idx = sorted({p['index'] for p in planes if p['axis'] == 0})
    mask = np.load(GTNPZ)['mask2d'].astype(bool)
    ny, nz = mask.shape
    fy = np.fft.fftshift(np.fft.fftfreq(ny))[:, None]
    fz = np.fft.fftshift(np.fft.fftfreq(nz))[None, :]
    r = np.sqrt(fy ** 2 + fz ** 2)
    band = np.digitize(r, EDGES) - 1
    methods = [('ZS-SSDU', None)] + [tuple(m.split('=', 1)) for m in a.method]
    out = {'planes_axis0': idx, 'edges': EDGES, 'mask_fraction': float(mask.mean()), 'scans': {}}
    for scan in ('pre', 'post'):
        gt = np.load(CMP / f'gt_{scan}_complex.npy', mmap_mode='r')[idx].astype(np.complex64)
        G = fft2c(gt)
        gte = float((np.abs(G) ** 2).sum())
        res = {}
        for label, d in methods:
            path = CMP / f'zsssdu_{scan}_complex.npy' if d is None else Path(d) / f'candidate_{scan}_complex.npy'
            rec = np.load(path, mmap_mode='r')[idx].astype(np.complex64)
            E = np.abs(fft2c(rec - gt)) ** 2
            M = np.abs(fft2c((np.abs(rec) - np.abs(gt)).astype(np.complex64))) ** 2
            row = {}
            for b in range(len(EDGES) - 1):
                sel = band == b
                row[b] = dict(sampled=float(E[:, sel & mask].sum() / gte), unsampled=float(E[:, sel & ~mask].sum() / gte),
                              mag=float(M[:, sel].sum() / gte))
            row['total'] = dict(sampled=sum(v['sampled'] for v in row.values()),
                                unsampled=sum(v['unsampled'] for v in row.values() if isinstance(v, dict)),
                                mag=float(M.sum() / gte))
            res[label] = row
        out['scans'][scan] = res
        labels = [m[0] for m in methods]
        print(f'\n== {scan}: error energy / GT energy (x1e-4), summed over {len(idx)} transverse lesion planes')
        print('band (|f|)        sampled%  ' + '  '.join(f'{l[:10]:>10s}:smp {l[:6]:>6s}:uns {l[:6]:>6s}:mag' for l in labels))
        for b in range(len(EDGES) - 1):
            sel = band == b
            frac = mask[sel].mean() if sel.any() else 0
            cells = '  '.join(f"{res[l][b]['sampled']*1e4:14.2f} {res[l][b]['unsampled']*1e4:10.2f} {res[l][b]['mag']*1e4:10.2f}"
                              for l in labels)
            print(f'[{EDGES[b]:.2f},{EDGES[b+1]:.2f})  {100*frac:7.1f}%  {cells}')
        cells = '  '.join(f"{res[l]['total']['sampled']*1e4:14.2f} {res[l]['total']['unsampled']*1e4:10.2f} "
                          f"{res[l]['total']['mag']*1e4:10.2f}" for l in labels)
        print(f'total              {100*mask.mean():6.1f}%  {cells}')
    art = os.environ.get('MCGS_ARTIFACT_DIR')
    if art:
        Path(art, 'diag_kspace.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
