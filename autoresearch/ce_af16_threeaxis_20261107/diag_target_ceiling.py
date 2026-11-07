#!/usr/bin/env python3
"""diag_target_ceiling.py -- can a BM4D prior supply the missing transverse 0.5 dB at all? (CPU, diagnostic only)

POST-HOC DIAGNOSTIC, NOT AN ADMISSIBLE METHOD OUTPUT (no post-hoc denoising is allowed as a result). It takes the
scored export of a finished run (relax_a1 candidates), replaces the magnitude by a blend of itself and BM4D of it
(phase kept), and scores that with the frozen evaluator through long_chain.score, exactly like the WS probes D1-D3.
If even this ceiling loses to ZS-SSDU on every transverse lesion plane, no NL-target tuning can close the gap with a
BM4D prior; if it wins, the gap is in how the Gaussian render follows the target. Outputs go to
ROOT/probes/<run>/ceiling_*, never into the run directory.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import threeaxis as ta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sigma-rel', type=float, nargs='+', default=[0.01, 0.02])
    ap.add_argument('--weights', type=float, nargs='+', default=[0.5, 1.0])
    ap.add_argument('--threads', type=int, default=24)
    a = ap.parse_args()
    ta.require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('-1', ''), 'CPU only')
    import bm4d
    lc = ta.long_chain()
    run = ta.ROOT / 'runs' / a.run
    src = run / f's{a.seed}' / 'm0625' / 'relax_a1'
    base = ta.axis_summary(run / 'scores' / f's{a.seed}_m0625')
    ta.log(f'{a.run} s{a.seed} as scored: {ta.short(base)}')
    vols = {scan: np.load(src / f'candidate_{scan}_complex.npy', allow_pickle=False) for scan in ('pre', 'post')}
    pro = bm4d.BM4DProfile()
    pro.num_threads = int(a.threads)
    out_root = ta.ROOT / 'probes' / a.run
    result = {'run': a.run, 'seed': a.seed, 'base': dict(m3=base['m3'], short=ta.short(base)),
              'note': 'post-hoc diagnostic, not an admissible method output', 'probes': {}}
    for sr in a.sigma_rel:
        den = {}
        for scan, x in vols.items():
            m = np.abs(x).astype(np.float32)
            sigma = float(sr * np.quantile(m[m > 0.05 * m.max()], 0.99))
            t0 = time.time()
            den[scan] = np.maximum(np.asarray(bm4d.bm4d(m, sigma, pro), dtype=np.float32), 0)
            ta.log(f'  BM4D {scan} sigma_rel {sr:g} (sigma {sigma:.4g}) {time.time() - t0:.0f}s')
        for w in a.weights:
            tag = f'ceiling_s{a.seed}_sr{sr:g}_w{w:g}'
            cand = out_root / tag / 'cand'
            if not (cand / 'relax.json').exists():
                cand.mkdir(parents=True, exist_ok=False)
                rec = {'note': result['note'], 'source': str(src), 'sigma_rel': sr, 'weight': w, 'scans': {}}
                for scan, x in vols.items():
                    m = np.abs(x)
                    mag = (1 - w) * m + w * den[scan]
                    y = (mag * np.exp(1j * np.angle(x))).astype(np.complex64)
                    path = cand / f'candidate_{scan}_complex.npy'
                    np.save(path, y, allow_pickle=False)
                    rec['scans'][scan] = {'path': str(path), 'sha256': ta.sha256(path)}
                ta.write_json(cand / 'relax.json', rec)
            sdir = out_root / tag / 'score'
            if not (sdir / 'objective.json').exists():
                lc.score(cand, sdir, f'{a.run}_{tag}', out_root / tag / f'score_{time.strftime("%H%M%S")}.log')
            s = ta.axis_summary(sdir)
            b0 = s['axes']['0']['best'] if 'axes' in s else None
            result['probes'][tag] = dict(m3=s['m3'], short=ta.short(s), a0_best=b0)
            ta.log(f'{tag}: M3 {s["m3"]:+.5f} ({s["m3"] - base["m3"]:+.5f}) {ta.short(s)}'
                   + (f' | a0 PSNR {b0["psnr"]}' if b0 else ''))
    art = os.environ.get('MCGS_ARTIFACT_DIR')
    if art:
        ta.write_json(Path(art) / f'ceiling_{a.run}_s{a.seed}.json', result)


if __name__ == '__main__':
    main()
