"""CPU-only: score accepted Tuned at OTHER seeds (existing pool) as if they were candidates.

Tests whether the frozen gates are passable by the accepted method itself under a seed
change. Reads the read-only pool; writes seed_gate_check.json here once. No inference.
"""
import json
from pathlib import Path
import sys

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as harness  # noqa: E402
from evaluate import load_evaluator  # noqa: E402

POOL = Path('/data0/zhuoxu/code/DualSync/outputs/autoresearch_snr_tuned_20261105/pool')


def main():
    harness.require_isolated_python()
    out = HERE / 'seed_gate_check.json'
    if out.exists():
        raise RuntimeError('exists')
    protocol, base = harness.load_protocols()
    ev = load_evaluator(protocol)
    rows = {}
    for case in protocol['tune_cases']:
        ref = sio.loadmat(str(Path(protocol['anchor_root']) / (case + '.mat')))
        anchor_pool = sio.loadmat(str(POOL / case / 'b1234.mat'))
        if not np.array_equal(np.squeeze(anchor_pool['diff']), np.squeeze(ref['diff'])):
            raise ValueError('pool b1234 differs from accepted anchor: ' + case)
        refs = {case: dict(cg_sense=np.squeeze(ref['cg_sense']), tuned=np.squeeze(ref['diff']),
                           coeff=float(np.squeeze(ref['coeff'])),
                           orig_shape=tuple(np.ravel(ref['orig_shape']).astype(int)))}
        for path in sorted((POOL / case).glob('b*.mat')):
            if path.name == 'b1234.mat':
                continue
            m = sio.loadmat(str(path))
            if not np.array_equal(np.squeeze(m['cg_sense']), refs[case]['cg_sense']):
                raise ValueError('preprocessing differs: ' + str(path))
            rep = ev.evaluate_images({case: np.squeeze(m['diff'])}, refs, [case], base)
            r = rep['per_case'][case]
            rows[case + '|' + path.stem] = dict(
                score=rep['score'], feasible=rep['feasible'],
                failed=sorted({v['metric'] for v in rep['violations']}),
                input_ssim_drop=r['tuned_input_coarse_ssim'] - r['input_coarse_ssim'],
                tuned_coarse_ssim=r['tuned_coarse_ssim'], bg=r['bg_snr_gain'], inband=r['inband_snr_gain'],
                source=str(path), sha256=harness.digest(path))
    out.write_text(json.dumps(rows, indent=1, sort_keys=True))
    for k, r in rows.items():
        print('%-30s score %.3f feasible %-5s drop %+.4f ssimT %.4f failed %s' % (
            k, r['score'], r['feasible'], r['input_ssim_drop'], r['tuned_coarse_ssim'], r['failed']))


if __name__ == '__main__':
    main()
