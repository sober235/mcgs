"""CPU-only comparison of the Yuantian port (sharp/balanced/legacy) with the registered final.

Uses the frozen SNR/structure evaluator from autoresearch_sigma_y_20261106 and writes
metrics.json, two PNG panels and viewer.html (Ctrl + left click observation) here.
"""
import json
from pathlib import Path
import sys

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SIG = HERE.parent / 'autoresearch_sigma_y_20261106'
sys.path.insert(0, str(SIG))
import run as harness  # noqa: E402
from evaluate import load_evaluator  # noqa: E402
import make_viewer  # noqa: E402

K2 = HERE.parent / 'autoresearch_snr_refine_20261106' / 'full38_outputs'
CASES = ['MSP-FSE-T/slice_06', 'DYF-FSE-T/slice_01']
CANDIDATES = [('K2 alpha=0.5 (seed avg)', K2),
              ('Yuantian sharp', HERE / 'sharp_seed1234'),
              ('Yuantian balanced', HERE / 'balanced_seed1234'),
              ('Yuantian legacy', HERE / 'legacy_seed1234')]
R = slice(112, 208)


def main():
    harness.require_isolated_python()
    for name in ('metrics.json', 'viewer.html'):
        if (HERE / name).exists():
            raise RuntimeError('refusing to overwrite ' + name)
    protocol, base = harness.load_protocols()
    ev = load_evaluator(protocol)
    metrics, viewer = {}, {}
    for case in CASES:
        ref = sio.loadmat(str(Path(protocol['anchor_root']) / (case + '.mat')))
        cg, tuned = np.squeeze(ref['cg_sense']), np.squeeze(ref['diff'])
        refs = {case: dict(cg_sense=cg, tuned=tuned, coeff=float(np.squeeze(ref['coeff'])),
                           orig_shape=tuple(np.ravel(ref['orig_shape']).astype(int)))}
        scale = float(np.percentile(np.abs(cg), 99.5))
        cols = [('Original input', np.abs(cg), '', None), ('Registered final (Tuned)', np.abs(tuned), 'reference', None)]
        for name, root in CANDIDATES:
            m = sio.loadmat(str(root / (case + '.mat')))
            if not np.array_equal(np.squeeze(m['cg_sense']), cg):
                raise ValueError('preprocessing differs from final: ' + name)
            img = np.squeeze(m['diff'])
            rep = ev.evaluate_images({case: img}, refs, [case], base)
            row = rep['per_case'][case]
            failed = sorted({v['metric'].replace('_', ' ') for v in rep['violations']})
            metrics.setdefault(case, {})[name] = dict(row, failed=failed, source=str(root / (case + '.mat')),
                                                     sha256=harness.digest(root / (case + '.mat')))
            label = 'bg x%.2f, in-band x%.2f, %s' % (row['bg_snr_gain'], row['inband_snr_gain'],
                                                     'passes all gates' if not failed else 'fails: ' + ', '.join(failed))
            cols.append((name, np.abs(img), label, not failed))
            print('%-20s %-24s %s | ssimT %.4f edge %.3f gradRet %.3f' % (
                case, name, label, row['tuned_coarse_ssim'], row['strong_edge_amplitude_ratio'],
                row['raw_gradient_retention']))
        # static PNG
        vmax = np.percentile(cols[0][1], 99.5)
        fig, ax = plt.subplots(2, len(cols), figsize=(3.6 * len(cols), 8.4))
        for k, (title, img, label, ok) in enumerate(cols):
            ax[0, k].imshow(img, cmap='gray', vmin=0, vmax=vmax)
            ax[0, k].set_title(title, fontsize=11, fontweight='bold')
            ax[1, k].imshow(img[R, R], cmap='gray', vmin=0, vmax=vmax, interpolation='nearest')
            ax[1, k].set_xlabel(label.replace(', fails', '\nfails').replace(', passes', '\npasses'), fontsize=8.5,
                                color='black' if ok is None else ('green' if ok else 'firebrick'))
            for a in ax[:, k]:
                a.set_xticks([]); a.set_yticks([])
        fig.suptitle(case + ' | Yuantian port vs registered final | same input p99.5 window | ROI 112:208')
        fig.tight_layout()
        fig.savefig(HERE / ('compare_' + case.replace('/', '_') + '.png'), dpi=110)
        viewer[case] = dict(size=int(cg.shape[0]), panels=[
            dict(name=t, label=l, ok=o, img=make_viewer.encode(img / scale)) for t, img, l, o in cols])
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=1, sort_keys=True))
    html = make_viewer.TEMPLATE.replace('__DATA__', json.dumps(viewer)).replace('__Q__', str(make_viewer.Q))
    (HERE / 'viewer.html').write_text(html.replace('最终图 vs 各候选', 'Yuantian 移植版 vs 最终图'))
    print('wrote metrics.json, compare_*.png, viewer.html')


if __name__ == '__main__':
    main()
