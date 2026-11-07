"""CPU-only side-by-side of all candidate results on the two registered slices.

Columns: input | registered final Tuned | K2 alpha .5 | sigma_y 0.35 | sigma_y 0.5 | eta 0.5.
Shared input p99.5 window, fixed ROI. Per-image labels come from the frozen evaluator
(gains vs final Tuned, gate pass/fail). Reads existing arrays only; no inference.
"""
from pathlib import Path
import sys

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as harness  # noqa: E402
from evaluate import load_evaluator  # noqa: E402

K2 = Path('/data0/zhuoxu/code/DualSync/outputs/autoresearch_snr_refine_20261106/full38_outputs')
CASES = ['MSP-FSE-T/slice_06', 'DYF-FSE-T/slice_01']
CANDIDATES = [('K2 alpha=0.5 (seed avg)', K2), ('sigma_y=0.35', HERE / 'runs/sy0.35'),
              ('sigma_y=0.5', HERE / 'runs/sy0.5'), ('eta=0.5', HERE / 'runs/eta0.5')]
R = slice(112, 208)


def main():
    harness.require_isolated_python()
    protocol, base = harness.load_protocols()
    ev = load_evaluator(protocol)
    for case in CASES:
        ref = sio.loadmat(str(Path(protocol['anchor_root']) / (case + '.mat')))
        refs = {case: dict(cg_sense=np.squeeze(ref['cg_sense']), tuned=np.squeeze(ref['diff']),
                           coeff=float(np.squeeze(ref['coeff'])),
                           orig_shape=tuple(np.ravel(ref['orig_shape']).astype(int)))}
        cols = [('Original input', np.abs(refs[case]['cg_sense']), ''),
                ('Registered final (Tuned)', np.abs(refs[case]['tuned']), 'reference')]
        for name, root in CANDIDATES:
            m = sio.loadmat(str(root / (case + '.mat')))
            if not np.array_equal(np.squeeze(m['cg_sense']), refs[case]['cg_sense']):
                raise ValueError('preprocessing differs: ' + name)
            img = np.squeeze(m['diff'])
            rep = ev.evaluate_images({case: img}, refs, [case], base)
            row = rep['per_case'][case]
            failed = sorted({v['metric'] for v in rep['violations']})
            label = 'bg x%.2f  in-band x%.2f\n%s' % (
                row['bg_snr_gain'], row['inband_snr_gain'],
                'PASSES all gates' if not failed else 'FAILS: ' + ', '.join(f.replace('_', ' ') for f in failed))
            cols.append((name, np.abs(img), label))
            print(case, name, label.replace('\n', ' | '))
        vmax = np.percentile(cols[0][1], 99.5)
        fig, ax = plt.subplots(2, len(cols), figsize=(3.6 * len(cols), 8.2))
        for k, (title, img, label) in enumerate(cols):
            ax[0, k].imshow(img, cmap='gray', vmin=0, vmax=vmax)
            ax[0, k].set_title(title, fontsize=11, fontweight='bold')
            ax[1, k].imshow(img[R, R], cmap='gray', vmin=0, vmax=vmax, interpolation='nearest')
            color = 'green' if label.startswith('bg') and 'PASSES' in label else ('black' if not label.startswith('bg') else 'firebrick')
            ax[1, k].set_xlabel(label, fontsize=9, color=color)
            for a in ax[:, k]:
                a.set_xticks([]); a.set_yticks([])
        fig.suptitle(case + ' | same input p99.5 window | ROI rows/cols 112:208 | gains vs registered final',
                     fontsize=12)
        fig.tight_layout()
        out = HERE / 'figures' / ('best_results_' + case.replace('/', '_') + '.png')
        if out.exists():
            raise RuntimeError('refusing to overwrite ' + str(out))
        fig.savefig(out, dpi=110)
        print('wrote', out)


if __name__ == '__main__':
    main()
