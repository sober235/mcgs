"""Original input vs current best (Yuantian sharp) only. CPU, reads registered arrays, no inference.

Writes original_vs_best_<case>.png and original_vs_best_viewer.html (Ctrl + left click observation).
Same input p99.5 window and ROI as the registration. Refuses to overwrite.
"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'autoresearch_sigma_y_20261106'))
import make_viewer  # noqa: E402


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    viewer_path = HERE / 'original_vs_best_viewer.html'
    assert not viewer_path.exists(), 'refusing to overwrite ' + str(viewer_path)
    data = {}
    for rec in manifest['records']:
        case, scale, (r, c) = rec['case'], rec['input_p99_5_scale'], rec['roi_start']
        mat = loadmat(rec['registered_mat'], simplify_cells=True)
        orig, best = np.abs(mat['cg_sense']) / scale, np.abs(mat['diff']) / scale
        m = rec['metrics_vs_previous_final']
        label = 'bg SNR x%.2f, in-band x%.2f (vs previous final)' % (m['bg_snr_gain'], m['inband_snr_gain'])
        out = HERE / ('original_vs_best_' + case.replace('/', '_') + '.png')
        assert not out.exists(), 'refusing to overwrite ' + str(out)
        fig, ax = plt.subplots(2, 2, figsize=(10, 10.6))
        for k, (title, img) in enumerate((('Original input', orig), ('Current best: Yuantian sharp', best))):
            ax[0, k].imshow(img, cmap='gray', vmin=0, vmax=1)
            ax[0, k].set_title(title, fontsize=13, fontweight='bold')
            ax[1, k].imshow(img[r:r + 96, c:c + 96], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax[1, k].set_xlabel('ROI rows %d:%d, cols %d:%d' % (r, r + 96, c, c + 96), fontsize=9)
            for a in ax[:, k]:
                a.set_xticks([]); a.set_yticks([])
        ax[0, 0].add_patch(plt.Rectangle((c - .5, r - .5), 96, 96, fill=False, ec='#ffd166', lw=1))
        ax[0, 1].add_patch(plt.Rectangle((c - .5, r - .5), 96, 96, fill=False, ec='#ffd166', lw=1))
        fig.suptitle(case + ' | same input p99.5 window | ' + label, fontsize=11)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        data[case] = dict(size=int(orig.shape[0]), panels=[
            dict(name='Original input', label='', ok=None, img=make_viewer.encode(orig)),
            dict(name='Current best: Yuantian sharp', label=label, ok=None, img=make_viewer.encode(best))])
        print('wrote', out)
    html = (make_viewer.TEMPLATE.replace('__DATA__', json.dumps(data)).replace('__Q__', str(make_viewer.Q))
            .replace('repeat(6,1fr)', 'repeat(2,1fr)')
            .replace('.grid{', '.grid{max-width:1200px;')
            .replace('最终图 vs 各候选', '原始输入 vs 当前最优（Yuantian sharp）'))
    viewer_path.write_text(html)
    print('wrote', viewer_path)


if __name__ == '__main__':
    main()
