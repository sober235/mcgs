"""Fixed-window comparison: input | accepted Tuned | sigma_y runs (whole image + centre ROI)."""
import sys
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ANCHOR = Path('/data0/zhuoxu/code/DualSync/outputs/full38_fourth_20261105/results/tuned')
TAGS = sys.argv[1].split(',')
CASES = ['MSP-FSE-T/slice_06', 'DYF-FSE-T/slice_01']
R = slice(112, 208)
for case in CASES:
    ref = sio.loadmat(str(ANCHOR / (case + '.mat')))
    cols = [('Original input', np.abs(np.squeeze(ref['cg_sense']))), ('Accepted Tuned (final)', np.abs(np.squeeze(ref['diff'])))]
    for tag in TAGS:
        m = sio.loadmat(str(HERE / 'runs' / tag / (case + '.mat')))
        cols.append(('sigma_y=%s' % tag[2:], np.abs(np.squeeze(m['diff']))))
    vmax = np.percentile(cols[0][1], 99.5)
    fig, ax = plt.subplots(2, len(cols), figsize=(4 * len(cols), 8.4))
    for k, (title, img) in enumerate(cols):
        ax[0, k].imshow(img, cmap='gray', vmin=0, vmax=vmax); ax[0, k].set_title(title)
        ax[1, k].imshow(img[R, R], cmap='gray', vmin=0, vmax=vmax, interpolation='nearest')
        for a in ax[:, k]: a.axis('off')
    fig.suptitle(case + ' | shared input p99.5 window | ROI rows/cols 112:208')
    fig.tight_layout()
    out = HERE / 'figures' / (case.replace('/', '_') + '_' + '_'.join(TAGS) + '.png')
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise RuntimeError('refusing to overwrite ' + str(out))
    fig.savefig(out, dpi=110); print('wrote', out)
