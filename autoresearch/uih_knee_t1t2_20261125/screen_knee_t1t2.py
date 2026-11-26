"""screen_knee_t1t2.py -- read-only QC of the UIH 3D knee set: T2-vs-T1 rigid offset per subject (CPU).

Same two contrast-insensitive metrics and search as the brain screening (screen_t1t2.py: NMI and NGF, 0.25-voxel
coarse then 0.05 fine), on the coil-RSS of the k-space as stored. Axes are the stored spatial axes
(n0, n2, n3) = (phase-encode, readout, phase-encode). Subjects are numbered S01..S16 in sorted path order; no
names are printed. S04, S10 and S16 are other exports (audit README) and are skipped by default.

    gpu.py run --cpu -- @python autoresearch/uih_knee_t1t2_20261125/screen_knee_t1t2.py [S-number ...]
"""
import glob
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "t1t2_aligned_subject_20261113"))
from screen_t1t2 import offset  # noqa: E402

ROOT = "/data2/congcong/data/3D/3D_data20261112/kspace_csm/knee"
DEFAULT = [s for s in range(1, 17) if s not in (4, 10, 16)]


def rss(si, tag):
    subs = sorted({os.path.dirname(p) for p in glob.glob(ROOT + "/**/*.mat", recursive=True)})
    p = glob.glob(subs[si - 1] + "/*_" + tag + "_mx3d*.mat")[0]
    with h5py.File(p, "r") as f:
        k = f["kspace"][()]
    ax = (0, 2, 3)
    acc = np.zeros((k.shape[0], k.shape[2], k.shape[3]))
    for c in range(k.shape[1]):
        x = sfft.fftshift(sfft.ifftn(sfft.ifftshift(k[:, c]), norm="ortho", workers=2))
        acc += np.abs(x) ** 2
    return np.sqrt(acc)


def one(si):
    try:
        t1, t2 = rss(si, "t1"), rss(si, "t2")
        assert t1.shape == t2.shape, (t1.shape, t2.shape)
        r = {"subject": "S%02d" % si, "grid": list(t1.shape),
             "T1-T2": {"ax%d" % a: offset(t1, t2, a, -2.5, 2.5) for a in range(3)}}
    except Exception as ex:
        r = {"subject": "S%02d" % si, "failed": type(ex).__name__}
    print("ROW " + json.dumps(r), flush=True)
    return r


if __name__ == "__main__":
    subs = [int(a) for a in sys.argv[1:]] or DEFAULT
    with Pool(5) as pool:
        rows = pool.map(one, subs)
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if art:
        json.dump(rows, open(os.path.join(art, "screen_knee_t1t2.json"), "w"), indent=1)
    print("SCREEN DONE", len(rows), flush=True)
