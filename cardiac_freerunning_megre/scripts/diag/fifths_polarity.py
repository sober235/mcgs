"""Diagnostic (2026-09-12, V03's criterion-(b) disagreement): sharpen the respiratory separation. Reconstruct the
extreme FIFTHS of the PT respiratory signal (echo 3, 112^3, like the thirds) and measure, in the subject's dome and
heart boxes, (i) the NCC shift of the high group against the low group (as s07) and (ii) the position of the
steepest intensity rise along axis2 (lung -> liver edge) of the mean profile in the dome box, for the thirds and the
fifths. Shift > 0 / edge further down = the high group sits towards the feet.  Usage: fifths_polarity.py <Vxx>"""
import json
import sys

import h5py
import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts")
from common import DERIVED_ROOT, LOGS, TOPIC, TRAJ_R, bart, recon_dir, work_dir  # noqa: E402
from cardiac_fr import GRID, KEEP  # noqa: E402

s = sys.argv[1]
ECHO = 3
LAGS = np.arange(-8, 9)
boxes = json.loads((TOPIC / "polarity_boxes.json").read_text())[s]
sp = np.load(DERIVED_ROOT / s / "spokes.npz")
use = sp["steady"] & ~sp["is_si"]
lin, resp = sp["lin"][use], sp["resp"][use]
o = np.argsort(resp)
n5 = resp.size // 5
groups = {"low5": np.sort(lin[o[:n5]]), "high5": np.sort(lin[o[-n5:]])}
wd = work_dir(s)
for name, ids in groups.items():
    if not (recon_dir(s) / f"{name}_e{ECHO}.npy").exists():
        k = np.load(TRAJ_R, mmap_mode="r")[:, KEEP][:, :, ids] * GRID
        bart.write_cfl(wd / f"traj_{name}", bart.bart_traj(np.asarray(k)))
        with h5py.File(DERIVED_ROOT / s / "raw.h5", "r") as h:
            y = h["ksp"][ECHO][ids][:, :, KEEP]
        bart.write_cfl(wd / f"ksp_{name}_e{ECHO}", bart.bart_ksp(y))
        info = bart.pics(wd / f"ksp_{name}_e{ECHO}", DERIVED_ROOT / s / "sens", wd / f"rec_{name}_e{ECHO}",
                         wd / f"traj_{name}", reg="l2", lam=1e-3, iters=100, log=LOGS / f"bart_{s}_{name}_e{ECHO}.log")
        np.save(recon_dir(s) / f"{name}_e{ECHO}.npy", bart.read_cfl(wd / f"rec_{name}_e{ECHO}").astype(np.complex64))
        print(f"{name}: {ids.size} spokes, {info['seconds']:.0f} s", flush=True)


def ncc(u, v):
    u = u - u.mean()
    v = v - v.mean()
    return float(np.sum(u * v) / np.sqrt(np.sum(u * u) * np.sum(v * v)))


def ncc_shift(a, b):                 # b(z) = a(z - d) along axis2, sub-voxel; > 0 = b sits towards the feet
    n = a.shape[-1]
    c = np.array([ncc(a[..., max(0, -L):n - max(0, L)], b[..., max(0, L):n - max(0, -L)]) for L in LAGS])
    i = int(np.argmax(c))
    f = 0.5 * (c[i - 1] - c[i + 1]) / (c[i - 1] - 2 * c[i] + c[i + 1]) if 0 < i < len(c) - 1 else 0.0
    return LAGS[i] + f, c[i]


def edge(v, box):                    # sub-voxel position of the steepest rise of the mean axis2 profile in the box
    (a0, a1), (b0, b1), (c0, c1) = box
    p = v[a0:a1, b0:b1, c0:c1].mean(axis=(0, 1))
    p = np.convolve(p, np.ones(3) / 3, mode="same")
    g = np.gradient(p)
    i = int(np.argmax(g[2:-2])) + 2
    w = g[i - 2:i + 3].clip(min=0)
    return c0 + float((np.arange(i - 2, i + 3) * w).sum() / w.sum())


for lo_name, hi_name, lab in (("low", "high", "thirds"), ("low5", "high5", "fifths")):
    lo = np.abs(np.load(recon_dir(s) / f"{lo_name}_e{ECHO}.npy"))
    hi = np.abs(np.load(recon_dir(s) / f"{hi_name}_e{ECHO}.npy"))
    out = [f"{s} {lab}:"]
    for bname, box in boxes.items():
        sl = tuple(slice(*r) for r in box)
        d, pk = ncc_shift(lo[sl], hi[sl])
        out.append(f"{bname} NCC shift {d:+.2f} vox (peak {pk:.2f})")
    out.append(f"dome edge (axis2) low {edge(lo, boxes['dome']):.2f} -> high {edge(hi, boxes['dome']):.2f} "
               f"({edge(hi, boxes['dome']) - edge(lo, boxes['dome']):+.2f} vox)")
    print(";  ".join(out), flush=True)
print("FIFTHS_DONE", flush=True)
