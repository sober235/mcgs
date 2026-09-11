#!/usr/bin/env python3
"""fig_three_planes.py -- the row figure of fig_row_final.py in the three planes, stacked into one image per contrast.

  python scripts/fig_three_planes.py --contrast T1 --slices 109 110 151
renders the transverse (axis 0), coronal (axis 1) and sagittal (axis 2) row figures at the given slices and stacks them
top to bottom into t1flair_v3_af16_<contrast>_row3planes_<tra>_<cor>_<sag>.png (served figs folder and figures/).
"""
import argparse, subprocess, sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1", "FLAIR"])
ap.add_argument("--slices", type=int, nargs=3, required=True, metavar=("TRA", "COR", "SAG"))
ap.add_argument("--err_complex", action="store_true", help="pass through to fig_row_final.py")
args = ap.parse_args()

R = Path(__file__).resolve().parents[1]
F = Path("/home/congcongliu/figs/t1flair_v3_af16")
parts = []
for axis, idx in enumerate(args.slices):
    subprocess.run([sys.executable, str(R / "scripts/fig_row_final.py"), "--contrast", args.contrast,
                    "--axis", str(axis), "--index", str(idx)] + (["--err_complex"] if args.err_complex else []), check=True)
    tag = "row_cplxerr" if args.err_complex else "row"
    parts.append(mpimg.imread(F / f"t1flair_v3_af16_{args.contrast}_{tag}_axis{axis}_{idx}.png")[..., :3])
w = max(p.shape[1] for p in parts)
blocks = []
for p in parts:
    blocks.append(np.concatenate([p, np.ones((p.shape[0], w - p.shape[1], 3), p.dtype)], axis=1))
    blocks.append(np.ones((24, w, 3), p.dtype))
img = np.concatenate(blocks[:-1], axis=0)
name = (f"t1flair_v3_af16_{args.contrast}_row3planes{'_cplxerr' if args.err_complex else ''}"
        f"_{'_'.join(map(str, args.slices))}.png")
for d in (F, R / "figures"):
    mpimg.imsave(d / name, img)
print("wrote", F / name, img.shape)
