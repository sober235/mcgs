#!/usr/bin/env python3
"""fig_lesion_surface.py -- the lesion boundary in 3-D: one surface, painted with each arm's error.

The surface is the truth's lesion boundary, extracted once by marching cubes from the core mask (lightly smoothed
so the rendering is not a staircase of voxels) and reused for every panel. Each panel paints that same surface with
that arm's |rec - GT| / max|GT|, sampled on the surface by trilinear interpolation, in the topic's error colours
(jet, [0, 0.1]).

Why the surface is fixed rather than re-extracted per arm: an arm-specific isosurface would need a threshold, and
the threshold operator's own ranking depends on which threshold is chosen (it crowns L1-SPIRiT on this data -- see
SEGMENTATION.md). With one shared geometry the only thing that varies between panels is the error, which is exactly
the quantity the shell RMSE of the axial figure reports as a single number. This is that number, resolved over the
boundary, so it shows *where* on the lesion each arm goes wrong rather than only by how much.

  env PYTHONNOUSERSITE=1 PYTHONPATH= <GSTMI python> scripts/fig_lesion_surface.py [--arms ...] [--elev 18 --azim -60]

Writes figures_seg/ce_retro_af16_lesion_surface.png, copied to ~/figs/ce_retro_seg/.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
PROS = Path(os.environ.get("CE_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast"))
ROI_NPZ = PROS / "lesion_candidates/enhance_rois_cssense.npz"
FIGDIR = R / "figures_seg"
FIGDIR_SERVED = Path("/home/congcongliu/figs/ce_retro_seg")

SCANS = ("pre", "post")
SMOOTH = 0.8            # voxels; enough to take the staircase off the mask, not enough to move the boundary
MCGS = "gs_joint"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="cssense_o200,jcssense,zsssdu,inr,mcinr,gs_single,gs_joint")
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--err_vmax", type=float, default=0.1)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure
    from scipy.ndimage import map_coordinates

    from arms import LABEL, arm_files
    keys = [k.strip() for k in a.arms.split(",") if k.strip()]

    core = np.load(ROI_NPZ, allow_pickle=True)["core"]
    idx = np.argwhere(core)
    pad = 4
    box = tuple(slice(max(int(idx[:, i].min()) - pad, 0), min(int(idx[:, i].max()) + pad + 1, core.shape[i]))
                for i in range(3))
    sub = ndi.gaussian_filter(core[box].astype(np.float64), SMOOTH)
    verts, faces, _, _ = measure.marching_cubes(sub, level=0.5)
    offset = np.array([s.start for s in box], float)
    vert_world = verts + offset            # back into volume coordinates
    print(f"surface: {len(verts)} vertices, {len(faces)} faces from a {int(core.sum())}-voxel lesion", flush=True)

    summary = json.load(open(R / "logs/summary.json"))
    err = {}
    for scan in SCANS:
        gt = np.abs(np.load(R / f"recon/gt_{scan}_complex.npy")).astype(np.float64)
        gmax = float(gt.max())
        for key, _, path in arm_files(scan, summary):
            if key not in keys:
                continue
            v = np.abs(np.load(path)).astype(np.float64)
            e = np.abs(v - gt) / gmax
            # trilinear sampling of the error field at the surface vertices
            err[(scan, key)] = map_coordinates(e, vert_world.T, order=1, mode="nearest")

    ncol = len(keys)
    fig = plt.figure(figsize=(2.35 * ncol, 5.6))
    norm = colors.Normalize(vmin=0.0, vmax=a.err_vmax)
    tri = vert_world[faces]
    for r_i, scan in enumerate(SCANS):
        for j, k in enumerate(keys):
            ax = fig.add_subplot(len(SCANS), ncol, r_i * ncol + j + 1, projection="3d")
            fv = err[(scan, k)][faces].mean(axis=1)          # one colour per triangle
            pc = Poly3DCollection(tri, facecolors=cm.jet(norm(fv)), linewidths=0, shade=True,
                                  lightsource=matplotlib.colors.LightSource(azdeg=225, altdeg=45))
            ax.add_collection3d(pc)
            for dim, setter in zip(range(3), (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
                lo, hi = vert_world[:, dim].min(), vert_world[:, dim].max()
                mid, half = 0.5 * (lo + hi), 0.5 * max(vert_world.ptp(axis=0))
                setter(mid - half, mid + half)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=a.elev, azim=a.azim)
            ax.set_axis_off()
            if r_i == 0:
                ax.set_title(LABEL[k], fontsize=11, pad=0,
                             fontweight="bold" if k == MCGS else "normal")
            if j == 0:
                ax.text2D(-0.06, 0.5, f"{scan}-contrast", transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=11)

    fig.subplots_adjust(left=0.035, right=0.93, top=0.90, bottom=0.03, wspace=0.0, hspace=0.0)
    cax = fig.add_axes([0.945, 0.12, 0.012, 0.72])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="jet"), cax=cax)
    cb.set_label("|rec $-$ GT| / max|GT| on the lesion boundary", fontsize=9)
    cb.ax.tick_params(labelsize=8.5)
    fig.suptitle("CE retro AF 16.00 -- the truth's lesion boundary, painted with each arm's error "
                 "(one surface, only the colour changes)", fontsize=11.5, fontweight="bold", y=0.975)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR_SERVED.mkdir(parents=True, exist_ok=True)
    png = FIGDIR / "ce_retro_af16_lesion_surface.png"
    fig.savefig(png, dpi=a.dpi)
    if a.pdf:
        fig.savefig(FIGDIR / "ce_retro_af16_lesion_surface.pdf")
    plt.close(fig)
    shutil.copy2(png, FIGDIR_SERVED / png.name)
    print(f"wrote {png}\n      http://localhost:8765/ce_retro_seg/{png.name}")
    for scan in SCANS:
        m = {k: float(err[(scan, k)].mean()) for k in keys}
        best = min(m, key=m.get)
        print(f"[{scan}] mean error on the surface: best {LABEL[best]} {m[best]:.5f}, "
              f"MC-GS {m[MCGS]:.5f}, GS-single {m['gs_single']:.5f}")


if __name__ == "__main__":
    main()
