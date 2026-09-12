#!/usr/bin/env python
"""Stage 1b for one contrast: ESPIRiT maps (house recipe, full resolution), SENSE combination of the
GRAPPA-completed k-space, CG-SENSE directly on the acquired data as the cross-check, vendor comparison
and figures.

    ./py stage1b_sense.py --contrast FA16_TE3.25 --device cuda:0

Outputs in data/<contrast>/: csm_espirit.h5 (dataset "map", (coil, RO, PE, SPE) complex64, unit sum of
squares over the whole FOV as `ecalib -c 0.0` gives), sense_grappa.npy and cgsense.npy ((RO, PE, SPE)
complex64), logs/stage1b_<contrast>.json, figures/<contrast>_sense.png and <contrast>_espirit_maps.png.
"""
import argparse, json, os, shutil, sys, time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

WT = "/data0/congcong/code/GS/mcgs_peds_wt"
sys.path.insert(0, WT)
from uih_mtp.bartio import read_cfl, run_bart, write_cfl      # noqa: E402
from uih_mtp.sense import cg_sense, sense_combine              # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_mtp_grappa import FIGS, TOPIC, VOXEL, corr, flatten, to_vendor_frame   # noqa: E402

ECALIB = ["ecalib", "-t", "0.001", "-c", "0.0", "-r", "24", "-m", "1"]        # frozen house recipe


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def espirit_maps(k: np.ndarray, out_dir: str, logf) -> np.ndarray:
    """(C, X, Y, Z) k-space -> (C, X, Y, Z) maps through BART ecalib on a scratch cfl pair."""
    path = os.path.join(out_dir, "csm_espirit.h5")
    if os.path.exists(path):
        with h5py.File(path, "r") as f:
            return f["map"][()]
    wd = os.path.join(out_dir, "bart"); os.makedirs(wd, exist_ok=True)
    t0 = time.time()
    write_cfl(os.path.join(wd, "ksp"), np.moveaxis(k, 0, 3))                 # BART: (X, Y, Z, C)
    run_bart(ECALIB + [os.path.join(wd, "ksp"), os.path.join(wd, "maps")], threads=8, nice=10, log=logf)
    maps = np.ascontiguousarray(np.moveaxis(np.squeeze(read_cfl(os.path.join(wd, "maps"))), 3, 0)).astype(np.complex64)
    log(f"ecalib done in {time.time() - t0:.0f} s, maps {maps.shape}")
    sos = np.sqrt((np.abs(maps) ** 2).sum(0))
    with h5py.File(path, "w") as f:
        d = f.create_dataset("map", data=maps, chunks=(1,) + maps.shape[1:])
        d.attrs["recipe"] = " ".join(ECALIB); d.attrs["axes"] = "coil, RO, PE, SPE"
        d.attrs["sos_min_mean_max"] = [float(sos.min()), float(sos.mean()), float(sos.max())]
    shutil.rmtree(wd, ignore_errors=True)
    return maps


def figure_sense(sense, cg, vendor, off, name, path):
    g = to_vendor_frame(np.abs(sense), off); c = to_vendor_frame(np.abs(cg), off); v = vendor
    diff = np.abs(to_vendor_frame(cg - sense, off)) / np.percentile(g, 99.9)
    Zn, R, Cn = g.shape; iz, ir, ic = Zn // 2, R // 2, Cn // 2
    cols = [("GRAPPA + ESPIRiT-SENSE combination", g), ("CG-SENSE on the acquired lines", c), ("vendor (uMR)", v)]
    fig, ax = plt.subplots(3, 4, figsize=(18, 13))
    for j, (lab, vol) in enumerate(cols):
        vmax = np.percentile(vol, 99.7)
        ax[0, j].imshow(vol[iz], cmap="gray", vmin=0, vmax=vmax); ax[0, j].set_title(f"{lab}\ntransverse {iz}/{Zn}", fontsize=10)
        ax[1, j].imshow(vol[:, ir, :], cmap="gray", vmin=0, vmax=vmax, aspect=VOXEL[2] / VOXEL[1]); ax[1, j].set_title(f"coronal (row {ir})", fontsize=10)
        ax[2, j].imshow(vol[:, :, ic], cmap="gray", vmin=0, vmax=vmax, aspect=VOXEL[2] / VOXEL[0]); ax[2, j].set_title(f"sagittal (col {ic})", fontsize=10)
    ims = [ax[0, 3].imshow(diff[iz], cmap="jet", vmin=0, vmax=0.1), ax[1, 3].imshow(diff[:, ir, :], cmap="jet", vmin=0, vmax=0.1, aspect=VOXEL[2] / VOXEL[1]),
           ax[2, 3].imshow(diff[:, :, ic], cmap="jet", vmin=0, vmax=0.1, aspect=VOXEL[2] / VOXEL[0])]
    ax[0, 3].set_title("|CG-SENSE - SENSE(GRAPPA)| / P99.9, jet [0, 0.1]", fontsize=10)
    for a in ax.ravel(): a.axis("off")
    fig.colorbar(ims[0], ax=ax[:, 3].tolist(), fraction=0.03, pad=0.01)
    fig.suptitle(f"{name}: two classical parallel-imaging reconstructions of the same prospective 2x data")
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def figure_maps(maps, name, path):
    z = maps.shape[3] // 2
    fig, ax = plt.subplots(2, 8, figsize=(20, 5.6))
    for j, c in enumerate(range(0, maps.shape[0], max(1, maps.shape[0] // 8))[:8]):
        m = maps[c, :, 2:302, z]
        ax[0, j].imshow(np.abs(m), cmap="gray", vmin=0, vmax=1); ax[0, j].set_title(f"coil {c} |S|", fontsize=9)
        ax[1, j].imshow(np.angle(m), cmap="twilight", vmin=-np.pi, vmax=np.pi); ax[1, j].set_title("phase", fontsize=9)
        ax[0, j].axis("off"); ax[1, j].axis("off")
    fig.suptitle(f"{name}: ESPIRiT maps ({' '.join(ECALIB)}), central partition"); fig.tight_layout()
    fig.savefig(path, dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()
    name = args.contrast; d = os.path.join(TOPIC, "data", name); device = torch.device(args.device)
    os.makedirs(os.path.join(TOPIC, "logs"), exist_ok=True); os.makedirs(os.path.join(TOPIC, "figures"), exist_ok=True)
    logf = open(os.path.join(TOPIC, "logs", f"stage1b_{name}_bart.log"), "a")
    with h5py.File(os.path.join(d, "kspace_grappa.h5"), "r") as f:
        k = f["kspace"][()]
    masks = np.load(os.path.join(d, "masks.npz")); m_acq = masks["m_acq"]
    log(f"{name}: k-space {k.shape} loaded, device {device}")

    maps = espirit_maps(k, d, logf)
    sos = np.sqrt((np.abs(maps) ** 2).sum(0))
    log(f"maps SoS min/mean/max {sos.min():.4f}/{sos.mean():.4f}/{sos.max():.4f}")
    figure_maps(maps, name, os.path.join(TOPIC, "figures", f"{name}_espirit_maps.png"))

    kt = torch.from_numpy(k); mt = torch.from_numpy(maps)
    t0 = time.time()
    sense = sense_combine(kt, mt, device=device).numpy()
    log(f"SENSE combination of the GRAPPA k-space {sense.shape}, {time.time() - t0:.0f} s")
    np.save(os.path.join(d, "sense_grappa.npy"), sense)

    t0 = time.time(); hist = []
    y = kt * torch.from_numpy(m_acq)[None, None]
    cg = cg_sense(y, mt, torch.from_numpy(m_acq), iters=args.iters, device=device, history=hist).numpy()
    log(f"CG-SENSE {args.iters} iterations, relative residual {hist[-1]:.3e}, {time.time() - t0:.0f} s")
    np.save(os.path.join(d, "cgsense.npy"), cg)

    head = np.abs(sense) > 0.1 * np.percentile(np.abs(sense), 99.9)
    nrmse_c = float(np.linalg.norm((cg - sense)[head]) / np.linalg.norm(sense[head]))
    nrmse_m = float(np.linalg.norm((np.abs(cg) - np.abs(sense))[head]) / np.linalg.norm(np.abs(sense)[head]))
    vendor = np.load(os.path.join(TOPIC, "data", "vendor", f"M_{name}.npy")).astype(np.float32)
    chk = json.load(open(os.path.join(TOPIC, "logs", f"vendor_check_{name}.json"))); off = chk["pe_offset"]
    grad = lambda v: np.sqrt(sum(np.gradient(v, axis=a) ** 2 for a in range(3)))
    res = {}
    for lab, vol in (("sense_grappa", np.abs(sense)), ("cgsense", np.abs(cg))):
        vv = to_vendor_frame(vol, off)
        res[f"corr_vendor_{lab}"] = corr(vv, vendor)
        res[f"corr_vendor_flat_{lab}"] = corr(flatten(vv), flatten(vendor))
        res[f"corr_vendor_gradient_{lab}"] = corr(grad(flatten(vv)), grad(flatten(vendor)))
    res.update(dict(contrast=name, nrmse_cg_vs_sense_complex_head=nrmse_c, nrmse_cg_vs_sense_magnitude_head=nrmse_m,
                    cg_iters=args.iters, cg_residual_history=hist, head_voxels=int(head.sum()),
                    maps_sos=[float(sos.min()), float(sos.mean()), float(sos.max())], ecalib=" ".join(ECALIB)))
    json.dump(res, open(os.path.join(TOPIC, "logs", f"stage1b_{name}.json"), "w"), indent=1)
    log(f"CG-SENSE vs SENSE(GRAPPA) in the head: complex NRMSE {nrmse_c:.4f}, magnitude NRMSE {nrmse_m:.4f}")
    log("vendor r: " + ", ".join(f"{k_}={v:.3f}" for k_, v in res.items() if k_.startswith("corr_")))
    p = os.path.join(TOPIC, "figures", f"{name}_sense.png"); figure_sense(sense, cg, vendor, off, name, p)
    for f_ in (p, os.path.join(TOPIC, "figures", f"{name}_espirit_maps.png")):
        shutil.copy2(f_, os.path.join(FIGS, os.path.basename(f_)))
    log(f"figures: {p}")


if __name__ == "__main__":
    main()
