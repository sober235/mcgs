"""G0 representation ceiling (spec §5), single instance.

Question: can the Gaussian representation itself approach the voxel-grid
reference under FULL sampling? If it can't even here, every downstream
"routing" conclusion in this project is confounded by a representation
bottleneck, not a routing failure.

Method: same instance, same CSM, mask all-ones (full sampling).
  reference arm : CG-SENSE (full sampling, lam=0) -> E_coil_full (denominator)
  Gaussian arm  : raw complex render from crisp/experiments/g0_train_gaussian.py
                  (cold-start JointTrainer on configs/g0_fullsample.yaml) -> E_coil_full
Pass line (fixed before the experiment): Gaussian E_coil_full <= 2x reference
E_coil_full, at <=1e6 primitives.

---- CSM source (load-bearing, verified empirically — do not "simplify" back to
     load_trainer_mat's csm) ----
`configs/g0_fullsample.yaml` (like the shipped `winner_v2_aligned.yaml`) points
`csm_path` at the ESPIRiT CSM files (`csm_espirit_{T1,FLAIR}.mat`); THAT is the
CSM the Gaussian trainer actually uses as its forward operator S. It is NOT the
same array as the trainer .mat's embedded `map` field: measured relative
Frobenius difference ||csm_trainer_map - csm_espirit|| / ||csm_espirit|| = 1.11
(111%) for T1 — these are two different CSM estimates, not a gauge/global-phase
equivalent pair. If the reference arm used `load_trainer_mat`'s `map` csm (as
the task-brief's illustrative starter code does) while the Gaussian arm trained
against the ESPIRiT csm, the two arms' E_coil_full would be evaluated under
different forward operators A=FS and would not be comparable at all — the
ratio would be meaningless regardless of which side "wins". So both arms here
load csm from the SAME ESPIRiT file the trainer config uses; `load_trainer_mat`
is used only for `ksp` (the raw full k-space, which the trainer config's
`data_path` points at too — matched by construction).

---- phase handicap ----
`tr._render()` (captured by g0_train_gaussian.py) is a genuine complex volume,
trained under the same F S forward convention `e_coil_full` evaluates, so no
phase is stripped and no magnitude-only handicap is needed for the Gaussian
arm. This script reports the complex-vs-complex comparison as the primary
number. If `--gauss_npz` ever points at a magnitude-only array (e.g. someone
substitutes arc/cycle.py's `--save_volumes` export by mistake), the script
detects this via `np.iscomplexobj` and automatically falls back to also
reporting a magnitude-only-handicapped reference figure, clearly labeled, so
the two sides are never silently compared unfairly.

Usage:
  # reference arm only
  PYTHONNOUSERSITE=1 PYTHONPATH=. .../python -m crisp.experiments.g0_ceiling \\
      --gpu 5 --contrast T1
  # with Gaussian arm (after crisp/experiments/g0_train_gaussian.py has run)
  PYTHONNOUSERSITE=1 PYTHONPATH=. .../python -m crisp.experiments.g0_ceiling \\
      --gpu 5 --contrast T1 --gauss_npz results_crisp/g0_full_render.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

from crisp.io import load_trainer_mat, normalize_p999
from crisp.metrics.coil_domain import e_coil_full, ifft3c
from crisp.metrics.image_domain import brain_gauge
from crisp.recon.cg_sense import cg_sense

ROOT = Path("/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
OUT = ROOT / "results_crisp"
BASE_KSP = ("/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/"
            "Train_3dBrain_huangyuwei_20230705-160927-1451_161015_%s_trainer.mat")
BASE_CSM = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_%s.mat"
SLICE_AXIS, SLICE_IDX = 2, 128   # match crisp/experiments/g2_voxel.py's readout-axis slice convention


def _load_espirit_csm(path: str) -> torch.Tensor:
    """Same file + same key + same axis convention as
    paired_gs/data/dataset.py:_load_external_csm (map shape (D,H,W,C) -> permute to (C,D,H,W))."""
    d = sio.loadmat(path)
    csm = torch.from_numpy(d["map"].astype(np.complex64))
    return csm.permute(3, 0, 1, 2).contiguous()


def _pe_slice(vol_mag: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(vol_mag[:, :, SLICE_IDX])


def _panel(gt_mag, zf_mag, rec_mag, peak, contrast, tag, ref_gauge, rec_gauge):
    imgs = [
        (_pe_slice(gt_mag) / peak, "gray", (0, 1.0), f"{contrast}\nGT (peak={peak:.3f})"),
        (_pe_slice(zf_mag) / peak, "gray", (0, 1.0), "zero-fill (mask=1 => adjoint of full k)"),
        (_pe_slice(rec_mag) / peak, "gray", (0, 1.0),
         f"Gaussian recon\nPSNR {rec_gauge['psnr_brain']:.2f}  NMSE {rec_gauge['nmse_brain']:.4f}"),
        (np.abs(_pe_slice(rec_mag) / peak - _pe_slice(gt_mag) / peak), "jet", (0, 0.1),
         "|Gaussian recon - GT|"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, (arr, cmap, (lo, hi), title) in zip(axes, imgs):
        im = ax.imshow(arr.T, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if cmap == "jet":
            fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    fig.suptitle(f"G0 representation ceiling: {contrast} readout slice {SLICE_IDX}, "
                 "error map jet fixed [0,0.1] of GT brain-P99.9 peak", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / f"g0_ceiling_huangyuwei_{contrast}{tag}_panel.png"
    plt.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--contrast", default="T1", choices=("T1", "FLAIR"))
    ap.add_argument("--gauss_npz", default="",
                     help="results_crisp/{exp}_render.npz written by g0_train_gaussian.py")
    ap.add_argument("--gauss_key", default="", help="npz key override (default renderT1/renderFL)")
    ap.add_argument("--max_primitives", type=int, default=1_000_000)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    torch.cuda.set_device(a.gpu)
    OUT.mkdir(exist_ok=True)

    ksp, _unused_trainer_csm = load_trainer_mat(BASE_KSP % a.contrast)
    csm = _load_espirit_csm(BASE_CSM % a.contrast)
    ksp, scale = normalize_p999(ksp, csm)
    ksp, csm = ksp.to(dev), csm.to(dev)
    mask = torch.ones(ksp.shape[1:], dtype=torch.float32, device=dev)

    x_ref = cg_sense(ksp, csm, mask, lam=0.0, iters=60)
    ref = e_coil_full(x_ref, csm, ksp)
    res = {
        "contrast": a.contrast,
        "ksp_source": BASE_KSP % a.contrast,
        "csm_source": BASE_CSM % a.contrast,
        "csm_source_note": ("ESPIRiT csm — the SAME file configs/g0_fullsample.yaml's csm_path "
                             "points at; deliberately NOT load_trainer_mat's embedded `map` field "
                             "(verified 111% relative Frobenius difference between the two, see "
                             "module docstring)"),
        "p999_scale": scale,
        "cg_sense_iters": 60,
        "reference_e_coil_full": ref,
        "pass_line": 2.0 * ref,
        "max_primitives": a.max_primitives,
    }

    if a.gauss_npz:
        key = a.gauss_key or ("renderT1" if a.contrast == "T1" else "renderFL")
        vol = np.load(a.gauss_npz)[key]
        is_complex = bool(np.iscomplexobj(vol))
        x_g = torch.from_numpy(vol.astype(np.complex64)).to(dev)
        res["gaussian_npz"] = a.gauss_npz
        res["gaussian_npz_key"] = key
        res["gaussian_render_is_complex"] = is_complex

        if is_complex:
            res["comparison_handicap"] = ("complex-vs-complex: Gaussian arm is the raw "
                                           "JointTrainer._render() output (real image-domain "
                                           "amplitude+phase, no DC substitution, no magnitude cast); "
                                           "reference arm is the full complex CG-SENSE solution. "
                                           "Neither side has had phase stripped.")
            res["gaussian_e_coil_full"] = e_coil_full(x_g, csm, ksp)
            verdict_ref, verdict_pass = ref, res["pass_line"]
        else:
            # fallback path (not expected given g0_train_gaussian.py's route, but handled honestly
            # per the task brief: never compare a magnitude-only arm against a complex reference)
            res["comparison_handicap"] = ("magnitude-only handicap applied to BOTH arms: the "
                                           "supplied Gaussian render has no phase, so the reference "
                                           "arm's error below is recomputed with its phase stripped "
                                           "too, for an apples-to-apples comparison.")
            x_ref_mag = x_ref.abs().to(torch.complex64)
            res["reference_e_coil_full_magnitude_only"] = e_coil_full(x_ref_mag, csm, ksp)
            res["pass_line_magnitude_only"] = 2.0 * res["reference_e_coil_full_magnitude_only"]
            res["gaussian_e_coil_full"] = e_coil_full(x_g, csm, ksp)
            verdict_ref = res["reference_e_coil_full_magnitude_only"]
            verdict_pass = res["pass_line_magnitude_only"]

        res["ratio_gaussian_over_reference"] = res["gaussian_e_coil_full"] / verdict_ref
        res["verdict"] = "PASS" if res["gaussian_e_coil_full"] <= verdict_pass else "FAIL"

        # mandated image panel: GT | zero-fill | recon | |recon-GT| (jet, fixed [0,0.1] of GT peak)
        gt = (csm.conj() * ifft3c(ksp)).sum(0)
        zf = (csm.conj() * ifft3c(mask.unsqueeze(0) * ksp)).sum(0)  # == gt when mask is all-ones
        gt_mag, zf_mag, rec_mag = gt.abs().cpu().numpy(), zf.abs().cpu().numpy(), x_g.abs().cpu().numpy()
        ref_gauge = brain_gauge(x_ref.abs().cpu().numpy(), gt_mag)
        rec_gauge = brain_gauge(rec_mag, gt_mag)
        res["reference_brain_gauge"] = ref_gauge
        res["gaussian_brain_gauge"] = rec_gauge
        panel_path = _panel(gt_mag, zf_mag, rec_mag, ref_gauge["peak_used"], a.contrast,
                             "", ref_gauge, rec_gauge)
        res["panel_png"] = str(panel_path)

    out_path = OUT / f"g0_ceiling_huangyuwei_{a.contrast}.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
