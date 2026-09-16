#!/usr/bin/env python3
"""ce_gs.py -- GS reconstruction of the UIH prospective CE pair (pre / post contrast).

Three modes, all through the SAME JointTrainer with the SAME frozen working point
(winner-v2 edge-TV levers), so "single" and "joint" differ ONLY in what sits in the
two channels:

  joint       : channel plain = PRE , channel contrast = POST   (shared geometry couples them)
  single_pre  : channel plain = PRE , channel contrast = PRE    (objective is 2x the single-scan
                objective; Adam is invariant to a global loss scale, so this is exactly
                "GS reconstructing PRE alone" with an identical loss form / hyperparameters /
                voxelizer / point budget)
  single_post : same with POST

Export: the RAW render (NO hard data consistency).  Hard-DC would force the on-mask
residual to be identically zero and make every error-vs-y map meaningless.
"""
import os, sys, json, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["joint", "single_pre", "single_post"])
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--iters", type=int, default=4000)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--points", type=int, default=600000)
ap.add_argument("--out", required=True)
ap.add_argument("--fit", action="store_true", help="train on the 95%% lines of the frozen held-out split (2026-09-14)")
ap.add_argument("--root", default="/data0/congcong/code/GS/Paired_ARC_GS_general_framework",
                help="directory holding paired_gs/ (2026-09-15: scripts/_trainer_3dd24aa pins the 09-07 trainer)")
ap.add_argument("--seed", type=int, default=42,
                help="initialisation and training randomness (2026-09-16, for the same-scan null test)")
args = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import torch
import yaml

ROOT = Path(args.root)
sys.path.insert(0, str(ROOT / "paired_gs"))
from seed import set_seed
from trainers.joint_trainer import JointTrainer
from data.transforms import fft3c

CE = Path("/data1/data1_congcong/data/3D/CE_Data")
PRE = dict(name="pre",
           data_path=str(CE / "t1_3D_gre_fsp_sag_iso_ACS_16887049370010/ksp.mat"),
           csm_path=str(CE / "t1_3D_gre_fsp_sag_iso_ACS_16887049370010/csm.mat"),
           mask_path=str(CE / "t1_3D_gre_fsp_sag_iso_ACS_16887049370010/mask.mat"))
POST = dict(name="post",
            data_path=str(CE / "t1_3D_gre_fsp_sag_iso_ACS_16887049370060/ksp2.mat"),
            csm_path=str(CE / "t1_3D_gre_fsp_sag_iso_ACS_16887049370060/csm2.mat"),
            mask_path=str(CE / "t1_3D_gre_fsp_sag_iso_ACS_16887049370060/mask2.mat"))

if args.fit:
    # 2026-09-14: the k-space with the 5% held-out lines zeroed (ce_make_fit_kspace.py, same layout as ksp*.mat).
    # The presampled loader infers its mask from the non-zero samples, so the held-out lines never enter the fit,
    # the zero-filled normalisation or the initialisation.
    HS = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/heldout_split")
    PRE = dict(PRE, data_path=str(HS / "ksp_fit_pre.h5"))
    POST = dict(POST, data_path=str(HS / "ksp_fit_post.h5"))

COMMON = dict(acceleration_factor=5.33, mask_type="external", normalization_percentile=99.9,
              normalize_kspace=True, phase_axes=[1, 2], readout_axis=0,
              presampled=True, skip_readout_check=True)

CHANNELS = {"joint": (PRE, POST), "single_pre": (PRE, PRE), "single_post": (POST, POST)}
plain_src, contrast_src = CHANNELS[args.mode]

cfg = {
    "data": {"plain": {**COMMON, **plain_src}, "contrast": {**COMMON, **contrast_src}},
    "gaussian": {"density_lr": 0.015, "init_scale": 0.008,
                 "initial_num_points": args.points,
                 "position_lr": 0.0003, "rotation_lr": 0.0003, "scale_lr": 0.002},
    "hardware": {"device": "cuda:0"},
    # frozen winner-v2 edge-TV working point (configs/winner_v2_edgeTV.yaml levers)
    "loss": {"loss_type": "l2", "tv_weight": 0.10, "kspace_freq_adaptive": True,
             "kspace_center_fraction": 0.05, "kspace_center_weight": 0.3,
             "kspace_mid_weight": 1.0, "kspace_high_weight": 2.2,
             "edge_tv": True, "edge_tv_kappa_q": 0.85, "edge_tv_wfloor": 0.10,
             "tv_eps": 0.04, "tv_vector": False},
    "mode": {"experimental_reproduction": False, "paper_faithful": False,
             "self_supervised_deploy": True},          # no GT on this data
    "output": {"exp_name": args.mode, "output_dir": args.out},
    "training": {"max_iterations": args.iters, "shared_warmup_iters": args.warmup,
                 "eval_every": 10**9, "log_every": 100, "save_every": 10**9,
                 "seed": args.seed, "max_grad_norm": 1.0,
                 "lr_scheduler": {"type": "exponential", "gamma": 0.9995}},
    "voxelizer": {"max_radius": 20, "tile_size": 8},
}

out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
set_seed(args.seed)
tr = JointTrainer(cfg)
tr.train()

# ---- export: RAW render + best-fit complex scale + relative k-space residual -------------
@torch.no_grad()
def export(which, tag):
    x = tr._render(which)                                   # complex volume, NO hard DC
    csm = getattr(tr, f"csm_{'p' if which == 'plain' else 'c'}")
    y   = getattr(tr, f"kspace_{'p' if which == 'plain' else 'c'}")
    m   = getattr(tr, f"mask_{'p' if which == 'plain' else 'c'}")
    k_model = fft3c(csm * x.unsqueeze(0))
    # alpha = argmin_a || M(a*A(x) - y) ||^2   (scale-invariant comparison across methods)
    num = (m * k_model.conj() * y).sum()
    den = (m * k_model.abs().square()).sum().clamp_min(1e-30)
    alpha = (num / den).item()
    res = m * (alpha * k_model - y)
    kres = float(torch.linalg.vector_norm(res) / torch.linalg.vector_norm(m * y))
    np.save(out_dir / f"{tag}_raw_complex.npy", (alpha * x).cpu().numpy().astype(np.complex64))
    meta = {"mode": args.mode, "channel": which, "tag": tag,
            "alpha_re": float(np.real(alpha)), "alpha_im": float(np.imag(alpha)),
            "kres_internal": kres, "shape": list(x.shape), "iters": args.iters,
            "points": args.points, "warmup": args.warmup}
    json.dump(meta, open(out_dir / f"{tag}_meta.json", "w"), indent=2)
    print(f"[export] {tag}: shape={tuple(x.shape)} |alpha|={abs(alpha):.4f} kres={kres:.5f}", flush=True)

if args.mode == "joint":
    export("plain", "gs_joint_pre")
    export("contrast", "gs_joint_post")
elif args.mode == "single_pre":
    export("plain", "gs_single_pre")
else:
    export("plain", "gs_single_post")
print("DONE", flush=True)
