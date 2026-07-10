#!/usr/bin/env python3
"""Load a trained ar_mcgs checkpoint and export reconstruction_{plain,contrast}.mat.

Builds the JointTrainer exactly like cycle.py (same base_cfg + kspace_high),
points warm_start_checkpoint at the winner best.pth so the trained Gaussians are
loaded, runs ZERO training iterations, and calls export_results(). The exported
reconstruction is the pure model render + hard-DC + SENSE-combine (self-CSM),
which we then re-score externally with the ESPIRiT CSM for apples-to-apples
comparison against the corrected-baseline / incoh-baseline numbers.
"""
import os, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=2)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--base_cfg", default="configs/r8_vdpois_af12_transverse.yaml")
ap.add_argument("--kspace_high", type=float, default=2.0)
ap.add_argument("--out", default="results_ar_mcgs/win_hi20_4k_export")
args = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

from pathlib import Path
import yaml, torch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from train import set_seed
from trainers.joint_trainer import JointTrainer
from data.private_paths import resolve_subject_paths

set_seed(42, deterministic=False)
cfg = yaml.safe_load(open(ROOT / args.base_cfg))
for s in ("plain", "contrast"):
    sc = cfg["data"][s]
    for k in ("data_path", "csm_path", "mask_path"):
        if sc.get(k):
            sc[k] = resolve_subject_paths(sc[k])
cfg.setdefault("loss", {})["kspace_high_weight"] = args.kspace_high
# load the trained Gaussians as warm start, do not train
cfg.setdefault("gaussian", {})["warm_start_checkpoint"] = str(Path(args.ckpt).resolve())
cfg.setdefault("training", {})["max_iterations"] = 2
cfg["training"]["shared_warmup_iters"] = 0
cfg.setdefault("output", {})["output_dir"] = str(ROOT / args.out)
cfg["output"]["exp_name"] = "win_hi20_4k_export"

dev = torch.device("cuda", 0)
tr = JointTrainer(cfg, device=dev)
try:
    tr.train()           # 0 iters -> just sets up, no optimization
except Exception as e:
    print("train() note:", e)
tr.export_results()
print("EXPORT_DONE ->", cfg["output"]["output_dir"])
