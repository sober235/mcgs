"""Smoke test for the planned four-arm density-parameterisation experiment harness.

Validates THREE properties of PairedARCTrainer (paired_arc_gs/train/trainer.py) before
spending 12 real training runs on it (see .superpowers/sdd/2026-08-12-fourarm-harness-smoke.md
for the write-up):

  P1. tr.holdout_kspace_nrmse() returns finite per-contrast values before AND after training,
      and training makes it go DOWN (the model fits something real).
  P2. training.lr.geometry = 0.0 gives EXACTLY zero movement of positions/scales/rotations
      (not merely small) -- geometry-frozen arms really are frozen.
  P3. model.decoder.enable / model.anatomy.latent_dim actually reach the model: an
      "independent" arm (decoder off, two density tables) and a "K2" arm (decoder on,
      latent_dim=2) produce genuinely different held-out scores, and their resolved configs
      + per-Gaussian free-parameter counts are printed so the ladder is auditable.

Data-path block copied verbatim from paired_arc_gs/train/_ablation_layer1.py, with ONE
change: that skeleton points mask_path at mask3d_vdpois_af12_transverse.mat, which project
memory (.claude/memory.md §8) documents as MISREGISTERED (ACS ~22px off the k-space DC,
capturing only 27%/31% of k-space energy -- this was the root cause of the old fake "AF=12
ceiling"). This script uses mask3d_vdpois_af12_transverse_ALIGNED.mat instead.

Read-only on the library: only merge_config / validate_config / PairedARCTrainer are
imported, unmodified, from paired_arc_gs.
"""
import sys, os, json, time, math
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/paired_gs")

import torch
from paired_arc_gs.train.config_schema import merge_config, validate_config
from paired_arc_gs.train.trainer import PairedARCTrainer

OUT = "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_arc/smoke_fourarm"
os.makedirs(OUT, exist_ok=True)

# ---- data-path block, verbatim from _ablation_layer1.py except MSK (see module docstring) ----
T1D = "/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_T1_trainer.mat"
FLD = "/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_FLAIR_trainer.mat"
CT1 = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_T1.mat"
CFL = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_FLAIR.mat"
MSK = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask/mask3d_vdpois_af12_transverse_ALIGNED.mat"  # ALIGNED, not the skeleton's misregistered one

DATA_CFG = {
    "plain":    {"name": "T1",    "data_path": T1D, "csm_path": CT1, "mask_path": MSK,
                 "readout_axis": 2, "phase_axes": [0, 1], "presampled": False, "acs_lines": 24},
    "contrast": {"name": "FLAIR", "data_path": FLD, "csm_path": CFL, "mask_path": MSK,
                 "readout_axis": 2, "phase_axes": [0, 1], "presampled": False, "acs_lines": 24},
}

N_POINTS = int(os.environ.get("SMOKE_N_POINTS", 200000))
ITERS = int(os.environ.get("SMOKE_ITERS", 300))


def snapshot(model):
    return {k: getattr(model, k).detach().clone() for k in ("positions", "scales", "rotations")}


def max_abs_delta(a, b):
    return float((a - b).abs().max())


def build_cfg(exp_name, decoder_enable, latent_dim, hidden_dim=24, n_layers=2, lr_geometry=None):
    model_cfg = {"num_points": N_POINTS,
                 "anatomy": {"latent_dim": latent_dim},
                 "decoder": {"enable": decoder_enable, "hidden_dim": hidden_dim, "n_layers": n_layers}}
    training_cfg = {"stages": {"s2_response_fit": {"iters": ITERS}},
                     "lr": {"latent": 1e-2, "decoder": 1e-2},
                     "eval": {"holdout_fraction": 0.10}}
    if lr_geometry is not None:
        training_cfg["lr"] = dict(training_cfg["lr"], geometry=lr_geometry)
    return merge_config({
        "data": DATA_CFG,
        "model": model_cfg,
        "training": training_cfg,
        "output": {"output_dir": OUT, "exp_name": exp_name, "eval_every": 100},
    })


def density_param_count(model):
    n = model.num_points
    if model.decoder_enable:
        dec_total = sum(p.numel() for p in model.decoder.parameters())
        return {
            "mode": "decoder",
            "per_gaussian_latent_dim": model.latent_dim,
            "per_gaussian_confidence": 1 if model.use_confidence else 0,
            "per_gaussian_dedicated_total": model.latent_dim + (1 if model.use_confidence else 0),
            "decoder_params_total_GLOBAL_shared_not_per_gaussian": dec_total,
            "decoder_params_amortized_per_gaussian": dec_total / n,
        }
    else:
        table_total = sum(p.numel() for p in model.density_tables.parameters())
        return {
            "mode": "independent_tables",
            "per_gaussian_table_params": table_total / n,
            "per_gaussian_confidence": 1 if model.use_confidence else 0,
            "per_gaussian_dedicated_total": table_total / n + (1 if model.use_confidence else 0),
            "table_params_total": table_total,
        }


def run_arm(exp_name, decoder_enable, latent_dim, hidden_dim=24, n_layers=2,
            lr_geometry=None, check_geometry_freeze=False):
    cfg = build_cfg(exp_name, decoder_enable, latent_dim, hidden_dim, n_layers, lr_geometry)
    errs = validate_config(cfg)
    print(f"\n===== ARM {exp_name} =====", flush=True)
    print(f"[{exp_name}] resolved model.anatomy.latent_dim={cfg['model']['anatomy']['latent_dim']} "
          f"model.decoder.enable={cfg['model']['decoder']['enable']} "
          f"training.lr.geometry={cfg['training']['lr']['geometry']}", flush=True)
    if errs:
        print(f"[{exp_name}] [validate_config] {len(errs)} issue(s) -- NOT enforced by the "
              f"trainer (informational only): {errs}", flush=True)

    t0 = time.time()
    tr = PairedARCTrainer(cfg)
    print(f"[{exp_name}] [build] {time.time()-t0:.1f}s  plan={tr.plan.summary()}", flush=True)

    geo_before = snapshot(tr.model) if check_geometry_freeze else None

    ho_pre = tr.holdout_kspace_nrmse()
    tr.train()
    ho_post = tr.holdout_kspace_nrmse()

    out = {
        "exp_name": exp_name,
        "resolved_cfg": {
            "latent_dim": cfg["model"]["anatomy"]["latent_dim"],
            "decoder_enable": cfg["model"]["decoder"]["enable"],
            "hidden_dim": cfg["model"]["decoder"]["hidden_dim"],
            "n_layers": cfg["model"]["decoder"]["n_layers"],
            "lr_geometry": cfg["training"]["lr"]["geometry"],
            "num_points": cfg["model"]["num_points"],
            "iters": ITERS,
        },
        "validate_config_errors": errs,
        "wall_s": round(time.time() - t0, 1),
        "holdout_pre": ho_pre,
        "holdout_post": ho_post,
        "params_summary": tr.model.parameters_summary(),
        "density_param_count": density_param_count(tr.model),
    }

    # ---- P1 check (finite + improved) ----
    p1 = {}
    for k in ho_pre:
        pre_v = ho_pre[k]["holdout_nrmse"]; post_v = ho_post[k]["holdout_nrmse"]
        p1[k] = {"pre": pre_v, "post": post_v,
                  "finite": bool(math.isfinite(pre_v) and math.isfinite(post_v)),
                  "improved": bool(post_v < pre_v)}
    out["p1_check"] = p1

    if check_geometry_freeze:
        geo_after = snapshot(tr.model)
        out["geometry_freeze_check"] = {k: max_abs_delta(geo_before[k], geo_after[k]) for k in geo_before}
        out["stage_opt_groups_s2"] = list(tr.plan.at(0).opt_groups)

    json.dump(out, open(f"{OUT}/{exp_name}.json", "w"), indent=2)

    ho_str = " ".join(f"{v['name']}={v['holdout_nrmse']:.4f}" for v in ho_pre.values())
    hp_str = " ".join(f"{v['name']}={v['holdout_nrmse']:.4f}" for v in ho_post.values())
    print(f"[{exp_name}] holdout NRMSE pre: {ho_str}  ->  post: {hp_str}  wall={out['wall_s']}s", flush=True)
    return out


results = {}
try:
    # P1 + P2 combined: default decoder-enabled arm, geometry LR forced to exactly 0.
    results["geom_freeze_p1p2"] = run_arm("geom_freeze_p1p2", decoder_enable=True, latent_dim=12,
                                           lr_geometry=0.0, check_geometry_freeze=True)
except Exception:
    import traceback; traceback.print_exc()
    results["geom_freeze_p1p2"] = {"error": traceback.format_exc()}
json.dump(results, open(f"{OUT}/summary.json", "w"), indent=2)

try:
    # P3 arm A: independent density tables (decoder disabled) -- the MC-GS-style baseline.
    results["independent"] = run_arm("independent", decoder_enable=False, latent_dim=12)
except Exception:
    import traceback; traceback.print_exc()
    results["independent"] = {"error": traceback.format_exc()}
json.dump(results, open(f"{OUT}/summary.json", "w"), indent=2)

try:
    # P3 arm B: decoder-enabled, latent_dim=2, hidden_dim=24, n_layers=2 (K2).
    results["K2"] = run_arm("K2", decoder_enable=True, latent_dim=2, hidden_dim=24, n_layers=2)
except Exception:
    import traceback; traceback.print_exc()
    results["K2"] = {"error": traceback.format_exc()}
json.dump(results, open(f"{OUT}/summary.json", "w"), indent=2)

print("\nSMOKE_FOURARM_DONE", flush=True)
