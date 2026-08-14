"""Driver for the four-arm density-parameterisation experiment (§ spec
docs/superpowers/specs/2026-08-12-shared-structure-response-design.md).

Question: does the STRUCTURE of the sharing between the two contrasts (a decoder mapping a
per-Gaussian latent + per-contrast condition code to density) carry information beyond a
shared spatial dictionary with fully independent per-contrast coefficients? Four arms, geometry
frozen and IDENTICAL (warm-started from the same MC-GS checkpoint, training.lr.geometry=0.0),
differing only in how each Gaussian's two complex densities are produced:

    independent : model.decoder.enable = False                     (independent tables)
    K12         : decoder.enable=True, anatomy.latent_dim=12
    K3          : decoder.enable=True, anatomy.latent_dim=3
    K2          : decoder.enable=True, anatomy.latent_dim=2

all with decoder.hidden_dim=24, decoder.n_layers=2.

Built directly on the validated harness (scripts/smoke_fourarm_harness.py + its report
.superpowers/sdd/2026-08-12-fourarm-harness-smoke.md): same data-path block (ALIGNED mask),
same config plumbing (merge_config / PairedARCTrainer), same evidence that
training.lr.geometry=0.0 freezes geometry exactly and that latent_dim reaches the model.
This script ADDS: warm-start from the shipped MC-GS checkpoint (so geometry is the real
working point, not an ACS re-init), per-run seeding, chunked training so the held-out NRMSE
TRAJECTORY can be recorded (not just pre/post), and the §5 lesion-leakage probe.

Single-run CLI (one arm x one seed x optional lesion per process, so waves of 4 can be
launched on the 4 free GPUs by an external orchestrator -- see run_fourarm_waves.sh):

    CUDA_VISIBLE_DEVICES=<gpu> PYTHONNOUSERSITE=1 PYTHONPATH= \
      /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/run_fourarm_response.py \
      --arm K2 --seed 42 --iters 1500 [--lesion]

Does not import or modify anything under paired_gs/, arc/, losses/, paired_arc_gs/,
diag_artefact/, crisp/, or config_schema.py -- only calls merge_config / validate_config /
PairedARCTrainer, all unmodified, exactly as the smoke harness does.
"""
import sys, os, json, time, argparse, traceback

ROOT = "/data0/congcong/code/GS/Paired_ARC_GS_general_framework"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/paired_gs")

import numpy as np
import torch

from paired_arc_gs.train.config_schema import merge_config, validate_config
from paired_arc_gs.train.trainer import PairedARCTrainer
from data.transforms import fft3c, ifft3c

# ---------------------------------------------------------------- data / paths (verbatim
# from scripts/smoke_fourarm_harness.py, which itself copied the block from
# paired_arc_gs/train/_ablation_layer1.py with ONE change: the ALIGNED mask, not the
# misregistered skeleton default -- see .claude/memory.md §8 / this experiment's spec §5.)
T1D = "/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_T1_trainer.mat"
FLD = "/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_FLAIR_trainer.mat"
CT1 = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_T1.mat"
CFL = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_FLAIR.mat"
MSK = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask/mask3d_vdpois_af12_transverse_ALIGNED.mat"  # ALIGNED (spec non-negotiable)
WARM = f"{ROOT}/results_arc/tvs4k_e040/tvs4k_e040/checkpoints/best.pth"  # spec §5 frozen geometry

DATA_CFG = {
    "plain":    {"name": "T1",    "data_path": T1D, "csm_path": CT1, "mask_path": MSK,
                 "readout_axis": 2, "phase_axes": [0, 1], "presampled": False, "acs_lines": 24},
    "contrast": {"name": "FLAIR", "data_path": FLD, "csm_path": CFL, "mask_path": MSK,
                 "readout_axis": 2, "phase_axes": [0, 1], "presampled": False, "acs_lines": 24},
}

# ---------------------------------------------------------------- the four arms (spec table)
ARMS = {
    "independent": dict(decoder_enable=False, latent_dim=12),   # latent_dim unused (no decoder)
    "K12":         dict(decoder_enable=True,  latent_dim=12),
    "K3":          dict(decoder_enable=True,  latent_dim=3),
    "K2":          dict(decoder_enable=True,  latent_dim=2),
}
HIDDEN_DIM = 24
N_LAYERS = 2

LESION_SEED = 42          # fixed seed the 4 lesion runs use; its no-lesion companion is
                           # reused from the 12-run main sweep (spec §8 run budget).
LESION_RADIUS_VOX = 6
LESION_MIN_BOUNDARY_DIST_VOX = 15
LESION_AMPLITUDE_FRAC = 0.30


def build_cfg(exp_name, decoder_enable, latent_dim, seed, iters, out_dir):
    model_cfg = {"warm_start_path": WARM,
                 "anatomy": {"latent_dim": latent_dim},
                 "decoder": {"enable": decoder_enable, "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS}}
    training_cfg = {"stages": {"s2_response_fit": {"iters": iters}},
                     "lr": {"latent": 1e-2, "decoder": 1e-2, "geometry": 0.0},  # spec §5 geometry frozen
                     "eval": {"holdout_fraction": 0.10}}
    return merge_config({
        "data": DATA_CFG,
        "model": model_cfg,
        "training": training_cfg,
        "output": {"output_dir": out_dir, "exp_name": exp_name, "eval_every": 500},
        "seed": seed,
    })


def snapshot(model):
    return {k: getattr(model, k).detach().clone() for k in ("positions", "scales", "rotations")}


def max_abs_delta(a, b):
    return float((a - b).abs().max())


def density_param_count(model):
    n = model.num_points
    if model.decoder_enable:
        dec_total = sum(p.numel() for p in model.decoder.parameters())
        return {"mode": "decoder", "per_gaussian_latent_dim": model.latent_dim,
                "per_gaussian_confidence": 1 if model.use_confidence else 0,
                "per_gaussian_dedicated_total": model.latent_dim + (1 if model.use_confidence else 0),
                "decoder_params_total_GLOBAL_shared_not_per_gaussian": dec_total,
                "decoder_params_amortized_per_gaussian": dec_total / n}
    table_total = sum(p.numel() for p in model.density_tables.parameters())
    return {"mode": "independent_tables", "per_gaussian_table_params": table_total / n,
            "per_gaussian_confidence": 1 if model.use_confidence else 0,
            "per_gaussian_dedicated_total": table_total / n + (1 if model.use_confidence else 0),
            "table_params_total": table_total}


# ---------------------------------------------------------------- §5 lesion leakage probe
def find_lesion_center(flair_abs_np, radius=LESION_RADIUS_VOX, min_boundary_dist=LESION_MIN_BOUNDARY_DIST_VOX):
    """Deterministic (no RNG) center in white matter, >= min_boundary_dist voxels from the WM
    mask's own boundary. WM = brain-percentile band [55,90], the same convention used for the
    WM mask in .claude/memory.md §14 ('gt in [P55,P90] of brain' + a sigma=2 opening)."""
    from scipy import ndimage
    brain = flair_abs_np > 0.05 * flair_abs_np.max()
    lo, hi = np.percentile(flair_abs_np[brain], [55, 90])
    wm = brain & (flair_abs_np >= lo) & (flair_abs_np <= hi)
    wm_smooth = ndimage.gaussian_filter(wm.astype(np.float32), sigma=2) > 0.5
    dist = ndimage.distance_transform_edt(wm_smooth)
    eligible = dist >= min_boundary_dist
    if not eligible.any():
        raise RuntimeError(f"no WM voxel >= {min_boundary_dist}vox from its own mask boundary "
                            f"(max available dist = {float(dist.max()):.1f}vox) -- lesion spec unsatisfiable")
    dist_masked = np.where(eligible, dist, -1.0)
    center = np.unravel_index(int(np.argmax(dist_masked)), dist_masked.shape)
    center = tuple(int(c) for c in center)
    return center, float(dist_masked[center])


def sphere_mask_np(volume_shape, center, radius):
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in volume_shape], indexing="ij")
    d2 = (ii - center[0]) ** 2 + (jj - center[1]) ** 2 + (kk - center[2]) ** 2
    return d2 <= radius ** 2


def inject_lesion(tr):
    """Add a sphere lesion to the FLAIR image, then re-forward through S_FLAIR/mask_FLAIR so it
    enters ONLY tr.ksp_c (FLAIR's measured k-space) -- 'before the forward operator' per spec §5.
    tr.gt_c / tr.ksp_p / tr.mask_p / tr.csm_p (T1 side) are untouched."""
    flair_abs = tr.gt_c.detach().abs().cpu().numpy()
    center, margin = find_lesion_center(flair_abs)
    mask_np = sphere_mask_np(tr.volume_shape, center, LESION_RADIUS_VOX)
    mask_t = torch.from_numpy(mask_np).to(tr.device).to(tr.gt_c.real.dtype)
    lesion_image = tr.gt_c * (1.0 + LESION_AMPLITUDE_FRAC * mask_t)   # +30% amplitude, phase kept
    with torch.no_grad():
        k_full_lesion = fft3c(tr.csm_c * lesion_image.unsqueeze(0))
        k_us_lesion = k_full_lesion * tr.mask_c.unsqueeze(0)
    tr.ksp_c = k_us_lesion.contiguous()
    info = {"center_vox": list(center), "radius_vox": LESION_RADIUS_VOX,
            "n_voxels": int(mask_np.sum()), "boundary_margin_vox": margin,
            "amplitude_frac": LESION_AMPLITUDE_FRAC}
    return info, mask_np


def reconstruct_dc(tr, ci):
    """Hard-DC + conj-SENSE combine of contrast ci's current render -- same physics as
    PairedARCTrainer.evaluate() (trainer.py:559-565), refactored here (read-only use, no
    edits to the trainer) so it can be called without requiring ground truth."""
    with torch.no_grad():
        x = tr.render(ci)
        csm, ksp, mask = (tr.csm_p, tr.ksp_p, tr.mask_p) if ci == 0 else (tr.csm_c, tr.ksp_c, tr.mask_c)
        k_model = fft3c(csm * x.unsqueeze(0))
        m = mask if mask.ndim == k_model.ndim else mask.unsqueeze(0)
        k_dc = m * ksp + (1 - m) * k_model
        x_dc = (ifft3c(k_dc) * csm.conj()).sum(0)
    return x_dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS.keys()))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--chunk", type=int, default=100, help="iters per tr.train() call, for the "
                     "held-out NRMSE trajectory (spec §4 convergence gate needs this, not just pre/post)")
    ap.add_argument("--lesion", action="store_true")
    ap.add_argument("--out_dir", default=f"{ROOT}/results_arc/fourarm")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_id = f"{args.arm}_seed{args.seed}" + ("_lesion" if args.lesion else "")
    arm_spec = ARMS[args.arm]

    record = {"run_id": run_id, "arm": args.arm, "seed": args.seed, "lesion": args.lesion,
              "iters": args.iters, "chunk": args.chunk}
    try:
        # seed controls model init (z_A, decoder weights / condition codes, or the independent
        # density tables' torch.randn init) -- NOT the fit/audit/test k-space split, which stays
        # at training.audit.seed's default (42) for every run so all arms/seeds are scored on
        # the EXACT SAME held-out points (seed variation must reflect only model-init noise).
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        cfg = build_cfg(run_id, arm_spec["decoder_enable"], arm_spec["latent_dim"],
                         args.seed, args.chunk, args.out_dir)
        errs = validate_config(cfg)
        exempt = [e for e in errs if "anatomy.latent_dim" in e]
        other_errs = [e for e in errs if e not in exempt]
        print(f"[{run_id}] resolved latent_dim={cfg['model']['anatomy']['latent_dim']} "
              f"decoder.enable={cfg['model']['decoder']['enable']} "
              f"hidden_dim={cfg['model']['decoder']['hidden_dim']} n_layers={cfg['model']['decoder']['n_layers']} "
              f"lr.geometry={cfg['training']['lr']['geometry']} warm_start={cfg['model']['warm_start_path']}", flush=True)
        if exempt:
            print(f"[{run_id}] validate_config: {exempt} -- EXEMPTED per spec §5b "
                  f"(K>=2 relaxation for THIS experiment's configs only; config_schema.py untouched, "
                  f"trainer never calls validate_config internally, so this is informational-only "
                  f"and does not block).", flush=True)
        if other_errs:
            print(f"[{run_id}] validate_config OTHER ERRORS (not exempted): {other_errs}", flush=True)
        record["validate_config_errors"] = errs
        record["validate_config_exempted"] = exempt

        t_build0 = time.time()
        tr = PairedARCTrainer(cfg)
        record["wall_build_s"] = round(time.time() - t_build0, 1)
        record["num_points"] = tr.model.num_points
        record["params_summary"] = tr.model.parameters_summary()
        record["density_param_count"] = density_param_count(tr.model)

        geo_before = snapshot(tr.model)

        lesion_info, lesion_mask_np = (None, None)
        if args.lesion:
            lesion_info, lesion_mask_np = inject_lesion(tr)
            record["lesion_info"] = lesion_info
            print(f"[{run_id}] LESION injected into FLAIR only: {lesion_info}", flush=True)

        traj = []
        ho0 = tr.holdout_kspace_nrmse()
        traj.append({"iter": 0, "T1": ho0["plain"]["holdout_nrmse"], "FLAIR": ho0["contrast"]["holdout_nrmse"]})
        print(f"[{run_id}] iter0 T1={traj[-1]['T1']:.4f} FLAIR={traj[-1]['FLAIR']:.4f}", flush=True)

        n_chunks = args.iters // args.chunk
        remainder = args.iters % args.chunk
        assert remainder == 0, f"iters={args.iters} not divisible by chunk={args.chunk}"

        t_train0 = time.time()
        cum = 0
        for c in range(n_chunks):
            tr.train()
            cum += args.chunk
            ho = tr.holdout_kspace_nrmse()
            traj.append({"iter": cum, "T1": ho["plain"]["holdout_nrmse"], "FLAIR": ho["contrast"]["holdout_nrmse"]})
            print(f"[{run_id}] iter{cum} T1={traj[-1]['T1']:.4f} FLAIR={traj[-1]['FLAIR']:.4f}", flush=True)
        record["wall_train_s"] = round(time.time() - t_train0, 1)
        record["trajectory"] = traj
        record["holdout_pre"] = ho0
        record["holdout_post"] = ho

        geo_after = snapshot(tr.model)
        record["geometry_freeze_check"] = {k: max_abs_delta(geo_before[k], geo_after[k]) for k in geo_before}

        # ---- §5b companion T1 recon: save (no-lesion, fixed-seed runs) or compare (lesion runs)
        if not args.lesion and args.seed == LESION_SEED:
            x_dc = reconstruct_dc(tr, 0)
            mag = x_dc.abs().cpu().numpy().astype(np.float32)
            np.save(f"{args.out_dir}/{args.arm}_seed{args.seed}_T1recon.npy", mag)
            record["t1_recon_saved"] = True
            record["t1_peak"] = float(tr.gt_p.abs().max())
            print(f"[{run_id}] saved companion T1 recon for lesion comparison, t1_peak={record['t1_peak']:.4f}", flush=True)

        if args.lesion:
            companion_json = f"{args.out_dir}/{args.arm}_seed{LESION_SEED}.json"
            companion_npy = f"{args.out_dir}/{args.arm}_seed{LESION_SEED}_T1recon.npy"
            if not (os.path.exists(companion_json) and os.path.exists(companion_npy)):
                raise RuntimeError(f"companion no-lesion run missing ({companion_json} / {companion_npy}); "
                                    f"run the 12-run main sweep (seed {LESION_SEED} in particular) first")
            companion = json.load(open(companion_json))
            t1_peak = companion["t1_peak"]
            t1_without = np.load(companion_npy)
            x_dc_with = reconstruct_dc(tr, 0)
            t1_with = x_dc_with.abs().cpu().numpy().astype(np.float32)
            diff = t1_with - t1_without
            rms_in_sphere = float(np.sqrt(np.mean(diff[lesion_mask_np] ** 2)))
            leakage_reading = rms_in_sphere / t1_peak
            record["leakage"] = {"rms_diff_in_sphere": rms_in_sphere, "t1_peak": t1_peak,
                                  "leakage_reading": leakage_reading, **lesion_info}
            print(f"[{run_id}] LEAKAGE reading (RMS(T1_with-T1_without)/T1_peak in sphere) = {leakage_reading:.6f}", flush=True)

        record["status"] = "ok"

    except Exception:
        record["status"] = "error"
        record["traceback"] = traceback.format_exc()
        print(f"[{run_id}] ERROR:\n{record['traceback']}", flush=True)

    with open(f"{args.out_dir}/{run_id}.json", "w") as f:
        json.dump(record, f, indent=2)
    print(f"[{run_id}] DONE status={record['status']}", flush=True)


if __name__ == "__main__":
    main()
