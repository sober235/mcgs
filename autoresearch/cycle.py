#!/usr/bin/env python3
"""ar_mcgs — one autoresearch cycle on the REAL MC-GS joint model (JointTrainer).

Trains the genuine shared-geometry + dual-density joint model (NOT the single-contrast
frozen-scaffold ablation in scheme_v1/ar) on HYW T1+FLAIR with the vd-Poisson mask, then
scores BOTH contrasts with the artifact-centric HFEN composite (hard-DC + SENSE-combine vs GT).

Anti-aliasing lever order (cs:s1-449): shared-geometry / cross-contrast coupling is PRIMARY
(shared_warmup_iters, lambda_dtv, densification); TV(lambda_geo) is a SECONDARY cleanup lever.

Metric contract mirrors scheme_v1/ar/prepare.py (DO NOT weaken):
  hfen = 3D-Laplacian high-freq error norm vs GT (covers axial+coronal+sagittal).
  per-contrast composite c = psnr + 25*ssim - 60*nmse - 25*hfen
  combined score = mean(c_T1, c_FLAIR); both contrasts must beat ZF on ssim/nmse/hfen.

Run (one cell, pinned to one GPU):
  PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python scheme_v1/ar_mcgs/cycle.py --exp A_base --gpu 1 --iters 1500
"""
import os, sys, time, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--gpu", type=int, default=1)
ap.add_argument("--iters", type=int, default=1500)
# --- tunable knobs (defaults = vdpois r8 baseline) ---
ap.add_argument("--lambda_geo", type=float, default=0.03)        # loss.tv_weight (-> lambda_geo); SECONDARY
ap.add_argument("--warmup", type=int, default=500)               # training.shared_warmup_iters; PRIMARY (guard <0.5*iters)
ap.add_argument("--lambda_dtv", type=float, default=0.0)         # loss.lambda_dtv cross-contrast coupling; PRIMARY
ap.add_argument("--dtv_eta_scale", type=float, default=1.0)
ap.add_argument("--max_radius", type=int, default=20)            # voxelizer.max_radius; geometry expressivity
ap.add_argument("--kspace_high", type=float, default=1.0)        # loss.kspace_high_weight (FLAIR/shared); SECONDARY
ap.add_argument("--kspace_high_plain", type=float, default=-1)   # decoupled T1(plain) high-band weight (-1=share)
ap.add_argument("--densify_until", type=int, default=-1)         # adaptive_control.densify_until_iter (-1=leave)
ap.add_argument("--max_points", type=int, default=-1)            # gaussian.max_num_points (-1=leave)
ap.add_argument("--base_cfg", default="configs/r8_vdpois_af12.yaml")  # mask/framework base (e.g. AF10 sweep)
ap.add_argument("--curriculum", action="store_true")            # loss.kspace_freq_curriculum (progressive high-freq)
args = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

from pathlib import Path
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent           # scheme_v1/ar_mcgs
ROOT = HERE.parents[1]                            # 3dgsVC
sys.path.insert(0, str(ROOT))
from train import set_seed
from trainers.joint_trainer import JointTrainer
from data.private_paths import resolve_subject_paths
from data.transforms import fft3c, ifft3c
from metrics import evaluate_reconstruction

BASE_CFG = ROOT / "configs/r8_vdpois_af12.yaml"
RES = ROOT / "results_ar_mcgs"
log = lambda *a: print(*a, flush=True)


def hfen(rec, gt):
    """3D-Laplacian high-freq error norm vs GT (lower = fewer aliasing/speckle residuals)."""
    def lap(v):
        return (6.0 * v - torch.roll(v, 1, -3) - torch.roll(v, -1, -3)
                - torch.roll(v, 1, -2) - torch.roll(v, -1, -2)
                - torch.roll(v, 1, -1) - torch.roll(v, -1, -1))
    g = gt / (gt.max() + 1e-9)
    r = rec / (gt.max() + 1e-9)
    lg = lap(g)
    return float(torch.linalg.norm(lap(r) - lg) / (torch.linalg.norm(lg) + 1e-9))


def sharp_ratio(rec, gt):
    """Edge-sharpness ratio: mean per-voxel gradient magnitude rec/GT over the brain mask.
    ~1.0 = matched sharpness; <1 = blurred (under-shoots GT edges); >1 = over-sharp/noisy.
    Explicit anti-blur diagnostic so the optimizer can't trade aliasing for edge softening."""
    def gm(v):
        gx = torch.zeros_like(v); gy = torch.zeros_like(v); gz = torch.zeros_like(v)
        gx[:-1] = v[1:] - v[:-1]; gy[:, :-1] = v[:, 1:] - v[:, :-1]; gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
        return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
    brain = gt > 0.05 * gt.max()
    return float(gm(rec)[brain].mean() / (gm(gt)[brain].mean() + 1e-9))


def grain(rec, gt):
    """Non-anatomical grain: std of the high-pass residual (rec - smooth(rec)) over GT-FLAT brain
    voxels (where GT gradient is in the lowest 30%). Isolates speckle/grain in homogeneous tissue
    from genuine edge structure, so it is separable from sharp_ratio (which lives at edges).
    Lower = cleaner flat regions. Normalised by GT max."""
    import torch.nn.functional as F
    g = gt / (gt.max() + 1e-9)
    r = rec / (gt.max() + 1e-9)
    sm = F.avg_pool3d(r[None, None], kernel_size=3, stride=1, padding=1)[0, 0]   # local mean
    hp = r - sm                                                                   # high-pass residual
    def gmag(v):
        gx = torch.zeros_like(v); gy = torch.zeros_like(v); gz = torch.zeros_like(v)
        gx[:-1] = v[1:] - v[:-1]; gy[:, :-1] = v[:, 1:] - v[:, :-1]; gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
        return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
    brain = g > 0.05 * g.max()
    gg = gmag(g)
    thr = torch.quantile(gg[brain], 0.30)            # GT-flat = lowest-30% gradient inside brain
    flat = brain & (gg <= thr)
    return float(hp[flat].std())


def build_cfg():
    cfg = yaml.safe_load(open(ROOT / args.base_cfg))
    for s in ("plain", "contrast"):
        sc = cfg["data"][s]
        for k in ("data_path", "csm_path", "mask_path"):
            if sc.get(k):
                sc[k] = resolve_subject_paths(sc[k])
    cfg.setdefault("training", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("voxelizer", {})
    cfg.setdefault("gaussian", {})
    cfg.setdefault("adaptive_control", {})
    cfg["training"]["max_iterations"] = args.iters
    cfg["training"]["shared_warmup_iters"] = args.warmup
    cfg["loss"]["tv_weight"] = args.lambda_geo
    cfg["loss"]["lambda_dtv"] = args.lambda_dtv
    cfg["loss"]["dtv_eta_scale"] = args.dtv_eta_scale
    cfg["loss"]["kspace_high_weight"] = args.kspace_high
    if args.kspace_high_plain > 0:
        cfg["loss"]["kspace_high_weight_plain"] = args.kspace_high_plain
    cfg["voxelizer"]["max_radius"] = args.max_radius
    if args.curriculum:
        cfg["loss"]["kspace_freq_curriculum"] = True
        cfg["loss"].setdefault("kspace_curriculum_frac", 0.7)
        cfg["loss"].setdefault("kspace_curriculum_rc_start", 0.12)
        cfg["loss"].setdefault("kspace_curriculum_tail_w", 0.05)
    if args.densify_until > 0:
        cfg["adaptive_control"]["densify_until_iter"] = args.densify_until
    if args.max_points > 0:
        cfg["gaussian"]["max_num_points"] = args.max_points
    cfg.setdefault("output", {})
    cfg["output"]["output_dir"] = str(RES / args.exp)
    cfg["output"]["exp_name"] = args.exp
    return cfg


def score_contrast(tr, which, dev):
    """hard-DC synth + SENSE-combine vs GT on raw model output, + HFEN composite."""
    if which == "plain":
        csm, y, mask, ds = tr.csm_p, tr.kspace_p, tr.mask_p, tr.ds_p
    else:
        csm, y, mask, ds = tr.csm_c, tr.kspace_c, tr.mask_c, tr.ds_c
    gt = ds.get_data()["ground_truth"].to(dev)
    m = (mask if mask.ndim == 3 else mask[0]).to(dev).float()
    with torch.no_grad():
        x = tr._render(which)
        k_model = fft3c(csm * x.unsqueeze(0))
        k_dc = m.unsqueeze(0) * y + (1.0 - m).unsqueeze(0) * k_model
        x_dc = (ifft3c(k_dc) * csm.conj()).sum(0)
        met = evaluate_reconstruction(x_dc, gt, compute_extended=False)
        hf = hfen(x_dc.abs(), gt.abs())
        sr = sharp_ratio(x_dc.abs(), gt.abs())
        gr = grain(x_dc.abs(), gt.abs())
        # ZF anchor (lower bound this contrast)
        x_zf = (ifft3c(m.unsqueeze(0) * y) * csm.conj()).sum(0)
        mz = evaluate_reconstruction(x_zf, gt, compute_extended=False)
        zf_hf = hfen(x_zf.abs(), gt.abs())
    psnr, ssim, nmse = float(met["psnr"]), float(met["ssim"]), float(met["nmse"])
    # tri-objective: anti-alias (low hfen) + anti-blur (sharp->1, one-sided) + anti-grain (low flat-
    # region speckle). grain and sharp pull opposite ways -> the optimum is edge-preserving denoise.
    comp = psnr + 25.0 * ssim - 60.0 * nmse - 25.0 * hf - 40.0 * max(0.0, 1.0 - sr) - 200.0 * gr
    beat = (ssim >= float(mz["ssim"])) and (nmse <= float(mz["nmse"])) and (hf <= zf_hf)
    return dict(psnr=psnr, ssim=ssim, nmse=nmse, hfen=hf, sharp=sr, grain=gr, comp=comp, beat=beat,
                zf=(float(mz["psnr"]), float(mz["ssim"]), float(mz["nmse"]), zf_hf))


def main():
    t0 = time.time()
    dev = torch.device("cuda", 0)
    set_seed(42, deterministic=False)
    assert args.warmup < args.iters * 0.5, f"warmup {args.warmup} must be < 0.5*iters {args.iters}"
    cfg = build_cfg()
    RES.mkdir(parents=True, exist_ok=True)
    log(f"[ar_mcgs {args.exp}] gpu={args.gpu} iters={args.iters} lambda_geo={args.lambda_geo} "
        f"warmup={args.warmup} lambda_dtv={args.lambda_dtv} max_radius={args.max_radius} "
        f"kspace_high={args.kspace_high}")
    tr = JointTrainer(cfg, device=dev)
    tr.train()

    sp = score_contrast(tr, "plain", dev)
    sc = score_contrast(tr, "contrast", dev)
    combined = 0.5 * (sp["comp"] + sc["comp"])
    both_beat = sp["beat"] and sc["beat"]
    vram = torch.cuda.max_memory_allocated() / 1e9
    npts = tr.model.num_points

    log(f"T1  psnr={sp['psnr']:.3f} ssim={sp['ssim']:.4f} nmse={sp['nmse']:.4f} hfen={sp['hfen']:.4f} "
        f"sharp={sp['sharp']:.3f} grain={sp['grain']:.4f} comp={sp['comp']:.3f} beat={sp['beat']}")
    log(f"FLAIR psnr={sc['psnr']:.3f} ssim={sc['ssim']:.4f} nmse={sc['nmse']:.4f} hfen={sc['hfen']:.4f} "
        f"sharp={sc['sharp']:.3f} grain={sc['grain']:.4f} comp={sc['comp']:.3f} beat={sc['beat']}")
    log(f"SCORE combined={combined:.3f} both_beat_zf={both_beat} N={npts} vram_gb={vram:.1f} "
        f"elapsed_s={time.time()-t0:.0f}")

    row = dict(exp=args.exp, iters=args.iters, lambda_geo=args.lambda_geo, warmup=args.warmup,
               lambda_dtv=args.lambda_dtv, max_radius=args.max_radius, kspace_high=args.kspace_high,
               T1=sp, FLAIR=sc, combined=combined, both_beat=both_beat, npts=int(npts), vram_gb=vram)
    (RES / f"{args.exp}_score.json").write_text(json.dumps(row, indent=2))
    tsv = RES / "results_grain.tsv"
    head = "exp\titers\tlambda_geo\twarmup\tlambda_dtv\tmax_radius\tkspace_high\tT1_psnr\tT1_ssim\tT1_nmse\tT1_hfen\tT1_sharp\tT1_grain\tFLAIR_psnr\tFLAIR_ssim\tFLAIR_nmse\tFLAIR_hfen\tFLAIR_sharp\tFLAIR_grain\tcombined\tboth_beat\tnpts\tvram_gb\n"
    line = (f"{args.exp}\t{args.iters}\t{args.lambda_geo}\t{args.warmup}\t{args.lambda_dtv}\t{args.max_radius}\t{args.kspace_high}\t"
            f"{sp['psnr']:.3f}\t{sp['ssim']:.4f}\t{sp['nmse']:.4f}\t{sp['hfen']:.4f}\t{sp['sharp']:.3f}\t{sp['grain']:.4f}\t"
            f"{sc['psnr']:.3f}\t{sc['ssim']:.4f}\t{sc['nmse']:.4f}\t{sc['hfen']:.4f}\t{sc['sharp']:.3f}\t{sc['grain']:.4f}\t"
            f"{combined:.3f}\t{both_beat}\t{int(npts)}\t{vram:.1f}\n")
    if not tsv.exists():
        tsv.write_text(head)
    with open(tsv, "a") as f:
        f.write(line)
    log(f"[ar_mcgs {args.exp}] DONE -> {RES}/results.tsv")


if __name__ == "__main__":
    main()
