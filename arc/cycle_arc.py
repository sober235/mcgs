#!/usr/bin/env python
"""arc/cycle_arc.py — one autoresearch cycle on the FRONTIER Paired-ARC-GS model (PairedARCTrainer).

Single-knob deltas from the `paired_arc_gs/train/_final_tf.py` warm-start baseline (T1 36.95 /
FLAIR 37.46 at full budget). Scored with the SAME artifact-centric, NON-alpha-fit composite as
arc/cycle.py (psnr/ssim/nmse + hfen/sharp/grain) — per arc/program.md "no alpha-fit on scored path".

Anti-alias objective (identical to arc/cycle.py): drive axial/coronal/sagittal error to clean.
  per-contrast  c = psnr + 25*ssim - 60*nmse - 25*hfen - 40*max(0,1-sharp) - 200*grain
  combined      = mean(c_T1, c_FLAIR);  both contrasts must beat their ZF anchor on ssim/nmse/hfen.

REAL ARC levers (verified against paired_arc_gs/losses/arc_losses.py + trainer.py):
  loss.lambda_geo, loss.lambda_change, loss.lambda_route ; training.geometry_lr_scale ;
  model.decoder.{hidden_dim,n_layers} ; model.change.audit_tau (the §10.3 measurement-audit
  threshold — the on-target anti-alias guard: change admitted only if audit gain ΔE > tau) ;
  stage budgets s2/s4/s5 ; model.gabor.enable.  (lambda_dtv is MC-GS-only; NOT read by ARC.)

Run one cycle:
  CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle_arc.py \
      --exp a_base --gpu 7 --budget 0.5
"""
import os, sys, time, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--gpu", type=int, default=7)
ap.add_argument("--budget", type=float, default=0.5,
                help="scales s2/s4/s5 stage iters (1.0 = full _final_tf 5500; 0.5 = fast sweep)")
# --- ARC levers (single-knob deltas; defaults == _final_tf.py baseline) ---
ap.add_argument("--lambda_geo", type=float, default=1.0e-5)          # geometry reg weight
ap.add_argument("--lambda_change", type=float, default=1.0e-3)       # change-branch sparsity
ap.add_argument("--lambda_route", type=float, default=1.0e-2)        # §10 routing loss
ap.add_argument("--geometry_lr_scale", type=float, default=0.15)     # s4 geometry gentleness
ap.add_argument("--decoder_hidden", type=int, default=48)
ap.add_argument("--decoder_layers", type=int, default=3)
ap.add_argument("--audit_tau", type=float, default=0.0)              # anti-alias admission guard
ap.add_argument("--gabor", action="store_true")                     # default OFF (deadlocks at 576902 pts)
ap.add_argument("--no_warm", action="store_true")                   # ablate the MC-GS warm-start
ap.add_argument("--smoke", action="store_true")                     # tiny stage iters to validate the pipeline
ap.add_argument("--save_volumes", action="store_true")              # dump recon+GT npz (both contrasts) for figures
ap.add_argument("--lambda_llr", type=float, default=0.0)             # P3 cross-contrast LLR weight
ap.add_argument("--llr_block", type=int, default=8)
ap.add_argument("--llr_stride", type=int, default=4)
ap.add_argument("--lambda_freqextrap", type=float, default=0.0)      # P1 extrapolation weight
ap.add_argument("--p1_rho", type=float, default=0.3)                 # P1 held-out fraction
ap.add_argument("--p1_K", type=int, default=50)                      # P1 Λ redraw period
ap.add_argument("--p1_r_acs", type=float, default=12.0)              # P1 ACS radius (always in Θ)
args = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import numpy as np
import torch

ROOT = Path("/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paired_gs"))
from paired_arc_gs.train.config_schema import merge_config
from paired_arc_gs.train.trainer import PairedARCTrainer, fft3c, ifft3c
from metrics import evaluate_reconstruction               # paired_gs/metrics.py (non-alpha-fit)

RES = ROOT / "results_arc_frontier"
RES.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)

T1D = "/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_T1_trainer.mat"
FLD = "/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_FLAIR_trainer.mat"
CT1 = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_T1.mat"
CFL = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_FLAIR.mat"
MSK = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask/mask3d_vdpois_af12_transverse_ALIGNED.mat"
WARM = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/results_espirit_winner_v2/mcgs_espirit_winner_v2_tv06hi22/checkpoints/best.pth"

# ------- artifact-centric metrics (copied verbatim from arc/cycle.py; identical composite) -------
def hfen(rec, gt):
    def lap(v):
        return (6.0 * v - torch.roll(v, 1, -3) - torch.roll(v, -1, -3)
                - torch.roll(v, 1, -2) - torch.roll(v, -1, -2)
                - torch.roll(v, 1, -1) - torch.roll(v, -1, -1))
    g = gt / (gt.max() + 1e-9); r = rec / (gt.max() + 1e-9); lg = lap(g)
    return float(torch.linalg.norm(lap(r) - lg) / (torch.linalg.norm(lg) + 1e-9))

def sharp_ratio(rec, gt):
    def gm(v):
        gx = torch.zeros_like(v); gy = torch.zeros_like(v); gz = torch.zeros_like(v)
        gx[:-1] = v[1:] - v[:-1]; gy[:, :-1] = v[:, 1:] - v[:, :-1]; gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
        return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
    brain = gt > 0.05 * gt.max()
    return float(gm(rec)[brain].mean() / (gm(gt)[brain].mean() + 1e-9))

def grain(rec, gt):
    import torch.nn.functional as F
    g = gt / (gt.max() + 1e-9); r = rec / (gt.max() + 1e-9)
    sm = F.avg_pool3d(r[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    hp = r - sm
    def gmag(v):
        gx = torch.zeros_like(v); gy = torch.zeros_like(v); gz = torch.zeros_like(v)
        gx[:-1] = v[1:] - v[:-1]; gy[:, :-1] = v[:, 1:] - v[:, :-1]; gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
        return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
    brain = g > 0.05 * g.max(); gg = gmag(g)
    thr = torch.quantile(gg[brain], 0.30); flat = brain & (gg <= thr)
    return float(hp[flat].std())

def psnr_unified(rec, gt, p=99.9):
    """Unified-gauge PSNR: data_range = P_p percentile of GT magnitude (robust to the
    single-voxel peak that inflates @max-PSNR). rec, gt: real magnitude tensors, same
    normalized scale. NMSE (gauge-free) remains the cross-contrast arbiter."""
    r = rec.detach().cpu().numpy().astype("float64")
    g = gt.detach().cpu().numpy().astype("float64")
    dr = float(np.percentile(g, p))
    mse = float(np.mean((r - g) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10(dr ** 2 / mse))

def build_cfg():
    """_final_tf.py baseline cfg, with single-knob overrides from args."""
    b = args.budget
    if args.smoke:
        s2, s4, s5 = 30, 20, 20             # pipeline-validation only
    else:
        s2 = max(200, int(round(3000 * b)))     # response-fit (the key warm-start stage)
        s4 = max(100, int(round(2000 * b)))     # low-LR geometry polish
        s5 = max(50,  int(round(300 * b)))      # change birth
    s3 = 10 if args.smoke else 200              # deformation
    data = {"plain":    {"name": "T1",    "data_path": T1D, "csm_path": CT1, "mask_path": MSK,
                         "readout_axis": 2, "phase_axes": [0, 1], "presampled": False, "acs_lines": 24},
            "contrast": {"name": "FLAIR", "data_path": FLD, "csm_path": CFL, "mask_path": MSK,
                         "readout_axis": 2, "phase_axes": [0, 1], "presampled": False, "acs_lines": 24}}
    model = {"num_points": 300000,
             "decoder": {"enable": True, "hidden_dim": args.decoder_hidden, "n_layers": args.decoder_layers},
             "change":  {"enable": True, "max_per_contrast": 20000, "propose_per_contrast": 4000,
                         "birth_fit_iters": 160, "audit_tau": args.audit_tau},
             "phase":   {"enable": True},
             "deformation": {"enable": True, "rigid": True, "nonrigid": True, "u_max": 0.05, "reference_index": 0},
             "gabor":   {"enable": bool(args.gabor), "max_primitives": 5000, "propose_per_birth": 100}}
    if not args.no_warm:
        model["warm_start_path"] = WARM
    cfg = merge_config({
        "data": data,
        "model": model,
        "loss": {"lambda_geo": args.lambda_geo, "lambda_change": args.lambda_change,
                 "lambda_route": args.lambda_route,
                 "lambda_llr": args.lambda_llr, "llr_block": args.llr_block, "llr_stride": args.llr_stride,
                 "lambda_freqextrap": args.lambda_freqextrap},
        "training": {
            "geometry_lr_scale": args.geometry_lr_scale,
            "stages": {"s2_response_fit": {"iters": s2}, "s3_deformation": {"iters": s3},
                       "s4_geometry_lowlr": {"iters": s4}, "s5_change_birth": {"iters": s5},
                       "s6_gabor_birth": {"iters": 120 if args.gabor else 0},
                       "s7_joint_prune": {"iters": 0}, "s8_stability": {"iters": 0},
                       "s9_full_refine": {"iters": 0}, "s10_cg_project": {"iters": 0}},
            "audit": {"enable": True, "fit_fraction": 0.75, "seed": 42, "bypass": False},
            "p1": {"rho": args.p1_rho, "K": args.p1_K, "r_acs": args.p1_r_acs,
                   "warmup_start": (0 if args.smoke else 500),
                   "warmup_ramp": (5 if args.smoke else 500)},
            "cg": {"enable": True, "eta": 1.0, "n_iters": 10},
            "eval": {"holdout_fraction": 0.10}},
        "output": {"output_dir": str(RES / args.exp), "exp_name": args.exp, "eval_every": 1000},
        "seed": 42,
    })
    return cfg, (s2 + s3 + s4 + s5)


@torch.no_grad()
def score_head(tr, ci, gt, csm, ksp, mask, dev):
    """hard-DC synth (FULL sampled mask) + SENSE-combine on raw render, artifact composite. Non-alpha."""
    x = tr.render(ci)
    k_model = fft3c(csm * x.unsqueeze(0))
    m = mask if mask.ndim == k_model.ndim else mask.unsqueeze(0)
    k_dc = m * ksp + (1.0 - m) * k_model
    x_dc = (ifft3c(k_dc) * csm.conj()).sum(0)
    met = evaluate_reconstruction(x_dc, gt, compute_extended=False)
    hf = hfen(x_dc.abs(), gt.abs()); sr = sharp_ratio(x_dc.abs(), gt.abs()); gr = grain(x_dc.abs(), gt.abs())
    x_zf = (ifft3c(m * ksp) * csm.conj()).sum(0)
    mz = evaluate_reconstruction(x_zf, gt, compute_extended=False); zf_hf = hfen(x_zf.abs(), gt.abs())
    psnr, ssim, nmse = float(met["psnr"]), float(met["ssim"]), float(met["nmse"])
    psnr_uni = psnr_unified(x_dc.abs(), gt.abs())
    comp = psnr + 25.0 * ssim - 60.0 * nmse - 25.0 * hf - 40.0 * max(0.0, 1.0 - sr) - 200.0 * gr
    beat = (ssim >= float(mz["ssim"])) and (nmse <= float(mz["nmse"])) and (hf <= zf_hf)
    return dict(psnr=psnr, ssim=ssim, nmse=nmse, hfen=hf, sharp=sr, grain=gr, comp=comp, beat=beat, psnr_uni=psnr_uni)


def main():
    t0 = time.time()
    dev = torch.device("cuda", 0)
    cfg, tot_iters = build_cfg()
    log(f"[arc_frontier {args.exp}] gpu={args.gpu} budget={args.budget} (~{tot_iters} iters) "
        f"lambda_geo={args.lambda_geo} lambda_change={args.lambda_change} audit_tau={args.audit_tau} "
        f"geom_lr_scale={args.geometry_lr_scale} decoder={args.decoder_hidden}/{args.decoder_layers} "
        f"gabor={args.gabor} warm={not args.no_warm}")
    tr = PairedARCTrainer(cfg)
    if getattr(tr, "gabor", None) is not None:
        tr.gabor.render_chunk = 64
    tr.train()
    tr._cg_finalize = True

    sp = score_head(tr, 0, tr.gt_p, tr.csm_p, tr.ksp_p, tr.mask_p, dev)
    sc = score_head(tr, 1, tr.gt_c, tr.csm_c, tr.ksp_c, tr.mask_c, dev)
    combined = 0.5 * (sp["comp"] + sc["comp"])
    both_beat = sp["beat"] and sc["beat"]
    vram = torch.cuda.max_memory_allocated() / 1e9
    npts = int(tr.model.num_points)
    try:
        chg = {"T1": int(tr.change.count(0)), "FLAIR": int(tr.change.count(1))}
    except Exception:
        chg = {"T1": -1, "FLAIR": -1}

    log(f"T1    psnr={sp['psnr']:.3f} ssim={sp['ssim']:.4f} nmse={sp['nmse']:.4f} hfen={sp['hfen']:.4f} "
        f"sharp={sp['sharp']:.3f} grain={sp['grain']:.4f} comp={sp['comp']:.3f} beat={sp['beat']}")
    log(f"FLAIR psnr={sc['psnr']:.3f} ssim={sc['ssim']:.4f} nmse={sc['nmse']:.4f} hfen={sc['hfen']:.4f} "
        f"sharp={sc['sharp']:.3f} grain={sc['grain']:.4f} comp={sc['comp']:.3f} beat={sc['beat']}")
    log(f"SCORE combined={combined:.3f} both_beat_zf={both_beat} change={chg} N={npts} vram_gb={vram:.1f} "
        f"elapsed_s={time.time()-t0:.0f}")

    try:
        ho = tr.holdout_kspace_nrmse()                       # fixed 10% test set, disjoint from Θ and Λ
        ho_t1, ho_fl = ho["plain"]["holdout_nrmse"], ho["contrast"]["holdout_nrmse"]
    except Exception:
        ho_t1 = ho_fl = float("nan")
    log(f"UNIFIED psnr_uni T1={sp['psnr_uni']:.3f} FLAIR={sc['psnr_uni']:.3f} | "
        f"holdout_kspace_nrmse T1={ho_t1:.4f} FLAIR={ho_fl:.4f}")

    row = dict(exp=args.exp, budget=args.budget, tot_iters=tot_iters, lambda_geo=args.lambda_geo,
               lambda_change=args.lambda_change, lambda_route=args.lambda_route,
               geometry_lr_scale=args.geometry_lr_scale, decoder=f"{args.decoder_hidden}/{args.decoder_layers}",
               audit_tau=args.audit_tau, gabor=bool(args.gabor), warm=not args.no_warm,
               T1=sp, FLAIR=sc, combined=combined, both_beat=both_beat, change=chg, npts=npts, vram_gb=vram)
    (RES / f"{args.exp}_score.json").write_text(json.dumps(row, indent=2))
    tsv = RES / "results_arc.tsv"
    head = ("exp\tbudget\tlambda_geo\tlambda_change\taudit_tau\tgeom_lr_scale\tdecoder\tgabor\twarm\t"
            "T1_psnr\tT1_ssim\tT1_nmse\tT1_hfen\tT1_sharp\tT1_grain\t"
            "FLAIR_psnr\tFLAIR_ssim\tFLAIR_nmse\tFLAIR_hfen\tFLAIR_sharp\tFLAIR_grain\t"
            "combined\tboth_beat\tnpts\tvram_gb\tT1_psnr_uni\tFLAIR_psnr_uni\n")
    line = (f"{args.exp}\t{args.budget}\t{args.lambda_geo}\t{args.lambda_change}\t{args.audit_tau}\t"
            f"{args.geometry_lr_scale}\t{args.decoder_hidden}/{args.decoder_layers}\t{bool(args.gabor)}\t{not args.no_warm}\t"
            f"{sp['psnr']:.3f}\t{sp['ssim']:.4f}\t{sp['nmse']:.4f}\t{sp['hfen']:.4f}\t{sp['sharp']:.3f}\t{sp['grain']:.4f}\t"
            f"{sc['psnr']:.3f}\t{sc['ssim']:.4f}\t{sc['nmse']:.4f}\t{sc['hfen']:.4f}\t{sc['sharp']:.3f}\t{sc['grain']:.4f}\t"
            f"{combined:.3f}\t{both_beat}\t{npts}\t{vram:.1f}\t{sp['psnr_uni']:.3f}\t{sc['psnr_uni']:.3f}\n")
    if not tsv.exists():
        tsv.write_text(head)
    with open(tsv, "a") as f:
        f.write(line)

    if args.save_volumes:
        @torch.no_grad()
        def _vol(ci, gt, csm, ksp, mask):
            x = tr.render(ci); k_model = fft3c(csm * x.unsqueeze(0))
            m = mask if mask.ndim == k_model.ndim else mask.unsqueeze(0)
            x_dc = (ifft3c(m * ksp + (1.0 - m) * k_model) * csm.conj()).sum(0)
            x_zf = (ifft3c(m * ksp) * csm.conj()).sum(0)              # zero-filled (aliased "before")
            f = lambda t: t.abs().cpu().numpy().astype("float32")
            return f(x_dc), f(gt), f(x_zf)
        rT, gT, zT = _vol(0, tr.gt_p, tr.csm_p, tr.ksp_p, tr.mask_p)
        rF, gF, zF = _vol(1, tr.gt_c, tr.csm_c, tr.ksp_c, tr.mask_c)
        np.savez_compressed(RES / f"{args.exp}_volumes.npz",
                            recT1=rT, gtT1=gT, zfT1=zT, recFL=rF, gtFL=gF, zfFL=zF)
        log(f"  saved volumes -> {RES}/{args.exp}_volumes.npz")
    log(f"[arc_frontier {args.exp}] DONE -> {tsv}")


if __name__ == "__main__":
    main()
