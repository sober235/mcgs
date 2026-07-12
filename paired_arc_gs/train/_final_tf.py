
import sys, os, json, time
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/paired_gs")
import numpy as np, torch
from paired_arc_gs.train.config_schema import merge_config
from paired_arc_gs.train.trainer import PairedARCTrainer
from paired_arc_gs.train.trainer import fft3c, ifft3c

OUT="/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_arc_final_tf"
os.makedirs(OUT, exist_ok=True)
T1D="/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_T1_trainer.mat"
FLD="/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_FLAIR_trainer.mat"
CT1="/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_T1.mat"
CFL="/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_FLAIR.mat"
MSK="/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask/mask3d_vdpois_af12_transverse.mat"

data={"plain":{"name":"T1","data_path":T1D,"csm_path":CT1,"mask_path":MSK,"readout_axis":2,"phase_axes":[0,1],"presampled":False,"acs_lines":24},
      "contrast":{"name":"FLAIR","data_path":FLD,"csm_path":CFL,"mask_path":MSK,"readout_axis":2,"phase_axes":[0,1],"presampled":False,"acs_lines":24}}

# FULL model: every module ON, full 10-stage schedule to convergence
cfg = merge_config({
  "data": data,
  "model": {"num_points": 300000,
            "decoder": {"enable": True},
            "change": {"enable": True, "max_per_contrast": 20000, "propose_per_contrast": 4000,
                       "birth_fit_iters": 160, "audit_tau": 0.0},
            "phase":  {"enable": True},
            "deformation": {"enable": True, "rigid": True, "nonrigid": True, "u_max": 0.05, "reference_index": 0},
            "gabor":  {"enable": True, "max_primitives": 5000, "propose_per_birth": 200}},
  "training": {
     # Gabor-active stages (s6, s9) are per-primitive full-volume + gradient-checkpointed, so
     # they are the runtime bottleneck; the anatomy+response converges in s2-s4 (DC 0.30->0.012).
     # Keep s6/s9 short — enough to birth + settle the Gabor high-freq branch, not to re-converge.
     "stages": {"s2_response_fit":{"iters":1800},"s3_deformation":{"iters":200},
                "s4_geometry_lowlr":{"iters":800},"s5_change_birth":{"iters":300},
                "s6_gabor_birth":{"iters":120},"s7_joint_prune":{"iters":0},
                "s8_stability":{"iters":0},"s9_full_refine":{"iters":0},
                "s10_cg_project":{"iters":0}},
     "audit": {"enable": True, "fit_fraction": 0.75, "seed": 42, "bypass": False},
     "cg": {"enable": True, "eta": 1.0, "n_iters": 10},
     "eval": {"holdout_fraction": 0.10}},
  "output": {"output_dir": OUT, "exp_name": "arc_final_tf", "eval_every": 400},
  "seed": 42,
})

t0=time.time()
tr = PairedARCTrainer(cfg)
# bigger Gabor render chunk => fewer checkpoint recomputations (22GB peak leaves ample 80GB headroom)
if getattr(tr, "gabor", None) is not None:
    tr.gabor.render_chunk = 64
tr.train()
tr._cg_finalize = True   # §13-S10 CG projection applied at final eval/render
train_s = time.time()-t0
print(f"TRAIN_DONE wall={train_s:.1f}s change_counts T1={tr.change.count(0)} FLAIR={tr.change.count(1)} gabor={getattr(tr.gabor,'count',lambda:0)() if getattr(tr,'gabor',None) else 0}", flush=True)

# ---- score (alpha-fit vs GT, hard-DC + SENSE combine — same口径 as MC-GS baseline) + capture volumes
@torch.no_grad()
def recon_and_score(ci, gt, csm, ksp, mask):
    x = tr.render(ci)
    k_model = fft3c(csm * x.unsqueeze(0))
    m = mask if mask.ndim == k_model.ndim else mask.unsqueeze(0)
    k_dc = m*ksp + (1-m)*k_model
    x_dc = (ifft3c(k_dc) * csm.conj()).sum(0)
    rec = x_dc.abs().cpu().numpy().astype("float64")
    g   = gt.abs().cpu().numpy().astype("float64")
    alpha = float((rec*g).sum()/(rec*rec).sum())
    r = rec*alpha; mx=float(g.max())
    from skimage.metrics import structural_similarity as ssim
    psnr = 10.0*np.log10(mx**2/np.mean((r-g)**2))
    ss = float(ssim(g, r, data_range=mx))
    return r.astype("float32"), g.astype("float32"), float(psnr), ss

recT1, gtT1, pT1, sT1 = recon_and_score(0, tr.gt_p, tr.csm_p, tr.ksp_p, tr.mask_p)
recFL, gtFL, pFL, sFL = recon_and_score(1, tr.gt_c, tr.csm_c, tr.ksp_c, tr.mask_c)
ho = tr.holdout_kspace_nrmse()
print(f"SCORE T1 psnr={pT1:.4f} ssim={sT1:.4f} | FLAIR psnr={pFL:.4f} ssim={sFL:.4f}", flush=True)

np.savez_compressed(f"{OUT}/final_volumes.npz",
    recT1=recT1, gtT1=gtT1, recFL=recFL, gtFL=gtFL)
scores = {"train_s": train_s, "num_points": 300000,
   "change_counts": {"T1": tr.change.count(0), "FLAIR": tr.change.count(1)},
   "T1":    {"psnr": pT1, "ssim": sT1, "holdout_nrmse": ho["plain"]["holdout_nrmse"]},
   "FLAIR": {"psnr": pFL, "ssim": sFL, "holdout_nrmse": ho["contrast"]["holdout_nrmse"]},
   "ref": {"T1_psnr": 37.738, "T1_ssim": 0.8755, "FLAIR_psnr": 38.143, "FLAIR_ssim": 0.894,
           "note": "MC-GS baseline winner-v2 (§17 row2) for context; NOT a target — ARC is a different, staged model"}}
json.dump(scores, open(f"{OUT}/final_scores.json","w"), indent=2)
torch.save({"model_state": tr.model.state_dict()}, f"{OUT}/arc_final_tf.pth")
print("FINAL_DONE", flush=True)
