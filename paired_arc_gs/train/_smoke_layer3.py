
import sys, os
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/paired_gs")
import torch
from paired_arc_gs.train.config_schema import merge_config, validate_config
from paired_arc_gs.train.trainer import PairedARCTrainer

T1D="/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_T1_trainer.mat"
FLD="/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_FLAIR_trainer.mat"
CT1="/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_T1.mat"
CFL="/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_FLAIR.mat"
MSK="/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask/mask3d_vdpois_af12_transverse.mat"

# enable phase (§9) + deformation (§8), run Stage2 then Stage3(deformation)
cfg = merge_config({
  "data": {"plain":{"name":"T1","data_path":T1D,"csm_path":CT1,"mask_path":MSK,"readout_axis":2,"phase_axes":[0,1],"presampled":False,"acs_lines":24},
           "contrast":{"name":"FLAIR","data_path":FLD,"csm_path":CFL,"mask_path":MSK,"readout_axis":2,"phase_axes":[0,1],"presampled":False,"acs_lines":24}},
  "model": {"num_points": 120000, "decoder": {"enable": True},
            "phase": {"enable": True, "control_points":[8,8,8]},
            "deformation": {"enable": True, "rigid": True, "nonrigid": True, "control_points":[6,6,6], "u_max":0.05, "reference_index":0}},
  "training": {"stages": {"s2_response_fit": {"iters": 60}, "s3_deformation": {"iters": 60}, "s4_geometry_lowlr": {"iters": 40}},
               "lr": {"latent": 1e-2, "decoder": 1e-2, "phase": 5e-3, "deformation": 5e-3},
               "audit": {"enable": False}, "eval": {"holdout_fraction": 0.10}},
  "output": {"output_dir":"/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_arc_l3smoke","exp_name":"l3", "eval_every":30},
})
errs=validate_config(cfg); print("cfg errs:", errs); assert not errs
tr = PairedARCTrainer(cfg)
print("phase enabled:", tr.phase.enabled, "| deform enabled:", tr.deform.enabled)
# check render applies phase+deform without error, produces finite complex volume
x = tr.render(1)
print(f"render(FLAIR): shape={tuple(x.shape)} dtype={x.dtype} finite={torch.isfinite(x.abs()).all().item()}")
# check reg terms present + finite in a stage that has them
loss, log = tr.compute_losses(("dc","geo","phase","def"))
print("loss terms:", {k:round(v,4) for k,v in log.items()})
assert "phase" in log and "def" in log, "phase/def reg missing"
assert all(torch.isfinite(torch.tensor(v)) for v in log.values()), "non-finite loss"
ho0 = tr.holdout_kspace_nrmse()
res = tr.train()
ho1 = tr.holdout_kspace_nrmse()
# verify deformation params actually moved (Stage 3 trained them)
theta_norm = float(tr.deform.theta[1].norm()); trans_norm = float(tr.deform.trans[1].norm())
u_norm = float(tr.deform.u_ctrl[1].norm())
phi_norm = float(tr.phase.phi_ctrl[1].norm())
print(f"AFTER TRAIN: deform θ={theta_norm:.4g} t={trans_norm:.4g} u={u_norm:.4g} | phase φ={phi_norm:.4g}")
print(f"holdout: pre={ {k:round(v['holdout_nrmse'],4) for k,v in ho0.items()} } post={ {k:round(v['holdout_nrmse'],4) for k,v in ho1.items()} }")
moved = (theta_norm>1e-6 or trans_norm>1e-6 or u_norm>1e-6) and phi_norm>1e-4
print("L3_SMOKE_OK" if moved else "L3_SMOKE_PARAMS_STATIC")
