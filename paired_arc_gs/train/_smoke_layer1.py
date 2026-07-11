
import sys, os, json, time
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/paired_gs")
import torch
from paired_arc_gs.train.config_schema import merge_config, validate_config
from paired_arc_gs.train.trainer import PairedARCTrainer

# T1/FLAIR pair (has real fully-sampled GT). Tiny run: 200000->20000 pts, 60 iters, to prove
# Stage1-2 plumbing end-to-end (ACS init -> fixed-geometry response fit -> eval).
T1D = "/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_T1_trainer.mat"
FLD = "/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_FLAIR_trainer.mat"
CT1 = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_T1.mat"
CFL = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_FLAIR.mat"
MSK = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask/mask3d_vdpois_af12_transverse.mat"

cfg = merge_config({
  "data": {
    "plain":    {"name":"T1",    "data_path":T1D, "csm_path":CT1, "mask_path":MSK,
                 "readout_axis":2, "phase_axes":[0,1], "presampled":False, "acs_lines":24},
    "contrast": {"name":"FLAIR", "data_path":FLD, "csm_path":CFL, "mask_path":MSK,
                 "readout_axis":2, "phase_axes":[0,1], "presampled":False, "acs_lines":24},
  },
  "model": {"num_points": 20000, "anatomy": {"latent_dim": 12}, "decoder": {"enable": True, "hidden_dim": 24}},
  "training": {"stages": {"s2_response_fit": {"iters": 60}}, "lr": {"latent": 1e-2, "decoder": 1e-2}},
  "output": {"output_dir":"/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_arc_smoke",
             "exp_name":"layer1_t1flair", "eval_every": 20},
})
errs = validate_config(cfg)
print("config errors:", errs)
assert not errs, errs
t0=time.time()
tr = PairedARCTrainer(cfg)
print(f"[build] {time.time()-t0:.1f}s")
sc0 = tr.evaluate()
print("PRE-FIT (random decoder):", {k:(round(v['psnr'],3),round(v['ssim'],4)) for k,v in sc0.items()})
res = tr.train()
sc1 = tr.evaluate()
print("POST-FIT (60 iter Stage2):", {k:(round(v['psnr'],3),round(v['ssim'],4)) for k,v in sc1.items()})
# did the response fit IMPROVE both contrasts? (plumbing success = monotone up from random init)
improved = all(sc1[k]['psnr'] >= sc0[k]['psnr'] for k in sc1)
os.makedirs(cfg["output"]["output_dir"], exist_ok=True)
json.dump({"pre":sc0,"post":sc1,"improved":improved,"history":res["history"]},
          open(cfg["output"]["output_dir"]+"/layer1_smoke.json","w"), indent=2)
print("LAYER1_PLUMBING", "OK" if improved else "NO_IMPROVE")
