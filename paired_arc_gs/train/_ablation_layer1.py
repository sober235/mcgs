
import sys, os, json, time
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/paired_gs")
import torch
from paired_arc_gs.train.config_schema import merge_config
from paired_arc_gs.train.trainer import PairedARCTrainer

OUT="/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_arc_layer1"
os.makedirs(OUT, exist_ok=True)
T1D="/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_T1_trainer.mat"
FLD="/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_FLAIR_trainer.mat"
CT1="/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_T1.mat"
CFL="/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_FLAIR.mat"
MSK="/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask/mask3d_vdpois_af12_transverse.mat"
CE="/data0/congcong/data/CE_data"
PAIRS = {
  "t1flair": {"ref_ok": True, "data": {
    "plain":    {"name":"T1","data_path":T1D,"csm_path":CT1,"mask_path":MSK,"readout_axis":2,"phase_axes":[0,1],"presampled":False,"acs_lines":24},
    "contrast": {"name":"FLAIR","data_path":FLD,"csm_path":CFL,"mask_path":MSK,"readout_axis":2,"phase_axes":[0,1],"presampled":False,"acs_lines":24}}},
  "ce": {"ref_ok": False, "data": {
    "plain":    {"name":"pre", "data_path":f"{CE}/t1_3D_gre_fsp_sag_iso_ACS_16887049370010/ksp.mat","csm_path":f"{CE}/t1_3D_gre_fsp_sag_iso_ACS_16887049370010/csm.mat","presampled":True,"acs_lines":24},
    "contrast": {"name":"post","data_path":f"{CE}/t1_3D_gre_fsp_sag_iso_ACS_16887049370060/ksp2.mat","csm_path":f"{CE}/t1_3D_gre_fsp_sag_iso_ACS_16887049370060/csm2.mat","presampled":True,"acs_lines":24}}},
}
ITERS=800; N=200000
def run(pair_key, decoder_on):
    P=PAIRS[pair_key]
    cfg = merge_config({
      "data": P["data"],
      "model": {"num_points": N, "anatomy": {"latent_dim": 12}, "decoder": {"enable": decoder_on, "hidden_dim": 24}},
      "training": {"stages": {"s2_response_fit": {"iters": ITERS}}, "lr": {"latent": 1e-2, "decoder": 1e-2},
                   "eval": {"holdout_fraction": 0.10}},
      "output": {"output_dir": OUT, "exp_name": f"{pair_key}_{'dec' if decoder_on else 'indep'}", "eval_every": 400},
    })
    t0=time.time()
    tr = PairedARCTrainer(cfg)
    ho0 = tr.holdout_kspace_nrmse()          # null-ish model held-out NRMSE (decoder~0)
    res = tr.train()
    ho1 = tr.holdout_kspace_nrmse()          # after fit
    ref1 = tr.evaluate(ref_ok=P["ref_ok"]) if P["ref_ok"] else None
    return {"decoder": decoder_on, "n_points": N, "iters": ITERS, "wall_s": round(time.time()-t0,1),
            "holdout_pre": ho0, "holdout_post": ho1, "ref_post": ref1,
            "dc_traj": [(h["iter"], round(h.get("dc",0),5)) for h in res["history"]]}
results={}
for pair_key in ["t1flair","ce"]:
    for decoder_on in [False, True]:
        key=f"{pair_key}_{'decoder' if decoder_on else 'indep'}"
        print(f"\n===== RUN {key} =====", flush=True)
        try:
            results[key]=run(pair_key, decoder_on)
            r=results[key]
            hp=r["holdout_post"]
            print(f"[{key}] holdout NRMSE post: " + " ".join(f"{v['name']}={v['holdout_nrmse']:.4f}" for v in hp.values()) +
                  (f" | ref " + " ".join(f"{v['name']}={v['psnr']:.2f}/{v['ssim']:.3f}" for v in r["ref_post"].values()) if r["ref_post"] else ""), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); results[key]={"error":str(e)}
        json.dump(results, open(f"{OUT}/ablation_layer1.json","w"), indent=2)
print("\nLAYER1_ABLATION_DONE")
