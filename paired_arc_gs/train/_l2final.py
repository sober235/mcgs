
import sys, os, json, time
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/paired_gs")
import torch
from paired_arc_gs.train.config_schema import merge_config
from paired_arc_gs.train.trainer import PairedARCTrainer

OUT="/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_arc_layer2"
os.makedirs(OUT, exist_ok=True)
T1D="/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_T1_trainer.mat"
FLD="/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_FLAIR_trainer.mat"
CT1="/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_T1.mat"
CFL="/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_FLAIR.mat"
MSK="/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask/mask3d_vdpois_af12_transverse.mat"
CE="/data0/congcong/data/CE_data"
TF={"plain":{"name":"T1","data_path":T1D,"csm_path":CT1,"mask_path":MSK,"readout_axis":2,"phase_axes":[0,1],"presampled":False,"acs_lines":24},
    "contrast":{"name":"FLAIR","data_path":FLD,"csm_path":CFL,"mask_path":MSK,"readout_axis":2,"phase_axes":[0,1],"presampled":False,"acs_lines":24}}
CEP={"plain":{"name":"pre","data_path":f"{CE}/t1_3D_gre_fsp_sag_iso_ACS_16887049370010/ksp.mat","csm_path":f"{CE}/t1_3D_gre_fsp_sag_iso_ACS_16887049370010/csm.mat","presampled":True,"acs_lines":24},
     "contrast":{"name":"post","data_path":f"{CE}/t1_3D_gre_fsp_sag_iso_ACS_16887049370060/ksp2.mat","csm_path":f"{CE}/t1_3D_gre_fsp_sag_iso_ACS_16887049370060/csm2.mat","presampled":True,"acs_lines":24}}

def make(data, change_enable, bypass):
    return merge_config({
      "data": data,
      "model": {"num_points": 200000, "decoder": {"enable": True},
                "change": {"enable": change_enable, "max_per_contrast": 8000, "propose_per_contrast": 4000, "birth_fit_iters": 160, "audit_tau": 0.0}},
      "training": {"stages": ({"s2_response_fit": {"iters": 400}, "s5_change_birth": {"iters": 300}} if change_enable else {"s2_response_fit": {"iters": 700}}),
                   "lr": {"latent": 1e-2, "decoder": 1e-2, "change": 1e-2},
                   "audit": {"enable": True, "fit_fraction": 0.75, "bypass": bypass}, "eval": {"holdout_fraction": 0.10}},
      "output": {"output_dir": OUT, "exp_name": "l2final"},
    })

def run(tag, data, change_enable, bypass):
    t0=time.time(); tr=PairedARCTrainer(make(data,change_enable,bypass)); tr.train()
    ho=tr.holdout_kspace_nrmse(); birth=getattr(tr,"_last_birth_report",{})
    return {"tag":tag,"holdout":{k:v["holdout_nrmse"] for k,v in ho.items()},
            "birth":birth,"counts":{tr.name_p:tr.change.count(0),tr.name_c:tr.change.count(1)},
            "change_enable":change_enable,"bypass":bypass,"wall_s":round(time.time()-t0,1)}

RUNS=[
  ("ce_audit",  CEP, True,  False),   # decisive: does audit accept genuine pre/post change?
  ("ce_bypass", CEP, True,  True),
  ("ce_nochange", CEP, False, False), # capacity control
  ("tf_nochange", TF, False, False),  # FLAIR 3-way control: disabled
  ("tf_audit",  TF, True,  False),    #                      audit (reject)
  ("tf_bypass", TF, True,  True),     #                      bypass (accept)
]
results={}
if os.path.exists(f"{OUT}/l2final.json"):
    try: results=json.load(open(f"{OUT}/l2final.json"))
    except: results={}
for tag,data,ce,by in RUNS:
    if tag in results and "error" not in results[tag]:
        print(f"skip {tag} (done)"); continue
    print(f"\n===== {tag} =====", flush=True)
    try:
        results[tag]=run(tag,data,ce,by); r=results[tag]
        print(f"[{tag}] holdout={ {k:round(v,4) for k,v in r['holdout'].items()} } counts={r['counts']}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); results[tag]={"error":str(e)}
    json.dump(results, open(f"{OUT}/l2final.json","w"), indent=2)
print("\nL2FINAL_DONE")
