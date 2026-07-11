
import sys, os, json, time
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/paired_gs")
import torch
from paired_arc_gs.train.config_schema import merge_config
from paired_arc_gs.train.trainer import PairedARCTrainer

OUT="/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_arc_ablation17"
os.makedirs(OUT, exist_ok=True)
T1D="/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_T1_trainer.mat"
FLD="/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_FLAIR_trainer.mat"
CT1="/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_T1.mat"
CFL="/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/csm_espirit/csm_espirit_FLAIR.mat"
MSK="/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask/mask3d_vdpois_af12_transverse.mat"
def spec(name,d,c): return {"name":name,"data_path":d,"csm_path":c,"mask_path":MSK,"readout_axis":2,"phase_axes":[0,1],"presampled":False,"acs_lines":24}
PLN=spec("T1",T1D,CT1); FLR=spec("FLAIR",FLD,CFL)

def cfg(plain,contrast,npts,decoder,change,bypass,exp):
    stages = ({"s2_response_fit":{"iters":400},"s5_change_birth":{"iters":300}} if change else {"s2_response_fit":{"iters":700}})
    return merge_config({
      "data":{"plain":plain,"contrast":contrast},
      "model":{"num_points":npts,"decoder":{"enable":decoder},
               "change":{"enable":change,"max_per_contrast":8000,"propose_per_contrast":4000,"birth_fit_iters":160,"audit_tau":0.0}},
      "training":{"stages":stages,"lr":{"latent":1e-2,"decoder":1e-2,"change":1e-2},
                  "audit":{"enable":True,"fit_fraction":0.75,"bypass":bypass},"eval":{"holdout_fraction":0.10}},
      "output":{"output_dir":OUT,"exp_name":exp},"seed":42,
    })

def holdout_of(tr): return {k:v["holdout_nrmse"] for k,v in tr.holdout_kspace_nrmse().items()}

def run_pair(tag,decoder,change,bypass,npts=200000):
    t0=time.time(); tr=PairedARCTrainer(cfg(PLN,FLR,npts,decoder,change,bypass,tag)); tr.train()
    ho=holdout_of(tr)
    return {"tag":tag,"holdout":ho,"counts":{tr.name_p:tr.change.count(0),tr.name_c:tr.change.count(1)},
            "flags":{"shared_anat":True,"decoder":decoder,"change":change,"audit":(change and not bypass)},
            "wall_s":round(time.time()-t0,1)}

def run_independent(npts=100000):
    # Row1: each contrast its OWN geometry (no sharing). Fit each contrast alone (both slots=that
    # contrast), decoder off. Half points/contrast so TOTAL budget = 200k matches shared rows.
    out={}; t0=time.time()
    for spc,key in [(PLN,"T1"),(FLR,"FLAIR")]:
        tr=PairedARCTrainer(cfg(spc,spc,npts,False,False,False,f"indep_{key}")); tr.train()
        out[key]=holdout_of(tr)["plain"]   # plain slot = this contrast, geometry dedicated to it
    return {"tag":"independent_gs","holdout":out,"counts":{},"flags":{"shared_anat":False,"decoder":False,"change":False,"audit":False},
            "note":"two independent single-contrast fits, 100k Gaussians each (200k total, matches shared rows)","wall_s":round(time.time()-t0,1)}

ROWS=[
  ("row1_independent_gs", "independent"),
  ("row2_mcgs",           dict(decoder=False,change=False,bypass=False)),
  ("row3_wo_change",      dict(decoder=True, change=False,bypass=False)),
  ("row4_wo_audit",       dict(decoder=True, change=True, bypass=True)),
  ("row5_full",           dict(decoder=True, change=True, bypass=False)),
]
results={}
if os.path.exists(f"{OUT}/ablation17.json"):
    try: results=json.load(open(f"{OUT}/ablation17.json"))
    except: results={}
for tag,spec_ in ROWS:
    if tag in results and "error" not in results[tag]:
        print(f"skip {tag} (done)"); continue
    print(f"\n===== {tag} =====", flush=True)
    try:
        results[tag]= run_independent() if spec_=="independent" else run_pair(tag,**spec_)
        r=results[tag]; print(f"[{tag}] holdout={ {k:round(v,4) for k,v in r['holdout'].items()} } flags={r['flags']}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); results[tag]={"error":str(e)}
    json.dump(results, open(f"{OUT}/ablation17.json","w"), indent=2)
print("\nABLATION17_DONE")
