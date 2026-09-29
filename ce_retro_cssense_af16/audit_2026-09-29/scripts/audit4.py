import numpy as np, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
sys.path.insert(0,"/data0/congcong/code/GS/mcgs_ce_retro_wt/ce_retro_cssense_af16/scripts")
import arms
T=arms.R; OUT=os.path.expanduser("~/figs/ce_retro_af16_audit")
d=json.load(open(f"{T}/logs/slice_scores.json"))
# 1. per-slice PSNR curves along the three axes
fig,ax=plt.subplots(2,3,figsize=(15,7))
for i,scan in enumerate(["pre","post"]):
    for a in range(3):
        rows=sorted([x for x in d[scan] if x["axis"]==a],key=lambda x:x["index"])
        idx=[x["index"] for x in rows]
        for arm,c in [("zf","k"),("cssense","tab:blue"),("spirit","tab:orange"),("zsssdu","tab:green"),("gs_single","tab:purple"),("gs_joint","tab:red")]:
            ax[i,a].plot(idx,[x["psnr"][arm] for x in rows],c,label=arms.LABEL.get(arm if arm!="cssense" else "cssense_o200",arm),lw=1)
        ax[i,a].set_title(f"{scan} axis{a} per-slice head PSNR"); ax[i,a].grid(alpha=.3)
        if i==0 and a==0: ax[i,a].legend(fontsize=7)
        worst=sorted(rows,key=lambda x:x["psnr"]["gs_joint"])[:3]
        print(scan,"axis",a,"worst MC-GS slices:",[(x["index"],round(x["psnr"]["gs_joint"],2),round(x["psnr"]["gs_single"],2),x["n_head"]) for x in worst])
        # slices where MC-GS loses most vs runner-up
        lose=sorted(rows,key=lambda x:x["joint_minus_runnerup"])[:3]
        print("   MC-GS smallest margin vs runner-up:",[(x["index"],round(x["joint_minus_runnerup"],3),x["best"]) for x in lose])
plt.tight_layout(); plt.savefig(f"{OUT}/E_perslice_psnr.png",dpi=110); plt.close()
# 2. Is the MC-GS residual "residual aliasing"? correlate residuals with the ZF residual, in head, after removing smooth part
for scan in ["pre","post"]:
    gt=np.load(f"{T}/recon/gt_{scan}_complex.npy"); gmag=np.abs(gt); p999=float(np.quantile(gmag,0.999)); head=gmag>0.05*p999
    zf=np.abs(np.load(f"{T}/recon/zf_{scan}_complex.npy"))
    ezf=zf-gmag
    out={}
    for k,l,p in arms.arm_files(scan):
        if k in ("zf",): continue
        e=np.abs(np.load(p))-gmag
        # split residual into low-pass (sigma 3) and high-pass parts
        lo=ndimage.gaussian_filter(e,3); hi=e-lo
        out[l]={"corr_full_vs_zf":float(np.corrcoef(e[head],ezf[head])[0,1]),
                "corr_hp_vs_zf_hp":float(np.corrcoef(hi[head],(ezf-ndimage.gaussian_filter(ezf,3))[head])[0,1]),
                "frac_energy_lowpass":float(np.sum(lo[head]**2)/np.sum(e[head]**2)),
                "rms_lo_rel":float(np.sqrt(np.mean(lo[head]**2))/p999),"rms_hi_rel":float(np.sqrt(np.mean(hi[head]**2))/p999)}
    print(scan,json.dumps(out,indent=1))
    json.dump(out,open(f"{OUT}/audit_residual_structure_{scan}.json","w"),indent=1)
