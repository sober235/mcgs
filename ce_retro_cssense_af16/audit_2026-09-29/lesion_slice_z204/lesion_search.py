import numpy as np, json
A="/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois_ar"
roi=np.load("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz")
names=[str(n) for n in roi["names"]]; core=roi["roi%d"%names.index("病灶核心")]>0
V={}; G={}
for sc in ("pre","post"):
    G[sc]=np.abs(np.load(f"{A}/recon/gt_{sc}_complex.npy"))
    V[(sc,"j")]=np.abs(np.load(f"{A}/recon/gs_joint_{sc}_complex.npy")); V[(sc,"s")]=np.abs(np.load(f"{A}/recon/gs_single_{sc}_complex.npy"))
out=[]
for ax in range(3):
    for i in range(core.shape[ax]):
        c=np.take(core,i,axis=ax)
        if c.sum()<200: continue
        ys,xs=np.where(c); pad=14
        sl=[slice(None)]*3; sl[ax]=i
        r={}
        for sc in ("pre","post"):
            gt=G[sc][tuple(sl)]; gmax=float(G[sc].max()); head=gt>0.05*float(np.quantile(G[sc],0.999))
            box=np.zeros_like(head); box[max(ys.min()-pad,0):ys.max()+pad, max(xs.min()-pad,0):xs.max()+pad]=True; box&=head
            ps=lambda v,m: 20*np.log10(gmax)-10*np.log10(np.mean((v[m]-gt[m])**2))
            j=V[(sc,"j")][tuple(sl)]; s=V[(sc,"s")][tuple(sl)]
            r[sc]=(ps(j,head)-ps(s,head), ps(j,box)-ps(s,box))
        out.append((min(r["pre"][1],r["post"][1]),min(r["pre"][0],r["post"][0]),ax,i,int(c.sum()),r))
ok=[o for o in out if o[0]>0 and o[1]>0]
ok.sort(reverse=True)
print(len(out),"lesion slices,",len(ok),"with MC-GS ahead of GS-single in the lesion box AND the whole slice on both scans")
for o in ok[:10]: print("axis %d idx %3d core %4d | box pre %+.3f post %+.3f | slice pre %+.3f post %+.3f"%(o[2],o[3],o[4],o[5]["pre"][1],o[5]["post"][1],o[5]["pre"][0],o[5]["post"][0]))
