import numpy as np, glob, os
T='/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois_cmp/recon/'
WS='/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_losstune_20261101/runs/'
def f2(x): return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x,axes=(-2,-1)),norm='ortho',axes=(-2,-1)),axes=(-2,-1))
names=['gs_joint','gs_single','zsssdu']
runs=['C1_khplain250/s42/m0375/relax_a0.5','C1_khplain250/s42/m0625/relax_a0.5','C7_vtv01/s42/m0625/relax_a0.5','C3_ptv0/s42/m0625/relax_a0.5',
 'C8_vtv005_long/s42/m0625/relax_a0.5','C8_vtv005_long/s42/m1000/relax_a0.5','C30_edge0/s42/m0625/relax_a0.5',
 'C54_bm4dpos_l4_once/s42/m0125/relax_a0.5','C54_bm4dpos_l4_once/s42/m0625/relax_a0.5',
 'C57_bm4dpos_l4_once_s1250/s42/m0625/relax_a0.5','C57_bm4dpos_l4_once_s1250/s42/m1250/relax_a0.5',
 'C77_bm4dpos_lp2c4_once_loc0_s0165_a1/s42/m0125/relax_a1','C77_bm4dpos_lp2c4_once_loc0_s0165_a1/s42/m0625/relax_a1',
 'FV01_nl_sigma_rel_0.01/s42/m0625/relax_a1','FV05_nl_sigma_rel_0.02/s42/m0625/relax_a1','FV24_nl_sigma_rel_0.03/s42/m0625/relax_a1',
 'FV06_lambda_nl_plain_0.5/s42/m0625/relax_a1','FV10_lambda_nl_contrast_1/s42/m0625/relax_a1','FV16_nl_refresh_100/s42/m0625/relax_a1',
 'FV37_lambda_wav_0/s42/m0625/relax_a1','FV50_lambda_ptv_0/s42/m0625/relax_a1','FV42_kspace_high_weight_4/s42/m0625/relax_a1','FV53_lambda_vtv_0/s42/m0625/relax_a1']
for scan in ('pre','post'):
    gt=np.abs(np.load(T+f'gt_{scan}_complex.npy')).astype(np.float32); gm=gt.max(); m3=gt>.05*np.quantile(gt,.999)
    Y,X=np.meshgrid(np.linspace(-1,1,gt.shape[1]),np.linspace(-1,1,gt.shape[2]),indexing='ij'); rr=np.sqrt(Y**2+X**2); hb=rr>.5; lb=rr<.1
    G=f2(gt); hfg=(np.abs(G[:,hb])**2).sum(); del G
    print(f'===== {scan}  {"name":58s} volPSNR  a0/201  HFratio  errHF  errLF')
    def rep(lbl,p):
        v=np.abs(np.load(p)).astype(np.float32)
        ps=20*np.log10(gm)-10*np.log10(((v-gt)[m3]**2).mean())
        g=gt[201]; mk=m3[201]; pa=20*np.log10(gm)-10*np.log10(((v[201]-g)[mk]**2).mean())
        F=f2(v); hf=(np.abs(F[:,hb])**2).sum()/hfg; E=f2(v-gt); P=np.abs(E)**2
        print(f'  {lbl:62s} {ps:7.3f} {pa:7.3f}  {hf:6.3f}  {P[:,hb].sum():7.0f} {P[:,lb].sum():6.1f}',flush=True)
    for n in names: rep(n,T+f'{n}_{scan}_complex.npy')
    for r in runs:
        p=WS+r+f'/candidate_{scan}_complex.npy'
        if os.path.exists(p): rep(r.replace('/s42',''),p)
