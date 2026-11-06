import numpy as np
T='/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois_cmp/recon/'
WS='/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_losstune_20261101/runs/FV37_lambda_wav_0/s42/m0625/relax_a1/'
for scan in ('pre','post'):
    gt=np.abs(np.load(T+f'gt_{scan}_complex.npy')).astype(np.float64); gm=gt.max(); m3=gt>.05*np.quantile(gt,.999)
    mc=np.abs(np.load(WS+f'candidate_{scan}_complex.npy')).astype(np.float64)
    zs=np.abs(np.load(T+f'zsssdu_{scan}_complex.npy')).astype(np.float64)
    a=(mc[m3]*gt[m3]).sum()/(mc[m3]**2).sum()
    print(scan,'volume best scale for MCGS %.4f'%a)
    for ax,i in ((0,201),(1,130),(2,48)):
        g=np.take(gt,i,ax); mk=np.take(m3,i,ax)
        P=lambda p:20*np.log10(gm)-10*np.log10(((p-g)[mk]**2).mean())
        p=np.take(mc,i,ax); z=np.take(zs,i,ax)
        print(f'  a{ax}/{i}: MCGS {P(p):.3f}  MCGS*a {P(a*p):.3f}  avg(MCGS,ZS) {P(.5*(p+z)):.3f}  ZS {P(z):.3f}  corr(errMC,errZS) {np.corrcoef((p-g)[mk],(z-g)[mk])[0,1]:.3f}')
