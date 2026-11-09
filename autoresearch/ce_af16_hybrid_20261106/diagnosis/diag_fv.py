"""Read-only diagnosis: why MC-GS FV37 loses axial/sagittal to ZS-SSDU. Prints only, writes nothing."""
import numpy as np
from scipy import ndimage

T = '/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois_cmp'
WS = '/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_losstune_20261101'
R = T + '/recon/'
M = {
    'pre': {'ZS-SSDU': R + 'zsssdu_pre_complex.npy', 'GS-single': R + 'gs_single_pre_complex.npy',
            'CS-SENSE': R + 'cssense_o200_lam0.0005_pre_complex.npy',
            'MCGS42': WS + '/runs/FV37_lambda_wav_0/s42/m0625/relax_a1/candidate_pre_complex.npy',
            'MCGS7': WS + '/runs/FV37_lambda_wav_0/s7/m0625/relax_a1/candidate_pre_complex.npy'},
    'post': {'ZS-SSDU': R + 'zsssdu_post_complex.npy', 'GS-single': R + 'gs_single_post_complex.npy',
             'CS-SENSE': R + 'cssense_o200_lam0.00035_post_complex.npy',
             'MCGS42': WS + '/runs/FV37_lambda_wav_0/s42/m0625/relax_a1/candidate_post_complex.npy',
             'MCGS7': WS + '/runs/FV37_lambda_wav_0/s7/m0625/relax_a1/candidate_post_complex.npy'},
}
VIEWS = {0: 201, 1: 130, 2: 48}
pe = np.load(T + '/sim/mask/mask_af16_pe.npy').astype(bool)
print('PE mask', pe.shape, 'frac sampled %.4f' % pe.mean(), 'center sampled', pe[pe.shape[0]//2, pe.shape[1]//2])


def f2(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho', axes=(-2, -1)), axes=(-2, -1))


for scan in ('pre', 'post'):
    gt = np.abs(np.load(R + f'gt_{scan}_complex.npy')).astype(np.float32)
    gmax = float(gt.max()); thr = .05 * float(np.quantile(gt, .999)); m3 = gt > thr
    vols = {k: np.abs(np.load(p)).astype(np.float32) for k, p in M[scan].items()}
    print('\n==========', scan, 'shape', gt.shape, 'mask frac %.3f' % m3.mean())
    # ---- 3D volume PSNR in head mask
    for k, v in vols.items():
        e = ((v - gt)[m3] ** 2).mean()
        print('  vol PSNR %-9s %.3f' % (k, 20*np.log10(gmax) - 10*np.log10(e)))
    # ---- per-slice PSNR along every axis, delta MCGS42 - ZS-SSDU
    for ax in (0, 1, 2):
        n = gt.shape[ax]; ps = {k: np.full(n, np.nan) for k in vols}
        for i in range(n):
            g = np.take(gt, i, ax); mk = np.take(m3, i, ax)
            if mk.sum() < 2000: continue
            for k, v in vols.items():
                e = ((np.take(v, i, ax) - g)[mk] ** 2).mean()
                ps[k][i] = 20*np.log10(gmax) - 10*np.log10(e)
        d = ps['MCGS42'] - ps['ZS-SSDU']; dg = ps['GS-single'] - ps['ZS-SSDU']; ok = np.isfinite(d)
        idx = VIEWS[ax]
        print(f'  axis{ax}: slices={ok.sum()} MCGS-ZS mean {np.nanmean(d):+.3f} median {np.nanmedian(d):+.3f} '
              f'win-frac {(d[ok] > 0).mean():.3f} | at {idx}: {d[idx]:+.3f} (rank pct {(d[ok] < d[idx]).mean():.2f}) '
              f'| GSs-ZS mean {np.nanmean(dg):+.3f} | MCGS-GSs mean {np.nanmean(ps["MCGS42"]-ps["GS-single"]):+.3f}')
        w = np.arange(max(0, idx-12), min(n, idx+13), 4)
        print('     around idx:', ' '.join(f'{i}:{d[i]:+.2f}' for i in w if np.isfinite(d[i])))
    # ---- whole-volume hybrid k-space split of error (exact: mask constant along readout axis0)
    print('  hybrid-k error energy (|pred|-|gt| magnitude error, all axial slices):')
    rr = None
    for k, v in vols.items():
        E = f2(v - gt); P = np.abs(E) ** 2; tot = P.sum()
        if rr is None:
            Y, X = np.meshgrid(np.linspace(-1, 1, gt.shape[1]), np.linspace(-1, 1, gt.shape[2]), indexing='ij')
            rr = np.sqrt(Y**2 + X**2)
            bands = [(0, .1), (.1, .25), (.25, .5), (.5, .75), (.75, 2)]
        smp = P[:, pe].sum() / tot
        bandstr = ' '.join(f'{lo}-{hi}:{P[:, (rr >= lo) & (rr < hi)].sum():.3g}' for lo, hi in bands)
        print(f'    {k:9s} total {tot:.4g}  sampled-frac {smp:.3f}  unsampled {P[:, ~pe].sum():.4g} sampled {P[:, pe].sum():.4g} | {bandstr}')
    # ---- axial 201 slice diagnostics
    z = 201; g = gt[z]; mk = m3[z]
    gr = ndimage.gaussian_gradient_magnitude(g, 1.0); edge = mk & (gr > np.quantile(gr[mk], .8)); flat = mk & ~edge
    G = f2(g); hfg = (np.abs(G[rr > .5]) ** 2).sum()
    print(f'  axial {z}: pixels {mk.sum()} edge {edge.sum()}')
    for k, v in vols.items():
        p = v[z]; e = p - g
        a = (p[mk] * g[mk]).sum() / (p[mk] ** 2).sum()
        hf = (np.abs(f2(p)[rr > .5]) ** 2).sum() / hfg
        bias = e[mk].mean() / g[mk].mean()
        print(f'    {k:9s} SSE edge {(e[edge]**2).sum():.4g} flat {(e[flat]**2).sum():.4g} '
              f'outside-mask {(e[~mk]**2).sum():.3g} | bestscale {a:.4f} bias {bias:+.4f} | HF-energy/GT {hf:.3f}')
    del vols, gt
