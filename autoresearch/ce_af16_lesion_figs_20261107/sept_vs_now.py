# read-only: compare MC-GS generations on the same GT with one metric code (slice metric = CE_retro fig_three_planes_compact)
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity
B = Path('/data0/congcong/code/GS/Results_20260908')
V = B / 'CE_retro_cssense_af16_vdpois/recon'
TS = Path('/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_threeaxis_20261107/runs/TS01_h2_voxel_iso_bases/s42/m0625/relax_a1')
def arms(c):
    return [('9月配方 @vdpois', V / f'gs_joint_{c}_mag.npy'),
            ('10月 ar_i1 @vdpois', V / f'ar_i1_k30p90_{c}_mag.npy'),
            ('TS01 s42 @vdpois', TS / f'candidate_{c}_complex.npy'),
            ('ZS-SSDU强 @vdpois', V / f'zsssdu_z2_b15_cg10_e400_{c}_complex.npy'),
            ('[9月配方 @9月掩膜]', B / f'CE_retro_cssense_af16/recon/gs_joint_{c}_mag.npy'),
            ('[ZS-SSDU旧 @9月掩膜]', B / f'CE_retro_cssense_af16/recon/zsssdu_{c}_complex.npy')]
SLS = [(0, 196), (1, 130), (2, 40), (0, 201), (1, 131), (2, 48)]
for c in ('pre', 'post'):
    gt = np.abs(np.load(V / f'gt_{c}_complex.npy')).astype(np.float64)
    gmax = gt.max(); head = gt > 0.05 * np.quantile(gt, 0.999); g = gt[head]
    print(f'\n===== {c}')
    print(f"{'':22s} {'volPSNR':>8s} {'volNMSE':>8s} {'tSSIM':>7s} | " + ' '.join(f'a{a}/{i:<4d}' for a, i in SLS))
    for lab, p in arms(c):
        x = np.abs(np.load(p)).astype(np.float64); e = x[head] - g
        ps = 20 * np.log10(gmax) - 10 * np.log10((e * e).mean()); nm = (e * e).sum() / (g * g).sum()
        ss = []
        for k in range(0, gt.shape[0], 4):
            m = head[k]
            if m.sum() < 2000: continue
            _, smap = structural_similarity(gt[k], x[k], data_range=gmax, full=True, win_size=7); ss.append(smap[m].mean())
        cells = []
        for a, i in SLS:
            sl = [slice(None)] * 3; sl[a] = i; sl = tuple(sl); m2 = head[sl]; ee = x[sl][m2] - gt[sl][m2]
            cells.append(20 * np.log10(gmax) - 10 * np.log10((ee * ee).mean()))
        print(f'{lab:22s} {ps:8.3f} {nm:8.5f} {np.mean(ss):7.4f} | ' + ' '.join(f'{v:8.3f}' for v in cells))
