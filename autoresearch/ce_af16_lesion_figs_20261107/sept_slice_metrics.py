# read-only: per-slice metrics as in CE_retro_cssense_af16/scripts/fig_three_planes_compact.py (no crop, uniform 7x7 SSIM)
import json, numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity
R = Path('/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16')
SL = (201, 131, 48)
roi = np.load('/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz')
core = None
for k in roi.files:
    if '病灶核心' in k or 'lesion_core' in k: core = roi[k] > 0; ck = k
print('roi keys', roi.files)
if core is not None:
    print('lesion core', ck, core.shape, 'voxels', int(core.sum()), 'pixels in planes', [int(core.take(i, axis=a).sum()) for a, i in enumerate(SL)])
for c in ('pre', 'post'):
    rows = {r['method']: r for r in json.load(open(R / 'logs/summary.json')) if r['contrast'] == c}
    sp = rows['spirit']
    arms = [('ZF', f'recon/zf_{c}_complex.npy'), ('CG-SENSE', f'recon/cgsense_{c}_complex.npy'),
            ('CS-SENSE', f"recon/cssense_o200_lam{rows['cssense_o200']['lam']:g}_{c}_complex.npy"),
            ('Joint SENSE', f"recon/jcssense_o200_lam{rows['jcssense']['lam']:g}_{c}_complex.npy"),
            ('L1-SPIRiT', f"recon/spirit_ks{sp['ksize']}_lam{sp['lam']:g}_{c}_complex.npy"),
            ('ZS-SSDU', f'recon/zsssdu_{c}_complex.npy'), ('3D-DIP', f'recon/dip_{c}_complex.npy'),
            ('3D-INR', f'recon/inr_{c}_complex.npy'), ('MC-INR', f'recon/mcinr_{c}_complex.npy'),
            ('GS-single', f'recon/gs_single_{c}_mag.npy'), ('MC-GS', f'recon/gs_joint_{c}_mag.npy')]
    gt = np.abs(np.load(R / f'recon/gt_{c}_complex.npy')).astype(np.float32)
    gmax = float(gt.max()); head = gt > 0.05 * float(np.quantile(gt, 0.999))
    vols = {lab: np.abs(np.load(R / p, mmap_mode='r')) for lab, p in arms}
    for axis, idx in enumerate(SL):
        sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
        g64 = gt[sl].astype(np.float64); m2 = head[sl]; g = g64[m2]
        res = {}
        for lab, _ in arms:
            r = np.asarray(vols[lab][sl], dtype=np.float32).astype(np.float64); e = r[m2] - g
            ps = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
            _, smap = structural_similarity(g64, r, data_range=gmax, full=True, win_size=7)
            res[lab] = (ps, float(smap[m2].mean()), float((e * e).sum() / (g * g).sum()))
        opp = {k: v for k, v in res.items() if k != 'MC-GS'}; m = res['MC-GS']
        wins = [m[0] > max(v[0] for v in opp.values()), m[1] > max(v[1] for v in opp.values()), m[2] < min(v[2] for v in opp.values())]
        print(f'\n== {c} axis {axis} slice {idx}  MC-GS wins psnr/ssim/nmse: {wins}')
        for lab, v in sorted(res.items(), key=lambda kv: -kv[1][0]):
            print(f'  {lab:12s} {v[0]:8.3f} {v[1]:7.4f} {v[2]:8.5f}')
