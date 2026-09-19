"""Noise estimate C: complex background of the SENSE-combined reference, far from the head.
Estimates A (k-space tail, still decaying = signal) and B (Haar HHH, whose passband sits on the
measured-zero corners, so it misses noise) were both unusable; see ceiling_noise.py."""
import sys, json
import numpy as np, h5py, scipy.io as sio
from scipy import ndimage
sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt")
from diag_artefact.headonly import head_mask
from uih_mtp.rescore import partition_region
T = "/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro/"
mask = sio.loadmat(T + "data/mask_af12_2d.mat"); mask = next(v for k, v in mask.items() if not k.startswith("__")).astype(bool)
with h5py.File(T + "data/support_2d.h5", "r") as f: sup = np.asarray(f[list(f.keys())[0]]).astype(bool)
N_u, N_s = int((sup & ~mask).sum()), int(sup.sum())
best = {"T1w": 38.848, "T2s": 38.848 and 34.175}
res = {}
for c in ("T1w", "T2s"):
    gt = np.load(T + f"recon/gt_{c}_complex.npy").astype(np.complex128); g = np.abs(gt)
    gmax = float(g.max())
    hm_all = head_mask(g.astype(np.float32))
    far = ~ndimage.binary_dilation(hm_all, iterations=12)
    # only low-signal background: below 1 % of the head threshold, to exclude ghosts / scalp tails
    thr = 0.01 * float(np.quantile(g, 0.999))
    bg = far & (g < thr)
    re, im = gt.real[bg], gt.imag[bg]
    s_re = float(np.median(np.abs(re - np.median(re))) / 0.6745); s_im = float(np.median(np.abs(im - np.median(im))) / 0.6745)
    v_total = s_re ** 2 + s_im ** 2
    # spatial variation: noise level per slab along SPE and along RO in the background
    head = hm_all & partition_region(g.shape, 2, 12, 92)
    mse_floor = v_total * (N_u / N_s) / 2
    ceil = 20 * np.log10(gmax) - 10 * np.log10(mse_floor)
    # where does the background sit? fraction of volume, and a check of whiteness along readout
    lag = [float(np.corrcoef(gt.real[1:][bg[1:] & bg[:-1]], gt.real[:-1][bg[1:] & bg[:-1]])[0, 1]),
           float(np.corrcoef(gt.real[:, 1:][bg[:, 1:] & bg[:, :-1]], gt.real[:, :-1][bg[:, 1:] & bg[:, :-1]])[0, 1]),
           float(np.corrcoef(gt.real[:, :, 1:][bg[:, :, 1:] & bg[:, :, :-1]], gt.real[:, :, :-1][bg[:, :, 1:] & bg[:, :, :-1]])[0, 1])]
    res[c] = dict(bg_frac=float(bg.mean()), sigma_re=s_re, sigma_im=s_im, lag1_corr_RO_PE_SPE=lag,
                  mse_floor=mse_floor, ceiling_C=ceil, gmax=gmax, head_voxels=int(head.sum()))
    print(c, json.dumps(res[c]))
json.dump(res, open("/tmp/claude-1002/-home-congcongliu--claude/55f338b6-6256-4c6f-9006-fd26f65d0f54/scratchpad/p4/ceiling_bg.json", "w"), indent=1)
