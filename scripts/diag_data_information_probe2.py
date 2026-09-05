"""Second probe: image-domain noise (FOV-corner air blocks), noise whiteness,
fine k-space energy profiles (PE-plane radius and readout axis) to detect vendor
band-limiting, unsampled signal-vs-noise, noise ceilings, background-hallucination
split, phase roughness on strong-signal voxels, and the residual energy of the
best aligned recon in the k-space region where the GT is (near) zero.
"""
import sys, time, glob, os
import numpy as np, h5py
from scipy import fft as sfft
from scipy.io import loadmat

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

ROOT = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC"
DATA = "/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat"
PATHS = {"T1": f"{DATA}/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_T1_trainer.mat",
         "FLAIR": f"{DATA}/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_FLAIR_trainer.mat"}
CSM = {"T1": f"{ROOT}/csm_espirit/csm_espirit_T1.mat", "FLAIR": f"{ROOT}/csm_espirit/csm_espirit_FLAIR.mat"}
MASK = f"{ROOT}/mask/mask3d_vdpois_af12_transverse_ALIGNED.mat"
ARCF = "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_arc/edgeTV_best_bboxfix_volumes.npz"
WORKERS = 16

def load_complex_h5(path, key):
    with h5py.File(path, "r") as f:
        d = f[key]; out = np.empty(d.shape, np.complex64); out.real = d["real"]; out.imag = d["imag"]
    return out
def coil_last(a, ncoil=12):
    ax = [i for i, s in enumerate(a.shape) if s == ncoil][0]; return np.moveaxis(a, ax, -1)
def ifft3c(k):
    return sfft.fftshift(sfft.ifftn(sfft.ifftshift(k, axes=(0,1,2)), axes=(0,1,2), norm="ortho", workers=WORKERS), axes=(0,1,2))
def fft3c(x):
    return sfft.fftshift(sfft.fftn(sfft.ifftshift(x, axes=(0,1,2)), axes=(0,1,2), norm="ortho", workers=WORKERS), axes=(0,1,2))
def bcast_pe(a2, ro_axis, sh):
    return np.moveaxis(np.broadcast_to(a2[..., None], (*a2.shape, sh[ro_axis])), -1, ro_axis)

mk = loadmat(MASK); mask = mk["mask"].astype(np.float32)
ro_axis = 2; m2 = mask[:, :, 0]; f_s = float(mask.mean())
arc = np.load(ARCF)

for con in ("T1", "FLAIR"):
    log(f"================ {con} ================")
    k = coil_last(load_complex_h5(PATHS[con], "ksp")); S = coil_last(loadmat(CSM[con])["map"]).astype(np.complex64)
    D, H, W, C = k.shape; sh = np.array([D, H, W]); N = D * H * W
    e = (np.abs(k) ** 2).sum(-1); dc = np.unravel_index(int(np.argmax(e)), e.shape)
    pe = [0, 1]; dc2 = [dc[0], dc[1]]
    idx = np.indices(m2.shape).astype(np.float32)
    rho = np.sqrt(((idx[0]-dc2[0])/(m2.shape[0]/2))**2 + ((idx[1]-dc2[1])/(m2.shape[1]/2))**2)
    # ---- fine PE-plane energy profile (per point, per coil, per readout sample) ----
    e_pe = e.sum(axis=2) / (C * W)
    log("  PE-plane mean |k|^2 per (coil,sample) by rho ring:")
    for lo, hi in [(0,0.05),(0.2,0.3),(0.4,0.5),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0),(1.0,1.1),(1.1,1.2),(1.2,1.28),(1.28,1.5)]:
        s = (rho>=lo)&(rho<hi)
        if s.sum(): log(f"    rho[{lo:.2f},{hi:.2f}) {e_pe[s].mean():.3e}")
    # ---- readout-axis energy profile ----
    e_ro = e.sum(axis=(0,1)) / (C * D * H)
    kro = np.abs(np.arange(W) - dc[2]) / (W/2)
    log("  readout-axis mean |k|^2 per (coil,sample) by |k_ro| band:")
    for lo, hi in [(0,0.05),(0.2,0.3),(0.5,0.6),(0.7,0.8),(0.8,0.9),(0.9,0.95),(0.95,1.0)]:
        s = (kro>=lo)&(kro<hi)
        if s.sum(): log(f"    kro[{lo:.2f},{hi:.2f}) {e_ro[s].mean():.3e}")
    # ---- GT, coil images ----
    gt = np.zeros(sh, np.complex64); coil_img = np.empty_like(k)
    for c in range(C):
        coil_img[..., c] = ifft3c(k[..., c]); gt += np.conj(S[..., c]) * coil_img[..., c]
    gtm = np.abs(gt); gmax = float(gtm.max()); p999 = float(np.percentile(gtm, 99.9))
    head = gtm > 0.05 * p999; f_head = float(head.mean()); Nh = int(head.sum())
    # ---- FOV-corner air blocks: noise ----
    B = 20; blocks = []
    for a in (slice(0,B), slice(D-B,D)):
        for b in (slice(0,B), slice(H-B,H)):
            for c_ in (slice(0,B), slice(W-B,W)):
                g = gt[a, b, c_]; blocks.append(g)
    rms = np.array([np.sqrt(np.mean(np.abs(g)**2)) for g in blocks]); rel = np.array([np.mean(np.abs(g)) for g in blocks]) / p999
    log(f"  corner blocks rms|gt|: {np.array2string(rms, precision=0, floatmode='fixed')}  (mean|gt|/P99.9: {np.array2string(rel*100, precision=2)}%)")
    sigma_gt = float(np.median(rms))
    gb = blocks[int(np.argsort(rms)[len(rms)//2])]
    ac = []
    for ax in range(3):
        x0 = gb - gb.mean(); x1 = np.roll(x0, 1, axis=ax)
        ac.append(float(np.real(np.vdot(x0, x1)) / np.vdot(x0, x0).real))
    log(f"  sigma_gt (median block, complex) = {sigma_gt:.4e}; lag-1 autocorr of GT noise per axis {np.round(ac,3)} (0 => white; >0.3 => low-pass filtered)")
    sig_c = np.array([np.median([np.sqrt(np.mean(np.abs(coil_img[a,b,c_,cc])**2)) for a in (slice(0,B),slice(D-B,D)) for b in (slice(0,B),slice(H-B,H)) for c_ in (slice(0,B),slice(W-B,W))]) for cc in range(C)])
    log(f"  per-coil image-domain noise std: mean {sig_c.mean():.3e} min {sig_c.min():.3e} max {sig_c.max():.3e}")
    log(f"  image SNR: head-mean|GT|/sigma_gt = {float(gtm[head].mean())/sigma_gt:.1f};  P99.9/sigma_gt = {p999/sigma_gt:.1f}")
    # ---- unsampled energy: signal vs noise ----
    E_tot = float(e.sum()); E_un = float((e*(1-mask)).sum()); n_un = float((1-mask).sum())*C
    E_noise_un_white = n_un * float(np.mean(sig_c**2))
    log(f"  unsampled multicoil energy {E_un/E_tot*100:.2f}% of total; white-noise floor there would be {E_noise_un_white/E_tot*100:.2f}% => unsampled SIGNAL >= {(E_un-E_noise_un_white)/E_tot*100:.2f}% of total (if noise white; less if vendor-filtered)")
    # ---- noise ceilings vs the noisy GT ----
    mse_floor_whole = (1-f_s)*sigma_gt**2*(f_head*0.5 + (1-f_head)*1.0)
    log(f"  NOISE CEILING (white-noise assumption): whole@max {20*np.log10(gmax)-10*np.log10(mse_floor_whole):.2f} dB ; head-only@max {20*np.log10(gmax)-10*np.log10((1-f_s)*sigma_gt**2*0.5):.2f} dB")
    # ---- phase roughness on strong-signal voxels ----
    K = fft3c(gt)
    for win in (24, 48):
        wpe = np.zeros(m2.shape, np.float32)
        wpe[dc2[0]-win//2:dc2[0]+win//2, dc2[1]-win//2:dc2[1]+win//2] = np.outer(np.hanning(win), np.hanning(win))
        r = gt * np.exp(-1j*np.angle(ifft3c(K*bcast_pe(wpe, ro_axis, sh))))
        for thr in (0.05, 0.3):
            m = gtm > thr*p999
            log(f"  low-res phase {win}x{win}: imag-residual energy on voxels |x|>{int(thr*100)}% P99.9 = {float((r.imag[m]**2).sum()/(np.abs(gt[m])**2).sum())*100:.2f}%  (n={int(m.sum())})")
    # ---- ARC recon residual decomposition ----
    tag = "T1" if con == "T1" else "FL"
    rec = arc["rec"+tag].astype(np.float32)
    a = float((rec[head]*gtm[head]).sum()/(rec[head]**2).sum()); rec_s = rec*a; dm = rec_s - gtm
    E_res = float((dm**2).sum()); E_res_bg = float((dm[~head]**2).sum()); E_res_head = E_res - E_res_bg
    E_gt = float((gtm**2).sum())
    bg_noise = (1-f_head)*N*sigma_gt**2; head_noise = Nh*(1-f_s)*sigma_gt**2*0.5
    log(f"  ARC recon NMSE (magnitude, whole) {E_res/E_gt*100:.3f}%  head-only NMSE {E_res_head/float((gtm[head]**2).sum())*100:.3f}%")
    log(f"  residual split: head {E_res_head/E_res*100:.0f}% | background {E_res_bg/E_res*100:.0f}% ;  GT background noise energy = {bg_noise/E_res_bg*100:.0f}% of background residual ; unavoidable in-head noise = {head_noise/E_res_head*100:.1f}% of head residual")
    log(f"  => PSNR if background residual were only GT noise: whole@max {20*np.log10(gmax)-10*np.log10((E_res_head+bg_noise)/N):.2f} dB (now {20*np.log10(gmax)-10*np.log10(E_res/N):.2f})")
    # residual in k-space regions where GT is ~zero (PE rho > 1.0 / > 1.1), magnitude-domain spectra
    Kr = fft3c(rec_s.astype(np.complex64)); Kg = fft3c(gtm.astype(np.complex64))
    dK = np.abs(Kr-Kg)**2; Eg = np.abs(Kg)**2; Er = np.abs(Kr)**2
    for thr in (0.9, 1.0, 1.1):
        s3 = bcast_pe((rho >= thr).astype(np.float32), ro_axis, sh)
        log(f"  PE rho>={thr:.1f}: GT energy {float((Eg*s3).sum())/E_gt*100:.3f}% | recon energy {float((Er*s3).sum())/E_gt*100:.3f}% | residual there {float((dK*s3).sum())/E_res*100:.1f}% of total residual")
    # readout high band
    s_ro = np.broadcast_to((kro >= 0.9), sh)
    log(f"  |k_ro|>=0.9: GT energy {float((Eg*s_ro).sum())/E_gt*100:.3f}% | recon {float((Er*s_ro).sum())/E_gt*100:.3f}% | residual {float((dK*s_ro).sum())/E_res*100:.1f}% of total residual")
    # ring-wise absolute GT-per-point vs noise-per-point (magnitude domain)
    ii = np.indices(sh).astype(np.float32); r3 = np.sqrt(sum(((ii[q]-dc[q])/(sh[q]/2))**2 for q in range(3)))
    noise_pt = sigma_gt**2*(f_head*0.5 + (1-f_head)*0.215)
    log(f"  magnitude-domain white-noise energy per k-point ≈ {noise_pt:.3e}")
    for lo, hi in [(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.8),(0.8,1.0),(1.0,1.3),(1.3,2.0)]:
        s = (r3>=lo)&(r3<hi)
        log(f"    r3[{lo:.1f},{hi:.1f}) GT/pt {Eg[s].mean():.3e} ({Eg[s].mean()/noise_pt:7.1f}x noise)  recon/pt {Er[s].mean():.3e}  resid/pt {dK[s].mean():.3e}  resid share {dK[s].sum()/E_res*100:5.1f}%")
    del k, S, coil_img, gt, K, Kr, Kg, dK, Eg, Er
log("DONE")
