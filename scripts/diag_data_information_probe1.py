"""Probe what information the undersampled acquisition still contains beyond the
sampled k-space values: noise floor / coil covariance, k-space shutter, head
support, low-res phase smoothness (phase-constraint feasibility), and the
conjugate-symmetry gain of the ALIGNED vd-Poisson mask.  Also splits the
remaining error of the newest aligned AF12 MC-GS reconstruction into
(mask ∪ mirror) vs outside k-space and gives a radial error profile.
Read-only.  CPU only.
"""
import sys, time, glob, os
import numpy as np, h5py
from scipy import fft as sfft
from scipy.io import loadmat
from scipy.ndimage import maximum_filter

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

ROOT = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC"
DATA = "/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat"
PATHS = {
    "T1":    f"{DATA}/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_T1_trainer.mat",
    "FLAIR": f"{DATA}/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_FLAIR_trainer.mat",
}
CSM = {"T1": f"{ROOT}/csm_espirit/csm_espirit_T1.mat", "FLAIR": f"{ROOT}/csm_espirit/csm_espirit_FLAIR.mat"}
MASK = f"{ROOT}/mask/mask3d_vdpois_af12_transverse_ALIGNED.mat"
ARC = "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_arc"
WORKERS = 16

def load_complex_h5(path, key):
    with h5py.File(path, "r") as f:
        d = f[key]
        if d.dtype.names:
            out = np.empty(d.shape, np.complex64)
            out.real = d["real"]; out.imag = d["imag"]
        else:
            out = d[()].astype(np.complex64)
    return out

def load_any(path, key):
    try:
        return load_complex_h5(path, key)
    except (OSError, KeyError):
        m = loadmat(path)
        k = key if key in m else [k for k in m if not k.startswith("__")][0]
        return m[k]

def coil_last(a, ncoil=12):
    ax = [i for i, s in enumerate(a.shape) if s == ncoil][0]
    return np.moveaxis(a, ax, -1)

def ifft3c(k):
    return sfft.fftshift(sfft.ifftn(sfft.ifftshift(k, axes=(0, 1, 2)), axes=(0, 1, 2), norm="ortho", workers=WORKERS), axes=(0, 1, 2))
def fft3c(x):
    return sfft.fftshift(sfft.fftn(sfft.ifftshift(x, axes=(0, 1, 2)), axes=(0, 1, 2), norm="ortho", workers=WORKERS), axes=(0, 1, 2))

def psnr_max(rec, gt):
    gt = np.abs(gt); rec = np.abs(rec)
    return 20 * np.log10(gt.max() / np.sqrt(np.mean((rec - gt) ** 2)))

def bcast_pe(a2, ro_axis, sh):
    """broadcast a 2-D PE-plane array along the readout axis to full 3-D shape."""
    return np.moveaxis(np.broadcast_to(a2[..., None], (*a2.shape, sh[ro_axis])), -1, ro_axis)

# ---------------- mask ----------------
mk = loadmat(MASK); mask = mk[[k for k in mk if not k.startswith("__")][0]].astype(np.float32)
ro_axis = int(np.argmin([mask.var(axis=a).mean() for a in range(3)]))
m2 = np.take(mask, 0, axis=ro_axis)
log("mask", mask.shape, "sampled frac", float(mask.mean()), "readout axis", ro_axis, "2D PE mask", m2.shape)

for con in ("T1", "FLAIR"):
    log(f"================ {con} ================")
    k = coil_last(load_any(PATHS[con], "ksp"))      # (D,H,W,C)
    S = coil_last(load_any(CSM[con], "map")).astype(np.complex64)
    D, H, W, C = k.shape
    sh = np.array([D, H, W])
    log("ksp", k.shape, "csm", S.shape, "SoS median", float(np.median(np.sqrt((np.abs(S) ** 2).sum(-1)))))

    # ---- DC position (data) vs array centre ----
    e = (np.abs(k) ** 2).sum(-1)
    dc = np.unravel_index(int(np.argmax(e)), e.shape)
    acs_peak = np.unravel_index(int(np.argmax(mask.sum(axis=ro_axis))), m2.shape)
    log("data DC index", dc, "array centre", tuple(sh // 2), "mask ACS-density peak (PE plane)", acs_peak)

    # ---- k-space shutter / corners ----
    pe = [a for a in range(3) if a != ro_axis]
    idx = np.indices(m2.shape).astype(np.float32)
    dc2 = [dc[pe[0]], dc[pe[1]]]
    rho = np.sqrt(((idx[0] - dc2[0]) / (m2.shape[0] / 2)) ** 2 + ((idx[1] - dc2[1]) / (m2.shape[1] / 2)) ** 2)
    e_pe = np.moveaxis(e, ro_axis, 0).sum(0)
    for lo, hi in [(0, 0.1), (0.5, 0.6), (0.9, 1.0), (1.0, 1.1), (1.1, 1.2), (1.2, 1.28), (1.28, 1.35), (1.35, 1.5)]:
        sel = (rho >= lo) & (rho < hi)
        if sel.sum():
            log(f"  PE ring rho[{lo:.2f},{hi:.2f}) n={int(sel.sum()):6d} mean|k|^2/pt={float(e_pe[sel].mean()):.4e} exact-zero frac={float((e_pe[sel]==0).mean()):.3f} sampled frac={float(m2[sel].mean()):.4f}")
    shutter_frac = float((rho >= 1.28).mean())
    log(f"  PE-plane fraction beyond rho>=1.28: {shutter_frac:.3f}")

    # ---- noise estimate from the outermost k-space band ----
    ro_rel = np.abs(np.arange(sh[ro_axis]) - dc[ro_axis]) / (sh[ro_axis] / 2)
    band_pe = (rho >= 1.10) & (rho < 1.26)
    band_ro = ro_rel >= 0.85
    sel3 = bcast_pe(band_pe, ro_axis, sh) & np.moveaxis(np.broadcast_to(band_ro, (*m2.shape, sh[ro_axis])), -1, ro_axis)
    samp = k[sel3]
    samp = samp[np.abs(samp).sum(-1) > 0]
    cov = (samp.conj().T @ samp) / samp.shape[0]
    sig_c = np.sqrt(np.real(np.diag(cov)))
    corr = np.abs(cov) / np.outer(sig_c, sig_c)
    off = corr[~np.eye(C, dtype=bool)]
    sigma_k = float(np.sqrt(np.mean(sig_c ** 2)))
    ev = np.linalg.eigvalsh(cov)
    log(f"  outer-band samples n={samp.shape[0]}; per-coil complex noise std mean {sigma_k:.4e} (min {sig_c.min():.3e} max {sig_c.max():.3e})")
    log(f"  coil noise correlation |corr| off-diag mean {off.mean():.3f} max {off.max():.3f}; cov eigen spread {ev.max()/ev.min():.1f}")

    # ---- energy retained by mask ----
    E_tot = float(e.sum()); E_samp = float((e * mask).sum())
    log(f"  multicoil k energy retained by mask: {E_samp/E_tot*100:.2f}%  (unsampled {100-E_samp/E_tot*100:.2f}%)")

    # ---- GT and ZF (unit-SoS ESPIRiT combine) ----
    gt = np.zeros(sh, np.complex64); zf = np.zeros(sh, np.complex64)
    for c in range(C):
        gt += np.conj(S[..., c]) * ifft3c(k[..., c])
        zf += np.conj(S[..., c]) * ifft3c(k[..., c] * mask)
    gtm = np.abs(gt); gmax = float(gtm.max()); p999 = float(np.percentile(gtm, 99.9))
    log(f"  GT max {gmax:.4e} P99.9 {p999:.4e};  ZF PSNR whole@max = {psnr_max(zf, gt):.3f} dB (anchor ~31.11 T1 / ~30.70 FLAIR)")
    head = gtm > 0.05 * p999
    f_head = float(head.mean())
    log(f"  head support fraction (|x|>5% P99.9): {f_head:.3f} => background known-zero voxels {1-f_head:.3f}")

    # ---- image-domain noise measured in the GT itself ----
    far_bg = maximum_filter(gtm, size=9) < 0.01 * p999
    sigma_gt = float(np.sqrt(np.mean(np.abs(gt[far_bg]) ** 2)))
    log(f"  deep-background voxels {int(far_bg.sum())}: sigma_gt (complex) = {sigma_gt:.4e} vs outer-k-band sigma_k = {sigma_k:.4e} (ratio {sigma_gt/sigma_k:.2f})")

    # ---- noise-limited PSNR ceiling vs noisy GT ----
    f_s = float(mask.mean())
    sigma_img2 = sigma_gt ** 2
    mse_floor = (1 - f_s) * sigma_img2 * (f_head * 0.5 + (1 - f_head) * 1.0)
    ceil_whole = 20 * np.log10(gmax) - 10 * np.log10(mse_floor)
    ceil_head = 20 * np.log10(gmax) - 10 * np.log10((1 - f_s) * sigma_img2 * 0.5)
    log(f"  NOISE CEILING (unsampled-k noise of GT unrecoverable): whole@max ≈ {ceil_whole:.2f} dB ; head-only@max ≈ {ceil_head:.2f} dB")
    log(f"  head-mean |GT| / sigma_gt = {float(gtm[head].mean())/sigma_gt:.1f} (image SNR)")

    # ---- phase smoothness: low-res phase from ACS-sized centre ----
    K = fft3c(gt)
    for win in (24, 32, 48, 64):
        wpe = np.zeros(m2.shape, np.float32)
        a0 = slice(dc2[0] - win // 2, dc2[0] + win // 2); a1 = slice(dc2[1] - win // 2, dc2[1] + win // 2)
        wpe[a0, a1] = np.outer(np.hanning(win), np.hanning(win))
        xlr = ifft3c(K * bcast_pe(wpe, ro_axis, sh))
        r = gt * np.exp(-1j * np.angle(xlr))
        e_im_head = float((r.imag[head] ** 2).sum() / (np.abs(gt[head]) ** 2).sum())
        neg = float((r.real[head] < 0).mean())
        log(f"  low-res phase {win}x{win} PE (full RO): imag-residual energy in head {e_im_head*100:.3f}%  neg-real frac {neg*100:.2f}%")
    wpe = np.zeros(m2.shape, np.float32)
    wpe[dc2[0]-16:dc2[0]+16, dc2[1]-16:dc2[1]+16] = np.outer(np.hanning(32), np.hanning(32))
    wro = np.zeros(sh[ro_axis], np.float32); wro[dc[ro_axis]-16:dc[ro_axis]+16] = np.hanning(32)
    W3 = bcast_pe(wpe, ro_axis, sh) * np.moveaxis(np.broadcast_to(wro, (*m2.shape, sh[ro_axis])), -1, ro_axis)
    r = gt * np.exp(-1j * np.angle(ifft3c(K * W3)))
    log(f"  low-res phase 32^3 (3-D window): imag-residual energy in head {float((r.imag[head]**2).sum()/(np.abs(gt[head])**2).sum())*100:.3f}%")

    # ---- conjugate-symmetry gain of the mask (reflect about data DC) ----
    i0 = np.arange(m2.shape[0]); i1 = np.arange(m2.shape[1])
    j0 = 2 * dc2[0] - i0; j1 = 2 * dc2[1] - i1
    ok0 = (j0 >= 0) & (j0 < m2.shape[0]); ok1 = (j1 >= 0) & (j1 < m2.shape[1])
    mconj = np.zeros_like(m2)
    mconj[np.ix_(ok0, ok1)] = m2[np.ix_(j0[ok0], j1[ok1])]
    union = np.maximum(m2, mconj)
    log(f"  mask frac {m2.mean():.4f} -> mask ∪ mirror {union.mean():.4f}  (effective AF {1/m2.mean():.2f} -> {1/union.mean():.2f} if phase known);  inside shutter: {m2[rho<1.28].mean():.4f} -> {union[rho<1.28].mean():.4f}")

    # ---- residual split of newest aligned AF12 reconstruction ----
    cands = (sorted(glob.glob(f"{ARC}/*bboxfix*volumes.npz"))
             + sorted(glob.glob(f"{ARC}/reg_aligned*volumes.npz"))
             + sorted(glob.glob(f"{ARC}/af12_*volumes.npz"), key=os.path.getmtime, reverse=True))
    if len(sys.argv) > 1:
        cands = [sys.argv[1]]
    if cands:
        f = np.load(cands[0]); keys = list(f.keys())
        tag = "t1" if con == "T1" else "fl"
        rk = [kk for kk in keys if kk.lower().startswith("rec") and tag in kk.lower()]
        gk = [kk for kk in keys if kk.lower().startswith("gt") and tag in kk.lower()]
        log("  ARC file", os.path.basename(cands[0]), "keys", keys, "->", rk, gk)
        if rk and f[rk[0]].shape == tuple(sh):
            rec = f[rk[0]].astype(np.float32)
            if gk:
                g2 = f[gk[0]].astype(np.float32)
                a = float((g2 * gtm).sum() / (g2 * g2).sum())
                log(f"    ARC gt vs my GT: rel diff after scale {np.linalg.norm(g2*a-gtm)/np.linalg.norm(gtm):.2e} (≈0 => conventions match)")
            a = float((rec[head] * gtm[head]).sum() / (rec[head] ** 2).sum()); rec_s = rec * a
            log(f"    ARC recon PSNR whole@max {psnr_max(rec_s, gt):.3f} dB;  head-only@max {20*np.log10(gmax/np.sqrt(np.mean((rec_s[head]-gtm[head])**2))):.3f} dB")
            dm = rec_s - gtm
            log(f"    magnitude residual energy: head {(dm[head]**2).sum():.3e} ({(dm[head]**2).sum()/(dm**2).sum()*100:.0f}%)  background {(dm[~head]**2).sum():.3e}")
            Kr = fft3c(rec_s.astype(np.complex64)); Kg = fft3c(gtm.astype(np.complex64))
            dK = np.abs(Kr - Kg) ** 2; Eg = np.abs(Kg) ** 2
            u3 = bcast_pe(union, ro_axis, sh)
            E_in = float((dK * u3).sum()); E_out = float((dK * (1 - u3)).sum()); Eg_out = float((Eg * (1 - u3)).sum())
            log(f"    |k| residual (magnitude images): inside mask∪mirror {E_in:.3e} ({E_in/(E_in+E_out)*100:.1f}%)  outside {E_out:.3e};  GT energy outside {Eg_out:.3e} => recovered {(1-E_out/Eg_out)*100:.1f}% of outside energy")
            ii = np.indices(sh).astype(np.float32)
            r3 = np.sqrt(sum(((ii[a] - dc[a]) / (sh[a] / 2)) ** 2 for a in range(3)))
            noise_pt = sigma_img2 * 0.5
            edges = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 2.0]
            log("    radial profile (3-D normalised radius about DC):")
            for lo, hi in zip(edges[:-1], edges[1:]):
                s = (r3 >= lo) & (r3 < hi); n = int(s.sum())
                if n:
                    log(f"      r[{lo:.2f},{hi:.2f}) n={n:8d}  GT-energy {Eg[s].sum()/Eg.sum()*100:6.2f}%  resid/GT {dK[s].sum()/max(Eg[s].sum(),1e-30):6.3f}  sampled {mask[s].mean():.3f}  GT-per-pt/noise-per-pt {Eg[s].mean()/noise_pt:9.1f}")
            del Kr, Kg, dK, Eg, rec, rec_s, dm
    del k, S, K, gt, zf, e
log("DONE")
