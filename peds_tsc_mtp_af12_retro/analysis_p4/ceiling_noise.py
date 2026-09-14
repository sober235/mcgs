"""Round 0 of the +1 dB campaign: how much head PSNR is even available on this reference?

The reference is GRAPPA-completed, SENSE-combined data, so it carries noise. The sampled positions are
matched by construction; the noise at the UNSAMPLED in-support positions is independent of every
measurement, so no method can predict it. Its energy is a floor on the head MSE of any reconstruction:
    max head PSNR  <=  20 log10(gmax) - 10 log10(MSE_noise_unsampled_in_head).
Two independent noise estimates, both biased HIGH (so the ceiling they give is conservative, i.e. low):
  A. k-space: the flat tail of the GT's power per phase-encode position at the outer edge of the support.
  B. image: MAD of the finest Haar HHH coefficients of Re/Im inside the head (anatomy leaks in).
"""
import sys, json
import numpy as np, h5py, scipy.io as sio
sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt")
from diag_artefact.headonly import head_mask
from uih_mtp.rescore import partition_region

T = "/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro/"
mask = sio.loadmat(T + "data/mask_af12_2d.mat")
mask = next(v for k, v in mask.items() if not k.startswith("__")).astype(bool)
with h5py.File(T + "data/support_2d.h5", "r") as f:
    sup = np.asarray(f[list(f.keys())[0]]).astype(bool)
print("mask", mask.shape, mask.sum(), "support", sup.shape, sup.sum(), "mask outside support", (mask & ~sup).sum())
N_u = int((sup & ~mask).sum()); N_s = int(sup.sum())
ARMS = {"T1w": {"MC-GS tuned": "gs_joint_p3_T1w_complex", "GS-single tuned": "gs_single_p3_T1w_complex",
                "L1-ESPIRiT": "l1espirit_lam0.001_T1w_complex"},
        "T2s": {"MC-GS tuned": "gs_joint_p3_T2s_complex", "GS-single": "gs_single_ar_T2s_complex",
                "L1-ESPIRiT": "l1espirit_lam0.003_T2s_complex"}}
out = {}
for c, arms in ARMS.items():
    gt = np.load(T + f"recon/gt_{c}_complex.npy").astype(np.complex128)
    g = np.abs(gt)
    reg = partition_region(g.shape, 2, 12, 92)
    head = head_mask(g.astype(np.float32)) & reg
    gmax = float(g.max())
    K = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(gt), norm="ortho"))
    P2 = (np.abs(K) ** 2).sum(0)                                   # (PE, SPE), summed over readout
    # which convention does the mask use? energy captured must be large and zero outside support
    cap = P2[mask].sum() / P2.sum(); outside = P2[~sup].sum() / P2.sum()
    pe, spe = np.meshgrid(np.arange(P2.shape[0]) - P2.shape[0] // 2, np.arange(P2.shape[1]) - P2.shape[1] // 2, indexing="ij")
    r = np.sqrt((pe / (P2.shape[0] / 2)) ** 2 + (spe / (P2.shape[1] / 2)) ** 2)
    prof = []
    for lo in np.arange(0.0, 1.0, 0.1):
        sel = sup & (r >= lo) & (r < lo + 0.1)
        if sel.sum():
            prof.append((round(float(lo), 1), int(sel.sum()), float(np.median(P2[sel])), float(np.percentile(P2[sel], 10))))
    ring = sup & (r >= 0.85)
    floorA = float(np.median(P2[ring])) if ring.sum() > 50 else float("nan")
    nz = int((g > 0).sum())                                        # SENSE support: noise lives here
    vA = N_u * floorA / nz                                         # complex noise var per voxel from unsampled positions
    mseA = vA / 2                                                  # magnitude, high-SNR approximation
    # B: image-domain MAD on the finest Haar HHH band (both Re and Im), inside the head
    def hhh(x):
        x = x[: x.shape[0] // 2 * 2, : x.shape[1] // 2 * 2, : x.shape[2] // 2 * 2]
        s = np.array([1, -1])
        w = np.einsum("i,j,k->ijk", s, s, s) / np.sqrt(8)
        return sum(w[i, j, k] * x[i::2, j::2, k::2] for i in range(2) for j in range(2) for k in range(2))
    hm = head[: head.shape[0] // 2 * 2 : 2, : head.shape[1] // 2 * 2 : 2, : head.shape[2] // 2 * 2 : 2]
    comp = np.concatenate([hhh(gt.real)[hm], hhh(gt.imag)[hm]])
    sig_comp = float(np.median(np.abs(comp)) / 0.6745)
    vB = 2 * sig_comp ** 2 * (N_u / N_s)
    mseB = vB / 2
    rows = {}
    for lab, key in arms.items():
        rec = np.abs(np.load(T + f"recon/{key}.npy"))
        mse = float(np.mean((rec[head].astype(np.float64) - g[head]) ** 2))
        rows[lab] = dict(head_psnr=20 * np.log10(gmax) - 10 * np.log10(mse), mse=mse)
    best_other = max(v["head_psnr"] for k, v in rows.items() if not k.startswith("MC-GS"))
    ceilA = 20 * np.log10(gmax) - 10 * np.log10(mseA); ceilB = 20 * np.log10(gmax) - 10 * np.log10(mseB)
    out[c] = dict(energy_captured_by_mask=cap, energy_outside_support=outside, N_u=N_u, N_s=N_s,
                  sense_support_frac=nz / g.size, radial_profile_median_p10=prof,
                  floorA_P2=floorA, ceiling_A=ceilA, sigma_comp_B=sig_comp, ceiling_B=ceilB,
                  arms=rows, best_other=best_other, target=best_other + 1.0,
                  headroom_A_over_best_other=ceilA - best_other, headroom_B_over_best_other=ceilB - best_other)
    print(f"\n== {c}: mask captures {cap:.3f} of GT k-space energy; outside support {outside:.2e}; SENSE support {nz / g.size:.3f}")
    print("   radial profile (r_lo, n, median P2, p10 P2):")
    for p in prof: print("     ", p)
    for lab, v in rows.items(): print(f"   {lab:16s} head PSNR {v['head_psnr']:.3f}")
    print(f"   noise floor A (k-space tail): ceiling {ceilA:.2f} dB  -> headroom over best other {ceilA - best_other:+.2f} dB")
    print(f"   noise floor B (HHH MAD):      ceiling {ceilB:.2f} dB  -> headroom over best other {ceilB - best_other:+.2f} dB")
    print(f"   target (best other + 1 dB): {best_other + 1:.3f}")
json.dump(out, open("/tmp/claude-1002/-home-congcongliu--claude/55f338b6-6256-4c6f-9006-fd26f65d0f54/scratchpad/p4/ceiling_noise.json", "w"), indent=1)
