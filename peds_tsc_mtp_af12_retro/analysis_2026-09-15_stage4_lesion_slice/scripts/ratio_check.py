"""Feasibility of a signal-ratio coupling, fast version (torch CPU FFTs): T2s = T1w * rho, rho smooth.
rho is restricted to the k-space band |k_i| <= frac * N_i / 2 on every axis and fitted by CG on the
ground truth (weighted by the head), then T1w*rho is scored against T2s in the primary head region on
magnitude, as the table does; likewise the reverse direction."""
import sys, json
import numpy as np, torch
torch.set_num_threads(16)
sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt")
from diag_artefact.headonly import head_mask
from uih_mtp.rescore import partition_region
T = "/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro/"
a = np.load(T + "recon/gt_T1w_complex.npy").astype(np.complex128)
b = np.load(T + "recon/gt_T2s_complex.npy").astype(np.complex128)
shape = a.shape
reg = partition_region(shape, 2, 12, 92)
ha = head_mask(np.abs(a).astype(np.float32)); hb = head_mask(np.abs(b).astype(np.float32))

def fit(src, dst, head, frac, iters=30):
    km = torch.ones(shape, dtype=torch.bool)
    for ax, n in enumerate(shape):
        s_ = [1, 1, 1]; s_[ax] = n
        km &= (torch.fft.fftfreq(n).abs() <= frac / 2).reshape(s_)
    km = km.to(torch.complex128)
    S, D = torch.from_numpy(src), torch.from_numpy(dst)
    w = torch.from_numpy(head.astype(np.float64))
    P = lambda x: torch.fft.ifftn(torch.fft.fftn(x) * km)
    SW = torch.conj(S) * w
    A = lambda r: P(SW * (S * r))
    rhs = P(SW * D)
    x = torch.zeros(shape, dtype=torch.complex128); r = rhs.clone(); p = r.clone()
    rs = torch.vdot(r.flatten(), r.flatten()).real; r0 = rs
    for _ in range(iters):
        Ap = A(p); al = rs / torch.vdot(p.flatten(), Ap.flatten()).real
        x += al * p; r -= al * Ap
        rn = torch.vdot(r.flatten(), r.flatten()).real
        if rn < 1e-10 * r0: break
        p = r + (rn / rs) * p; rs = rn
    return x.numpy()

def hpsnr(rec, gt, head):
    g = np.abs(gt); h = head & reg
    return float(20 * np.log10(g.max()) - 10 * np.log10(np.mean((np.abs(rec)[h] - g[h]) ** 2)))

out = {}
# (1) convergence: same complex fit at cutoff 1/4 with 120 CG iterations
pb = hpsnr(a * fit(a, b, hb, 0.25, iters=120), b, hb)
print(f"complex, cutoff 0.25, 120 it: T2s ~ T1w*rho head PSNR {pb:.2f} dB", flush=True)
# (2) magnitude-only ratio: |T2s| ~ |T1w| * r, r real and smooth
am, bm = np.abs(a).astype(np.complex128), np.abs(b).astype(np.complex128)
for frac in (0.125, 0.25):
    r = fit(am, bm, hb, frac, iters=120)
    pm = hpsnr(am * r.real, b, hb)
    print(f"magnitude, cutoff {frac}: |T2s| ~ |T1w|*r head PSNR {pm:.2f} dB", flush=True)
    out[f"mag_{frac}"] = pm
# (3) reference points: a global scalar, and the current reconstructions
c = np.vdot(am[hb & reg], bm[hb & reg]) / np.vdot(am[hb & reg], am[hb & reg])
print(f"global scalar |T2s| ~ c|T1w|: {hpsnr(am * c.real, b, hb):.2f} dB", flush=True)
out["complex_0.25_120it"] = pb
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
