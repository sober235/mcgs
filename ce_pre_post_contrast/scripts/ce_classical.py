#!/usr/bin/env python3
"""ce_classical.py -- classical / self-supervised baselines on the UIH prospective CE pair.

  zf      : zero-filled SENSE combine            x = sum_c conj(S_c) . ifft3c(y_c)
  cgsense : CG on the Tikhonov normal equations  (tik=0.01, 60 iters)   [June contract]
  cssense : ADMM isotropic-3D-TV SENSE           (lam=1e-3, mu=0.1, 35 outer x 8 CG)
  zsssdu  : zero-shot self-supervised DL (Yaman et al.) -- unrolled MoDL-style network
            trained on THIS subject only, k-space split Theta(DC)/Lambda(loss)/Gamma(val).

The readout axis (256) is fully sampled and the mask is constant along it, so zsssdu runs
as a 2D network in hybrid (x, ky, kz) space -- exact, not an approximation.

Every method saves the RAW (pre-hard-DC) complex volume, rescaled by the best-fit complex
alpha = argmin_a ||M(a.A(x) - y)||^2, so all methods land on the measured-data scale and the
error-vs-y maps are directly comparable.
"""
import os, sys, json, time, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=["zf", "cgsense", "cssense", "zsssdu"])
ap.add_argument("--scan", required=True, choices=["pre", "post"])
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=60)         # zsssdu
ap.add_argument("--unrolls", type=int, default=10)        # zsssdu
ap.add_argument("--cg_iters_net", type=int, default=6)    # zsssdu inner DC solver
ap.add_argument("--fit", action="store_true",
                help="use only the 95%% lines of heldout_split/split_95_5_seed0.npz (added 2026-09-14; tag gets _fit)")
args = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.fft as tfft

torch.manual_seed(0); np.random.seed(0)
dev = torch.device("cuda:0")
RR = Path("/data0/congcong/code/NV-Generate-CTMR/recon_real")
# 2026-09-16: CE_WORK_PRE / CE_WORK_POST override the source, so the retrospective simulation
# (Results_20260908/CE_retro_gt) runs through exactly this code path.
WORK = {"pre": Path(os.environ.get("CE_WORK_PRE", RR / "work_ce_t1_16887049370010")),
        "post": Path(os.environ.get("CE_WORK_POST", RR / "work_ce_t1_16887049370060"))}[args.scan]
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)

# ---------------- centered orthonormal FFTs ----------------
def _c(fn, x, dims):
    return tfft.fftshift(fn(tfft.ifftshift(x, dim=dims), dim=dims, norm="ortho"), dim=dims)
def fft3c(x):  return _c(tfft.fftn,  x, (-3, -2, -1))
def ifft3c(x): return _c(tfft.ifftn, x, (-3, -2, -1))
def fft2c(x):  return _c(tfft.fftn,  x, (-2, -1))
def ifft2c(x): return _c(tfft.ifftn, x, (-2, -1))
def ifft1c(x, d): return _c(tfft.ifftn, x, (d,))

# ---------------- data ----------------
log(f"loading {WORK} ...")
y_np = np.load(WORK / "ksp_full.npy")            # (42,256,232,176) complex64, zeros off-mask
S_np = np.load(WORK / "csm.npy")                 # (42,256,232,176) complex64
M_np = np.load(WORK / "mask2d.npy")              # (256,232,176) float32, constant along axis 0
y = torch.from_numpy(y_np).to(dev); del y_np
S = torch.from_numpy(S_np).to(dev); del S_np
m3 = torch.from_numpy(M_np).to(dev) > 0.5        # (256,232,176) bool
del M_np
if args.fit:                                      # held-out lines never enter the fit (split frozen by ce_split.py)
    _fit = torch.from_numpy(np.load("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/"
                                    "heldout_split/split_95_5_seed0.npz")["fit"]).to(dev)
    m3 = m3 & _fit
    log(f"--fit: restricted to {int(_fit.sum())} of the acquired lines")
NC = y.shape[0]
assert bool((m3 == m3[0:1]).all()), "readout axis is not uniformly sampled"
m2 = m3[0]                                        # (232,176)
y = y * m3                                        # enforce exact zeros off-mask
log(f"coils={NC} grid={tuple(m3.shape)} AF={m3.numel()/m3.sum().item():.4f}")

CHUNK = 6
def AH_k(k):                                      # (NC,X,Y,Z) k-space -> (X,Y,Z) image
    acc = torch.zeros(m3.shape, dtype=torch.complex64, device=dev)
    for i in range(0, NC, CHUNK):
        acc += (S[i:i+CHUNK].conj() * ifft3c(k[i:i+CHUNK] * m3)).sum(0)
    return acc
def AHA(x):                                       # normal operator, coil-chunked
    acc = torch.zeros_like(x)
    for i in range(0, NC, CHUNK):
        kk = fft3c(S[i:i+CHUNK] * x) * m3
        acc += (S[i:i+CHUNK].conj() * ifft3c(kk)).sum(0)
    return acc

zf = AH_k(y)
scale = torch.quantile(zf.abs().flatten().float()[::7], 0.999).item()
log(f"scale (P99.9 of |ZF|) = {scale:.4e}")
yn = y / scale                                    # work in normalised units (lam/mu match June)

# ---------------- solvers ----------------
def cg(applyA, b, x0=None, iters=60, tol=1e-10):
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - applyA(x); p = r.clone()
    rs = torch.vdot(r.flatten(), r.flatten()).real
    for _ in range(iters):
        Ap = applyA(p)
        a = rs / torch.vdot(p.flatten(), Ap.flatten()).real.clamp_min(1e-30)
        x = x + a * p; r = r - a * Ap
        rs_new = torch.vdot(r.flatten(), r.flatten()).real
        if rs_new.sqrt() < tol: break
        p = r + (rs_new / rs.clamp_min(1e-30)) * p; rs = rs_new
    return x

def grad3(x):
    return torch.stack([x - torch.roll(x, 1, 0), x - torch.roll(x, 1, 1), x - torch.roll(x, 1, 2)])
def div3(g):
    return sum(g[d] - torch.roll(g[d], -1, d) for d in range(3))

# ---------------- methods ----------------
t0 = time.time()
if args.method == "zf":
    xn = AH_k(yn)

elif args.method == "cgsense":
    TIK = 0.01
    xn = cg(lambda v: AHA(v) + TIK * v, AH_k(yn), iters=60)

elif args.method == "cssense":
    LAM, MU, TIK, OUTER, CGIT = 1e-3, 0.1, 0.01, 35, 8
    rhs0 = AH_k(yn)
    xn = cg(lambda v: AHA(v) + TIK * v, rhs0, iters=CGIT)
    z = grad3(xn); u = torch.zeros_like(z)
    for it in range(OUTER):
        rhs = rhs0 + MU * div3(z - u)
        xn = cg(lambda v: AHA(v) + TIK * v + MU * div3(grad3(v)), rhs, x0=xn, iters=CGIT)
        gx = grad3(xn) + u
        mag = gx.abs().square().sum(0).sqrt()                     # isotropic
        z = gx * (1.0 - (LAM / MU) / mag.clamp_min(1e-12)).clamp_min(0.0)
        u = gx - z
        if (it + 1) % 7 == 0:
            log(f"  [admm {it+1}/{OUTER}] |x|max={xn.abs().max().item():.4f}")

elif args.method == "zsssdu":
    # ---- hybrid space: readout is fully sampled -> 256 independent 2D problems ----
    yh = torch.empty_like(yn)
    for i in range(0, NC, CHUNK):
        yh[i:i+CHUNK] = ifft1c(yn[i:i+CHUNK], -3)
    NX = yh.shape[1]

    def A2(x, Sr, mk):        # x (B,Y,Z) -> (B,NC,Y,Z)
        return fft2c(Sr * x.unsqueeze(1)) * mk
    def AH2(k, Sr, mk):
        return (Sr.conj() * ifft2c(k * mk)).sum(1)
    def AHA2(x, Sr, mk):
        return AH2(A2(x, Sr, mk), Sr, mk)

    def cg2(Sr, mk, mu, b, iters):
        op = lambda v: AHA2(v, Sr, mk) + mu * v
        x = torch.zeros_like(b); r = b.clone(); p = r.clone()
        rs = (r.conj() * r).real.sum((-2, -1), keepdim=True)
        for _ in range(iters):
            Ap = op(p)
            a = rs / (p.conj() * Ap).real.sum((-2, -1), keepdim=True).clamp_min(1e-30)
            x = x + a * p; r = r - a * Ap
            rs2 = (r.conj() * r).real.sum((-2, -1), keepdim=True)
            p = r + (rs2 / rs.clamp_min(1e-30)) * p; rs = rs2
        return x

    class DC(torch.autograd.Function):
        """x = (A^H A + mu I)^-1 b.  Operator is self-adjoint PD and parameter-free,
        so the backward pass is the same solve applied to the incoming gradient."""
        @staticmethod
        def forward(ctx, b, Sr, mk, mu, it):
            ctx.save_for_backward(Sr, mk); ctx.mu = mu; ctx.it = it
            return cg2(Sr, mk, mu, b, it)
        @staticmethod
        def backward(ctx, g):
            Sr, mk = ctx.saved_tensors
            return cg2(Sr, mk, ctx.mu, g.contiguous(), ctx.it), None, None, None, None

    class ResNet(nn.Module):                       # SSDU/ZS-SSDU denoiser: 15 layers, 64 ch
        def __init__(self, ch=64, nb=5):
            super().__init__()
            self.inp = nn.Conv2d(2, ch, 3, padding=1)
            self.blk = nn.ModuleList([nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1)) for _ in range(nb)])
            self.out = nn.Conv2d(ch, 2, 3, padding=1)
        def forward(self, x):                       # x complex (B,Y,Z)
            h = torch.stack([x.real, x.imag], 1)
            h0 = self.inp(h)
            h = h0
            for b in self.blk:
                h = h + 0.1 * b(h)
            o = self.out(h + h0)
            return torch.complex(o[:, 0], o[:, 1])

    net = ResNet().to(dev)
    MU = 0.05
    K, CGN = args.unrolls, args.cg_iters_net

    def unrolled(y_dc, Sr, mk):
        rhs0 = AH2(y_dc, Sr, mk)
        x = rhs0
        for _ in range(K):
            z = net(x)
            x = DC.apply(rhs0 + MU * z, Sr, mk, MU, CGN)
        return x

    # ---- k-space splits over the 2D acquired mask ----
    idx = torch.nonzero(m2, as_tuple=False)                       # (Nacq,2)
    Na = idx.shape[0]
    g = torch.Generator(device="cpu").manual_seed(0)
    perm = torch.randperm(Na, generator=g)
    n_val = int(0.10 * Na)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    def mk_mask(sel):
        mm = torch.zeros_like(m2)
        mm[idx[sel, 0], idx[sel, 1]] = True
        return mm
    m_val = mk_mask(val_idx)                                      # Gamma: early stopping
    m_tr = mk_mask(tr_idx)
    tr_pool = tr_idx.clone()

    def split_theta_lambda(step):
        gg = torch.Generator(device="cpu").manual_seed(1234 + step)
        p = torch.randperm(tr_pool.numel(), generator=gg)
        cut = int(0.80 * p.numel())
        return mk_mask(tr_pool[p[:cut]]), mk_mask(tr_pool[p[cut:]])

    def nloss(pred_k, targ_k, mk):
        d = (pred_k - targ_k) * mk
        t = targ_k * mk
        l2 = torch.linalg.vector_norm(d) / torch.linalg.vector_norm(t).clamp_min(1e-12)
        l1 = d.abs().sum() / t.abs().sum().clamp_min(1e-12)
        return l1 + l2

    BS = 4
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    best, best_state, patience, bad = 1e9, None, 10, 0
    steps_per_ep = NX // BS
    log(f"zs-ssdu: slices={NX} bs={BS} steps/ep={steps_per_ep} epochs={args.epochs} K={K}")
    for ep in range(args.epochs):
        net.train()
        order = torch.randperm(NX, generator=torch.Generator().manual_seed(100 + ep))
        run = 0.0
        for s in range(steps_per_ep):
            sl = order[s*BS:(s+1)*BS]
            Sr = S[:, sl].permute(1, 0, 2, 3).contiguous()          # (B,NC,Y,Z)
            yr = yh[:, sl].permute(1, 0, 2, 3).contiguous()
            m_th, m_lam = split_theta_lambda(ep * steps_per_ep + s)
            x = unrolled(yr * m_th, Sr, m_th)
            loss = nloss(A2(x, Sr, m_lam), yr * m_lam, m_lam)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); run += loss.item()
        # ---- validation on Gamma (never seen by DC or loss) ----
        net.eval(); vs = 0.0; nv = 0
        with torch.no_grad():
            for s in range(0, NX, 16):
                sl = torch.arange(s, min(s+16, NX))
                Sr = S[:, sl].permute(1, 0, 2, 3).contiguous()
                yr = yh[:, sl].permute(1, 0, 2, 3).contiguous()
                x = unrolled(yr * m_tr, Sr, m_tr)
                vs += nloss(A2(x, Sr, m_val), yr * m_val, m_val).item() * sl.numel(); nv += sl.numel()
        v = vs / nv
        log(f"  [ep {ep+1}/{args.epochs}] train={run/steps_per_ep:.4f} val={v:.4f} "
            f"({time.time()-t0:.0f}s)")
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"  early stop at epoch {ep+1} (best val {best:.4f})"); break
    if best_state is not None: net.load_state_dict(best_state)

    # ---- inference with the FULL acquired mask ----
    net.eval()
    xn = torch.zeros(m3.shape, dtype=torch.complex64, device=dev)
    with torch.no_grad():
        for s in range(0, NX, 16):
            sl = torch.arange(s, min(s+16, NX))
            Sr = S[:, sl].permute(1, 0, 2, 3).contiguous()
            yr = yh[:, sl].permute(1, 0, 2, 3).contiguous()
            xn[sl] = unrolled(yr * m2, Sr, m2)
    del yh

# ---------------- best-fit scale + k-space residual vs the measured y ----------------
x = xn * scale
num = torch.zeros((), dtype=torch.complex64, device=dev)
den = torch.zeros((), device=dev)
for i in range(0, NC, CHUNK):
    km = fft3c(S[i:i+CHUNK] * x) * m3
    num = num + (km.conj() * y[i:i+CHUNK]).sum()
    den = den + km.abs().square().sum()
alpha = (num / den.clamp_min(1e-30)).item()
num_r = 0.0; den_r = 0.0
for i in range(0, NC, CHUNK):
    km = fft3c(S[i:i+CHUNK] * x) * m3
    num_r += float(torch.linalg.vector_norm(alpha * km - y[i:i+CHUNK]).item() ** 2)
    den_r += float(torch.linalg.vector_norm(y[i:i+CHUNK]).item() ** 2)
kres = (num_r / den_r) ** 0.5

xa = (alpha * x).cpu().numpy().astype(np.complex64)
tag = f"{args.method}_{args.scan}" + ("_fit" if args.fit else "")
np.save(OUT / f"{tag}_complex.npy", xa)
json.dump({"method": args.method, "scan": args.scan, "shape": list(xa.shape),
           "scale": scale, "alpha_re": float(np.real(alpha)), "alpha_im": float(np.imag(alpha)),
           "kres": kres, "seconds": time.time() - t0},
          open(OUT / f"{tag}_meta.json", "w"), indent=2)
log(f"[{tag}] kres={kres:.5f} |alpha|={abs(alpha):.4f} in {time.time()-t0:.0f}s -> {OUT/(tag+'_complex.npy')}")
log("DONE")
