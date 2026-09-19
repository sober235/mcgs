#!/usr/bin/env python3
"""recon_zsssdu.py -- zero-shot self-supervised deep learning (Yaman et al.) on this pair.

Transcribed from Results_20260908/CE_pre_post_contrast/scripts/ce_classical.py (method
"zsssdu"), with three changes and nothing else:
  1. data comes from the h5 exports here, retrospectively undersampled, and the GT is available;
  2. scoring is the shared GT gauge in t1t2_common, not the GT-free k-space residual;
  3. the readout is axis 0 of a (C, X, Y, Z) tensor with 26 coils instead of 42.

Network and protocol are unchanged: a 15-layer 64-channel ResNet denoiser unrolled K times
around a conjugate-gradient data-consistency block, trained ONLY on this subject. The acquired
mask is split into Gamma (10 %, held out for early stopping) and a training pool that is
re-split Theta / Lambda (80 / 20) at every step -- Theta drives data consistency, Lambda is the
loss. No ground truth is used anywhere in fitting or model selection.

The readout axis is fully sampled and the mask is constant along it, so the problem factorises
into NX independent 2-D problems in hybrid (x, ky, kz) space. That is exact, not an approximation.
"""
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1w", "T2s"])
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--unrolls", type=int, default=10)
ap.add_argument("--cg_iters_net", type=int, default=6)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--mu", type=float, default=0.05)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--dry_run", action="store_true", help="print the mask and split counts, then exit")
args = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np, torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1t2_common import Pair, R, fft2c, ifft2c, ifft1c

torch.manual_seed(0); np.random.seed(0)
dev = torch.device("cuda:0")
log = lambda *a: print(*a, flush=True)

t0 = time.time()
P = Pair(args.contrast, dev)
NC, G, m2 = P.NC, P.G, P.m2
log(f"[{args.contrast}] coils={NC} grid={G} forward-support frac={P.mask_frac:.4f} sampling AF={1/P.mask_frac_omega:.2f} scale={P.scale:.4e}")

# ---- hybrid space: the readout is fully sampled -> NX independent 2-D problems ---------------
yh = torch.empty_like(P.y)
for i in range(0, NC, P.chunk):
    yh[i:i + P.chunk] = ifft1c(P.y[i:i + P.chunk], -3)
NX = yh.shape[1]
S = P.S


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
    """x = (A^H A + mu I)^-1 b. Self-adjoint PD and parameter-free, so the backward pass is the
    same solve applied to the incoming gradient."""
    @staticmethod
    def forward(ctx, b, Sr, mk, mu, it):
        ctx.save_for_backward(Sr, mk); ctx.mu = mu; ctx.it = it
        return cg2(Sr, mk, mu, b, it)

    @staticmethod
    def backward(ctx, g):
        Sr, mk = ctx.saved_tensors
        return cg2(Sr, mk, ctx.mu, g.contiguous(), ctx.it), None, None, None, None


class ResNet(nn.Module):                       # SSDU / ZS-SSDU denoiser: 15 layers, 64 channels
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
MU, K, CGN = args.mu, args.unrolls, args.cg_iters_net


def unrolled(y_dc, Sr, mk):
    rhs0 = AH2(y_dc, Sr, mk)
    x = rhs0
    for _ in range(K):
        z = net(x)
        x = DC.apply(rhs0 + MU * z, Sr, mk, MU, CGN)
    return x


# ---- k-space splits over the SAMPLING mask Omega (spec section 5) ----------------------------
# Omega | Z is the forward operator, but Z carries measured zeros: training or validating on it
# would score a trivially correct prediction, so every split is drawn from Omega alone.
m_dc = P.m2                                    # Omega | Z: data consistency and inference
m_z = P.m2 & ~P.m2_omega                       # Z alone: the measured zeros
withz = lambda mm: mm | m_z                    # every DC operator sees Omega-subset | Z
idx = torch.nonzero(P.m2_omega, as_tuple=False)
Na = idx.shape[0]
perm = torch.randperm(Na, generator=torch.Generator(device="cpu").manual_seed(0))
n_val = int(0.10 * Na)
val_idx, tr_idx = perm[:n_val], perm[n_val:]


def mk_mask(sel):
    mm = torch.zeros_like(P.m2_omega)
    mm[idx[sel, 0], idx[sel, 1]] = True
    return mm


m_val = mk_mask(val_idx)          # Gamma: early stopping only
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


BS = args.batch
opt = torch.optim.Adam(net.parameters(), lr=args.lr)
best, best_state, bad = 1e9, None, 0
steps_per_ep = NX // BS
log(f"zs-ssdu: slices={NX} bs={BS} steps/ep={steps_per_ep} epochs={args.epochs} K={K} "
    f"acquired lines={Na} gamma={int(m_val.sum())}")
log(f"[masks] union {int(P.m2.sum())} sampling {int(P.m2_omega.sum())} known-zero {int(m_z.sum())} "
    f"gamma {int(m_val.sum())} train {int(m_tr.sum())} (gamma + train must equal the sampling mask: "
    f"{int(m_val.sum()) + int(m_tr.sum())})")
if args.dry_run:
    raise SystemExit(0)
for ep in range(args.epochs):
    net.train()
    order = torch.randperm(NX, generator=torch.Generator().manual_seed(100 + ep))
    run = 0.0
    for s in range(steps_per_ep):
        sl = order[s * BS:(s + 1) * BS]
        Sr = S[:, sl].permute(1, 0, 2, 3).contiguous()          # (B,NC,Y,Z)
        yr = yh[:, sl].permute(1, 0, 2, 3).contiguous()
        m_th, m_lam = split_theta_lambda(ep * steps_per_ep + s)
        x = unrolled(yr * withz(m_th), Sr, withz(m_th))
        loss = nloss(A2(x, Sr, m_lam), yr * m_lam, m_lam)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); run += loss.item()
    net.eval(); vs = 0.0; nv = 0
    with torch.no_grad():
        for s in range(0, NX, 16):
            sl = torch.arange(s, min(s + 16, NX))
            Sr = S[:, sl].permute(1, 0, 2, 3).contiguous()
            yr = yh[:, sl].permute(1, 0, 2, 3).contiguous()
            x = unrolled(yr * withz(m_tr), Sr, withz(m_tr))
            vs += nloss(A2(x, Sr, m_val), yr * m_val, m_val).item() * sl.numel(); nv += sl.numel()
    v = vs / nv
    log(f"  [ep {ep+1}/{args.epochs}] train={run/steps_per_ep:.4f} val={v:.4f} ({time.time()-t0:.0f}s)")
    if v < best - 1e-4:
        best, bad = v, 0
        best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
    else:
        bad += 1
        if bad >= args.patience:
            log(f"  early stop at epoch {ep+1} (best val {best:.4f})"); break
if best_state is not None:
    net.load_state_dict(best_state)

# ---- inference with the FULL acquired mask ---------------------------------------------------
net.eval()
xn = torch.zeros(G, dtype=torch.complex64, device=dev)
with torch.no_grad():
    for s in range(0, NX, 16):
        sl = torch.arange(s, min(s + 16, NX))
        Sr = S[:, sl].permute(1, 0, 2, 3).contiguous()
        yr = yh[:, sl].permute(1, 0, 2, 3).contiguous()
        xn[sl] = unrolled(yr * m_dc, Sr, m_dc)
del yh

row = P.score(xn, "zsssdu", seconds=time.time() - t0,
              val_loss=best, epochs_run=ep + 1, unrolls=K, cg_iters_net=CGN)
np.save(R / f"recon/zsssdu_{args.contrast}_complex.npy", xn.cpu().numpy().astype(np.complex64))
json.dump({"contrast": args.contrast, "rows": [row], "scale_p999": P.scale, "coils": int(NC)},
          open(R / f"logs/zsssdu_{args.contrast}.json", "w"), indent=2)
log(f"[{args.contrast}] zsssdu psnr={row['psnr']:.3f} ssim={row['ssim']:.4f} nmse={row['nmse']:.5f} "
    f"hfen={row['hfen']:.4f} head_psnr={row['head_psnr']:.3f} head_ssim={row['head_ssim']:.4f} "
    f"({row['seconds']:.0f} s)")
log("DONE")
