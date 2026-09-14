#!/usr/bin/env python3
"""recon_dip_inr.py -- true-3D DIP and INR baselines on this retrospective AF24 pair.

Transcribed from Results_20260908/CE_pre_post_contrast/scripts/ce_dip_inr.py, changed only in
where the data comes from and how the result is scored:
  1. data comes from the h5 exports here, retrospectively undersampled;
  2. the exported iterate is scored against the ground truth through t1t2_common's shared gauge.

MODEL SELECTION STILL USES NO GROUND TRUTH. As in the CE study, the acquired PE lines are split
95 / 5; the fit only ever sees the 95 %, and the 5 % held-out relative k-space residual picks the
checkpoint that gets exported. That keeps DIP and INR usable exactly as they would be on data
with no reference, and keeps the comparison with the CE numbers meaningful.

  dip : 3D ConvDecoder-style untrained decoder (Darestani & Heckel, IEEE TCI 2021, 3D-ised the
        way ML-DIP arXiv:2507.19404 does it). Deliberately UNDER-parameterised.
  inr : 3D coordinate network, hash-grid encoding -> small ReLU MLP -> (re, im), i.e. the SUMMIT
        (Lao et al., MedIA 2025) backbone with a complex weighted-image head.
"""
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=["dip", "inr"])
ap.add_argument("--contrast", required=True, choices=["T1", "FLAIR"])
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--eval_every", type=int, default=25)
ap.add_argument("--lr", type=float, default=-1.0)          # -1 -> per-method default
ap.add_argument("--holdout", type=float, default=0.05)
ap.add_argument("--tv", type=float, default=0.0)           # faithful default: architecture IS the prior
ap.add_argument("--dip_base", type=int, default=64)
ap.add_argument("--dip_levels", type=int, default=4)
ap.add_argument("--dip_in_ch", type=int, default=32)
ap.add_argument("--inr_levels", type=int, default=16)
ap.add_argument("--inr_log2T", type=int, default=19)
ap.add_argument("--inr_feat", type=int, default=2)
ap.add_argument("--inr_nmax", type=int, default=512)
ap.add_argument("--inr_hidden", type=int, default=64)
ap.add_argument("--inr_layers", type=int, default=3)
ap.add_argument("--render_chunk", type=int, default=1_500_000)
args = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1t2_common import Pair, R, fft3c

torch.manual_seed(0); np.random.seed(0)
dev = torch.device("cuda:0")
log = lambda *a: print(*a, flush=True)

P = Pair(args.contrast, dev)
NC, G, CHUNK = P.NC, P.G, P.chunk
S, yn, m3, m2 = P.S, P.y, P.m3, P.m2
zfn = P.sense_adj(yn)
log(f"[{args.contrast}] coils={NC} grid={G} AF_rect={1/m3.float().mean():.4f} scale={P.scale:.4e}")

# ---- 95 / 5 held-out split at the PE-line level ----------------------------------------------
idx = torch.nonzero(m2, as_tuple=False)
Nl = idx.shape[0]
perm = torch.randperm(Nl, generator=torch.Generator().manual_seed(0))
nho = int(round(args.holdout * Nl))


def build(sel):
    mm = torch.zeros_like(m2)
    mm[idx[sel, 0], idx[sel, 1]] = True
    return mm.unsqueeze(0).expand(G).contiguous()


m_ho = build(perm[:nho])
m_fit = build(perm[nho:])
log(f"lines: total {Nl}  fit {Nl-nho}  held-out {nho} ({100*nho/Nl:.1f} %)  "
    f"-> fit pts {int(m_fit.sum())}  ho pts {int(m_ho.sum())}")
y_fit = yn * m_fit
nrm_fit = torch.linalg.vector_norm(y_fit)
nrm_ho = torch.linalg.vector_norm(yn * m_ho)


def kres_on(x, mk, nrm):
    """relative residual of the best-fit-scaled model on mask mk (no grad)."""
    num = torch.zeros((), dtype=torch.complex64, device=dev); den = torch.zeros((), device=dev)
    for i in range(0, NC, CHUNK):
        km = fft3c(S[i:i + CHUNK] * x) * mk
        num = num + (km.conj() * yn[i:i + CHUNK] * mk).sum(); den = den + km.abs().square().sum()
    a = num / den.clamp_min(1e-30)
    acc = 0.0
    for i in range(0, NC, CHUNK):
        km = fft3c(S[i:i + CHUNK] * x) * mk
        acc += float(torch.linalg.vector_norm(a * km - yn[i:i + CHUNK] * mk).item() ** 2)
    return (acc ** 0.5) / float(nrm), a


def grad_of_fit_loss(xd):
    """dL/dx and L for L = ||M_fit .(A(x) - y)||^2 / ||M_fit . y||^2, coil-chunked."""
    gx = torch.zeros_like(xd); tot = 0.0
    for i in range(0, NC, CHUNK):
        r = (fft3c(S[i:i + CHUNK] * xd) - yn[i:i + CHUNK]) * m_fit
        l = r.abs().square().sum() / (nrm_fit ** 2)
        g, = torch.autograd.grad(l, xd)
        gx += g; tot += float(l.item())
    return gx, tot


def tv_grad(xd):
    if args.tv <= 0:
        return torch.zeros_like(xd), 0.0
    d = [xd - torch.roll(xd, 1, a) for a in range(3)]
    l = args.tv * sum(torch.sqrt(t.abs().square() + 1e-8).sum() for t in d) / xd.numel()
    g, = torch.autograd.grad(l, xd)
    return g, float(l.item())


class ConvDecoder3D(nn.Module):
    """3D ConvDecoder: fixed low-res noise -> (upsample -> Conv3d -> LeakyReLU) x L -> (re,im)."""
    def __init__(self, shape, in_ch, base, levels):
        super().__init__()
        self.shape = tuple(int(s) for s in shape); self.levels = int(levels)
        div = 2 ** self.levels
        self.low = tuple(max(2, int(np.ceil(s / div))) for s in self.shape)
        self.stages, chans, c_in, blocks = [], [], int(in_ch), []
        for i in range(self.levels):
            sd = 2 ** (self.levels - 1 - i)
            self.stages.append(self.shape if i == self.levels - 1
                               else tuple(max(2, int(np.ceil(s / sd))) for s in self.shape))
            chans.append(max(8, int(base) * 2 ** max(self.levels - 2 - i, 0)))
        for c_out in chans:
            blocks.append(nn.Sequential(nn.Conv3d(c_in, c_out, 3, padding=1),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.BatchNorm3d(c_out),
                                        nn.Conv3d(c_out, c_out, 3, padding=1),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.BatchNorm3d(c_out)))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Conv3d(c_in, 2, 1)
        nn.init.zeros_(self.final.weight); nn.init.zeros_(self.final.bias)
        self.chans = chans

    def forward(self, z):
        x = z
        for b, s in zip(self.blocks, self.stages):
            x = b(F.interpolate(x, size=s, mode="trilinear", align_corners=False))
        o = self.final(x)[0]
        return torch.complex(o[0], o[1])


_P = (1, 2654435761, 805459861)


class HashGrid3D(nn.Module):
    def __init__(self, levels, log2T, feat, nmin, nmax):
        super().__init__()
        self.L, self.T, self.F = levels, 1 << log2T, feat
        g = np.exp((np.log(nmax) - np.log(nmin)) / max(levels - 1, 1))
        self.res = [int(np.floor(nmin * g ** l)) for l in range(levels)]
        self.emb = nn.ParameterList([nn.Parameter(torch.empty(self.T, feat).uniform_(-1e-4, 1e-4))
                                     for _ in range(levels)])
        self.register_buffer("corners", torch.tensor(
            [[(i >> a) & 1 for a in range(3)] for i in range(8)], dtype=torch.long))
        self.register_buffer("primes", torch.tensor(_P, dtype=torch.long))
        self.out_dim = levels * feat

    def forward(self, c):                                        # (N,3) in [-1,1]
        u = (c * 0.5 + 0.5).clamp(0, 1); outs = []
        for l in range(self.L):
            p = u * self.res[l]
            p0 = torch.floor(p).long(); fr = p - p0.float()
            ic = p0.unsqueeze(1) + self.corners.unsqueeze(0)
            h = ((ic[..., 0] * self.primes[0]) ^ (ic[..., 1] * self.primes[1])
                 ^ (ic[..., 2] * self.primes[2]))
            f = self.emb[l][(h & 0x7FFFFFFFFFFFFFFF) % self.T]
            w = torch.ones(c.shape[0], 8, device=c.device)
            for a in range(3):
                ca = self.corners[:, a].float()
                w = w * (ca * fr[:, a:a + 1] + (1 - ca) * (1 - fr[:, a:a + 1]))
            outs.append((f * w.unsqueeze(-1)).sum(1))
        return torch.cat(outs, -1)


class HashINR(nn.Module):
    def __init__(self, levels, log2T, feat, nmax, hidden, layers):
        super().__init__()
        self.enc = HashGrid3D(levels, log2T, feat, 16, nmax)
        d = [self.enc.out_dim] + [hidden] * layers; net = []
        for i in range(len(d) - 1):
            net += [nn.Linear(d[i], d[i + 1]), nn.ReLU(inplace=True)]
        self.mlp = nn.Sequential(*net, nn.Linear(hidden, 2))

    def forward(self, c):
        o = self.mlp(self.enc(c))
        return torch.complex(o[:, 0], o[:, 1])


t0 = time.time()
out_scale = zfn.abs().max().item()
if args.method == "dip":
    net = ConvDecoder3D(G, args.dip_in_ch, args.dip_base, args.dip_levels).to(dev)
    z = (torch.randn((1, args.dip_in_ch, *net.low), device=dev) * 0.1)
    lr = args.lr if args.lr > 0 else 1e-3
    npar = sum(p.numel() for p in net.parameters())
    log(f"[dip] chans={net.chans} low={net.low} params={npar/1e6:.2f}M "
        f"vs outputs={2*int(np.prod(G))/1e6:.2f}M -> ratio {npar/(2*np.prod(G)):.3f}")
    render = lambda: net(z) * out_scale
else:
    net = HashINR(args.inr_levels, args.inr_log2T, args.inr_feat,
                  args.inr_nmax, args.inr_hidden, args.inr_layers).to(dev)
    lr = args.lr if args.lr > 0 else 5e-3
    npar = sum(p.numel() for p in net.parameters())
    log(f"[inr] hash L={args.inr_levels} T=2^{args.inr_log2T} F={args.inr_feat} "
        f"res 16..{args.inr_nmax} MLP {args.inr_hidden}x{args.inr_layers} params={npar/1e6:.2f}M")
    ax = [torch.linspace(-1, 1, s, device=dev) for s in G]
    COORD = torch.stack(torch.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)

    def render_nograd():
        with torch.no_grad():
            return torch.cat([net(COORD[i:i + args.render_chunk])
                              for i in range(0, COORD.shape[0], args.render_chunk)]).reshape(G) * out_scale

opt = torch.optim.Adam(net.parameters(), lr=lr)
best = dict(ho=1e9, it=-1, vol=None)
log(f"[{args.method}] lr={lr} iters={args.iters} tv={args.tv}")
for it in range(1, args.iters + 1):
    opt.zero_grad(set_to_none=True)
    if args.method == "dip":
        x = render()
        xd = x.detach().requires_grad_(True)
        gx, l = grad_of_fit_loss(xd); gt, lt = tv_grad(xd)
        x.backward(gx + gt)
    else:
        x = render_nograd()
        xd = x.detach().requires_grad_(True)
        gx, l = grad_of_fit_loss(xd); gt, lt = tv_grad(xd)
        gflat = ((gx + gt) * out_scale).reshape(-1)
        for i in range(0, COORD.shape[0], args.render_chunk):
            xc = net(COORD[i:i + args.render_chunk])
            xc.backward(gradient=gflat[i:i + args.render_chunk])
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    if it % args.eval_every == 0 or it == 1:
        with torch.no_grad():
            xe = render() if args.method == "dip" else render_nograd()
        kho, _ = kres_on(xe, m_ho, nrm_ho)
        kfit, _ = kres_on(xe, m_fit, nrm_fit)
        star = ""
        if kho < best["ho"] - 1e-5:
            best = dict(ho=kho, it=it, vol=xe.detach().clone()); star = "  *"
        log(f"[{args.method}/{args.contrast} it {it:5d}/{args.iters}] fit={l:.4e} tv={lt:.2e} "
            f"kres_fit={kfit:.5f} kres_ho={kho:.5f}{star}  ({time.time()-t0:.0f}s)")

log(f"[{args.method}/{args.contrast}] best held-out kres {best['ho']:.5f} @ iter {best['it']}")
xn = best["vol"]

row = P.score(xn, args.method, seconds=time.time() - t0,
              kres_heldout=best["ho"], best_iter=best["it"], params=int(npar),
              holdout_lines=int(nho), total_lines=int(Nl), iters=args.iters, lr=lr)
np.save(R / f"recon/{args.method}_{args.contrast}_complex.npy", xn.cpu().numpy().astype(np.complex64))
json.dump({"contrast": args.contrast, "rows": [row], "scale_p999": P.scale, "coils": int(NC)},
          open(R / f"logs/{args.method}_{args.contrast}.json", "w"), indent=2)
log(f"[{args.contrast}] {args.method} psnr={row['psnr']:.3f} ssim={row['ssim']:.4f} "
    f"nmse={row['nmse']:.5f} hfen={row['hfen']:.4f} head_psnr={row['head_psnr']:.3f} "
    f"head_ssim={row['head_ssim']:.4f} ({row['seconds']:.0f} s)")
log("DONE")
