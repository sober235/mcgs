#!/usr/bin/env python3
"""mcgs_zsref.py -- joint MC-GS + zero-shot self-supervised unrolled refinement (CE AF16, 2026-11-06).

Why: the fixed-view diagnosis (README.md here) shows MC-GS FV37 already beats ZS-SSDU at low
frequency and on the acquired k-space samples, but loses at mid/high frequency (it keeps ~60 % of
the GT high-frequency energy, ZS-SSDU ~92 %). Neither Gaussian count (densify study, 1.2-4.0 M
points: +/-0.02 dB) nor longer training nor regulariser weights move that. The missing piece is a
learned prior for the unacquired high frequencies.

What: the ZS-SSDU protocol of the registered opponent arm (recon_zsssdu.py, zsssdu_z2_b15_cg10_e400:
15-block 64-ch ResNet, K=10 unrolls, CG 10, mu .05, lr 5e-4, patience 40, Gamma 10 %, Theta 80 %),
with three changes, and nothing else:
  1. JOINT: one network sees pre and post together (4 channels) and outputs both; each scan keeps
     its own coils/data in its own data-consistency block (same acquired mask);
  2. MC-GS PRIOR: the unroll starts from the MC-GS joint reconstruction instead of A^H y, and the
     MC-GS volumes are fed to the network as 4 extra input channels at every unroll;
  3. both scans are trained in one loop (loss = sum of the two normalised k-space losses).
Fitting and model selection use ONLY acquired k-space (Theta/Lambda/Gamma splits). No GT, no ROI.
The single GT-derived quantity is the per-scan P99.9 gauge scalar (sim/gt.npz scale_<scan>), used
exactly as the opponent arms use it, so outputs land on the frozen evaluator's scale.

Readout (axis 0) is fully sampled and the mask is constant along it, so the problem factorises into
NX independent 2-D (ky,kz) problems in hybrid space -- exact, as in the opponent.

Outputs candidate_pre_complex.npy / candidate_post_complex.npy (score with the frozen
fixed_views_score.py --candidate DIR) and fit_log.json.
Smoke mode (--smoke): CPU-sized subset of readout slices, a few steps, no volume written.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--init-dir", required=True, type=Path, help="dir with MC-GS candidate_{pre,post}_complex.npy")
ap.add_argument("--topic", type=Path, default=Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois"))
ap.add_argument("--out", type=Path, default=None, help="output dir (default $MCGS_ARTIFACT_DIR)")
ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
ap.add_argument("--gpu", type=int, default=None, help="explicit physical GPU id (required with --device cuda)")
ap.add_argument("--epochs", type=int, default=400)
ap.add_argument("--patience", type=int, default=40)
ap.add_argument("--unrolls", type=int, default=10)
ap.add_argument("--cg", type=int, default=10)
ap.add_argument("--blocks", type=int, default=15)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--mu", type=float, default=0.05)
ap.add_argument("--rho-val", type=float, default=0.10)
ap.add_argument("--theta-frac", type=float, default=0.80)
ap.add_argument("--no-prior-input", action="store_true", help="ablation: start from MC-GS but do not feed it as channels")
ap.add_argument("--separate", action="store_true", help="ablation: two independent 2-ch networks (not joint)")
ap.add_argument("--no-residual", action="store_true", help="ablation: opponent-style direct output instead of identity-initialised residual")
ap.add_argument("--split-file", type=Path, default=None,
                help="leak-free mode: npz with fixed 2-D theta/lam/gamma masks (theta_prior.py split). The --init-dir "
                     "prior must have been fitted on theta only; DC uses theta, loss lam, early stop gamma")
ap.add_argument("--lam-dc-frac", type=float, default=None,
                help="with --split-file: every step move a fresh random fraction of Lambda into DC "
                     "(DC = theta + that part, loss = rest of Lambda); the prior never saw Lambda, so it stays "
                     "leak-free while the loss set changes every step like the opponent's")
ap.add_argument("--infer-init-dir", type=Path, default=None,
                help="optional second prior for inference only (e.g. the full-data MC-GS); written to OUT/infer_alt/")
ap.add_argument("--smoke", action="store_true", help="CPU smoke: few slices, 2 epochs x 2 steps, no volumes")
ap.add_argument("--smoke-slices", default="197,199,201,203,205,207,209,211")
args = ap.parse_args()

if args.device == "cuda":
    if args.gpu is None:
        sys.exit("--device cuda needs an explicit --gpu ID (no internal GPU selection)")
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited not in (None, str(args.gpu)):
        sys.exit(f"launcher exposed CUDA_VISIBLE_DEVICES={inherited} but --gpu {args.gpu}; refusing")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # physical id = nvidia-smi index
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.fft as tfft  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.manual_seed(0); np.random.seed(0)
if args.device == "cpu":
    torch.set_num_threads(int(os.environ.get("ZSREF_THREADS", "8")))
dev = torch.device("cuda:0" if args.device == "cuda" else "cpu")
OUT = args.out or Path(os.environ.get("MCGS_ARTIFACT_DIR", ""))
if args.device == "cuda":
    pr = torch.cuda.get_device_properties(0)
    print(f"device: physical {args.gpu} {pr.name} uuid=GPU-{pr.uuid} {pr.total_memory/2**30:.0f} GiB", flush=True)
assert str(OUT), "give --out or run under gpu.py (MCGS_ARTIFACT_DIR)"
OUT.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)
t0 = time.time()
SCANS = ("pre", "post")


def _c(fn, x, dims):
    return tfft.fftshift(fn(tfft.ifftshift(x, dim=dims), dim=dims, norm="ortho"), dim=dims)


def fft2c(x): return _c(tfft.fftn, x, (-2, -1))
def ifft2c(x): return _c(tfft.ifftn, x, (-2, -1))
def ifft1c(x, d): return _c(tfft.ifftn, x, (d,))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------- data (same files and gauge as the opponent arms) ----------------
T = args.topic
with h5py.File(T / "sim/mask/mask_af16_3d.h5", "r") as f:
    m3 = np.asarray(f["mask"][:]) > 0
assert (m3 == m3[0:1]).all(), "mask not constant along readout"
m2 = torch.from_numpy(m3[0]).to(dev)
NX = m3.shape[0]
sl_all = [int(s) for s in args.smoke_slices.split(",")] if args.smoke else list(range(NX))
with np.load(T / "sim/gt.npz") as g:          # gauge scalars only; GT volumes are never read
    scale = {s: float(g[f"scale_{s}"]) for s in SCANS}

def load_prior(d):
    out = {}
    for s in SCANS:
        v = np.load(d / f"candidate_{s}_complex.npy", mmap_mode="r")
        assert v.shape == m3.shape and np.iscomplexobj(v)
        out[s] = torch.from_numpy(np.ascontiguousarray(v[sl_all])).to(torch.complex64).to(dev)
    return out


YH, S, X0 = {}, {}, {}
for s in SCANS:
    with h5py.File(T / f"sim/gs/{s}_kspace.h5", "r") as f:
        y = torch.from_numpy(f["kspace"][:]).to(torch.complex64)
    y = y * torch.from_numpy(m3).unsqueeze(0)           # retrospectively acquired data only
    y = ifft1c(y, -3)[:, sl_all] / scale[s]              # hybrid (C, x, ky, kz), gauge
    YH[s] = y.to(dev)
    with h5py.File(T / f"sim/gs/{s}_csm.h5", "r") as f:
        S[s] = torch.from_numpy(f["map"][:, sl_all] if args.smoke else f["map"][:]).to(torch.complex64).to(dev)
    del y
X0.update(load_prior(args.init_dir))
NC = YH["pre"].shape[0]
NS = len(sl_all)
log(f"coils={NC} slices={NS} grid={m3.shape} acquired={m2.float().mean():.4f} scale={scale} "
    f"load {time.time()-t0:.0f}s")


# ---------------- operators (verbatim from the opponent) ----------------
def A2(x, Sr, mk): return fft2c(Sr * x.unsqueeze(1)) * mk
def AH2(k, Sr, mk): return (Sr.conj() * ifft2c(k * mk)).sum(1)


def cg2(Sr, mk, mu, b, iters):
    op = lambda v: AH2(A2(v, Sr, mk), Sr, mk) + mu * v
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
    @staticmethod
    def forward(ctx, b, Sr, mk, mu, it):
        ctx.save_for_backward(Sr, mk); ctx.mu = mu; ctx.it = it
        return cg2(Sr, mk, mu, b, it)

    @staticmethod
    def backward(ctx, g):
        Sr, mk = ctx.saved_tensors
        return cg2(Sr, mk, ctx.mu, g.contiguous(), ctx.it), None, None, None, None


RESID = not args.no_residual


class ResNet(nn.Module):              # opponent denoiser, input/output channel counts generalised
    def __init__(self, cin, cout, ch=64, nb=15):
        super().__init__()
        self.inp = nn.Conv2d(cin, ch, 3, padding=1)
        self.blk = nn.ModuleList([nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
                                                nn.Conv2d(ch, ch, 3, padding=1)) for _ in range(nb)])
        self.out = nn.Conv2d(ch, cout, 3, padding=1)
        if RESID:                     # residual mode: start as the identity, so epoch 0 = MC-GS + DC
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, h):
        h0 = self.inp(h); h = h0
        for b in self.blk:
            h = h + 0.1 * b(h)
        return self.out(h + h0)


ri = lambda z: [z.real, z.imag]
PRIOR = not args.no_prior_input
if args.separate:
    nets = {s: ResNet(2 + 2 * PRIOR, 2, nb=args.blocks).to(dev) for s in SCANS}
    params = [p for n in nets.values() for p in n.parameters()]
else:
    net = ResNet(4 + 4 * PRIOR, 4, nb=args.blocks).to(dev)
    params = list(net.parameters())
log(f"{'separate' if args.separate else 'joint'} residual={RESID} prior_input={PRIOR} blocks={args.blocks} "
    f"params={sum(p.numel() for p in params)/1e6:.3f}M K={args.unrolls} CG={args.cg} mu={args.mu}")


def denoise(x, x0):
    """x, x0: dict scan -> (B,Y,Z) complex. Returns dict scan -> complex."""
    if args.separate:
        out = {}
        for s in SCANS:
            o = nets[s](torch.stack(ri(x[s]) + (ri(x0[s]) if PRIOR else []), 1))
            out[s] = torch.complex(o[:, 0], o[:, 1]) + (x[s] if RESID else 0)
        return out
    h = torch.stack(ri(x["pre"]) + ri(x["post"]) + ((ri(x0["pre"]) + ri(x0["post"])) if PRIOR else []), 1)
    o = net(h)
    out = {"pre": torch.complex(o[:, 0], o[:, 1]), "post": torch.complex(o[:, 2], o[:, 3])}
    return {s: out[s] + x[s] for s in SCANS} if RESID else out


def unrolled(sl, mk):
    Sr = {s: S[s][:, sl].permute(1, 0, 2, 3).contiguous() for s in SCANS}
    yr = {s: YH[s][:, sl].permute(1, 0, 2, 3).contiguous() for s in SCANS}
    x0 = {s: X0[s][sl] for s in SCANS}
    rhs0 = {s: AH2(yr[s] * mk, Sr[s], mk) for s in SCANS}
    x = dict(x0)                                       # start from MC-GS, not A^H y
    for _ in range(args.unrolls):
        z = denoise(x, x0)
        x = {s: DC.apply(rhs0[s] + args.mu * z[s], Sr[s], mk, args.mu, args.cg) for s in SCANS}
    return x, Sr, yr


def nloss(pred_k, targ_k, mk):
    d = (pred_k - targ_k) * mk; t = targ_k * mk
    return (torch.linalg.vector_norm(d) / torch.linalg.vector_norm(t).clamp_min(1e-12)
            + d.abs().sum() / t.abs().sum().clamp_min(1e-12))


def lossk(x, Sr, yr, mk):
    return sum(nloss(A2(x[s], Sr[s], mk), yr[s] * mk, mk) for s in SCANS)


# ---------------- k-space splits (opponent protocol) ----------------
idx = torch.nonzero(m2, as_tuple=False); Na = idx.shape[0]
perm = torch.randperm(Na, generator=torch.Generator(device="cpu").manual_seed(0))
n_val = int(args.rho_val * Na); val_idx, tr_pool = perm[:n_val], perm[n_val:]


def mk_mask(sel):
    mm = torch.zeros_like(m2); mm[idx[sel, 0], idx[sel, 1]] = True; return mm


m_val, m_tr = mk_mask(val_idx), mk_mask(tr_pool)
FIXED = None
if args.split_file is not None:          # leak-free mode: one fixed split shared with the prior fit
    with np.load(args.split_file) as z:
        FIXED = {k: torch.from_numpy(z[k]).to(dev) for k in ("theta", "lam", "gamma")}
    th, la, ga = FIXED["theta"], FIXED["lam"], FIXED["gamma"]
    assert not (th & la).any() and not (th & ga).any() and not (la & ga).any(), "split overlaps"
    assert torch.equal(th | la | ga, m2), "split is not a partition of the acquired mask"
    m_val, m_tr = ga, th | la
    log(f"fixed split {args.split_file}: theta {int(th.sum())} lam {int(la.sum())} gamma {int(ga.sum())} "
        f"(--rho-val/--theta-frac ignored)")
    LAM_IDX = torch.nonzero(la, as_tuple=False)
    if args.lam_dc_frac is not None:
        assert 0.0 < args.lam_dc_frac < 1.0, "--lam-dc-frac must be in (0, 1)"
        log(f"Lambda resampled every step: {args.lam_dc_frac:.2f} of it joins DC, the rest is the loss")
elif args.lam_dc_frac is not None:
    sys.exit("--lam-dc-frac needs --split-file")


def split(step):
    if FIXED is not None:
        if args.lam_dc_frac is None:
            return FIXED["theta"], FIXED["lam"]
        q = torch.randperm(LAM_IDX.shape[0], generator=torch.Generator(device="cpu").manual_seed(1234 + step))
        sel = LAM_IDX[q[:int(round(args.lam_dc_frac * q.numel()))].to(LAM_IDX.device)]
        a = torch.zeros_like(m2); a[sel[:, 0], sel[:, 1]] = True
        return FIXED["theta"] | a, FIXED["lam"] & ~a
    p = torch.randperm(tr_pool.numel(), generator=torch.Generator(device="cpu").manual_seed(1234 + step))
    cut = int(args.theta_frac * p.numel())
    return mk_mask(tr_pool[p[:cut]]), mk_mask(tr_pool[p[cut:]])


if FIXED is not None and args.lam_dc_frac is not None:   # one-off check of the resampled split
    (d0, l0), (d1, l1) = split(0), split(1)
    for d, l in ((d0, l0), (d1, l1)):
        assert not (d & l).any() and torch.equal(d | l, FIXED["theta"] | FIXED["lam"]), "resampled split broken"
        assert not (l & FIXED["theta"]).any() and not ((d | l) & FIXED["gamma"]).any(), "loss/val leak"
    assert not torch.equal(l0, l1), "loss set does not change between steps"
    log(f"resample check: DC {int(d0.sum())} loss {int(l0.sum())} per step; loss overlap step0/step1 {int((l0 & l1).sum())}")

def val_loss():
    with torch.no_grad():
        vs = 0.0
        for s0 in range(0, NS, 16):
            sl = torch.arange(s0, min(s0 + 16, NS))
            x, Sr, yr = unrolled(sl, m_tr)
            vs += lossk(x, Sr, yr, m_val).item() * sl.numel()
        return vs / NS


def set_mode(train):
    for m in (nets.values() if args.separate else [net]):
        m.train(train)


opt = torch.optim.Adam(params, lr=args.lr)
BS = min(args.batch, NS)
steps = max(1, NS // BS)
epochs = int(os.environ.get("ZSREF_SMOKE_EPOCHS", "2")) if args.smoke else args.epochs
set_mode(False); v0 = val_loss()
log(f"val(Gamma) before training = {v0:.4f}  ({time.time()-t0:.0f}s)")
best, best_state, bad, hist = v0, None, 0, [dict(epoch=0, val=v0)]
for ep in range(epochs):
    set_mode(True); run = 0.0
    order = torch.randperm(NS, generator=torch.Generator().manual_seed(100 + ep))
    for st in range(steps):
        sl = order[st * BS:(st + 1) * BS]
        m_th, m_lam = split(ep * steps + st)
        x, Sr, yr = unrolled(sl, m_th)
        loss = lossk(x, Sr, yr, m_lam)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); run += loss.item()
    set_mode(False); v = val_loss()
    hist.append(dict(epoch=ep + 1, train=run / steps, val=v))
    log(f"  [ep {ep+1}/{epochs}] train={run/steps:.4f} val={v:.4f} ({time.time()-t0:.0f}s)")
    if v < best - 1e-4:
        best, bad = v, 0
        best_state = [p.detach().clone() for p in params]
    else:
        bad += 1
        if bad >= args.patience:
            log(f"  early stop at epoch {ep+1} (best val {best:.4f})"); break
if best_state is not None:
    with torch.no_grad():
        for p, b in zip(params, best_state):
            p.copy_(b)

# ---------------- inference with the full acquired mask ----------------
def infer(out_dir, tag):
    set_mode(False)
    res = {s: torch.zeros((NS,) + tuple(m2.shape), dtype=torch.complex64, device=dev) for s in SCANS}
    with torch.no_grad():
        for s0 in range(0, NS, 16):
            sl = torch.arange(s0, min(s0 + 16, NS))
            x, _, _ = unrolled(sl, m2)
            for s in SCANS:
                res[s][sl] = x[s]
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in SCANS:
        r = res[s].cpu().numpy()
        assert np.isfinite(r).all()
        d = np.abs(r) - np.abs(X0[s].cpu().numpy())
        log(f"  [{tag}] {s}: |out|-|prior| rms {np.sqrt((d**2).mean()):.4g}, max |out| {np.abs(r).max():.4g}")
        if not args.smoke:
            np.save(out_dir / f"candidate_{s}_complex.npy", r.astype(np.complex64))


infer(OUT, "prior=init-dir")
if args.infer_init_dir is not None:      # same trained network, unroll started from / fed with another prior
    X0.update(load_prior(args.infer_init_dir))
    infer(OUT / "infer_alt", f"prior={args.infer_init_dir}")
meta = dict(args={k: str(v) for k, v in vars(args).items()}, coils=NC, slices=NS, val_before=v0, best_val=best,
            history=hist, seconds=time.time() - t0,
            init_sha256={s: sha(args.init_dir / f"candidate_{s}_complex.npy") for s in SCANS} if not args.smoke else None,
            split_sha256=sha(args.split_file) if args.split_file else None,
            note="no GT/ROI used; selection by Gamma k-space loss only")
(OUT / "fit_log.json").write_text(json.dumps(meta, indent=2))
log("DONE", f"{time.time()-t0:.0f}s")
