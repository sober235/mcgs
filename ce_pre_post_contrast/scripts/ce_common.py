"""ce_common.py -- data, held-out split and scoring shared by the 2026-09-14 comparison arms
(converged CS-SENSE / Joint SENSE, L1-ESPIRiT, MC-INR) on the UIH prospective CE pair.

Only torch + numpy, so it imports under both the GSTMI (torch 2.7) and GR_pt1.10 (torch 1.13) envs.

Scoring is the protocol every other column of this topic uses, computed in one pass over the coils:
    alpha_m = <A_m(x), y>_m / ||A_m(x)||_m^2          (best-fit complex scale on mask m)
    kres_m  = || m .(alpha_m A(x) - y) || / || m . y ||  =  sqrt((Y_m - |N_m|^2 / D_m) / Y_m)
for m in {full acquired mask, 95% fit lines, 5% held-out lines}. The held-out number is defined exactly as
ce_dip_inr.py's kres_on (alpha refit on the held-out lines), so it is comparable with the 3D-DIP / 3D-INR rows.
Sums are accumulated in float64.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.fft as tfft

TOPIC = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast")
RR = Path("/data0/congcong/code/NV-Generate-CTMR/recon_real")
WORK = {"pre": RR / "work_ce_t1_16887049370010", "post": RR / "work_ce_t1_16887049370060"}
SPLIT = TOPIC / "heldout_split" / "split_95_5_seed0.npz"
SCANS = ("pre", "post")


def _c(fn, x, dims):
    return tfft.fftshift(fn(tfft.ifftshift(x, dim=dims), dim=dims, norm="ortho"), dim=dims)


def fft3c(x):
    return _c(tfft.fftn, x, (-3, -2, -1))


def ifft3c(x):
    return _c(tfft.ifftn, x, (-3, -2, -1))


def load_scan(scan, device="cpu"):
    """y (42,256,232,176) with exact zeros off the acquired mask, S (same shape), m2 (232,176) bool plane."""
    y = torch.from_numpy(np.load(WORK[scan] / "ksp_full.npy"))
    S = torch.from_numpy(np.load(WORK[scan] / "csm.npy"))
    m3 = torch.from_numpy(np.load(WORK[scan] / "mask2d.npy")) > 0.5
    assert bool((m3 == m3[0:1]).all()), "readout axis is not uniformly sampled"
    m2 = m3[0].clone()
    y = y * m2
    return y.to(device), S.to(device), m2.to(device)


def classical_scale(scan):
    """P99.9 of |ZF| as computed by ce_classical.py (read from its meta, so the normalisation is identical)."""
    return float(json.load(open(TOPIC / "classical" / f"cssense_{scan}_meta.json"))["scale"])


def make_split(m2, holdout=0.05, seed=0):
    """ce_dip_inr.py's 95/5 split at the readout-line level, reproduced verbatim."""
    idx = torch.nonzero(m2, as_tuple=False)
    nl = idx.shape[0]
    perm = torch.randperm(nl, generator=torch.Generator().manual_seed(seed))
    nho = int(round(holdout * nl))

    def build(sel):
        mm = torch.zeros_like(m2)
        mm[idx[sel, 0], idx[sel, 1]] = True
        return mm

    return build(perm[nho:]), build(perm[:nho]), nl, nho


def load_split(device="cpu"):
    d = np.load(SPLIT)
    return torch.from_numpy(d["fit"]).to(device), torch.from_numpy(d["ho"]).to(device)


def kres_metrics(x, y, S, planes, chunk=6):
    """x (256,232,176) complex, y/S (42,256,232,176) on x's device; planes: {name: (232,176) bool}.
    Returns {name: {"kres": float, "alpha": complex}}."""
    acc = {n: [0j, 0.0, 0.0] for n in planes}
    with torch.no_grad():
        for i in range(0, y.shape[0], chunk):
            k = fft3c(S[i:i + chunk] * x)
            yy = y[i:i + chunk]
            for n, p in planes.items():
                km = k[..., p].to(torch.complex128)
                ym = yy[..., p].to(torch.complex128)
                acc[n][0] += complex((km.conj() * ym).sum().item())
                acc[n][1] += float((km.abs() ** 2).sum().item())
                acc[n][2] += float((ym.abs() ** 2).sum().item())
            del k
    out = {}
    for n, (N, D, Y) in acc.items():
        out[n] = {"kres": float(np.sqrt(max(Y - abs(N) ** 2 / D, 0.0) / Y)), "alpha": N / D}
    return out


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)
