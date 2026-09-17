#!/usr/bin/env python3
"""sim_cssense_af16.py -- synthetic pre/post k-space: the CS-SENSE volumes as truth, the real 42-coil maps,
the Poisson-disc AF16 mask, complex Gaussian noise at the level the real acquisition showed.

Per scan:  sigma  from r = M0 (y_meas - alpha F(S x_GT)) on the prospective mask M0 (alpha = least-squares complex
           scale), sigma = sqrt(mean|r|^2 / 2); the relative residual ||r||/||M0 y|| must reproduce the archived
           CS-SENSE kres (0.18487 pre / 0.14969 post) or the run stops -- that is the FFT/CSM convention gate.
           y_noisy = F(S x_GT) + sigma (N + iN) on every k-space point (seed 100 pre / 200 post), then
           work_<scan>/ksp_full.npy = mask . y_noisy   and   gs/<scan>_kspace.h5 = y_noisy (the trainer masks itself).
  python scripts/sim_cssense_af16.py --gpu 0        (about 10 min; --gpu -1 for CPU, slower)
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
CE = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast")
GT = {s: CE / f"tv_sense/cssense_o200_lam0.002_full_{s}_complex.npy" for s in ("pre", "post")}
MEAS = {"pre": Path("/data0/congcong/code/NV-Generate-CTMR/recon_real/work_ce_t1_16887049370010"),
        "post": Path("/data0/congcong/code/NV-Generate-CTMR/recon_real/work_ce_t1_16887049370060")}
ROI = CE / "lesion_candidates/enhance_rois_cssense.npz"
KRES_REF = {"pre": 0.18487, "post": 0.14969}
NOISE_SEED = {"pre": 100, "post": 200}
NX, NY, NZ = 256, 232, 176


def _c(fn, x, dims):
    return torch.fft.fftshift(fn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm="ortho"), dim=dims)


def fft3c(x):  return _c(torch.fft.fftn, x, (-3, -2, -1))
def ifft3c(x): return _c(torch.fft.ifftn, x, (-3, -2, -1))


def sigma_from_residual(S, x, y_meas, m0, dev, chunk=6):
    """alpha = <A x, y>_M0 / ||A x||_M0^2 ; kres = ||M0 (alpha A x - y)|| / ||M0 y|| ; sigma per real component."""
    x = x.to(dev); m0 = m0.to(dev).bool()
    num = 0j; den = 0.0; yy = 0.0
    for i in range(0, S.shape[0], chunk):
        k = fft3c(S[i:i + chunk].to(dev) * x)[..., m0].to(torch.complex128)
        ym = y_meas[i:i + chunk].to(dev)[..., m0].to(torch.complex128)
        num += complex((k.conj() * ym).sum().item()); den += float((k.abs() ** 2).sum().item()); yy += float((ym.abs() ** 2).sum().item())
    alpha = num / den
    res = 0.0
    for i in range(0, S.shape[0], chunk):
        k = fft3c(S[i:i + chunk].to(dev) * x)[..., m0].to(torch.complex128)
        ym = y_meas[i:i + chunk].to(dev)[..., m0].to(torch.complex128)
        res += float(((alpha * k - ym).abs() ** 2).sum().item())
    n = int(m0.sum().item()) * S.shape[0] * S.shape[1]  # BUGFIX vs brief: [..., m0] keeps the readout axis (S.shape[1]);
    return alpha, float(np.sqrt(res / yy)), float(np.sqrt(res / (2.0 * n)))              # n must count it or sigma is inflated by sqrt(S.shape[1]).


def synth_kspace(S, x, sigma, rng, dev, chunk=6):
    """F(S x) + sigma (N + iN) on every point, coil by coil in a fixed order so the seed fixes the noise."""
    x = x.to(dev)
    out = np.empty(tuple(S.shape), np.complex64)
    for i in range(0, S.shape[0], chunk):
        k = fft3c(S[i:i + chunk].to(dev) * x).cpu().numpy()
        n = rng.standard_normal(k.shape, dtype=np.float32) + 1j * rng.standard_normal(k.shape, dtype=np.float32)
        out[i:i + chunk] = (k + np.float32(sigma) * n).astype(np.complex64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cpu") if a.gpu < 0 else torch.device(f"cuda:{a.gpu}")
    if a.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu); dev = torch.device("cuda:0")
    t0 = time.time()
    mask2d = np.load(R / "sim/mask/mask_af16_pe.npy") > 0
    assert mask2d.shape == (NY, NZ) and int(mask2d.sum()) == NY * NZ // 16
    with np.load(ROI) as r:
        wm = r["roi2"]
    (R / "sim/gs").mkdir(parents=True, exist_ok=True)
    gt_out, meta = {"mask2d": mask2d}, dict(grid=[NX, NY, NZ], pe_points=int(mask2d.sum()), pe_total=NY * NZ,
                                           af=NY * NZ / int(mask2d.sum()), scans={}, gt=GT_str(), fft="centred orthonormal (ce_common.fft3c)")
    for scan in ("pre", "post"):
        x = torch.from_numpy(np.load(GT[scan])).to(torch.complex64)
        S = torch.from_numpy(np.load(MEAS[scan] / "csm.npy")).to(torch.complex64)
        y_meas = torch.from_numpy(np.load(MEAS[scan] / "ksp_full.npy")).to(torch.complex64)
        m0 = torch.from_numpy(np.load(MEAS[scan] / "mask2d.npy")[0] > 0.5)
        assert x.shape == (NX, NY, NZ) and S.shape == y_meas.shape == (42, NX, NY, NZ)
        alpha, kres, sigma = sigma_from_residual(S, x, y_meas, m0, dev)
        print(f"[{scan}] alpha={alpha:.5f} kres={kres:.5f} (ref {KRES_REF[scan]}) sigma={sigma:.5e}  {time.time() - t0:.0f}s", flush=True)
        if abs(kres - KRES_REF[scan]) >= 1e-3:
            raise SystemExit(f"[{scan}] kres {kres:.5f} does not reproduce the archived {KRES_REF[scan]}: FFT/CSM convention mismatch")
        del y_meas
        y = synth_kspace(S, x, sigma, np.random.default_rng(NOISE_SEED[scan]), dev)
        clean_sample = fft3c(S[:2].to(dev) * x.to(dev)).cpu().numpy()
        sigma_meas = float((y[:2] - clean_sample).real.std())
        assert abs(sigma_meas / sigma - 1) < 0.02, (sigma_meas, sigma)
        with h5py.File(R / f"sim/gs/{scan}_kspace.h5", "w") as f:
            f.create_dataset("kspace", data=y)
        with h5py.File(R / f"sim/gs/{scan}_csm.h5", "w") as f:
            f.create_dataset("map", data=S.numpy())
        w = R / f"sim/work_{scan}"; w.mkdir(exist_ok=True)
        np.save(w / "ksp_full.npy", (y * mask2d[None, None]).astype(np.complex64))
        shutil.copyfile(MEAS[scan] / "csm.npy", w / "csm.npy")
        np.save(w / "mask2d.npy", np.broadcast_to(mask2d[None], (NX, NY, NZ)).astype(np.float32))
        xa = x.abs().numpy()
        sos = torch.sqrt((S.abs() ** 2).sum(0)).numpy()
        snr_wm = float(np.median(xa[wm]) / np.median(sigma * sos[wm]))
        gt_out[scan] = x.numpy(); gt_out[f"scale_{scan}"] = float(np.quantile(xa, 0.999))
        meta["scans"][scan] = dict(alpha_re=alpha.real, alpha_im=alpha.imag, kres_on_prospective_mask=kres, kres_ref=KRES_REF[scan],
                                   sigma=sigma, sigma_measured_first_two_coils=sigma_meas, noise_seed=NOISE_SEED[scan],
                                   wm_snr_sense_adjoint=snr_wm, scale_p999=gt_out[f"scale_{scan}"], csm=str(MEAS[scan] / "csm.npy"))
        print(f"[{scan}] wrote gs/{scan}_kspace.h5 work_{scan}/  WM SNR {snr_wm:.1f}  {time.time() - t0:.0f}s", flush=True)
        del S, y, x
    np.savez(R / "sim/gt.npz", **gt_out)
    json.dump(meta, open(R / "sim/meta.json", "w"), indent=1)
    (R / "logs").mkdir(exist_ok=True)
    af = meta["af"]
    json.dump(dict(mask2d_shape=[NY, NZ], mask2d_samples=int(mask2d.sum()), mask2d_AF_rect=af,
                   contrasts={s: dict(AF_vs_own_support=af, AF_vs_full_rect=af) for s in ("pre", "post")}),
              open(R / "logs/prep_04_export.json", "w"), indent=1)
    print(f"DONE {time.time() - t0:.0f}s -> sim/gt.npz sim/meta.json logs/prep_04_export.json")


def GT_str():
    return {s: str(p) for s, p in GT.items()}


if __name__ == "__main__":
    main()
