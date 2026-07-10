#!/usr/bin/env python3
"""Score an MC-GS reconstruction against an ESPIRiT-CSM self-consistent ground truth.

The GT is generated from the fully-sampled multi-coil k-space using the SAME
ESPIRiT CSM the reconstruction used:
    GT = | sum_c conj(CSM_c) * ifft3c(ksp_c) |
where ifft3c is a centered 3D inverse FFT (ortho norm). A global amplitude
alpha is least-squares fit before computing PSNR / SSIM, removing scale
ambiguity. This keeps the metric apples-to-apples across runs.

Usage:
    python score_espirit_csm.py \
        --recon reconstruction_plain.mat \
        --ksp   T1_trainer.mat \
        --csm   csm_espirit_T1.mat
"""
import argparse
import numpy as np
import h5py
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim


def ifft3c(x):
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(x, axes=(0, 1, 2)), axes=(0, 1, 2), norm="ortho"),
        axes=(0, 1, 2),
    )


def load_ksp(path):
    with h5py.File(path, "r") as f:
        arr = np.asarray(f["ksp"])
        if arr.dtype.names and {"real", "imag"} <= set(arr.dtype.names):
            arr = arr["real"] + 1j * arr["imag"]
    return np.asarray(arr)


def load_key(path, key):
    try:
        return np.asarray(loadmat(path)[key])
    except Exception:
        with h5py.File(path, "r") as f:
            return np.asarray(f[key])


def load_recon(path):
    m = loadmat(path)
    for k in m:
        if not k.startswith("__"):
            return np.asarray(m[k])


def gen_gt(ksp, csm):
    coil_imgs = np.stack([ifft3c(ksp[..., c]) for c in range(ksp.shape[-1])], axis=-1)
    return np.abs(np.sum(np.conj(csm) * coil_imgs, axis=-1)).astype(np.float32)


def score(recon, gt):
    recon = np.abs(recon).astype(np.float32)
    alpha = np.sum(recon * gt) / np.sum(recon * recon)
    r = recon * alpha
    mx = gt.max()
    psnr = 10 * np.log10(mx ** 2 / np.mean((r - gt) ** 2))
    return psnr, ssim(gt, r, data_range=mx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--ksp", required=True)
    ap.add_argument("--csm", required=True)
    args = ap.parse_args()

    ksp = load_ksp(args.ksp)                 # (X, Y, Z, coils)
    csm = load_key(args.csm, "map")          # (X, Y, Z, coils)
    gt = gen_gt(ksp, csm)
    recon = load_recon(args.recon)
    psnr, ss = score(recon, gt)
    print(f"PSNR = {psnr:.2f} dB   SSIM = {ss:.4f}")


if __name__ == "__main__":
    main()
