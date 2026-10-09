"""A tiny synthetic T1/FLAIR pair on disk for the sg2x2 trainer tests: one phantom geometry rendered at two
contrast scalings, a smooth two-coil CSM with unit sum-of-squares, a 40 % random phase-encode mask with a fully
sampled 6x6 centre (constant along the readout axis 0), written as the h5 files MRIDataset reads (keys 'kspace',
'map', 'mask'), plus the frozen init support of the phantom (.npz key 'support')."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
from data.transforms import fft3c  # noqa: E402

RECIPE_REG = {"tv_weight": 0.10, "tv_eps": 0.04, "edge_tv": True, "edge_tv_kappa_q": 0.85, "edge_tv_wfloor": 0.1,
              "edge_tv_kappa_source": "zf", "lambda_wav": 0.10, "lambda_wav_plain": 0.10, "wav_name": "db4",
              "wav_levels": 1, "lambda_ptv": 0.03}
RECIPE_OFF = {"tv_weight": 0.0, "tv_eps": 0.04, "edge_tv": False, "lambda_wav": 0.0, "lambda_wav_plain": 0.0,
              "lambda_ptv": 0.0}
DC = {"loss_type": "l2", "kspace_freq_adaptive": True, "kspace_center_fraction": 0.05, "kspace_center_weight": 0.3,
      "kspace_mid_weight": 1.0, "kspace_high_weight": 2.2}


def make_pair(tmp: Path, shape=(32, 32, 24), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d, h, w = shape
    zz, yy, xx = np.meshgrid(np.linspace(-1, 1, d), np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    r2 = zz ** 2 + yy ** 2 + xx ** 2
    base = np.exp(-r2 / 0.18) * (1.0 + 0.3 * np.cos(4.0 * xx))
    base[r2 > 0.8] = 0.0
    csm = np.stack([np.exp(-((xx - 0.5) ** 2) / 0.8), np.exp(-((xx + 0.5) ** 2) / 0.8)]).astype(np.complex64)
    csm = csm / np.sqrt((np.abs(csm) ** 2).sum(0, keepdims=True))
    mask2d = rng.random((h, w)) < 0.4
    mask2d[h // 2 - 3:h // 2 + 3, w // 2 - 3:w // 2 + 3] = True
    mask = np.broadcast_to(mask2d[None], shape).astype(np.float32).copy()
    out = {"shape": shape}
    for name, scale, ph in (("T1", 1.0, 0.3), ("FLAIR", 0.6, -0.2)):
        img = (scale * base * np.exp(1j * ph * xx)).astype(np.complex64)
        ksp = fft3c(torch.from_numpy(csm * img[None])).numpy().astype(np.complex64)
        kp, cp, mp = tmp / f"ksp_{name}.h5", tmp / f"csm_{name}.h5", tmp / "mask.h5"
        with h5py.File(kp, "w") as f:
            f.create_dataset("kspace", data=ksp)
        with h5py.File(cp, "w") as f:
            f.create_dataset("map", data=csm)
        if not mp.exists():
            with h5py.File(mp, "w") as f:
                f.create_dataset("mask", data=mask)
        out[name] = {"name": name, "data_path": str(kp), "csm_path": str(cp), "mask_path": str(mp)}
    support = base > 0.05 * base.max()
    sp = tmp / "support.npz"
    np.savez_compressed(sp, support=support)
    out["support"] = str(sp)
    return out


def _data_block(spec: dict) -> dict:
    return {"name": spec["name"], "data_path": spec["data_path"], "csm_path": spec["csm_path"],
            "mask_path": spec["mask_path"], "mask_type": "external", "readout_axis": 0, "phase_axes": [1, 2],
            "presampled": False, "skip_readout_check": True, "normalize_kspace": True, "normalization_percentile": 99.9,
            "acceleration_factor": 2}


def base_cfg(pair: dict, tmp: Path, exp: str, reg: bool = True, single: str | None = None,
             init_mode: str = "support_file", npts: int = 3000, seed: int = 42, iters: int = 1) -> dict:
    data = {"plain": _data_block(pair[single])} if single else {"plain": _data_block(pair["T1"]),
                                                                  "contrast": _data_block(pair["FLAIR"])}
    loss = dict(DC)
    loss.update(RECIPE_REG if reg else RECIPE_OFF)
    return {
        "data": data,
        "gaussian": {"initial_num_points": npts, "init_mode": init_mode, "init_support_path": pair["support"],
                     "init_scale_mode": "3nn", "init_scale": 0.06, "position_lr": 3e-4, "rotation_lr": 3e-4,
                     "scale_lr": 2e-3, "density_lr": 1.5e-2},
        "loss": loss,
        "training": {"max_iterations": iters, "shared_warmup_iters": 0, "seed": seed, "log_every": 1,
                     "save_every": 100000, "eval_every": 100000, "max_grad_norm": 1.0,
                     "lr_scheduler": {"type": "exponential", "gamma": 0.9995}},
        "voxelizer": {"type": "tile_cuda", "tile_size": 8, "max_radius": 20},
        "output": {"output_dir": str(tmp / "out"), "exp_name": exp, "use_tensorboard": False},
    }
