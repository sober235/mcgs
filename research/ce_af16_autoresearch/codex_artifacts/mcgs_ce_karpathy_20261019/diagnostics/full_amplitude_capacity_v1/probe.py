#!/usr/bin/env python3
"""CPU-only G2 coefficient teacher fit; never reads MRI measurements or targets."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

CHECKPOINT = Path("/home/congcongliu/codex_artifacts/mcgs_ce_autoresearch_20261017/frequency_adam/outputs/adamflat_g2_v1/training/ar_i1_k30p90/checkpoints/latest.pth")
EXPECTED_SHA = "282e0303c46a6c18775d6224c032611072bd465b4ef975ba7a7e2ea636230fd0"
OUT = Path(__file__).resolve().parent
SEED = 20261019
TRAIN_N, TEST_N, STEPS, BATCH = 60000, 20000, 1000, 1024
NAMES = ("positions", "scales", "rotations", "density_real", "density_imag", "density_real_contrast", "density_imag_contrast")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(x: torch.Tensor) -> dict:
    x = x.to(torch.float64)
    return {"min": x.amin(0).tolist(), "max": x.amax(0).tolist(),
            "mean": x.mean(0).tolist(), "std_population": x.std(0, correction=0).tolist(),
            "quantile_levels": [0., .01, .25, .5, .75, .99, 1.],
            "quantiles": torch.quantile(x, torch.tensor([0., .01, .25, .5, .75, .99, 1.], dtype=x.dtype), dim=0).tolist()}


def features(xyz: torch.Tensor, bands: int, scale: torch.Tensor | None = None) -> torch.Tensor:
    frequencies = math.pi * 2.0 ** torch.arange(bands, dtype=xyz.dtype)
    angles = xyz.unsqueeze(-1) * frequencies
    pieces = [xyz, angles.sin().flatten(1), angles.cos().flatten(1)]
    if scale is not None:
        pieces.append(scale)
    return torch.cat(pieces, 1).detach()


def metrics(pred_log: torch.Tensor, true_log: torch.Tensor, true_mag: torch.Tensor) -> dict:
    p, t, m = (z.to(torch.float64) for z in (pred_log, true_log, true_mag))
    pred_mag = p.exp()
    if not torch.isfinite(pred_mag).all():
        raise ValueError("Nonfinite decoded coefficient modulus")
    r2 = 1 - (t - p).square().sum(0) / (t - t.mean(0)).square().sum(0)
    rel = (pred_mag - m).square().sum(0).sqrt() / m.square().sum(0).sqrt()
    return {"log_r2": r2.tolist(), "relative_modulus_l2": rel.tolist(),
            "log_rmse": (p-t).square().mean(0).sqrt().tolist()}


def main() -> None:
    if (OUT / "results.json").exists():
        raise FileExistsError("Refusing to overwrite completed diagnostic")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    checkpoint_sha = digest(CHECKPOINT)
    if checkpoint_sha != EXPECTED_SHA:
        raise ValueError("G2 checkpoint digest mismatch")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state = checkpoint["model_state"]
    if set(state) != set(NAMES) or state["positions"].shape != (3600000, 3):
        raise ValueError("Unexpected G2 model schema")
    if any(v.device.type != "cpu" or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Nonfinite or non-CPU G2 state")
    # Sampling without replacement; partition is a coefficient approximation holdout only.
    g = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(state["positions"].shape[0], generator=g)[:TRAIN_N+TEST_N]
    xyz = state["positions"][indices].clone().detach()
    logscale = state["scales"][indices].clone().detach()
    mag = torch.stack((torch.complex(state["density_real"][indices], state["density_imag"][indices]).abs(),
                       torch.complex(state["density_real_contrast"][indices], state["density_imag_contrast"][indices]).abs()), 1).detach()
    target = mag.clamp_min(1e-12).log()
    mu, sd = target[:TRAIN_N].mean(0), target[:TRAIN_N].std(0, correction=0).clamp_min(1e-8)
    scale_mu = logscale[:TRAIN_N].mean(0)
    scale_sd = logscale[:TRAIN_N].std(0, correction=0).clamp_min(1e-8)
    standardized_scales = (logscale-scale_mu)/scale_sd
    index_sha = hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    del checkpoint, state
    result = {
        "purpose": "G2-internal coefficient-modulus teacher fit only; no MRI quality or formal trial",
        "checkpoint": {"path": str(CHECKPOINT), "sha256": checkpoint_sha},
        "script_sha256": digest(Path(__file__).resolve()), "torch_version": torch.__version__,
        "device": "cpu", "threads": 1, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "seed": SEED, "train_count": TRAIN_N, "holdout_count": TEST_N,
        "sample_indices_sha256": index_sha, "coefficient_channels": ["plain", "enhanced"],
        "split": "Random distinct Gaussian indices, not unseen MRI data; G2 was optimized from all acquired measurements",
        "teacher": "log(max(abs(density_real+i*density_imag),1e-12)), independently per contrast",
        "target_train_mean": mu.tolist(), "target_train_std_population": sd.tolist(),
        "scale_train_mean": scale_mu.tolist(), "scale_train_std_population": scale_sd.tolist(),
        "scale_definition": "frozen G2 scales stores log Gaussian sigma; append train-standardized log sigma xyz",
        "coordinate_features": "raw_xyz,sin(pi*2**k*xyz),cos(pi*2**k*xyz);k=0..bands-1; original normalized G2 coordinates",
        "input_source": "Frozen positions and optionally log scales only; no coefficient modulus input",
        "sample_stats": {"positions": stats(xyz), "log_scales": stats(logscale),
                         "moduli": stats(mag), "log_moduli": stats(target)},
        "source_tensor_shapes": {"positions": [3600000, 3], "scales": [3600000, 3],
                                 "rotations": [3600000, 4], "complex_coefficients_per_contrast": [3600000]},
        "arms": [], "formal_mri_experiment_count_increment": 0,
    }
    constant = mag[:TRAIN_N].median(0).values.clamp_min(1e-12).log()
    result["constant_train_median_modulus"] = {
        "values": constant.exp().tolist(),
        "train": metrics(constant.expand(TRAIN_N, -1), target[:TRAIN_N], mag[:TRAIN_N]),
        "holdout": metrics(constant.expand(TEST_N, -1), target[TRAIN_N:], mag[TRAIN_N:])}
    weights = {"target_mean": mu, "target_std": sd, "scale_mean": scale_mu, "scale_std": scale_sd}
    prep_seconds = time.monotonic() - started
    result["preparation_seconds"] = prep_seconds
    for name, bands, use_scales in (("coordinates_bands4", 4, False), ("coordinates_bands8", 8, False),
                                     ("coordinates_bands8_logscales", 8, True)):
        t0 = time.monotonic()
        x = features(xyz, bands, standardized_scales if use_scales else None)
        y = ((target[:TRAIN_N]-mu)/sd).detach()
        torch.manual_seed(SEED)
        net = nn.Sequential(nn.Linear(x.shape[1], 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        sample_g = torch.Generator().manual_seed(SEED+1)
        losses = []
        for step in range(STEPS):
            rows = torch.randint(TRAIN_N, (BATCH,), generator=sample_g)
            optimizer.zero_grad(set_to_none=True)
            loss = (net(x[rows])-y[rows]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss in {name}")
            loss.backward()
            optimizer.step()
            if step == 0 or (step+1) % 100 == 0:
                losses.append({"step": step+1, "standardized_log_mse": loss.item()})
        with torch.no_grad():
            predicted = torch.cat([net(part) for part in x.split(8192)]) * sd + mu
        arm = {"name": name, "bands": bands, "use_scales": use_scales,
               "input_dimensions": x.shape[1], "hidden_layers": [64,64], "activation": "SiLU",
               "parameter_count": sum(p.numel() for p in net.parameters()),
               "optimizer": "Adam", "learning_rate": 1e-3, "betas": [.9,.999],
               "actual_updates": STEPS, "batch_size": BATCH, "seed": SEED,
               "batch_seed": SEED+1, "loss_history": losses,
               "train": metrics(predicted[:TRAIN_N], target[:TRAIN_N], mag[:TRAIN_N]),
               "holdout": metrics(predicted[TRAIN_N:], target[TRAIN_N:], mag[TRAIN_N:]),
               "elapsed_seconds_including_feature_build_and_metric_eval": time.monotonic()-t0}
        result["arms"].append(arm)
        weights[name] = net.state_dict()
        print(json.dumps(arm), flush=True)
    torch.save(weights, OUT/"probe_weights.pt")
    result["weights_sha256"] = digest(OUT/"probe_weights.pt")
    result["total_seconds"] = time.monotonic()-started
    (OUT/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"completed": True, "total_seconds": result["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
