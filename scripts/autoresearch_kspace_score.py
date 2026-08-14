#!/usr/bin/env python3
"""Score a joint checkpoint on truly unobserved complex multicoil k-space.

This is a checkpoint-replay metric.  It deliberately reconstructs the complex
volumes from ``model_state`` and compares their multicoil forward projections
against ``MRIDataset.kspace_multicoil``.  Magnitude-only exported NPZ files are
not sufficient for this metric and are never read here.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results_arc"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a JointTrainer checkpoint and score each modality on "
            "unobserved complex multicoil k-space."
        )
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="experiment name under results_arc/<experiment>/<experiment>",
    )
    parser.add_argument(
        "--gpu",
        required=True,
        type=int,
        help="physical GPU index to expose as cuda:0",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new JSON result path; existing paths are refused",
    )
    return parser.parse_args()


def _sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _resolve_run_paths(experiment: str) -> tuple[Path, Path]:
    # The layout is intentionally fixed.  Disallow path traversal and aliases so
    # the provenance paths cannot silently point at a different run.
    if not experiment or Path(experiment).name != experiment or experiment in {".", ".."}:
        raise ValueError(
            "--experiment must be one directory name, not a path: "
            f"{experiment!r}"
        )
    run_dir = RESULTS_ROOT / experiment / experiment
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "checkpoints" / "latest.pth"
    if not config_path.is_file():
        raise FileNotFoundError(f"resolved config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"latest checkpoint not found: {checkpoint_path}")
    return config_path, checkpoint_path


def _require_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


def _validate_shared_state(state: Mapping[str, Any]) -> int:
    required_shapes = {
        "positions": (None, 3),
        "scales": (None, 3),
        "rotations": (None, 4),
        "density_real": (None,),
        "density_imag": (None,),
        "density_real_contrast": (None,),
        "density_imag_contrast": (None,),
    }
    missing = sorted(set(required_shapes).difference(state))
    if missing:
        raise KeyError(f"checkpoint model_state missing keys: {missing}")

    positions = state["positions"]
    if not hasattr(positions, "shape") or len(positions.shape) != 2:
        raise ValueError("model_state['positions'] must be a rank-2 tensor")
    n_points = int(positions.shape[0])
    if n_points <= 0:
        raise ValueError("checkpoint contains no shared Gaussian points")

    for key, template in required_shapes.items():
        value = state[key]
        expected = tuple(n_points if dim is None else dim for dim in template)
        actual = tuple(value.shape) if hasattr(value, "shape") else None
        if actual != expected:
            raise ValueError(
                f"model_state[{key!r}] shape {actual} != expected {expected}"
            )
    return n_points


def _write_shared_bootstrap(torch: Any, state: Mapping[str, Any], path: Path) -> None:
    # JointTrainer builds its model before load_state_dict.  A final checkpoint
    # may have more points than the original warm start after densification, so
    # bootstrap the constructor with the final shared geometry and then perform
    # the authoritative strict model_state load below.
    gaussian_state = {
        key: state[key].detach().cpu()
        for key in ("positions", "scales", "rotations", "density_real", "density_imag")
    }
    torch.save({"gaussian_state": gaussian_state}, path)


def _configure_private_bootstrap(
    torch: Any,
    cfg: dict[str, Any],
    state: Mapping[str, Any],
    temp_dir: Path,
) -> None:
    private_mapping = {
        "private_positions": "positions",
        "private_scales": "scales",
        "private_rotations": "rotations",
        "private_density_real": "density_real",
        "private_density_imag": "density_imag",
    }
    present = set(private_mapping).intersection(state)
    gaussian_cfg = cfg.setdefault("gaussian", {})
    if not present:
        # The checkpoint is authoritative.  Prevent a stale config from adding
        # parameters that would make strict=True fail with unexpected keys.
        gaussian_cfg["num_private_points"] = 0
        gaussian_cfg.pop("private_warm_start_checkpoint", None)
        return
    if present != set(private_mapping):
        missing = sorted(set(private_mapping).difference(present))
        raise KeyError(f"incomplete private residual in model_state; missing {missing}")

    n_private = int(state["private_positions"].shape[0])
    expected_shapes = {
        "private_positions": (n_private, 3),
        "private_scales": (n_private, 3),
        "private_rotations": (n_private, 4),
        "private_density_real": (n_private,),
        "private_density_imag": (n_private,),
    }
    for key, expected in expected_shapes.items():
        actual = tuple(state[key].shape)
        if actual != expected:
            raise ValueError(f"model_state[{key!r}] shape {actual} != {expected}")

    private_state = {
        target_key: state[source_key].detach().cpu()
        for source_key, target_key in private_mapping.items()
    }
    private_path = temp_dir / "private_bootstrap.pth"
    torch.save({"gaussian_state": private_state}, private_path)
    gaussian_cfg["num_private_points"] = n_private
    gaussian_cfg["private_warm_start_checkpoint"] = str(private_path)
    # Constructor-time density scaling is irrelevant after strict load, but 1.0
    # avoids introducing needless overflow/underflow before that load.
    gaussian_cfg["private_density_init_factor"] = 1.0


def _complex_energy(torch: Any, value: Any) -> Any:
    # Float64 accumulation; inputs remain float32/complex64 to limit memory.
    return value.real.square().sum(dtype=torch.float64) + value.imag.square().sum(
        dtype=torch.float64
    )


def _score_modality(
    torch: Any,
    fft3c: Any,
    trainer: Any,
    channel: str,
    display_name: str,
) -> dict[str, Any]:
    if channel == "plain":
        csm, mask, dataset = trainer.csm_p, trainer.mask_p, trainer.ds_p
    elif channel == "contrast":
        csm, mask, dataset = trainer.csm_c, trainer.mask_c, trainer.ds_c
    else:  # defensive: this function is intentionally not a generic channel API
        raise ValueError(f"unknown joint channel: {channel!r}")

    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 3:
        raise ValueError(f"{channel} mask must be 3D, got shape {tuple(mask.shape)}")
    if not bool(torch.isfinite(mask).all().item()):
        raise ValueError(f"{channel} mask contains NaN or Inf")
    rounded_mask = mask.round()
    if not bool(torch.allclose(mask, rounded_mask, atol=1e-6, rtol=0.0)):
        raise ValueError(f"{channel} mask is not binary")

    full_kspace = dataset.kspace_multicoil
    if full_kspace.ndim != 4 or not torch.is_complex(full_kspace):
        raise ValueError(
            f"{channel} dataset.kspace_multicoil must be complex (C,D,H,W), "
            f"got shape={tuple(full_kspace.shape)}, dtype={full_kspace.dtype}"
        )
    if tuple(full_kspace.shape) != tuple(csm.shape):
        raise ValueError(
            f"{channel} full k-space shape {tuple(full_kspace.shape)} != "
            f"CSM shape {tuple(csm.shape)}"
        )
    if tuple(mask.shape) != tuple(full_kspace.shape[-3:]):
        raise ValueError(
            f"{channel} mask shape {tuple(mask.shape)} != spatial k-space shape "
            f"{tuple(full_kspace.shape[-3:])}"
        )

    unsampled = 1.0 - mask
    unsampled_count = int(torch.count_nonzero(unsampled > 0.5).item())
    total_count = int(mask.numel())
    if unsampled_count <= 0:
        raise ValueError(f"{channel} mask has no unobserved k-space locations")

    numerator = torch.zeros((), dtype=torch.float64, device=trainer.device)
    denominator = torch.zeros((), dtype=torch.float64, device=trainer.device)
    with torch.inference_mode():
        x = trainer._render(channel)
        if tuple(x.shape) != tuple(mask.shape) or not torch.is_complex(x):
            raise ValueError(
                f"{channel} render must be complex with shape {tuple(mask.shape)}, "
                f"got shape={tuple(x.shape)}, dtype={x.dtype}"
            )

        # Coil-wise execution is algebraically identical to
        # fft3c(csm * x.unsqueeze(0)); it avoids materializing another complete
        # multicoil prediction/target pair on GPU.  No coil is averaged away.
        for coil in range(int(csm.shape[0])):
            csm_coil = csm[coil] * x
            predicted = fft3c(csm_coil)
            target = full_kspace[coil].to(trainer.device)
            residual_unobs = (predicted - target) * unsampled
            target_unobs = target * unsampled
            numerator += _complex_energy(torch, residual_unobs)
            denominator += _complex_energy(torch, target_unobs)

    numerator_value = float(numerator.item())
    denominator_value = float(denominator.item())
    scalars_finite = math.isfinite(numerator_value) and math.isfinite(denominator_value)
    denominator_positive = denominator_value > 0.0
    if scalars_finite and denominator_positive:
        nmse_unobs = numerator_value / denominator_value
        e_unobs = math.sqrt(nmse_unobs)
        finite = math.isfinite(nmse_unobs) and math.isfinite(e_unobs)
    else:
        nmse_unobs = None
        e_unobs = None
        finite = False

    return {
        "channel": channel,
        "name": display_name,
        "e_unobs": e_unobs,
        "nmse_unobs": nmse_unobs,
        "numerator_energy": numerator_value if math.isfinite(numerator_value) else None,
        "denominator_energy": denominator_value if math.isfinite(denominator_value) else None,
        "finite": finite,
        "mask_unsampled_count": unsampled_count,
        "mask_sampled_count": total_count - unsampled_count,
        "mask_total_count": total_count,
        "mask_unsampled_multicoil_count": unsampled_count * int(csm.shape[0]),
        "num_coils": int(csm.shape[0]),
    }


def main() -> None:
    args = _parse_args()
    if args.gpu < 0:
        raise ValueError(f"--gpu must be non-negative, got {args.gpu}")
    output_path = args.output.expanduser().resolve()
    _require_new_output(output_path)
    config_path, checkpoint_path = _resolve_run_paths(args.experiment)

    # Pin the selected physical GPU before importing torch or any project module
    # that imports torch.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    import torch
    import yaml

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is unavailable after CUDA_VISIBLE_DEVICES={args.gpu}; "
            "checkpoint replay requires the configured CUDA voxelizer"
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    sys.path.insert(0, str(ROOT / "paired_gs"))
    from data.private_paths import resolve_subject_paths
    from data.transforms import fft3c
    from trainers.joint_trainer import JointTrainer

    config_bytes = config_path.read_bytes()
    cfg_loaded = yaml.safe_load(config_bytes)
    if not isinstance(cfg_loaded, dict):
        raise ValueError(f"config must contain a YAML mapping: {config_path}")
    cfg = copy.deepcopy(cfg_loaded)
    for sub in ("plain", "contrast"):
        sub_cfg = cfg.get("data", {}).get(sub, {})
        if bool(sub_cfg.get("presampled", False)):
            raise ValueError(
                f"{sub} is presampled: dataset.kspace_multicoil contains no "
                "unobserved reference samples, so e_unobs would be invalid"
            )
        for key in ("data_path", "csm_path", "mask_path"):
            value = sub_cfg.get(key)
            if value:
                sub_cfg[key] = resolve_subject_paths(value)

    checkpoint_sha256 = _sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint root must be a mapping")
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise KeyError("checkpoint has no mapping-valued 'model_state'")
    n_points = _validate_shared_state(state)
    if "iter" not in checkpoint:
        raise KeyError("checkpoint has no 'iter' field")
    checkpoint_iteration = int(checkpoint["iter"])
    if checkpoint_iteration < 0:
        raise ValueError(f"checkpoint iteration must be non-negative, got {checkpoint_iteration}")

    with tempfile.TemporaryDirectory(prefix=f"kspace_replay_{args.experiment}_") as tmp:
        temp_dir = Path(tmp)
        shared_bootstrap = temp_dir / "shared_bootstrap.pth"
        _write_shared_bootstrap(torch, state, shared_bootstrap)

        cfg.setdefault("gaussian", {})["warm_start_checkpoint"] = str(shared_bootstrap)
        _configure_private_bootstrap(torch, cfg, state, temp_dir)
        cfg.setdefault("output", {})["output_dir"] = str(temp_dir / "trainer_output")
        cfg["output"]["exp_name"] = "checkpoint_replay"
        cfg["output"]["use_tensorboard"] = False

        trainer = JointTrainer(cfg, device=device)
        incompatible = trainer.model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            # strict=True should have raised already; retain an explicit invariant
            # in case PyTorch's return contract changes.
            raise RuntimeError(
                "strict model load returned incompatibilities: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        trainer.model.eval()

        plain_result = _score_modality(
            torch, fft3c, trainer, "plain", trainer.name_plain
        )
        contrast_result = _score_modality(
            torch, fft3c, trainer, "contrast", trainer.name_contrast
        )

        result = {
            "schema_version": 1,
            "metric": "complex_multicoil_unobserved_kspace_checkpoint_replay",
            "formula": {
                "residual": "(1-mask) * (fft3c(csm*x) - kspace_multicoil)",
                "nmse_unobs": "sum(|residual|^2) / sum(|(1-mask)*kspace_multicoil|^2)",
                "e_unobs": "sqrt(nmse_unobs)",
                "aggregation": "none; reported separately per modality",
            },
            "experiment": args.experiment,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config": {
                "path": str(config_path.resolve()),
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": checkpoint_sha256,
                "iteration": checkpoint_iteration,
                "shared_num_points": n_points,
            },
            "model_load": {
                "strict": True,
                "voxelizer_backend": getattr(trainer.voxelizer, "backend", None),
            },
            "device": {
                "requested_physical_gpu": args.gpu,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "logical_device": "cuda:0",
                "name": torch.cuda.get_device_name(0),
            },
            "modalities": {
                "plain": plain_result,
                "contrast": contrast_result,
            },
            "finite": bool(plain_result["finite"] and contrast_result["finite"]),
            "source_guards": {
                "checkpoint_replayed": True,
                "complex_multicoil_reference": "MRIDataset.kspace_multicoil",
                "magnitude_npz_used": False,
            },
        }

    # Exclusive creation is the final race-safe overwrite guard.  No partial JSON
    # is left behind if dataset/model initialization or rendering fails.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"KSPACE_SCORE_DONE -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
