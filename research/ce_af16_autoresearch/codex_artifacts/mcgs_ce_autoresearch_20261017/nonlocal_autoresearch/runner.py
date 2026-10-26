#!/usr/bin/env python3
"""Formal CE AF16 i1 continuation with an optional scan-internal patch low-rank prior."""
from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parent
TOPIC = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois")
SOURCE_TRAINER = TOPIC / "trainer/_trainer_e778bf2_ar"
SOURCE_RUN = SOURCE_TRAINER / "results_arc/ar_i1_k30p90/ar_i1_k30p90"
CHECKPOINT = SOURCE_RUN / "checkpoints/latest.pth"
BASE_CONFIG = SOURCE_RUN / "config.yaml"
SCRIPTS = Path("/data0/congcong/code/GS/mcgs_ce_mask_wt/ce_retro_cssense_af16/scripts")
TRAINER = ROOT / "trainer"
SOURCE_ITERATION = 4000
METHOD = "ar_i1_k30p90"
NAMES = ("positions", "scales", "rotations", "density_real", "density_imag",
         "density_real_contrast", "density_imag_contrast")
CHECK_METRICS = ("psnr", "ssim", "nmse", "hfen", "head_psnr", "head_ssim", "head_nmse", "head_hfen")


PATCH_LOWRANK_PRESET = {"patch_size": 5, "search_radius": 8, "search_stride": 2,
                        "anchor_stride": 3, "group_size": 16, "min_group_size": 4,
                        "distance_factor": 3.0, "threshold_scale": 1.0,
                        "batch_size": 256, "device": "cuda:0"}


def build_config(mode, steps, run_dir, *, local_factor=1., nlambda=0., sigma_rel=.015):
    """Keep the frozen recipe; record each explicit local/NLR/output/resume override."""
    if mode not in ("audit", "all", "density"):
        raise ValueError("mode must be audit, all, or density")
    if mode != "audit" and (not isinstance(steps, int) or not 1 <= steps <= 500):
        raise ValueError("continued steps must be an integer from 1 through the approved 500-step budget")
    for name, value in (("local_factor", local_factor), ("nlambda", nlambda), ("sigma_rel", sigma_rel)):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    with BASE_CONFIG.open() as stream:
        cfg = yaml.safe_load(stream)
    training = cfg["training"]
    if (training["max_iterations"] != SOURCE_ITERATION or training["seed"] != 42
            or cfg["gaussian"]["initial_num_points"] != 3600000
            or training["lr_scheduler"] != {"type": "exponential", "gamma": .9995}
            or training["shared_warmup_iters"] != 500
            or cfg.get("adaptive_control", {}).get("enable", False)
            or cfg["gaussian"].get("num_private_points", 0)
            or cfg["gaussian"].get("warm_start_checkpoint")):
        raise ValueError("source config is not the frozen 4000-step, seven-group i1 recipe")
    for channel in ("plain", "contrast"):
        if cfg["data"][channel]["acceleration_factor"] != 16:
            raise ValueError("source config must use the formal AF16 acquisitions")
    overrides = {}
    def override(section, key, value):
        overrides[f"{section}.{key}"] = {"source_present": key in cfg[section],
                                         "source": cfg[section].get(key), "effective": value}
        cfg[section][key] = value
    loss = cfg["loss"]
    # The registered YAML uses loss.tv_weight (the trainer's lambda_geo alias),
    # loss.lambda_wav and loss.lambda_wav_plain; contrast inherits lambda_wav.
    # training.lambda_tv and lambda_wavelet are not keys consumed by this trainer.
    for key in ("tv_weight", "lambda_wav", "lambda_wav_plain"):
        override("loss", key, float(loss[key]) * float(local_factor))
    for key, value in {"lambda_nl": float(nlambda), "lambda_nl_plain": float(nlambda),
                       "lambda_nl_contrast": float(nlambda), "nl_denoiser": "patch_lowrank",
                       "nl_sigma_rel": float(sigma_rel), "nl_start": SOURCE_ITERATION + 1,
                       "nl_refresh": 250, "nl_denoiser_kwargs": dict(PATCH_LOWRANK_PRESET)}.items():
        override("loss", key, value)
    override("training", "resume_offset", SOURCE_ITERATION)
    override("training", "max_iterations", SOURCE_ITERATION + (0 if mode == "audit" else steps))
    override("output", "output_dir", str(Path(run_dir) / "training"))
    cfg["campaign_overrides"] = {
        "local_factor": float(local_factor), "nlambda": float(nlambda), "sigma_rel": float(sigma_rel),
        "local_regularization": {
            "tv": {"source_key": "loss.tv_weight", **overrides["loss.tv_weight"]},
            "plain_wavelet": {"source_key": "loss.lambda_wav_plain", **overrides["loss.lambda_wav_plain"]},
            "contrast_wavelet": {"source_key": "loss.lambda_wav (inherited)", **overrides["loss.lambda_wav"]}},
        "sigma_source": "current detached per-scan rendered magnitude P99 inside fixed ZF head support",
        "changes": overrides}
    return cfg


def reserve_run(output, exp):
    """Atomically claim a fresh run inside this isolated campaign."""
    output = Path(output).expanduser().resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError(f"output must be inside {ROOT}")
    if not exp or exp in (".", "..") or Path(exp).name != exp:
        raise ValueError("exp must be a single directory name")
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / exp
    run_dir.mkdir(exist_ok=False)
    return run_dir


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_copy_record():
    """Record all code-copy hashes and a reviewable delta, ignoring generated caches."""
    def files(root):
        return {p.relative_to(root).as_posix(): p for p in root.rglob("*")
                if p.is_file() and not any(part in ("results_arc", "__pycache__", "build")
                                          for part in p.relative_to(root).parts)
                and p.suffix not in (".pyc", ".so", ".o")}
    original, copied = files(SOURCE_TRAINER), files(TRAINER)
    manifest, delta = {}, []
    changed = set()
    for name in sorted(set(original) | set(copied)):
        a, b = original.get(name), copied.get(name)
        left, right = sha256_file(a) if a else None, sha256_file(b) if b else None
        manifest[name] = {"source_sha256": left, "copy_sha256": right}
        if left != right:
            changed.add(name)
            delta.extend(difflib.unified_diff(a.read_text().splitlines(True) if a else [],
                                             b.read_text().splitlines(True) if b else [],
                                             fromfile=f"source/{name}", tofile=f"copy/{name}"))
    allowed = {"paired_gs/trainers/joint_trainer.py", "paired_gs/trainers/resume_protocol.py",
               "paired_gs/losses/nonlocal_lowrank.py", "paired_gs/losses/nonlocal_red.py"}
    if changed != allowed:
        raise ValueError(f"unexpected source-to-copy delta: {sorted(changed)}; expected {sorted(allowed)}")
    return {"source": str(SOURCE_TRAINER), "copy": str(TRAINER), "files": manifest,
            "changed_files": sorted(changed)}, "".join(delta)


def state_snapshot(trainer):
    """Hash actual current tensors and Adam states by their resolved model names."""
    import torch
    from trainers.resume_protocol import validate_parameter_groups
    named = validate_parameter_groups(trainer.model, trainer.optimizer)

    def digest(tensor):
        array = tensor.detach().cpu().contiguous().numpy()
        return hashlib.sha256(memoryview(array).cast("B")).hexdigest()

    parameters = {name: digest(param) for name, param in named.items()}
    moments = {}
    for name, param in named.items():
        state = trainer.optimizer.state[param]
        moments[name] = {"step": int(state["step"].item()),
                         "exp_avg_sha256": digest(state["exp_avg"]),
                         "exp_avg_sq_sha256": digest(state["exp_avg_sq"])}
    joined = lambda names: hashlib.sha256("".join(parameters[n] for n in names).encode()).hexdigest()
    return {"parameter_sha256": parameters, "adam": moments,
            "model_sha256": joined(NAMES), "geometry_sha256": joined(NAMES[:3]),
            "density_sha256": joined(NAMES[3:])}


class StepTrace:
    """Observe before masking, then synchronize after the original Adam/scheduler step."""
    def __init__(self, trainer, mode, stream):
        self.trainer, self.mode, self.stream = trainer, mode, stream
        self.count = 0
        self.last_iteration = SOURCE_ITERATION
        self.training_step_seconds = 0.

    def synchronize(self):
        if self.trainer.device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.trainer.device)

    def __call__(self, phase, iteration, data):
        import torch
        if phase == "start":
            self.synchronize()
            self.started = time.perf_counter()
        elif phase == "pre_step":
            named = dict(self.trainer.model.named_parameters())
            active = [name for name in NAMES if named[name].grad is not None]
            with torch.no_grad():
                norm = torch.linalg.vector_norm(torch.stack([named[n].grad.detach().norm() for n in active]))
            if not bool(torch.isfinite(norm)):
                raise RuntimeError(f"nonfinite gradient at iteration {iteration}; Adam step refused")
            updated = [name for name in active if self.mode == "all" or name in NAMES[3:]]
            self.pending = {"mode": self.mode, "absolute_iteration": iteration, **data,
                            "postclip_gradient_norm": float(norm),
                            "gradients_finite": True,
                            "full_gradient_group_count": len(active),
                            "full_gradient_parameter_count": sum(named[n].numel() for n in active),
                            "updated_parameter_groups": updated,
                            "updated_parameter_count": sum(named[n].numel() for n in updated),
                            "learning_rates_used": {name: g["lr"] for name, g in
                                                    zip(NAMES, self.trainer.optimizer.param_groups)}}
        elif phase == "end":
            self.synchronize()
            seconds = time.perf_counter() - self.started
            self.pending.update(wall_seconds=seconds, learning_rates_next={name: g["lr"] for name, g in
                                zip(NAMES, self.trainer.optimizer.param_groups)})
            if self.trainer.device.type == "cuda":
                self.pending["cuda_memory_bytes"] = {
                    "allocated": torch.cuda.memory_allocated(self.trainer.device),
                    "reserved": torch.cuda.memory_reserved(self.trainer.device),
                    "peak_allocated": torch.cuda.max_memory_allocated(self.trainer.device),
                    "peak_reserved": torch.cuda.max_memory_reserved(self.trainer.device)}
            self.stream.write(json.dumps(self.pending, allow_nan=False) + "\n")
            self.stream.flush()
            self.training_step_seconds += seconds
            self.count += 1
            self.last_iteration = iteration
        else:
            raise ValueError(f"unknown observation phase {phase}")


def export_trainer_volume(trainer, channel):
    """Exact arc/cycle.py export: raw -> measured-kspace replacement -> coil combine."""
    import torch
    from data.transforms import fft3c, ifft3c
    tag = "p" if channel == "plain" else "c"
    if channel not in ("plain", "contrast"):
        raise ValueError("channel must be plain or contrast")
    csm, y, mask, mask_fit, y_dc = (getattr(trainer, f"{name}_{tag}")
                                   for name in ("csm", "kspace", "mask", "mask_fit", "kspace_dc"))
    with torch.no_grad():
        m = (mask if mask.ndim == 3 else mask[0]).to(trainer.device if hasattr(trainer, "device") else y.device).float().unsqueeze(0)
        mf = (mask_fit if mask_fit.ndim == 3 else mask_fit[0]).to(y.device).float().unsqueeze(0)
        x = trainer._render(channel)
        k_model = fft3c(csm * x.unsqueeze(0))
        x_dc = (ifft3c(mf * y_dc + (1.0 - mf) * k_model) * csm.conj()).sum(0)
        x_zf = (ifft3c(m * y) * csm.conj()).sum(0)
    return x_dc, x_zf


def audit_metric_check(row, reference):
    """The unchanged score_volume relative gate plus the registered 0.001 dB gate."""
    import math
    bad = [key for key in CHECK_METRICS if not math.isfinite(row[key])
           or abs(row[key] - reference[key]) > 2e-4 * max(1., abs(reference[key]))]
    difference = abs(row["head_psnr"] - reference["head_psnr"])
    return {"passed": not bad and difference <= .001,
            "score_volume_relative_tolerance": 2e-4, "mismatched_metrics": bad,
            "head_psnr_absolute_difference_db": difference, "head_psnr_tolerance_db": .001}


def export_and_score(trainer, run_dir, mode, elapsed):
    """Use topic gauge and original CPU score helpers; GT enters only here for scoring."""
    import numpy as np
    import torch
    # Pair imports original CPU scoring code, but resolves trainer helpers from the copy.
    os.environ["RETRO_TOPIC"] = str(TOPIC)
    os.environ["RETRO_TRAINER"] = str(TRAINER)
    sys.path.insert(0, str(SCRIPTS))
    from t1t2_common import Pair
    rows, checks = {}, {}
    for channel, scan, dataset in (("plain", "pre", trainer.ds_p), ("contrast", "post", trainer.ds_c)):
        dc, zf = export_trainer_volume(trainer, channel)
        candidate = dc.detach().cpu().numpy().astype(np.complex64)
        zf_trainer = zf.detach().cpu()
        del dc, zf
        pair = Pair(scan, torch.device("cpu"))
        factor = float(dataset.normalization_scale) / pair.scale
        ratio = float(pair.sense_adj(pair.y).abs().norm() / zf_trainer.abs().norm())
        zf_error = abs(ratio / factor - 1.)
        if zf_error > 1e-3:
            raise ValueError(f"[{scan}] ZF norm ratio {ratio} != scale factor {factor}: wrong data/gauge/FFT")
        candidate = (candidate * np.float32(factor)).astype(np.complex64)
        path = run_dir / f"candidate_{scan}_complex.npy"
        np.save(path, candidate)
        # Exactly score_volume.py --gpu -1: avoid float32 CPU norm drift.
        pair.gt_c = pair.gt_c.to(torch.complex128)
        pair.gt = pair.gt.to(torch.float64)
        x = torch.from_numpy(candidate).to(torch.complex128)
        rows[scan] = pair.score(x, METHOD, seconds=elapsed)
        reference_path = TOPIC / f"logs/{METHOD}_{scan}.json"
        with reference_path.open() as stream:
            reference = next(r for r in json.load(stream)["rows"] if r["method"] == METHOD)
        reference_array_path = TOPIC / f"recon/{METHOD}_{scan}_complex.npy"
        reference_array = torch.from_numpy(np.load(reference_array_path)).to(torch.complex128)
        if tuple(reference_array.shape) != tuple(x.shape):
            raise ValueError(f"[{scan}] registered reconstruction has a different shape")
        check = audit_metric_check(rows[scan], reference)
        checks[scan] = {**check, "required_for_this_mode": mode == "audit",
                        "source_json": str(reference_path), "source_json_sha256": sha256_file(reference_path),
                        "source_array": str(reference_array_path),
                        "source_array_sha256": sha256_file(reference_array_path),
                        "source_array_relative_l2": float((x - reference_array).norm() / reference_array.norm()),
                        "scale_trainer": float(dataset.normalization_scale), "scale_gauge": pair.scale,
                        "factor_scale": factor, "zf_norm_ratio": ratio, "zf_ratio_relative_error": zf_error,
                        "zf_ratio_tolerance": 1e-3}
        (run_dir / f"scores_{scan}.json").write_text(json.dumps({"contrast": scan, "rows": [rows[scan]]}, indent=2) + "\n")
        print(f"[{scan}] mode={mode} head_psnr={rows[scan]['head_psnr']:.6f}; audit_reference_pass={check['passed']}", flush=True)
        del pair, x, candidate, reference_array, zf_trainer
    return rows, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "all", "density"), default="all")
    parser.add_argument("--gpu", type=int, required=True, help="physical PCI-order GPU; process uses logical cuda:0")
    parser.add_argument("--exp", required=True)
    parser.add_argument("--steps", type=int, default=500, help="additional full forward/backward evaluations (1..500); audit uses zero")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs", help="fresh experiment parent inside this campaign")
    parser.add_argument("--local-factor", type=float, default=1., help="factor for original TV and both wavelet strengths only")
    parser.add_argument("--nlambda", type=float, default=0., help="lagged magnitude RED coefficient for each scan")
    parser.add_argument("--sigma-rel", type=float, default=.015, help="sigma relative to each current magnitude head P99")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("gpu must be a nonnegative physical index")
    try:
        cfg = build_config(args.mode, args.steps, Path(args.output).expanduser().resolve() / args.exp,
                           local_factor=args.local_factor, nlambda=args.nlambda, sigma_rel=args.sigma_rel)
        run_dir = reserve_run(args.output, args.exp)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    (run_dir / "actual_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(args.gpu),
                      RETRO_TOPIC=str(TOPIC), RETRO_TRAINER=str(TRAINER), PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    for key, relative in (("TMPDIR", "tmp"), ("TORCH_EXTENSIONS_DIR", "cache/torch"),
                          ("CUPY_CACHE_DIR", "cache/cupy"), ("CUDA_CACHE_PATH", "cache/cuda"),
                          ("MPLCONFIGDIR", "cache/matplotlib"), ("XDG_CACHE_HOME", "cache/xdg")):
        path = run_dir / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.chdir(TRAINER)
    sys.path[:0] = [str(TRAINER / "paired_gs"), str(TRAINER)]
    start = time.perf_counter()
    provenance = {"status": "starting", "mode": args.mode, "exp": args.exp,
                  "physical_gpu": args.gpu, "logical_device": "cuda:0", "argv": sys.argv[1:],
                  "source_iteration": SOURCE_ITERATION,
                  "end_iteration": cfg["training"]["max_iterations"],
                  "step_budget": 0 if args.mode == "audit" else args.steps,
                  "source_checkpoint": str(CHECKPOINT), "source_config": str(BASE_CONFIG),
                  "isolated_trainer": str(TRAINER),
                  "overrides": cfg["campaign_overrides"],
                  "normalization_protocol": "Original retrospective scalar normalization retained; this is not a newly constructed strict y-only pipeline. Pair ground truth is scoring only.",
                  "export_protocol": "arc/cycle.py raw -> original masked-kspace hard DC -> coil combine; dataset.normalization_scale / Pair.scale",
                  "checkpoint_selection": "current latest state, never best.pth"}
    provenance_path = run_dir / "provenance.json"
    with (run_dir / "runner.log").open("w", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                import torch
                # Claim the selected card before reading checkpoint/data or importing the trainer.
                gpu_context_hold = torch.zeros(1, device="cuda:0")
                provenance.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                                  device_name=torch.cuda.get_device_name(0),
                                  source_checkpoint_sha256=sha256_file(CHECKPOINT),
                                  source_config_sha256=sha256_file(BASE_CONFIG),
                                  runner_sha256=sha256_file(Path(__file__)))
                manifest, delta = source_copy_record()
                (run_dir / "source_to_copy_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                (run_dir / "source_to_copy.diff").write_text(delta)
                checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
                provenance["source_model_shapes_before_restore"] = {name: list(t.shape) for name, t in checkpoint["model_state"].items()}
                if checkpoint.get("iter") != SOURCE_ITERATION or any(t.shape[0] != 3600000 for t in checkpoint["model_state"].values()):
                    raise ValueError("not the registered 4000-step 3.6M-point checkpoint")
                provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
                from seed import set_seed
                from trainers.joint_trainer import JointTrainer
                from trainers.resume_protocol import restore_checkpoint, install_update_mask
                set_seed(42, deterministic=False)
                trainer = JointTrainer(cfg, device=torch.device("cuda", 0))
                provenance["resume"] = restore_checkpoint(trainer.model, trainer.optimizer, trainer.scheduler,
                                                          checkpoint, expected_iteration=SOURCE_ITERATION)
                del checkpoint
                provenance["before"] = state_snapshot(trainer)
                provenance["tf32"] = {"matmul": torch.backends.cuda.matmul.allow_tf32,
                                      "cudnn": torch.backends.cudnn.allow_tf32,
                                      "cudnn_benchmark": torch.backends.cudnn.benchmark}
                provenance["status"] = "restored"
                provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
                print(f"mode={args.mode} physical_gpu={args.gpu}->cuda:0; source={CHECKPOINT}", flush=True)
                with (run_dir / "step_trace.jsonl").open("w", buffering=1) as stream:
                    trace = StepTrace(trainer, args.mode, stream)
                    hook = install_update_mask(args.mode, trainer.model, trainer.optimizer) if args.mode != "audit" else None
                    torch.cuda.synchronize()
                    training_started = time.perf_counter()
                    if args.mode != "audit":
                        trainer.train(step_callback=trace)
                    elif trainer.writer is not None:
                        trainer.writer.close()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - training_started if args.mode != "audit" else 0.
                    if hook is not None:
                        hook.remove()
                if trace.count != provenance["step_budget"] or trace.last_iteration != provenance["end_iteration"]:
                    raise RuntimeError("observed forward/backward budget or final absolute iteration differs from request")
                checkpoint_path = trainer.out_dir / "checkpoints/latest.pth"
                if args.mode == "audit":
                    checkpoint_path.parent.mkdir(exist_ok=True)
                    torch.save({"iter": SOURCE_ITERATION, "model_state": trainer.model.state_dict(),
                                "optim_state": trainer.optimizer.state_dict()}, checkpoint_path)
                if not checkpoint_path.is_file():
                    raise RuntimeError("absolute final latest checkpoint was not saved")
                after = state_snapshot(trainer)
                before = provenance["before"]
                invariant = {"geometry_parameters_unchanged": before["geometry_sha256"] == after["geometry_sha256"],
                             "geometry_adam_state_unchanged": all(before["adam"][n] == after["adam"][n] for n in NAMES[:3]),
                             "density_parameters_changed": {n: before["parameter_sha256"][n] != after["parameter_sha256"][n] for n in NAMES[3:]},
                             "model_unchanged": before["model_sha256"] == after["model_sha256"]}
                provenance.update(after=after, invariants=invariant, forward_backward_evaluations=trace.count,
                                  training_elapsed_seconds=elapsed, summed_synchronized_step_seconds=trace.training_step_seconds,
                                  nlr=getattr(trainer, "nlr_summary", {"enabled": False, "audit_only": True}),
                                  cuda_memory_bytes={"peak_allocated": torch.cuda.max_memory_allocated(),
                                                     "peak_reserved": torch.cuda.max_memory_reserved()},
                                  final_checkpoint=str(checkpoint_path))
                provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
                if args.mode == "density" and (not invariant["geometry_parameters_unchanged"]
                        or not invariant["geometry_adam_state_unchanged"]
                        or not all(invariant["density_parameters_changed"].values())):
                    raise RuntimeError("density mode did not preserve geometry/value moments or update all four density tensors")
                if args.mode == "audit" and not invariant["model_unchanged"]:
                    raise RuntimeError("zero-step audit changed model tensors")
                rows, checks = export_and_score(trainer, run_dir, args.mode, elapsed)
                metrics = {"mode": args.mode, "source_iteration": SOURCE_ITERATION,
                           "end_iteration": trace.last_iteration, "forward_backward_evaluations": trace.count,
                           "training_elapsed_seconds": elapsed, "rows": rows, "audit_checks": checks,
                           "before": before, "after": after, "invariants": invariant,
                           "nlr": provenance["nlr"], "overrides": cfg["campaign_overrides"],
                           "source_checkpoint_sha256": provenance["source_checkpoint_sha256"],
                           "cuda_memory_bytes": provenance["cuda_memory_bytes"],
                           "final_checkpoint": str(checkpoint_path), "provenance": str(provenance_path)}
                (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
                if args.mode == "audit" and not all(check["passed"] for check in checks.values()):
                    raise RuntimeError("zero-step registered-output audit failed; inspect metrics.json; do not widen tolerances")
                provenance.update(status="complete", total_elapsed_seconds=time.perf_counter() - start)
                provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
            except Exception as exc:
                provenance.update(status="failed", error=repr(exc), total_elapsed_seconds=time.perf_counter() - start)
                provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
                traceback.print_exc()
                raise
    print(f"Saved {args.mode} continuation outputs: {run_dir}")


if __name__ == "__main__":
    main()
