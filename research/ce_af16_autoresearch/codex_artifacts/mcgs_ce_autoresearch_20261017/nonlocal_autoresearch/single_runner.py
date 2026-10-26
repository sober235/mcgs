#!/usr/bin/env python3
"""Matched scan-internal NLR continuations of the two frozen fair-single arms.

Each historical single arm is a seven-group, two-head model fed the same scan
in both channels. The historical export is the plain head for both pre and post.
This wrapper changes only source selection, documented continuation weights,
internal display/target names, and single-scan export; runner.py remains intact.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import util
import json
import math
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent
TOPIC = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois")
SOURCE_TRAINER = TOPIC / "trainer/_trainer_e778bf2_ar"
METHODS = {"pre": "ar_s_pre_wf00v", "post": "ar_s_post_wf00bv"}


@dataclass(frozen=True)
class SingleSource:
    scan: str
    method: str
    checkpoint: Path
    config: Path
    reference_array: Path
    reference_json: Path
    channel: str = "plain"


def source_for_scan(scan):
    if scan not in METHODS:
        raise ValueError("single source whitelist contains only pre and post")
    method = METHODS[scan]
    run = SOURCE_TRAINER / "results_arc" / method / method
    return SingleSource(scan, method, run / "checkpoints/latest.pth", run / "config.yaml",
                        TOPIC / f"recon/{method}_{scan}_complex.npy",
                        TOPIC / f"logs/{method}_{scan}.json")


def validate_source(source):
    expected = source_for_scan(source.scan)
    if source != expected:
        raise ValueError("source paths/method/export channel differ from the frozen single whitelist")
    for path in (source.checkpoint, source.config, source.reference_array, source.reference_json):
        if not path.is_file() or path.resolve() != path:
            raise ValueError(f"whitelist source is missing or redirected: {path}")


def configured_runner(scan):
    """Load a private runner module; no Torch import, CUDA access, or file mutation."""
    source = source_for_scan(scan)
    validate_source(source)
    spec = util.spec_from_file_location(f"_fair_single_runner_{scan}", ROOT / "runner.py")
    runner = util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.SOURCE_RUN = source.checkpoint.parent.parent
    runner.CHECKPOINT, runner.BASE_CONFIG, runner.METHOD = source.checkpoint, source.config, source.method
    original_builder = runner.build_config

    def build_config(mode, steps, run_dir, *, local_factor=1., nlambda=0., sigma_rel=.015):
        if mode not in ("audit", "all"):
            raise ValueError("single control mode must be audit or all")
        cfg = original_builder(mode, steps, run_dir, local_factor=local_factor,
                               nlambda=nlambda, sigma_rel=sigma_rel)
        original = yaml.safe_load(source.config.read_text())
        for channel in ("plain", "contrast"):
            data = cfg["data"][channel]
            expected = {"data_path": str(TOPIC / f"sim/gs/{scan}_kspace.h5"),
                        "csm_path": str(TOPIC / f"sim/gs/{scan}_csm.h5"),
                        "mask_path": str(TOPIC / "sim/mask/mask_af16_3d.h5")}
            if any(data.get(key) != value for key, value in expected.items()):
                raise ValueError("single control must duplicate only its registered AF16 acquisition")
            cfg["campaign_overrides"]["changes"][f"data.{channel}.name"] = {
                "source_present": "name" in data, "source": data.get("name"),
                "effective": f"{scan}_{channel}"}
            data["name"] = f"{scan}_{channel}"
        # Match the corresponding joint scan coefficient in BOTH same-scan heads.
        # Explicit per-head keys prevent inheritance from hiding an unequal weight.
        wavelet_base = .12 if scan == "pre" else .1
        for key in ("lambda_wav", "lambda_wav_plain", "lambda_wav_contrast"):
            value = wavelet_base * float(local_factor)
            record = {"source_present": key in original["loss"],
                      "source": original["loss"].get(key), "effective": value}
            if key == "lambda_wav_contrast" and key not in original["loss"]:
                record["inherited_source"] = original["loss"]["lambda_wav"]
            cfg["loss"][key] = value
            cfg["campaign_overrides"]["changes"][f"loss.{key}"] = record
        regularization = cfg["campaign_overrides"]["local_regularization"]
        for channel in ("plain", "contrast"):
            key = f"lambda_wav_{channel}"
            regularization[f"{channel}_wavelet"] = {
                "source_key": f"loss.{key}",
                **cfg["campaign_overrides"]["changes"][f"loss.{key}"]}
        cfg["campaign_overrides"]["single_control"] = {
            "scan": scan, "method": source.method, "export_channel": source.channel,
            "internal_to_acquisition": {f"{scan}_plain": scan, f"{scan}_contrast": scan},
            "historical_model": "two independently parameterized density heads on the same acquisition; all seven Adam groups active",
            "source_checkpoint": str(source.checkpoint), "source_config": str(source.config),
            "source_reference_array": str(source.reference_array),
            "source_reference_json": str(source.reference_json),
            "joint_scan_wavelet_base": wavelet_base,
            "matched_weights_apply_on_optimizer_steps": mode != "audit",
            "initial_iterations_per_model": 4000,
            "continued_iterations_per_model": 0 if mode == "audit" else steps,
            "full_pre_post_control_requires_two_models": True,
            "wrapper_path": str(Path(__file__).resolve()),
            "wrapper_sha256": runner.sha256_file(Path(__file__))}
        return cfg

    runner.build_config = build_config
    runner.export_and_score = lambda trainer, run_dir, mode, elapsed: export_single_and_score(
        runner, source, trainer, run_dir, mode, elapsed)
    return runner


def export_single_and_score(runner, source, trainer, run_dir, mode, elapsed):
    """Original hard-DC/Pair gauge/scorer, evaluated only for the selected scan.

    Ground truth is loaded by Pair here after optimization, exclusively to score.
    It never enters build_config, a target, a gradient, or checkpoint selection.
    """
    import numpy as np
    import torch
    os.environ["RETRO_TOPIC"] = str(TOPIC)
    os.environ["RETRO_TRAINER"] = str(runner.TRAINER)
    sys.path.insert(0, str(runner.SCRIPTS))
    from t1t2_common import Pair
    scan = source.scan
    dc, zf = runner.export_trainer_volume(trainer, source.channel)
    candidate = dc.detach().cpu().numpy().astype(np.complex64)
    zf_trainer = zf.detach().cpu()
    del dc, zf
    pair = Pair(scan, torch.device("cpu"))
    factor = float(trainer.ds_p.normalization_scale) / pair.scale
    ratio = float(pair.sense_adj(pair.y).abs().norm() / zf_trainer.abs().norm())
    zf_error = abs(ratio / factor - 1.)
    if not math.isfinite(zf_error) or zf_error > 1e-3:
        raise ValueError(f"[{scan}] ZF norm ratio {ratio} != scale factor {factor}: wrong data/gauge/FFT")
    candidate = (candidate * np.float32(factor)).astype(np.complex64)
    np.save(run_dir / f"candidate_{scan}_complex.npy", candidate)
    pair.gt_c = pair.gt_c.to(torch.complex128)
    pair.gt = pair.gt.to(torch.float64)
    x = torch.from_numpy(candidate).to(torch.complex128)
    row = pair.score(x, source.method, seconds=elapsed)
    reference = next(row for row in json.loads(source.reference_json.read_text())["rows"]
                     if row["method"] == source.method)
    reference_array = torch.from_numpy(np.load(source.reference_array)).to(torch.complex128)
    if tuple(reference_array.shape) != tuple(x.shape):
        raise ValueError(f"[{scan}] registered single reconstruction has a different shape")
    check = {**runner.audit_metric_check(row, reference), "required_for_this_mode": mode == "audit",
             "source_json": str(source.reference_json), "source_json_sha256": runner.sha256_file(source.reference_json),
             "source_array": str(source.reference_array), "source_array_sha256": runner.sha256_file(source.reference_array),
             "source_array_relative_l2": float((x - reference_array).norm() / reference_array.norm()),
             "export_channel": source.channel, "scan": scan,
             "scale_trainer": float(trainer.ds_p.normalization_scale), "scale_gauge": pair.scale,
             "factor_scale": factor, "zf_norm_ratio": ratio, "zf_ratio_relative_error": zf_error,
             "zf_ratio_tolerance": 1e-3}
    (run_dir / f"scores_{scan}.json").write_text(json.dumps({"contrast": scan, "rows": [row]}, indent=2, allow_nan=False) + "\n")
    print(f"[{scan}] single={source.method} channel=plain mode={mode} head_psnr={row['head_psnr']:.6f}; audit_reference_pass={check['passed']}", flush=True)
    return {scan: row}, {scan: check}


def validate_observations(metrics, trace, scan, mode, steps, nlambda):
    """Refuse a completed control lacking its observed budget or per-head prior."""
    source_for_scan(scan)
    if mode not in ("audit", "all"):
        raise ValueError("single control mode must be audit or all")
    expected_steps = 0 if mode == "audit" else steps
    expected_iterations = list(range(4001, 4001 + expected_steps))
    if (metrics["forward_backward_evaluations"] != expected_steps
            or metrics["end_iteration"] != 4000 + expected_steps
            or [row["absolute_iteration"] for row in trace] != expected_iterations):
        raise ValueError("single control observed continuation budget differs from its request")
    if set(metrics["rows"]) != {scan} or set(metrics["audit_checks"]) != {scan}:
        raise ValueError("single control must contain only its selected scan")
    heads = [f"{scan}_plain", f"{scan}_contrast"]
    refreshes = [it for it in expected_iterations if (it - 4001) % 250 == 0]
    if mode == "audit":
        if not metrics["invariants"]["model_unchanged"] or not metrics["audit_checks"][scan]["passed"]:
            raise ValueError("zero-step single audit changed tensors or failed the registered reference")
    else:
        for row in trace:
            if row.get("gradients_finite") is not True or row.get("full_gradient_group_count") != 7:
                raise ValueError("single control requires finite gradients for all seven groups")
        if nlambda > 0:
            summary = metrics["nlr"]
            if (not summary.get("enabled") or not summary.get("validated_active")
                    or summary.get("active_step_count") != expected_steps
                    or summary.get("all_gradients_finite") is not True):
                raise ValueError("single control has no fully active finite NLR continuation")
            for name in heads:
                history = summary.get("refresh_history", {}).get(name, [])
                if [row["iter"] for row in history] != refreshes:
                    raise ValueError(f"single control missing expected refreshes for {name}")
                for observation in history:
                    for key in ("sigma", "target_relative_norm", "target_rms_difference"):
                        value = observation[key]
                        if not math.isfinite(value) or value < 0:
                            raise ValueError(f"single control nonfinite target observation {name}.{key}")
            for row in trace:
                nl = row.get("nonlocal", {})
                raw = nl.get("raw_by_channel", {})
                if (not nl.get("enabled") or set(raw) != set(heads)
                        or not math.isfinite(nl.get("weighted", float("nan")))
                        or any(not math.isfinite(raw[name]) or raw[name] < 0 for name in heads)):
                    raise ValueError("single control lacks finite distinct per-head NLR terms")
    return {"passed": True, "scan": scan, "export_channel": "plain", "observed_steps": expected_steps,
            "internal_heads": heads, "expected_refresh_iterations": refreshes if nlambda > 0 else [],
            "nlambda": nlambda, "source_iteration": 4000, "end_iteration": 4000 + expected_steps}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", choices=("pre", "post"), required=True)
    parser.add_argument("--mode", choices=("audit", "all"), default="all")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--local-factor", type=float, default=1.)
    parser.add_argument("--nlambda", type=float, default=0.)
    parser.add_argument("--sigma-rel", type=float, default=.015)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    runner = configured_runner(args.scan)
    run_dir = args.output.expanduser().resolve() / args.exp
    forwarded = ["--mode", args.mode, "--gpu", str(args.gpu), "--exp", args.exp,
                 "--steps", str(args.steps), "--output", str(args.output),
                 "--local-factor", str(args.local_factor), "--nlambda", str(args.nlambda),
                 "--sigma-rel", str(args.sigma_rel)]
    previous_argv = sys.argv
    try:
        sys.argv = [str(ROOT / "single_runner.py"), *forwarded]
        runner.main()
    finally:
        sys.argv = previous_argv
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    # The delegated parser consumes the arguments without --scan; restore the
    # complete wrapper invocation so the recorded command can be replayed.
    provenance["argv"] = ["--scan", args.scan, *forwarded]
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text())
        trace = [json.loads(line) for line in (run_dir / "step_trace.jsonl").read_text().splitlines() if line.strip()]
        validation = validate_observations(metrics, trace, args.scan, args.mode, args.steps, args.nlambda)
        validation.update(source_checkpoint_sha256=provenance["source_checkpoint_sha256"],
                          source_config_sha256=provenance["source_config_sha256"],
                          wrapper_sha256=runner.sha256_file(Path(__file__)),
                          source_to_copy_manifest=str(run_dir / "source_to_copy_manifest.json"))
        (run_dir / "single_control_validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n")
        provenance["single_control_validation"] = validation
        provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        provenance.update(status="failed", single_control_error=repr(error))
        provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
        raise


if __name__ == "__main__":
    main()
