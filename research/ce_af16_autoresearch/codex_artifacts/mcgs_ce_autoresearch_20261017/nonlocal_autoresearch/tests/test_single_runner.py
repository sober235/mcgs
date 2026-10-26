"""CPU acceptance checks for copied, matched-weight fair-single controls."""
import copy
from dataclasses import replace
from importlib import util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_wrapper():
    path = ROOT / "single_runner.py"
    if not path.exists():
        pytest.fail("fair single wrapper is not implemented")
    spec = util.spec_from_file_location("single_runner_test_module", path)
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("scan,method", [("pre", "ar_s_pre_wf00v"), ("post", "ar_s_post_wf00bv")])
def test_source_whitelist_and_historical_plain_export(scan, method):
    wrapper = load_wrapper()
    source = wrapper.source_for_scan(scan)
    assert source.method == method
    assert source.checkpoint == wrapper.SOURCE_TRAINER / "results_arc" / method / method / "checkpoints/latest.pth"
    assert source.config == source.checkpoint.parent.parent / "config.yaml"
    assert source.reference_array == wrapper.TOPIC / f"recon/{method}_{scan}_complex.npy"
    assert source.reference_json == wrapper.TOPIC / f"logs/{method}_{scan}.json"
    assert source.channel == "plain"
    wrapper.validate_source(source)
    with pytest.raises(ValueError, match="whitelist"):
        wrapper.validate_source(replace(source, checkpoint=Path("/tmp/unregistered.pth")))
    with pytest.raises(ValueError, match="whitelist"):
        wrapper.validate_source(replace(source, reference_array=Path("/tmp/other.npy")))


@pytest.mark.parametrize("scan,wavelet", [("pre", .12), ("post", .1)])
def test_matching_weights_and_distinct_internal_names_preserve_other_recipe(scan, wavelet, tmp_path):
    wrapper = load_wrapper()
    runner = wrapper.configured_runner(scan)
    original = yaml.safe_load(runner.BASE_CONFIG.read_text())
    cfg = runner.build_config("all", 500, tmp_path, local_factor=.4, nlambda=.03, sigma_rel=.02)
    assert cfg["training"]["resume_offset"] == 4000
    assert cfg["training"]["max_iterations"] == 4500
    assert cfg["loss"]["tv_weight"] == pytest.approx(.04)
    for key in ("lambda_wav", "lambda_wav_plain", "lambda_wav_contrast"):
        assert cfg["loss"][key] == pytest.approx(wavelet * .4)
        record = cfg["campaign_overrides"]["changes"][f"loss.{key}"]
        assert record["effective"] == cfg["loss"][key]
        assert record["source_present"] == (key in original["loss"])
        assert record["source"] == original["loss"].get(key)
    for channel in ("plain", "contrast"):
        assert cfg["data"][channel]["name"] == f"{scan}_{channel}"
        unchanged = copy.deepcopy(cfg["data"][channel])
        unchanged["name"] = scan
        assert unchanged == original["data"][channel]
    assert cfg["data"]["plain"]["data_path"] == cfg["data"]["contrast"]["data_path"]
    allowed = {"tv_weight", "lambda_wav", "lambda_wav_plain", "lambda_wav_contrast"}
    for key, value in original["loss"].items():
        if key not in allowed:
            assert cfg["loss"][key] == value
    assert cfg["loss"]["lambda_nl_plain"] == cfg["loss"]["lambda_nl_contrast"] == .03
    assert cfg["loss"]["nl_start"] == 4001 and cfg["loss"]["nl_refresh"] == 250
    control = cfg["campaign_overrides"]["single_control"]
    assert control["scan"] == scan and control["export_channel"] == "plain"
    assert control["joint_scan_wavelet_base"] == wavelet
    assert len(control["wrapper_sha256"]) == 64
    assert "ground_truth" not in cfg["loss"] and "reference_array" not in cfg["loss"]
    cfg2 = wrapper.configured_runner("post" if scan == "pre" else "pre").build_config("audit", 500, tmp_path)
    assert cfg2["training"]["max_iterations"] == 4000
    assert runner.METHOD == wrapper.source_for_scan(scan).method


@pytest.mark.parametrize("scan", ["joint", "PRE", "../post", "/tmp/custom"])
def test_unregistered_scan_is_rejected(scan):
    with pytest.raises(ValueError):
        load_wrapper().source_for_scan(scan)


def test_density_mode_is_rejected_and_cli_requires_scan(tmp_path):
    wrapper = load_wrapper()
    with pytest.raises(ValueError):
        wrapper.configured_runner("pre").build_config("density", 500, tmp_path)
    with pytest.raises(SystemExit):
        wrapper.parse_args(["--mode", "all", "--gpu", "0", "--exp", "test"])
    with pytest.raises(SystemExit):
        wrapper.parse_args(["--scan", "pre", "--mode", "density", "--gpu", "0", "--exp", "test"])


@pytest.mark.parametrize("scan", ["pre", "post"])
def test_export_uses_only_plain_scan_and_original_scale_gauge(scan, tmp_path, monkeypatch):
    wrapper = load_wrapper()
    runner = wrapper.configured_runner(scan)
    source = wrapper.source_for_scan(scan)
    # Toy scoring dependency exercises conversion/exports without loading real scan data.
    class Pair:
        def __init__(self, actual_scan, device):
            if actual_scan != scan:
                raise AssertionError("scoring another scan")
            self.scale = 2.0
            self.y = torch.full((2, 2, 2), 2.0, dtype=torch.complex64)
            self.gt = torch.ones((2, 2, 2))
            self.gt_c = self.gt.to(torch.complex64)
        def sense_adj(self, y):
            return y
        def score(self, x, method, seconds):
            return {"method": method, "head_psnr": float(x.abs().mean())}
    monkeypatch.setitem(sys.modules, "t1t2_common", SimpleNamespace(Pair=Pair))
    def exported(trainer, channel):
        if channel != "plain":
            raise AssertionError("single control exported an unintended head")
        return torch.full((2, 2, 2), 3.0, dtype=torch.complex64), torch.ones((2, 2, 2), dtype=torch.complex64)
    monkeypatch.setattr(runner, "export_trainer_volume", exported)
    monkeypatch.setattr(runner, "audit_metric_check", lambda row, ref: {"passed": True})
    # Redirect only scoring references for the controlled fixture; source selection remains whitelisted.
    tiny_reference = tmp_path / "reference.npy"
    np.save(tiny_reference, np.full((2, 2, 2), 6.0, np.complex64))
    tiny_json = tmp_path / "reference.json"
    tiny_json.write_text(json.dumps({"rows": [{"method": source.method, "head_psnr": 6.0}]}))
    reference = replace(source, reference_array=tiny_reference, reference_json=tiny_json)
    trainer = SimpleNamespace(ds_p=SimpleNamespace(normalization_scale=4.0))
    rows, checks = wrapper.export_single_and_score(runner, reference, trainer, tmp_path, "audit", 0.)
    assert list(rows) == list(checks) == [scan]
    np.testing.assert_array_equal(np.load(tmp_path / f"candidate_{scan}_complex.npy"), np.full((2, 2, 2), 6., np.complex64))
    assert rows[scan]["head_psnr"] == 6.0
    assert checks[scan]["source_array_relative_l2"] == 0
    assert checks[scan]["factor_scale"] == 2.0
    assert not (tmp_path / f"candidate_{'post' if scan == 'pre' else 'pre'}_complex.npy").exists()


def valid_observations(scan="pre", steps=500):
    names = [f"{scan}_plain", f"{scan}_contrast"]
    iterations = list(range(4001, 4001 + steps))
    refreshes = [it for it in iterations if (it - 4001) % 250 == 0]
    metrics = {"forward_backward_evaluations": steps, "end_iteration": 4000 + steps,
               "rows": {scan: {}}, "audit_checks": {scan: {"passed": True}},
               "invariants": {"model_unchanged": False},
               "nlr": {"enabled": True, "validated_active": True, "active_step_count": steps,
                       "all_gradients_finite": True,
                       "refresh_history": {name: [{"iter": it, "target_relative_norm": .01, "target_rms_difference": .01,
                                                    "sigma": .02} for it in refreshes] for name in names}}}
    trace = [{"absolute_iteration": it, "gradients_finite": True, "full_gradient_group_count": 7,
              "nonlocal": {"enabled": True, "weighted": .001, "raw_by_channel": {name: .01 for name in names}}}
             for it in iterations]
    return metrics, trace


def test_observation_gate_requires_real_budget_both_heads_and_finite_gradients():
    wrapper = load_wrapper()
    metrics, trace = valid_observations()
    check = wrapper.validate_observations(metrics, trace, "pre", "all", 500, .03)
    assert check["observed_steps"] == 500
    assert check["expected_refresh_iterations"] == [4001, 4251]
    assert check["internal_heads"] == ["pre_plain", "pre_contrast"]
    for mutation in ("missing_step", "bad_grad", "missing_head", "missing_refresh", "nan_term", "wrong_scan"):
        m, t = copy.deepcopy(metrics), copy.deepcopy(trace)
        if mutation == "missing_step": t.pop()
        elif mutation == "bad_grad": t[0]["gradients_finite"] = False
        elif mutation == "missing_head": del t[0]["nonlocal"]["raw_by_channel"]["pre_contrast"]
        elif mutation == "missing_refresh": m["nlr"]["refresh_history"]["pre_contrast"].pop()
        elif mutation == "nan_term": t[0]["nonlocal"]["weighted"] = float("nan")
        else: m["rows"]["post"] = {}
        with pytest.raises(ValueError):
            wrapper.validate_observations(m, t, "pre", "all", 500, .03)


def test_audit_requires_zero_steps_unchanged_model_and_source_comparison():
    wrapper = load_wrapper()
    metrics, _ = valid_observations(steps=0)
    metrics["invariants"]["model_unchanged"] = True
    check = wrapper.validate_observations(metrics, [], "pre", "audit", 500, .03)
    assert check["observed_steps"] == 0 and check["expected_refresh_iterations"] == []
    metrics["audit_checks"]["pre"]["passed"] = False
    with pytest.raises(ValueError):
        wrapper.validate_observations(metrics, [], "pre", "audit", 500, .03)


@pytest.mark.parametrize("scan", ["pre", "post"])
def test_actual_500_cpu_steps_keep_both_internal_head_histories_separate(scan, tmp_path, monkeypatch):
    import io
    # Reuse the existing actual-trainer AST harness, excluding CUDA renderer imports.
    path = ROOT / "tests/test_integration.py"
    spec = util.spec_from_file_location("single_control_training_harness", path)
    harness = util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    from losses import nonlocal_red
    monkeypatch.setattr(nonlocal_red, "make_denoiser", lambda *args, **kwargs: lambda z, sigma: z * .9)
    wrapper = load_wrapper()
    runner = wrapper.configured_runner(scan)
    cfg = runner.build_config("all", 500, tmp_path, nlambda=.03)
    trainer = harness.tiny_trainer(tmp_path / scan,
                                  {key: cfg["loss"][key] for key in ("lambda_nl", "lambda_nl_plain", "lambda_nl_contrast", "nl_start", "nl_refresh")},
                                  max_iterations=4500)
    trainer.name_plain = cfg["data"]["plain"]["name"]
    trainer.name_contrast = cfg["data"]["contrast"]["name"]
    trainer.optimizer = torch.optim.Adam([{"params": [parameter], "lr": .001} for parameter in trainer.model.parameters()])
    stream = io.StringIO()
    observer = runner.StepTrace(trainer, "all", stream)
    harness.train_function(ROOT / "trainer/paired_gs/trainers/joint_trainer.py")(trainer, observer)
    trace = [json.loads(line) for line in stream.getvalue().splitlines()]
    metrics = {"forward_backward_evaluations": observer.count, "end_iteration": observer.last_iteration,
               "rows": {scan: {}}, "audit_checks": {scan: {"passed": True}},
               "invariants": {"model_unchanged": False}, "nlr": trainer.nlr_summary}
    result = wrapper.validate_observations(metrics, trace, scan, "all", 500, .03)
    assert result["observed_steps"] == 500
    for name in (f"{scan}_plain", f"{scan}_contrast"):
        assert [row["iter"] for row in trainer.nlr_summary["refresh_history"][name]] == [4001, 4251]
        assert trainer.nlr_summary["raw_R_nl_by_channel_max"][name] > 0
    assert all(row["gradients_finite"] for row in trace)


@pytest.mark.parametrize("bad_observation", [False, True])
def test_wrapper_records_replayable_scan_cli_and_refuses_invalid_completed_run(tmp_path, monkeypatch, bad_observation):
    wrapper = load_wrapper()
    runner = wrapper.configured_runner("pre")
    run_dir = tmp_path / "toy"
    metrics, trace = valid_observations(steps=3)
    if bad_observation:
        trace.pop()
    def cpu_only_main():
        run_dir.mkdir()
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
        (run_dir / "step_trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace))
        (run_dir / "provenance.json").write_text(json.dumps({"status": "complete", "source_checkpoint_sha256": "a" * 64,
                                                             "source_config_sha256": "b" * 64}))
    runner.main = cpu_only_main
    monkeypatch.setattr(wrapper, "configured_runner", lambda scan: runner)
    argv = ["--scan", "pre", "--gpu", "0", "--exp", "toy", "--steps", "3", "--nlambda", ".03", "--output", str(tmp_path)]
    if bad_observation:
        with pytest.raises(ValueError):
            wrapper.main(argv)
        provenance = json.loads((run_dir / "provenance.json").read_text())
        assert provenance["status"] == "failed"
        assert "budget" in provenance["single_control_error"]
    else:
        wrapper.main(argv)
        provenance = json.loads((run_dir / "provenance.json").read_text())
        assert provenance["argv"][:2] == ["--scan", "pre"]
        assert provenance["single_control_validation"]["source_checkpoint_sha256"] == "a" * 64
        assert provenance["single_control_validation"]["observed_steps"] == 3
        assert (run_dir / "single_control_validation.json").is_file()
