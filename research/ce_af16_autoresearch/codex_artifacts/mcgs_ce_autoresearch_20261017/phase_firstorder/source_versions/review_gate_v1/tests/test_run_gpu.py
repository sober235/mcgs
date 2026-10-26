"""CPU launch-boundary tests. No GPU resources or clinical inputs are queried."""
import copy
import fcntl
import json
import os
from pathlib import Path
import types
import hashlib
import numpy as np

import pytest

import run_gpu as launch


def put(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))


def card(index, *, used=0, utilization=0, loading=(), compute=()):
    return dict(gpu=index, uuid=f"GPU-{index}", used_mib=used, total_mib=81920,
                utilization=utilization, compute_pids=list(compute), loading_claim_pids=list(loading),
                free=used < 100 and utilization == 0 and not loading and not compute)


def test_real_source_selector_prefers_idle_respects_visibility_and_claims(monkeypatch):
    campaign = launch.load_campaign()
    cards = [card(0, used=10000, utilization=10, compute=(123,)), card(1), card(2, loading=(456,))]
    monkeypatch.setattr(campaign, "gpu_snapshot", lambda: copy.deepcopy(cards))
    assert launch.select_resource(campaign, preferred=0, environment={})["gpu"] == 1
    assert launch.select_resource(campaign, preferred=1, environment={"CUDA_VISIBLE_DEVICES": "0"})["gpu"] == 0
    with pytest.raises(ValueError):
        launch.select_resource(campaign, preferred=2, environment={"CUDA_VISIBLE_DEVICES": "2"})


def test_claim_lock_excludes_second_claim_and_releases_on_exception(tmp_path):
    path = tmp_path / "global.lock"
    with pytest.raises(RuntimeError):
        with launch.claim_lock(path) as descriptor:
            assert os.get_inheritable(descriptor)
            with path.open("a") as other:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise RuntimeError("launch failed")
    with launch.claim_lock(path):
        pass


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    root = tmp_path / "frequency_mm"
    verify = root / "verification"
    verify.mkdir(parents=True)
    hashes = {str(root / name): "a" * 64 for name in launch.PRODUCTION}
    invariants = {"frozen_definitions_sha256": {"input_definition": "d" * 64, "source_definition": "e" * 64},
                  "input_manifest": {"files": {"checkpoint": "c" * 64}, "source_checkpoint": "checkpoint"},
                  "campaign_sources": {"gated_code_sha256": {}, "production_code_sha256": {}},
                  "prebuilt_extension": {"sha256": "b" * 64}}
    review = verify / "review.json"
    put(review, dict(status="pass", actionable_findings=0, unresolved_findings=0, code_sha256=hashes,
                     frozen_definitions_sha256=invariants["frozen_definitions_sha256"],invariants=invariants))
    monkeypatch.setattr(launch, "ROOT", root)
    monkeypatch.setattr(launch, "VERIFY", verify)
    monkeypatch.setattr(launch, "CLAIM_LOCK", tmp_path / "claim.lock")
    monkeypatch.setattr(launch, "source_hashes", lambda: copy.deepcopy(hashes))
    monkeypatch.setattr(launch, "snapshot_invariants", lambda: copy.deepcopy(invariants))
    monkeypatch.setattr(launch,"parent_runtime",lambda:{"python":str(launch.CPU_PYTHON),"python_version":[3,11,15],
                                                       "torch_imported":False,"weights_imported":False})
    campaign = types.SimpleNamespace(gpu_snapshot=lambda: [card(3)],
                   select_gpu=launch.load_campaign().select_gpu)
    monkeypatch.setattr(launch, "load_campaign", lambda: campaign)
    monkeypatch.setattr(launch, "initial_visibility", lambda: "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    args = launch.parser().parse_args(["--stage", "tiny", "--exp", "trial", "--review-gate", str(review)])
    return types.SimpleNamespace(root=root, verify=verify, hashes=hashes, invariants=invariants,
                                 review=review, args=args)


def own_selection(fixture):
    args = fixture.args
    args.reserved_gpu = 3
    args.reservation_token = "owned-token"
    descriptor = os.open(launch.CLAIM_LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    args.claim_fd = descriptor
    value = {"parent_pid": os.getpid(), "process_start": launch.process_start(),
             "reservation_token": args.reservation_token, "resource": card(3),
             "stage": args.stage, "request": launch._request(args),
             "code_sha256": fixture.hashes, "invariants": fixture.invariants}
    put(fixture.verify / "trial_selection.json", value)
    return value


def frequency_fixture(output,t=0.):
    folder=Path(output)/"frequency_weights";folder.mkdir()
    scans={}
    for scan in ("pre","post"):
        source=np.array([.3,.3,1.,1.,1.,2.2 if scan == "pre" else 2.8,2.2 if scan == "pre" else 2.8,
                         2.2 if scan == "pre" else 2.8],dtype=np.float32).reshape(2,2,2)
        mask=np.array([1,0,1,0,0,1,0,0],dtype=np.float32).reshape(2,2,2)
        mass=float(np.mean(source[mask.astype(bool)].astype(np.float64)**2));c=mass**.5
        mixed=(1.-t)*source.astype(np.float64)+t*c
        effective=source.copy() if t == 0. else np.full_like(source,c) if t == 1. else (
            mixed*c/np.mean(mixed[mask.astype(bool)]**2)**.5).astype(source.dtype)
        effective_mass=float(np.mean(effective[mask.astype(bool)].astype(np.float64)**2))
        scans[scan]={"acquired_count":3,"c":c,"source_mean_squared_weight":mass,"effective_mean_squared_weight":effective_mass,
                     "relative_mass_error":abs(effective_mass/mass-1.),"t0_exact_equal":True if t == 0. else None}
        for name in ("source_weight","effective_weight","acquired_mask"):
            value=source if name == "source_weight" else effective if name == "effective_weight" else mask
            path=folder/f"{scan}_{name}.npy"
            np.save(path,value,allow_pickle=False)
            scans[scan][name]={"path":str(path),"file_sha256":launch.sha256(path),"byte_sha256":hashlib.sha256(value.tobytes()).hexdigest(),
                               "shape":[2,2,2],"dtype":"torch.float32","device":"cuda:0"}
    result={"version":1,"frequency_t":t,"mass_relative_tolerance":2e-6,"scans":scans}
    put(folder/"manifest.json",result)
    return result


def fake_worker(monkeypatch, fixture, *, returncode=0, partial=False, invalid_json=False):
    class Worker:
        pid = 4567

        def __init__(self, command, **kwargs):
            self.returncode = None
            self.final_code = returncode
            output = fixture.root / "outputs/trial"
            output.mkdir(parents=True)
            artifact = output / "preflight.json"
            frequency=frequency_fixture(output,fixture.args.frequency_t)
            reference=copy.deepcopy(frequency)
            for row in reference["scans"].values():
                for name in ("source_weight","effective_weight","acquired_mask"):
                    row[name].pop("path");row[name].pop("file_sha256")
            numeric_runtime={"python":str(launch.PYTHON),"torch_version":"2.7.1+cu118","cuda_version":"11.8"}
            array_check={"passed":True,"frequency_t":fixture.args.frequency_t,"mass_relative_tolerance":2e-6,
                "numerical_runtime":numeric_runtime,"scans":{scan:{
                    **{key:row[key] for key in ("acquired_count","c","source_mean_squared_weight","effective_mean_squared_weight","relative_mass_error")},
                    "original_acquired_mask_exact":True,"original_source_weight_exact":True,
                    "interpolation_max_relative_error":0.,"interpolation_relative_tolerance":8.*2.**-23,
                    "effective_identity":fixture.args.frequency_t == 0.,
                    "source_levels":[.3,1.,2.2 if scan == "pre" else 2.8],
                    "acquired_source_levels":[.3,1.,2.2 if scan == "pre" else 2.8]} for scan,row in frequency["scans"].items()}}
            value = {"passed": not partial, "status": "failed" if partial else "pass",
                     **launch.phase_protocol(),
                     "wavelet_mode": "uniform8", "fixed_wavelet_shift": None, "wavelet_phase_count": 8,
                     "local_factor":fixture.args.local_factor,
                     "cg_steps":96,"true_residual_diagnostics":True,
                     "frequency_t":fixture.args.frequency_t,"frequency_evidence":frequency,"reference_frequency_summary":reference,
                     "runtime":numeric_runtime,"frequency_array_check":array_check,
                     "t0_source_parity":{"passed":True,"weights_exact_equal":True,"source_loss_absolute_difference":0.,
                         "zero_loss_absolute_difference":0.,"normal_relative_difference":1e-7,"normal_relative_tolerance":2e-5},
                     "prior_checks": {"passed": True, "wavelet_mode": "uniform8",
                       "local_factor":fixture.args.local_factor,"lambda_tv":[0.,0.],"lambda_phase":[0.,0.],
                       "lambda_wavelet":[.12*fixture.args.local_factor,.1*fixture.args.local_factor],
                       "tested_terms":["wavelet_mean8"],"nonzero_TV_phase_tested":False,
                       "tolerance": 2e-5, "errors": {"value_relative": 1e-7, "gradient_relative": 1e-7,
                       "quadratic_gradient_relative": 1e-7}, "wavelet_transform_batches": 40},
                     "code_sha256": fixture.hashes, "invariants": fixture.invariants,
                     "device": "cuda:0", "tolerance": 2e-5,
                     "errors": dict(superposition=1e-7, adjoint=1e-7, normal_linearity=1e-7, normal_hermitian=1e-7),
                     "normal_curvature": {"real": 1., "imag": 0.},
                     "parameters_unchanged": True,
                     "factory": {"cuda_backend": "prebuilt C tile extension","frequency_bands":{
                         scan:{"fit_only_count":1,"effective_identity":fixture.args.frequency_t == 0.,
                               "bands":[{"weight":weight,"volume_count":count,"acquired_count":1,"fit_count":fit}
                                        for weight,count,fit in ((.3,2,2),(1.,3,1),(2.2 if scan == "pre" else 2.8,3,1))]}
                         for scan in ("pre","post")}}}
            # This unit fixture mocks the child process/device boundary. Its
            # numerical phase evidence is generated by actual CPU source math;
            # it is never an observed CUDA gate or written outside tmp_path.
            from test_phase_protocol import numerical_tiny_fixture
            phase_check=copy.deepcopy(numerical_tiny_fixture())
            phase_check["device"]="cuda:0"
            value["phase_firstorder_checks"]=phase_check
            artifact.write_text("{" if invalid_json else json.dumps(value))

        def poll(self):
            return self.returncode

        def wait(self):
            self.returncode = self.final_code
            return self.returncode

    monkeypatch.setattr(launch.subprocess, "Popen", Worker)


def test_initial_selection_reexec_publishes_exact_source_owned_loading_lease(supervisor, monkeypatch):
    observed = {}

    def intercept(binary, argv, environment):
        observed.update(binary=binary, argv=argv, environment=environment)
        descriptor = int(argv[argv.index("--claim-fd") + 1])
        assert os.get_inheritable(descriptor)
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(launch.os, "execve", intercept)
    with pytest.raises(RuntimeError, match="exec intercepted"):
        launch.execute(supervisor.args, ["--stage", "tiny", "--exp", "trial", "--review-gate", str(supervisor.review)])
    selection = launch.read(supervisor.verify / "trial_selection.json")
    assert selection["parent_pid"] == os.getpid()
    assert selection["code_sha256"] == supervisor.hashes
    assert selection["invariants"] == supervisor.invariants
    assert selection["reservation_token"]
    assert observed["environment"]["CUDA_VISIBLE_DEVICES"] == "3"


def test_successful_child_requires_verified_post_run_invariants(supervisor, monkeypatch):
    own_selection(supervisor)
    fake_worker(monkeypatch, supervisor)
    launch.execute(supervisor.args, [])
    completed = launch.read(supervisor.verify / "trial_completed.json")
    proof = launch.read(supervisor.verify / "trial_post_run_invariants.json")
    assert completed["passed"] is True and completed["invariants"] == supervisor.invariants
    assert proof["status"] == "verified" and proof["before"] == proof["after"]
    # The observed completion is the gate itself; validate artifacts and proof.
    launch.validate_observed_gate(supervisor.verify / "trial_completed.json", "tiny",
                                  supervisor.hashes, supervisor.invariants)


@pytest.mark.parametrize("kind", ["source", "input", "campaign", "checkpoint"])
def test_post_run_changes_cannot_publish_success(supervisor, monkeypatch, kind):
    own_selection(supervisor)
    fake_worker(monkeypatch, supervisor)
    calls = []

    def snapshot():
        calls.append(1)
        value = copy.deepcopy(supervisor.invariants)
        if len(calls) > 1:
            value[kind] = "changed"
        return value

    monkeypatch.setattr(launch, "snapshot_invariants", snapshot)
    with pytest.raises(ValueError, match="changed"):
        launch.execute(supervisor.args, [])
    assert not (supervisor.verify / "trial_completed.json").exists()
    assert launch.read(supervisor.verify / "trial_post_run_invariants.json")["status"] == "failed"


@pytest.mark.parametrize("invalid_json", [False, True])
def test_child_failure_preserves_exit_code_partial_evidence_and_owned_lease(supervisor, monkeypatch, invalid_json):
    own_selection(supervisor)
    fake_worker(monkeypatch, supervisor, returncode=17, partial=True, invalid_json=invalid_json)
    with pytest.raises(launch.subprocess.CalledProcessError) as observed:
        launch.execute(supervisor.args, [])
    launch.record_failure(supervisor.args, observed.value)
    failure = launch.read(supervisor.verify / "trial_failure.json")
    assert failure["returncode"] == 17 and failure["worker_pid"] == 4567
    assert (supervisor.verify / "trial_selection.json").exists()
    assert not (supervisor.verify / "trial_completed.json").exists()
    assert launch.read(supervisor.verify / "trial_post_run_invariants.json")["status"] == "verified"
    if invalid_json:
        assert "JSONDecodeError" in failure["artifact_read_error"]


def test_nonowned_stale_reservation_never_receives_failure_record(supervisor):
    put(supervisor.verify / "trial_selection.json", {"parent_pid": os.getpid(), "reservation_token": "stale"})
    launch.record_failure(supervisor.args, RuntimeError("refused"))
    assert not (supervisor.verify / "trial_failure.json").exists()


def test_review_and_gate_hashes_are_required_before_resource_selection(supervisor):
    value = launch.read(supervisor.review)
    value["code_sha256"][next(iter(value["code_sha256"]))] = "stale"
    put(supervisor.review, value)
    with pytest.raises(ValueError, match="review"):
        launch.validate_review(supervisor.review, supervisor.hashes)


def test_wait_interruption_keeps_parent_until_owned_child_exits():
    class Worker:
        pid, returncode, calls = 123, None, 0

        def poll(self):
            return self.returncode

        def wait(self):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt()
            self.returncode = 19

    worker = Worker()
    with pytest.raises(KeyboardInterrupt) as observed:
        launch.wait_owned_worker(worker)
    assert worker.returncode == 19 and worker.calls == 2
    assert observed.value.worker_returncode == 19


@pytest.mark.parametrize("stage", ["tiny", "zero", "solve"])
def test_all_child_stages_force_utf8_interpreter_before_script(stage, tmp_path):
    args = launch.parser().parse_args(["--stage", stage, "--exp", "utf8_fixture",
                                     "--review-gate", str(tmp_path / "review.json"),
                                     "--cuda-operator-gate", str(tmp_path / "cuda.json"),
                                     "--zero-audit-gate", str(tmp_path / "zero.json")])
    command = launch._command(args, tmp_path / "launch.json")
    assert command[:4] == [str(launch.PYTHON), "-X", "utf8", "-u"]
    assert command[4] == str(launch.ROOT / ("gpu_preflight.py" if stage == "tiny" else "runner.py"))


@pytest.mark.parametrize("stage", ["tiny", "zero", "solve"])
def test_mode_is_bound_to_child_argv_and_reservation_without_single_shift(stage, tmp_path):
    args = launch.parser().parse_args(["--stage", stage, "--exp", "cycle_fixture", "--wavelet-mode", "uniform8",
                                     "--review-gate", str(tmp_path / "review.json")])
    command = launch._command(args, tmp_path / "launch.json")
    assert command[command.index("--wavelet-mode")+1] == "uniform8"
    assert "--wavelet-shift" not in command
    request = launch._request(args)
    assert request["wavelet_mode"] == "uniform8" and request["wavelet_shift"] is None
    assert len(launch.PRODUCTION) == 9 and "cycle_average.py" in launch.PRODUCTION and "frequency_weights.py" in launch.PRODUCTION


def test_uniform8_cannot_ignore_an_explicit_individual_shift(supervisor):
    supervisor.args.wavelet_mode = "uniform8"
    supervisor.args.wavelet_shift = (1, 1, 1)
    with pytest.raises(ValueError, match="shift"):
        launch.validate_stage_gates(supervisor.args, supervisor.hashes, supervisor.invariants)


def test_new_preflight_proof_cannot_be_reused_for_another_wavelet_mode(supervisor, monkeypatch):
    own_selection(supervisor)
    fake_worker(monkeypatch, supervisor)
    launch.execute(supervisor.args, [])
    with pytest.raises(ValueError, match="uniform8|wavelet"):
        launch.validate_observed_gate(supervisor.verify / "trial_completed.json", "tiny",
            supervisor.hashes, supervisor.invariants, wavelet_mode="single")
