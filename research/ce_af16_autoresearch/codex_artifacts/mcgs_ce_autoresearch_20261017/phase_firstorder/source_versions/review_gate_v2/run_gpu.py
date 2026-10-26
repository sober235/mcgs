#!/usr/bin/env python3
"""CPU-only GPU reservation supervisor with immutable review/input/source gates.

The initial process selects under a shared claim lock, then reexecutes the SAME
PID with one initial CUDA_VISIBLE_DEVICES reservation. That CPU parent remains
alive until its child finishes, including clinical export/scoring. No Torch is
imported here, no source/data file is written, and existing runs are preserved.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "nonlocal_autoresearch"
VERIFY = ROOT / "verification"
SOURCE_DEFINITION = ROOT.parent / "optimizer_probe/verification/source_integrity_before.json"
INPUT_DEFINITION = BASE / "verification/loop_input_freeze.json"
PYTHON = Path("/home/congcongliu/anaconda3/envs/GSTMI/bin/python")
CPU_PYTHON=Path("/home/congcongliu/anaconda3/envs/nvgen/bin/python")
SOURCE_TRAINER=Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois/trainer/_trainer_e778bf2_ar")
CLAIM_LOCK = Path(f"/tmp/mcgs_ce_gpu_claim_{os.getuid()}.lock")
PRODUCTION = ("runner.py", "coupled_edge.py", "density_operator.py", "surrogates.py", "pcg.py",
              "gpu_preflight.py", "run_gpu.py", "cycle_average.py", "frequency_weights.py")


def parent_runtime():
    require("torch" not in sys.modules and "frequency_weights" not in sys.modules,
            "frequency supervisor must remain stdlib-only; no Torch or weight utility")
    require(Path(sys.executable).resolve() == CPU_PYTHON.resolve() and sys.version_info[:3] == (3,11,15),
            "frequency supervisor requires NVGEN Python3.11.15 stdlib-only")
    return {"python":sys.executable,"python_version":list(sys.version_info[:3]),"torch_imported":False,"weights_imported":False}


def frequency_protocol(t,wavelet_mode="uniform8",wavelet_shift=None):
    require(not isinstance(t,bool) and isinstance(t,(int,float)) and math.isfinite(t) and t in (0.,.5,1.),
            "frequency_t must be exactly0/.5/1")
    require(wavelet_mode == "uniform8" and wavelet_shift is None,"frequency route requires uniform8 without individual shift")
    return {"frequency_t":float(t),**wavelet_protocol(wavelet_mode,wavelet_shift)}


def strength_protocol(frequency_t,local_factor,wavelet_mode="uniform8",wavelet_shift=None):
    require(not isinstance(local_factor,bool) and isinstance(local_factor,(int,float)) and math.isfinite(local_factor)
            and local_factor == 1. and frequency_t == 1.,"phase-first-order route requires fixedt1/factor1")
    return {**frequency_protocol(frequency_t,wavelet_mode,wavelet_shift),"local_factor":float(local_factor),
            "cg_steps":96,"true_residual_diagnostics":True,**phase_protocol()}


def phase_protocol():
    return {"phase_curvature_mode":"first_order","damping_curvature_mode":"full",
            "update_class":"phase_first_order_quasi_newton","full_MM_majorization":False}


def wavelet_protocol(mode="single", shift=None):
    require(mode in ("single", "uniform8"), "invalid wavelet mode")
    require(mode != "uniform8" or shift is None, "uniform8 forbids an individual wavelet shift")
    if mode == "single":
        shift = (1, 1, 1) if shift is None else tuple(shift)
        require(len(shift) == 3 and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 15 for v in shift),
                "invalid fixed wavelet shift")
    return {"wavelet_mode": mode, "fixed_wavelet_shift": list(shift) if mode == "single" else None,
            "wavelet_phase_count": 1 if mode == "single" else 8}


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write_new(path, value):
    """Atomic immutable publication, including failure observations."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_hashes():
    return {str(ROOT / name): sha256(ROOT / name) for name in PRODUCTION}


def source_files(root):
    return {p.relative_to(root).as_posix():p for p in root.rglob("*")
            if p.is_file() and not any(part in ("results_arc","__pycache__","build") for part in p.relative_to(root).parts)
            and p.suffix not in (".pyc",".so",".o")}


def snapshot_invariants():
    """Verify existing immutable C/O definitions; never create/refreeze one."""
    definitions = {str(path): sha256(path) for path in (SOURCE_DEFINITION, INPUT_DEFINITION)}
    source, manifest = read(SOURCE_DEFINITION), read(INPUT_DEFINITION)
    observed = {}
    for name, count in (("gated_code_sha256", 13), ("production_code_sha256", 103)):
        expected = source[name]
        require(isinstance(expected, dict) and len(expected) == count, f"expected frozen C source count {count}")
        observed[name] = {path: sha256(path) for path in expected}
        require(observed[name] == expected, f"frozen campaign {name} changed")
    require(isinstance(manifest.get("files"), dict) and len(manifest["files"]) == 35, "expected exactly 35 frozen inputs")
    actual_inputs = {path: sha256(path) for path in manifest["files"]}
    require(actual_inputs == manifest["files"], "frozen input/checkpoint bytes changed")
    checkpoint = manifest["source_checkpoint"]
    require(checkpoint in manifest["files"] and manifest["source_config"] in manifest["files"],
            "source checkpoint/config must be among frozen inputs")
    original,copied=source_files(SOURCE_TRAINER),source_files(BASE/"trainer")
    require(bool(original) and bool(copied),"original/copy source maps must be nonempty")
    source_definition={"original_root":str(SOURCE_TRAINER),"copy_root":str(BASE/"trainer"),
        "original_sha256":{name:sha256(path) for name,path in sorted(original.items())},
        "copy_sha256":{name:sha256(path) for name,path in sorted(copied.items())},
        "missing_original_sources":sorted(set(copied)-set(original)),"missing_copy_sources":sorted(set(original)-set(copied))}
    require(not source_definition["missing_copy_sources"],"copied trainer lost original files")
    production=set(source["production_code_sha256"])
    outside={name for name,path in copied.items() if str(path) not in production}
    require(outside <= {"SOURCE.txt","SOURCE_AR.txt","arc/program.md"},"new copied source outside frozen103 definition")
    source_definition["copy_files_outside_production_subset"]=sorted(outside)
    folder = BASE / "trainer/paired_gs/gaussian/cuda_ext"
    pieces = [hashlib.md5((folder / name).read_bytes()).hexdigest()[:8]
              for name in ("tile_voxelize_cuda.cpp", "tile_voxelize_cuda_kernel.cu")]
    binary = folder / "build" / ("tile_voxelize_cuda_" + "_".join(pieces) + ".so")
    require(binary.is_file(), "source-hash-named prebuilt extension is missing; compilation is forbidden")
    require(definitions == {str(path): sha256(path) for path in (SOURCE_DEFINITION, INPUT_DEFINITION)},
            "frozen definitions changed while being checked")
    return {"frozen_definitions_sha256": definitions, "campaign_sources": observed,
            "source_definition":source_definition,"source_config_sha256":actual_inputs[manifest["source_config"]],
            "input_manifest": manifest, "source_checkpoint_sha256": actual_inputs[checkpoint],
            "prebuilt_extension": {"path": str(binary), "sha256": sha256(binary)}}


def validate_review(path, hashes):
    review = read(path)
    require(review.get("status") == "pass" and review.get("actionable_findings") == 0
            and review.get("unresolved_findings") == 0, "fresh independent clean review is required")
    require(review.get("code_sha256") == hashes, "review source hashes are missing/stale")
    return review


def validate_observed_gate(path, stage, hashes, invariants, *, wavelet_mode="uniform8", wavelet_shift=None,frequency_t=1.,local_factor=1.,
                           cg_steps=96,true_residual_diagnostics=True):
    require(cg_steps == 96 and true_residual_diagnostics is True,"precision gate requires96 and true residual diagnostics")
    gate = read(path)
    require(gate.get("passed") is True and gate.get("status") == "complete" and gate.get("stage") == stage,
            f"passing observed {stage} gate is required")
    require(gate.get("code_sha256") == hashes and gate.get("invariants") == invariants,
            f"stale {stage} gate source/input/checkpoint/extension bindings")
    protocol = strength_protocol(frequency_t,local_factor,wavelet_mode, wavelet_shift)
    require(all(gate.get(k) == v for k,v in protocol.items()), "observed gate wavelet protocol mismatch")
    artifacts = gate.get("artifacts", {})
    require(bool(artifacts), f"{stage} gate lacks observed child artifacts")
    require(all(sha256(file) == digest for file, digest in artifacts.items()), f"{stage} gate artifacts changed")
    proof_path = gate.get("post_run_invariants")
    require(proof_path and sha256(proof_path) == gate.get("post_run_invariants_sha256"), f"{stage} post-run proof changed")
    proof = read(proof_path)
    expected = {"code_sha256": hashes, "invariants": invariants}
    require(proof.get("status") == "verified" and proof.get("before") == proof.get("after") == expected,
            f"{stage} gate has no matching verified before/after proof")
    bindings=gate.get("control_bindings",{})
    require(bool(bindings) and proof.get("control_bindings_before") == proof.get("control_bindings_after") == bindings
            and all(sha256(path) == digest for path,digest in bindings.items()),"observed review/control bindings changed")
    if stage == "tiny":
        candidates = [file for file in artifacts if Path(file).name == "preflight.json"]
        require(len(candidates) == 1, "preflight gate needs one actual preflight artifact")
        validate_child_artifacts("tiny", Path(candidates[0]).parent, hashes, invariants, **protocol)
    else:
        candidates = [file for file in artifacts if Path(file).name == "provenance.json"]
        require(len(candidates) == 1, "audit gate needs one clinical provenance artifact")
        validate_child_artifacts("zero", Path(candidates[0]).parent, hashes, invariants, **protocol)
    return gate


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", required=True, choices=("tiny", "zero", "solve"))
    result.add_argument("--exp", required=True)
    result.add_argument("--review-gate", type=Path, required=True)
    result.add_argument("--cuda-operator-gate", type=Path)
    result.add_argument("--zero-audit-gate", type=Path)
    result.add_argument("--selected-recipe",type=Path)
    result.add_argument("--frequency-t",type=float,choices=(0.,.5,1.),default=1.)
    result.add_argument("--preferred-gpu", type=int, default=5)
    result.add_argument("--outer-steps", type=int, default=5)
    result.add_argument("--cg-steps", type=int, default=96)
    result.add_argument("--max-vjp-pairs", type=int, default=500)
    result.add_argument("--max-render-pairs", type=int, default=700)
    result.add_argument("--damping-rel", type=float, default=.001)
    result.add_argument("--local-factor", type=float, default=1.)
    result.add_argument("--wavelet-mode", choices=("single", "uniform8"), default="uniform8")
    result.add_argument("--wavelet-shift", type=int, nargs=3, default=None)
    result.add_argument("--phase-curvature-mode",choices=("first_order",),default="first_order")
    result.add_argument("--reserved-gpu", type=int, help=argparse.SUPPRESS)
    result.add_argument("--reservation-token", help=argparse.SUPPRESS)
    result.add_argument("--claim-fd", type=int, help=argparse.SUPPRESS)
    return result


def validate_stage_gates(args, hashes, invariants):
    require(args.exp not in ("", ".", "..") and Path(args.exp).name == args.exp, "exp must be one fresh directory name")
    require(args.preferred_gpu >= 0 and (args.reserved_gpu is None or args.reserved_gpu >= 0), "invalid GPU index")
    # These are bounded research controls, not an unbounded CLI escape hatch.
    for name, maximum in (("outer_steps", 5), ("cg_steps", 96), ("max_vjp_pairs", 500), ("max_render_pairs", 700)):
        require(1 <= getattr(args, name) <= maximum, f"{name} exceeds the reviewed bounded probe")
    require(math.isfinite(args.damping_rel) and 0 < args.damping_rel <= 1., "invalid damping")
    require(math.isfinite(args.local_factor) and 0 <= args.local_factor <= 4., "invalid local prior factor")
    strength_protocol(args.frequency_t,args.local_factor,args.wavelet_mode, args.wavelet_shift)
    require(args.phase_curvature_mode == "first_order","clinical phase mode must be first_order")
    require(args.cg_steps == 96,"precision controls and solve require cg_steps96")
    review = validate_review(args.review_gate, hashes)
    require(review.get("frozen_definitions_sha256") == invariants["frozen_definitions_sha256"],
            "review lacks matching frozen C/O source/input definitions")
    require(review.get("invariants") == invariants,"review lacks complete original/copy/input/binary bindings")
    if args.stage != "tiny":
        require(args.cuda_operator_gate is not None, "audit/solve requires actual tiny CUDA preflight gate")
        validate_observed_gate(args.cuda_operator_gate, "tiny", hashes, invariants,
                               wavelet_mode=args.wavelet_mode, wavelet_shift=args.wavelet_shift,frequency_t=args.frequency_t,local_factor=args.local_factor)
    if args.stage == "solve":
        require(args.zero_audit_gate is not None, "solve requires actual clinical zero-step audit gate")
        validate_observed_gate(args.zero_audit_gate, "zero", hashes, invariants,
                               wavelet_mode=args.wavelet_mode, wavelet_shift=args.wavelet_shift,frequency_t=args.frequency_t,local_factor=args.local_factor)
        require(args.local_factor == 1. and args.outer_steps == 5
                and args.max_vjp_pairs == 500 and args.max_render_pairs == 700
                and args.damping_rel == .001,"formal phase-first-order solve requires t1/factor1 and fixed5x96 budget")
        require(args.selected_recipe is not None,"formal frequency solve requires a root-selected current recipe")
        recipe=validate_recipe(args.selected_recipe,hashes,invariants)
        require(args.local_factor in recipe["local_factor_grid"],"requested factor differs from root-selected lower-prior recipe")
    return review


def validate_recipe(path,hashes,invariants):
    recipe=read(path)
    require(recipe.get("status") == "selected" and recipe.get("selected_by") == "root"
            and recipe.get("code_sha256") == hashes and recipe.get("invariants") == invariants
            and recipe.get("frequency_t") == 1. and recipe.get("local_factor_grid") == [1.]
            and "local_factor" not in recipe and recipe.get("source_iteration") == 4000
            and recipe.get("outer_steps") == 5 and recipe.get("cg_steps") == 96 and recipe.get("true_residual_diagnostics") is True
            and recipe.get("max_vjp_pairs") == 500 and recipe.get("max_render_pairs") == 700
            and recipe.get("max_merit_forwards") == 100
            and recipe.get("nlambda") == 0. and recipe.get("wavelet_mode") == "uniform8"
            and recipe.get("damping_rel") == .001 and recipe.get("cg_rtol") == .001
            and recipe.get("minimum_score_improvement") == 1e-4
            and all(recipe.get(k) == v for k,v in phase_protocol().items()),"selected phase recipe differs")
    incumbent=recipe.get("incumbent",{}); source=ROOT.parent/"development_incumbent.json"
    require(incumbent.get("path") == str(source) and incumbent.get("sha256") == sha256(source),"latest incumbent bytes differ")
    actual=read(source)
    require(incumbent.get("score") == actual.get("score") and math.isfinite(actual["score"])
            and actual.get("strict_success") is False,"incumbent score changed or strict-win confirmation takes priority")
    summary=Path(actual["validated_summary"])
    prior=recipe.get("completed_prior_summary",{})
    require(prior.get("path") == str(summary) and prior.get("sha256") == actual["validated_summary_sha256"]
            and summary.resolve().is_relative_to(ROOT.parent.resolve()) and sha256(summary) == actual["validated_summary_sha256"],
            "selected latest completed campaign summary differs")
    previous=read(summary);q=previous.get("inner_precision_family",{});names=q.get("completed_candidates",[])
    trials=previous.get("trials",[]);qrows=[row for row in trials if row.get("name") in names]
    require(previous.get("status") == "completed_real_trial_snapshot" and q.get("frequency_t") == 1.
            and q.get("cg_steps") == 96 and q.get("true_residual_diagnostics") is True
            and type(names) is list and len(names) == len(qrows) == 1
            and qrows[0].get("method_variant") == "inner_precision_mm"
            and qrows[0].get("cg_steps") == 96 and qrows[0].get("true_residual_diagnostics") is True,
            "formal lower-prior recipe requires completed Q96 family")
    h=previous.get("frequency_adam_family",{});hrows=[row for row in trials if row.get("method_variant") == "frequency_adam"]
    require(h.get("frequency_t") == 1. and h.get("declared_geometry_factors") == [1.,2.]
            and h.get("completed_geometry_factors") == [1.,2.] and len(hrows) == 2
            and [row.get("geometry_factor") for row in hrows] == [1.,2.]
            and all(row.get("frequency_t") == 1. for row in hrows),
            "formal lower-prior recipe requires completed ordered H1/2 family")
    l=previous.get("frequency_lowreg_family",{});lnames=l.get("completed_candidates",[])
    lrows=[row for row in trials if row.get("name") in lnames]
    require(l.get("frequency_t") == 1. and l.get("declared_local_factors") == [.75,.5]
            and l.get("completed_local_factors") == [.75,.5] and l.get("cg_steps") == 96
            and l.get("true_residual_diagnostics") is True
            and type(lnames) is list and len(lnames) == len(lrows) == 2
            and [row.get("name") for row in lrows] == lnames
            and [row.get("local_factor") for row in lrows] == [.75,.5]
            and all(row.get("method_variant") == "frequency_lowreg_mm" and row.get("frequency_t") == 1.
                    and row.get("cg_steps") == 96 and row.get("true_residual_diagnostics") is True for row in lrows),
            "formal phase-first-order recipe requires completed ordered L.75/.5 family")
    require(all(row.get("strict_success") is False for row in [*qrows,*hrows,*lrows]),
            "Q/H/L provisional strict win requires confirmation before phase-first-order trials")
    return recipe


def load_campaign():
    """Load only C's CPU scheduler functions, without source bytecode writes."""
    original_path = list(sys.path)
    original_objective = sys.modules.get("objective")
    try:
        sys.path.insert(0, str(BASE))
        objective_spec = importlib.util.spec_from_file_location("objective", BASE / "objective.py")
        objective = importlib.util.module_from_spec(objective_spec)
        sys.modules["objective"] = objective
        objective_spec.loader.exec_module(objective)
        spec = importlib.util.spec_from_file_location("_density_mm_campaign", BASE / "autoresearch.py")
        campaign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(campaign)
        require(Path(campaign.__file__).resolve() == BASE / "autoresearch.py", "unexpected source GPU selector")
        return campaign
    finally:
        sys.path[:] = original_path
        if original_objective is None:
            sys.modules.pop("objective", None)
        else:
            sys.modules["objective"] = original_objective


def select_resource(campaign, *, preferred, environment):
    return campaign.select_gpu(campaign.gpu_snapshot(), preferred, environment)


@contextmanager
def claim_lock(path):
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        require(os.fstat(descriptor).st_uid == os.getuid(), "claim lock has a different owner")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.set_inheritable(descriptor, True)
        yield descriptor
    finally:
        os.close(descriptor)


def initial_visibility():
    for item in Path("/proc/self/environ").read_bytes().split(b"\0"):
        if item.startswith(b"CUDA_VISIBLE_DEVICES="):
            return item.split(b"=", 1)[1].decode()
    return None


def process_start():
    return Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]


def _request(args):
    require(args.phase_curvature_mode == "first_order","request phase curvature must be first_order")
    names=("stage","exp","outer_steps","cg_steps","max_vjp_pairs","max_render_pairs","damping_rel","local_factor",
           "wavelet_mode","wavelet_shift","frequency_t","phase_curvature_mode","review_gate","cuda_operator_gate","zero_audit_gate","selected_recipe")
    return {**{name:(str(value.resolve()) if isinstance(value,Path) else list(value) if name == "wavelet_shift" and value is not None else value)
            for name in names for value in (getattr(args,name),)},"true_residual_diagnostics":True,**phase_protocol()}


def control_bindings(args):
    files=[getattr(args,name) for name in ("review_gate","cuda_operator_gate","zero_audit_gate","selected_recipe")
           if getattr(args,name) is not None]
    if args.stage == "solve":
        incumbent=ROOT.parent/"development_incumbent.json"
        files += [incumbent,Path(read(incumbent)["validated_summary"])]
    return {str(path.resolve()):sha256(path) for path in files}


def validate_frequency_evidence(value,directory,frequency_t):
    """Stdlib byte/metadata check; numerical array reconstruction is a child check."""
    directory=Path(directory).resolve(); folder=directory/"frequency_weights"
    require(value.get("version") == 1 and value.get("frequency_t") == frequency_t
            and value.get("mass_relative_tolerance") == 2e-6 and set(value.get("scans",{})) == {"pre","post"},
            "actual per-t two-scan frequency evidence missing/mismatched")
    files={}
    for scan,row in value["scans"].items():
        require(type(row.get("acquired_count")) is int and row["acquired_count"] > 0,"original acquired mask is empty")
        for key in ("source_mean_squared_weight","effective_mean_squared_weight","c","relative_mass_error"):
            require(isinstance(row.get(key),(int,float)) and not isinstance(row[key],bool) and math.isfinite(row[key]),"nonfinite frequency mass evidence")
        source,effective=row["source_mean_squared_weight"],row["effective_mean_squared_weight"]
        require(source > 0 and effective > 0 and row["c"] > 0 and math.isclose(row["c"]**2,source,rel_tol=1e-12)
                and 0 <= row["relative_mass_error"] <= 2e-6
                and math.isclose(row["relative_mass_error"],abs(effective/source-1.),rel_tol=1e-12,abs_tol=1e-15),
                "acquired squared-weight conservation metadata differs")
        arrays=[]
        for name in ("source_weight","effective_weight","acquired_mask"):
            info=row[name]; path=Path(info["path"]); expected=folder/f"{scan}_{name}.npy"
            require(path == expected and path.resolve().parent == folder and path.is_file()
                    and sha256(path) == info.get("file_sha256"),"saved actual weight/mask array hash/path changed")
            require(isinstance(info.get("byte_sha256"),str) and len(info["byte_sha256"]) == 64
                    and isinstance(info.get("shape"),list) and len(info["shape"]) == 3
                    and all(type(n) is int and n > 0 for n in info["shape"])
                    and info.get("device") == "cuda:0","frequency array descriptor missing or not actual CUDA")
            files[str(path)]=info["file_sha256"]; arrays.append(info)
        require(arrays[0]["shape"] == arrays[1]["shape"] == arrays[2]["shape"]
                and arrays[0]["dtype"] == arrays[1]["dtype"] and row["acquired_count"] <= math.prod(arrays[2]["shape"]),
                "source/effective/original mask shape/dtype/count differ")
        if frequency_t == 0.:
            require(row.get("t0_exact_equal") is True and arrays[0]["byte_sha256"] == arrays[1]["byte_sha256"]
                    and source == effective,"t0 source weights were changed")
    manifest=folder/"manifest.json"
    require(read(manifest) == value,"saved frequency manifest differs from actual child evidence")
    files[str(manifest)]=sha256(manifest)
    return files


def validate_numeric_runtime(runtime):
    require(Path(runtime.get("python","")).resolve() == PYTHON.resolve()
            and runtime.get("torch_version") == "2.7.1+cu118" and runtime.get("cuda_version") == "11.8",
            "actual numerical child is not fixed GSTMI Torch/CUDA")


def validate_frequency_array_check(check,evidence,frequency_t,*,tiny=False):
    require(check.get("passed") is True and check.get("frequency_t") == frequency_t
            and check.get("mass_relative_tolerance") == 2e-6 and set(check.get("scans",{})) == {"pre","post"},
            "independent saved-six-array numerical check missing")
    validate_numeric_runtime(check.get("numerical_runtime",{}))
    for scan,row in check["scans"].items():
        original=evidence["scans"][scan]
        tolerance=8.*(2.**-23 if original["source_weight"]["dtype"] == "torch.float32" else 2.**-52)
        require(row.get("original_acquired_mask_exact") is True and row.get("original_source_weight_exact") is True
                and row.get("acquired_count") == original["acquired_count"]
                and row.get("interpolation_relative_tolerance") == tolerance
                and math.isfinite(row.get("interpolation_max_relative_error",float("nan")))
                and 0 <= row["interpolation_max_relative_error"] <= tolerance
                and all(math.isclose(row.get(key,float("nan")),original[key],rel_tol=1e-12)
                        for key in ("source_mean_squared_weight","effective_mean_squared_weight","c"))
                and 0 <= row.get("relative_mass_error",float("nan")) <= 2e-6,"saved-array source/mask/full interpolation check failed")
        if frequency_t == 0.:
            require(row.get("effective_identity") is True,"t0 saved-array identity failed")
        if tiny:
            expected=(.3,1.,2.2 if scan == "pre" else 2.8)
            require(len(row.get("source_levels",[])) == len(row.get("acquired_source_levels",[])) == 3
                    and all(math.isclose(a,b,rel_tol=1e-6) for a,b in zip(row["source_levels"],expected))
                    and all(math.isclose(a,b,rel_tol=1e-6) for a,b in zip(row["acquired_source_levels"],expected))
                    and (frequency_t == 0. or row.get("effective_identity") is False),
                    "tiny must exercise acquired center/mid/high and a nonidentity t>0 transform")


def validate_reservation(args, selection, hashes, invariants):
    require(args.reservation_token and selection.get("reservation_token") == args.reservation_token
            and selection.get("parent_pid") == os.getpid() and selection.get("process_start") == process_start(),
            "reservation is not owned by this exact CPU parent")
    require(selection.get("code_sha256") == hashes and selection.get("invariants") == invariants
            and selection.get("stage") == args.stage, "reservation source/input/stage changed")
    require(selection.get("request") == _request(args), "reservation bounded request changed")
    require(selection.get("resource", {}).get("gpu") == args.reserved_gpu
            and initial_visibility() == os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.reserved_gpu),
            "reservation must be in the CPU parent's INITIAL visibility")
    require(args.claim_fd is not None and args.claim_fd >= 3, "inherited global claim lock is required")
    actual, expected = os.fstat(args.claim_fd), CLAIM_LOCK.stat()
    require((actual.st_dev, actual.st_ino, actual.st_uid) == (expected.st_dev, expected.st_ino, os.getuid()),
            "inherited claim lock descriptor differs")
    fcntl.flock(args.claim_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def wait_owned_worker(worker):
    interrupted = None
    while worker.poll() is None:
        try:
            worker.wait()
        except KeyboardInterrupt as error:
            interrupted = error
    if interrupted is not None:
        interrupted.worker_returncode, interrupted.worker_pid = worker.returncode, worker.pid
        raise interrupted
    return worker.returncode


def validate_precision_trace(trace,work,solver_config,*,zero=False,expected_cg_steps=96):
    """Read-only scalar/work verification; no numerical library in parent.

    The discrepancy compares residual norms, not unavailable PCG vectors.
    True residuals diagnose the same frozen damped system and never replace
    the PCG convergence flag or control acceptance.
    """
    require(solver_config.get("cg_steps") == expected_cg_steps and solver_config.get("true_residual_diagnostics") is True,
            "precision solver config requires96 and true residual diagnostics")
    require(type(trace) is list and type(work) is dict,"precision trace/work missing")
    if zero:
        require(solver_config.get("outer_steps") == 0 and not trace
                and all(work.get(k) == 0 for k in ("render_pairs","renderer_calls","vjp_pairs","gradient_evaluations",
                    "normal_evaluations","merit_forward_evaluations","support_forward_pairs","wavelet_transform_batches",
                    "true_residual_evaluations","true_residual_elapsed_seconds")),"zero precision work must be empty")
        return
    require(len(trace) == solver_config.get("outer_steps") > 0,"precision trace outer count differs")
    total_matvecs=total_merits=0
    total_time=0.
    close=lambda a,b: math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15)
    for index,row in enumerate(trace,1):
        require(row.get("outer_step") == index and row.get("accepted") is True,"precision trace incomplete/unaccepted")
        iterations,matvecs=row.get("cg_iterations"),row.get("cg_matvecs")
        require(type(iterations) is int and 0 <= iterations <= expected_cg_steps and matvecs == iterations,"precision CG work differs")
        history=row.get("cg_residual_history",[])
        require(len(history) == iterations+1 and all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in history)
                and history[0] > 0,"precision recursive residual history invalid")
        converged=history[-1] <= solver_config["cg_rtol"]*history[0]
        require(row.get("cg_converged") is converged
                and row.get("cg_status") == ("converged" if converged else "maxiter")
                and (converged or iterations == expected_cg_steps),"precision CG termination differs")
        diagnostic=row.get("true_residual",{})
        require(diagnostic.get("diagnostic_only") is True and diagnostic.get("normal_evaluations") == 1,
                "true residual diagnostic missing")
        names=("rhs_norm","norm","relative","recursive_norm","recursive_relative","relative_norm_discrepancy","elapsed_seconds")
        require(all(type(diagnostic.get(k)) in (int,float) and math.isfinite(diagnostic[k]) and diagnostic[k] >= 0 for k in names)
                and diagnostic["rhs_norm"] > 0 and diagnostic["elapsed_seconds"] > 0,"true residual numbers invalid")
        rhs,norm,recursive=diagnostic["rhs_norm"],diagnostic["norm"],diagnostic["recursive_norm"]
        require(close(rhs,history[0]) and close(rhs,.5*row["gradient_norm"]) and recursive == history[-1]
                and close(diagnostic["relative"],norm/rhs) and close(diagnostic["recursive_relative"],recursive/rhs)
                and close(diagnostic["relative_norm_discrepancy"],abs(norm-recursive)/rhs),"true residual scalar evidence inconsistent")
        total_matvecs += matvecs
        total_merits += len(row.get("backtracking",[]))
        total_time += diagnostic["elapsed_seconds"]
        if row.get("wavelet_mode") == "uniform8":
            require(row.get("wavelet_transform_batches") == 8*(6+matvecs+len(row.get("backtracking",[]))),
                    "precision true residual wavelet work differs")
    n=len(trace)
    require(work.get("true_residual_evaluations") == work.get("gradient_evaluations") == n
            and close(work.get("true_residual_elapsed_seconds",float("nan")),total_time)
            and work.get("normal_evaluations") == total_matvecs+2*n
            and work.get("vjp_pairs") == total_matvecs+3*n
            and work.get("merit_forward_evaluations") == total_merits
            and work.get("support_forward_pairs") in (0,1)
            and work.get("render_pairs") == work["vjp_pairs"]+total_merits+work["support_forward_pairs"]
            and work.get("renderer_calls") == 2*work["render_pairs"]
            and work.get("wavelet_transform_batches") == sum(row["wavelet_transform_batches"] for row in trace),
            "precision diagnostic/work totals inconsistent")
    require(work["vjp_pairs"] <= solver_config["max_vjp_pairs"] <= 500
            and work["render_pairs"] <= solver_config["max_render_pairs"] <= 700
            and total_merits <= solver_config["max_merit_forwards"] <= 100,"precision work exceeded unchanged caps")


def validate_precision_callbacks(callbacks,trace,work,solver_config,*,zero=False,expected_cg_steps=96):
    """Bind real JSONL rows to result trace and their cumulative measured work."""
    require(type(callbacks) is list and len(callbacks) == len(trace),"precision callback trace length differs")
    require([{k:v for k,v in row.items() if k != "work"} for row in callbacks] == trace,
            "precision callback trace records differ")
    validate_precision_trace(trace,work,solver_config,zero=zero,expected_cg_steps=expected_cg_steps)
    for index,row in enumerate(callbacks,1):
        validate_precision_trace(trace[:index],row.get("work",{}),{**solver_config,"outer_steps":index},expected_cg_steps=expected_cg_steps)
    require(not callbacks or callbacks[-1].get("work") == work,"precision final callback work differs")


def validate_phase_trace(trace,work,*,zero=False):
    """No numeric imports: bind actual full-probe/reduced-solve role counts."""
    protocol=phase_protocol()
    require(all(work.get(k) == v for k,v in protocol.items()) and work.get("full_MM_majorization") is False,"phase work protocol differs")
    require(type(trace) is list and (not trace if zero else len(trace)>0),"phase trace missing")
    total=0
    for row in trace:
        require(all(row.get(k) == v for k,v in protocol.items()) and row.get("full_MM_majorization") is False,"phase trace protocol differs")
        count=row.get("cg_matvecs")
        require(type(count) is int and count >= 0
                and row.get("curvature_evaluations") == {"full_rayleigh":1,"reduced_cg":count,"reduced_endpoint":1}
                and row.get("true_residual",{}).get("phase_curvature_mode") == "first_order",
                "phase full/reduced normal roles differ")
        require(type(row.get("damping")) in (int,float) and math.isfinite(row["damping"]) and row["damping"]>0
                and type(row.get("damping_rayleigh")) in (int,float) and math.isfinite(row["damping_rayleigh"])
                and row["damping_rayleigh"]>=0
                and math.isclose(row["damping"],.001*max(row["damping_rayleigh"],1e-12),rel_tol=1e-12,abs_tol=1e-15),
                "phase damping differs from original full Rayleigh")
        total+=count+1
    require(work.get("full_curvature_normal_evaluations") == len(trace)
            and work.get("phase_first_order_normal_evaluations") == total
            and work.get("normal_evaluations") == len(trace)+total,
            "phase cumulative full/reduced normal work differs")


def validate_tiny_phase_check(check,*,device="cuda:0"):
    require(check.get("passed") is True and all(check.get(k) == v for k,v in phase_protocol().items())
            and check.get("local_factor") == 1. and check.get("device") == device
            and check.get("lambda_tv") == [.1,.1] and check.get("lambda_phase") == [.03,.03]
            and check.get("lambda_wavelet") == [.12,.1] and check.get("tolerance") == 2e-5
            and check.get("source_density_unchanged") is True and check.get("phase_image_vjp_evaluations") == 4,
            "actual tiny phase-first-order check missing/wrong profile")
    for name in ("phase_gradient_norm","phase_full_probe_gradient_norm"):
        require(type(check.get(name)) in (float,int) and math.isfinite(check[name]) and check[name]>0,
                "tiny nonzero phase gradient/curvature missing")
    require(check.get("phase_reduced_probe_gradient_norm") == 0.,"tiny reduced phase curvature is nonzero")
    for name in ("source_phase_gradient_relative_error","source_phase_value_relative_error","full_control_eta_relative_error"):
        require(type(check.get(name)) in (float,int) and math.isfinite(check[name]) and 0<=check[name]<=2e-5,
                "tiny source phase/full eta check failed")
    config=check.get("solver_config",{});control=check.get("full_control_solver_config",{})
    require(config == {"outer_steps":2,"cg_steps":4,"cg_rtol":.001,"damping_rel":.001,
                        "max_vjp_pairs":40,"max_render_pairs":100,"max_merit_forwards":24,
                        "backtracks":12,"backtrack_factor":.5,"armijo":.0001,"true_residual_diagnostics":True}
            and control == {**config,"outer_steps":1,"cg_steps":1},"tiny actual phase solver configs differ")
    require(check.get("solver_status") == check.get("full_control_solver_status") == "complete"
            and check.get("accepted_outer_steps") == 2,"tiny physical phase solver did not complete")
    trace=check.get("trace",[]);work=check.get("solver_counters",{})
    validate_precision_trace(trace,work,config,expected_cg_steps=4)
    validate_phase_trace(trace,work)
    full_trace=check.get("full_control_trace",[]);full_work=check.get("full_control_solver_counters",{})
    validate_precision_trace(full_trace,full_work,control,expected_cg_steps=1)
    require(math.isclose(trace[0]["damping"],full_trace[0]["damping"],rel_tol=2e-5,abs_tol=0.),
            "tiny first-order eta differs from actual original full helper")
    before=check.get("operator_counters_before",{});after=check.get("operator_counters_after",{})
    setup=check.get("source_check_operator_delta",{})
    require(all(type(row.get(k)) is int and row[k]>=0 for row in (before,after,setup) for k in ("render_pairs","renderer_calls"))
            and after["render_pairs"]-before["render_pairs"] == setup["render_pairs"]+work["render_pairs"]+full_work["render_pairs"]
            and after["renderer_calls"]-before["renderer_calls"] == 2*(after["render_pairs"]-before["render_pairs"]),
            "tiny phase work hid actual renderer calls")


def validate_child_artifacts(stage, directory, hashes, invariants, *, wavelet_mode="uniform8",
                             fixed_wavelet_shift=None, wavelet_phase_count=8,frequency_t=1.,local_factor=1.,
                             cg_steps=96,true_residual_diagnostics=True,phase_curvature_mode="first_order",
                             damping_curvature_mode="full",update_class="phase_first_order_quasi_newton",full_MM_majorization=False):
    require({"phase_curvature_mode":phase_curvature_mode,"damping_curvature_mode":damping_curvature_mode,
             "update_class":update_class,"full_MM_majorization":full_MM_majorization} == phase_protocol(),"phase artifact protocol differs")
    require(cg_steps == 96 and true_residual_diagnostics is True,"precision artifacts require96 and true residual diagnostics")
    directory = Path(directory)
    protocol = strength_protocol(frequency_t,local_factor,wavelet_mode, fixed_wavelet_shift)
    require(protocol["wavelet_phase_count"] == wavelet_phase_count, "wavelet phase count mismatch")
    if stage == "tiny":
        path = directory / "preflight.json"
        artifact = read(path)
        require(artifact.get("passed") is True and artifact.get("status") == "pass", "tiny CUDA preflight failed/incomplete")
        require(artifact.get("code_sha256") == hashes and artifact.get("invariants") == invariants,
                "preflight artifact bindings changed")
        require(all(artifact.get(k) == v for k,v in protocol.items()), "preflight wavelet protocol mismatch")
        validate_numeric_runtime(artifact.get("runtime",{}))
        frequency_files=validate_frequency_evidence(artifact.get("frequency_evidence",{}),directory,frequency_t)
        validate_frequency_array_check(artifact.get("frequency_array_check",{}),artifact["frequency_evidence"],frequency_t,tiny=True)
        band_evidence=artifact.get("factory",{}).get("frequency_bands",{})
        require(set(band_evidence) == {"pre","post"},"actual tiny frequency-band coverage missing")
        for scan,row in band_evidence.items():
            bands=row.get("bands",[]);saved=artifact["frequency_evidence"]["scans"][scan]
            require(len(bands) == 3 and type(row.get("fit_only_count")) is int and row["fit_only_count"] > 0
                    and all(type(b.get(k)) is int and 0 < b[k] <= b["volume_count"]
                            for b in bands for k in ("volume_count","acquired_count","fit_count"))
                    and sum(b["volume_count"] for b in bands) == math.prod(saved["source_weight"]["shape"])
                    and sum(b["acquired_count"] for b in bands) == saved["acquired_count"]
                    and row.get("effective_identity") == (frequency_t == 0.),
                    "tiny must retain three acquired bands and distinct fit-only samples")
        stripped=json.loads(json.dumps(artifact["frequency_evidence"]))
        for row in stripped["scans"].values():
            for name in ("source_weight","effective_weight","acquired_mask"):
                row[name].pop("path");row[name].pop("file_sha256")
        require(artifact.get("reference_frequency_summary") == stripped,"tiny source/reference frequency arrays differ")
        parity=artifact.get("t0_source_parity",{})
        require(parity.get("passed") is True and parity.get("weights_exact_equal") is True
                and parity.get("source_loss_absolute_difference") == 0. and parity.get("zero_loss_absolute_difference") == 0.
                and parity.get("normal_relative_tolerance") == 2e-5
                and math.isfinite(parity.get("normal_relative_difference",float("nan")))
                and 0 <= parity["normal_relative_difference"] <= 2e-5,"tiny source t0 loss/normal parity missing")
        prior = artifact.get("prior_checks", {})
        validate_tiny_phase_check(artifact.get("phase_firstorder_checks",{}))
        prior_errors = prior.get("errors", {})
        require(prior.get("passed") is True and prior.get("wavelet_mode") == wavelet_mode
                and prior.get("local_factor") == local_factor
                and prior.get("lambda_tv") == prior.get("lambda_phase") == [0.,0.]
                and prior.get("lambda_wavelet") == [.12*local_factor,.1*local_factor]
                and prior.get("tested_terms") == ["wavelet_mean8"] and prior.get("nonzero_TV_phase_tested") is False
                and prior.get("tolerance") == 2e-5
                and set(prior_errors) == {"value_relative", "gradient_relative", "quadratic_gradient_relative"}
                and all(math.isfinite(error) and 0 <= error <= 2e-5 for error in prior_errors.values())
                and prior.get("wavelet_transform_batches") == 5 * wavelet_phase_count,
                "new prior CUDA value/gradient checks failed or incomplete")
        require(artifact.get("device") == "cuda:0" and artifact.get("parameters_unchanged") is True
                and artifact.get("factory", {}).get("cuda_backend") == "prebuilt C tile extension",
                "preflight did not exercise actual source CUDA")
        tolerance = artifact.get("tolerance", float("nan"))
        errors = artifact.get("errors", {})
        require(math.isfinite(tolerance) and 0 < tolerance <= 2e-5
                and set(errors) == {"superposition", "adjoint", "normal_linearity", "normal_hermitian"}
                and all(math.isfinite(error) and 0 <= error <= tolerance for error in errors.values()),
                "actual CUDA checks exceed predeclared tolerance")
        curvature = artifact.get("normal_curvature", {})
        require(math.isfinite(curvature.get("real", float("nan"))) and curvature["real"] > 0,
                "preflight lacks positive normal curvature")
        return {str(path): sha256(path),**frequency_files}
    provenance_path, metrics_path = directory / "provenance.json", directory / "metrics.json"
    provenance, metrics = read(provenance_path), read(metrics_path)
    require(all(provenance.get(k) == metrics.get(k) == v for k,v in protocol.items()),
            "clinical wavelet protocol mismatch")
    require(provenance.get("status") == "complete", "clinical child failed/partial/no_update; retain observed artifacts")
    require(provenance.get("mode") == metrics.get("mode") == ("zero" if stage == "zero" else "frequency_mm")
            and provenance.get("method_variant") == metrics.get("method_variant") == "phase_firstorder"
            and provenance.get("code_sha256") == hashes and provenance.get("source_iteration") == 4000,
            "clinical mode/source hashes/source iteration mismatch")
    require(provenance.get("source_checkpoint_sha256") == invariants["source_checkpoint_sha256"],
            "clinical source checkpoint differs")
    require(provenance.get("original_adam_unchanged") is True and provenance.get("geometry_unchanged") is True,
            "clinical original Adam/geometry changed")
    require(provenance.get("frequency_evidence") == metrics.get("frequency_evidence"),"actual frequency records disagree")
    frequency_files=validate_frequency_evidence(provenance.get("frequency_evidence",{}),directory,frequency_t)
    require(provenance.get("frequency_array_check") == metrics.get("frequency_array_check"),"independent array-check records disagree")
    validate_frequency_array_check(provenance.get("frequency_array_check",{}),provenance["frequency_evidence"],frequency_t)
    runtime=provenance.get("numerical_runtime",{})
    require(runtime == metrics.get("numerical_runtime"),"numerical runtime records disagree")
    validate_numeric_runtime(runtime)
    execution=provenance.get("execution",{})
    launch=read(VERIFY/f"{directory.name}_launch.json")
    require(execution == metrics.get("execution") and execution.get("child_command") == launch.get("command")
            and execution.get("source_code_sha256") == hashes
            and all(execution.get("declared_parameters",{}).get(k) == v for k,v in protocol.items()),
            "actual child command/source/declared frequency parameters differ")
    config=provenance.get("solver_config",{})
    require(config == metrics.get("solver_config") == execution.get("declared_parameters",{}).get("solver_config"),
            "precision actual solver configs disagree")
    require(config.get("outer_steps") == (0 if stage == "zero" else 5)
            and config.get("max_vjp_pairs") == 500 and config.get("max_render_pairs") == 700
            and config.get("max_merit_forwards") == 100 and config.get("damping_rel") == .001
            and config.get("cg_rtol") == .001,"precision actual solver config differs")
    trace=[json.loads(line) for line in (directory/"outer_trace.jsonl").read_text().splitlines() if line.strip()]
    require(all(row.get("local_factor") == local_factor for row in trace)
            and all(row.get("local_factor") == local_factor for row in provenance.get("trace",[])),
            "clinical trace factor differs from actual requested factor")
    require(provenance.get("solver_counters") == metrics.get("solver_counters"),"precision counter records differ")
    validate_precision_callbacks(trace,provenance.get("trace",[]),provenance["solver_counters"],config,zero=stage == "zero")
    validate_phase_trace(provenance.get("trace",[]),provenance["solver_counters"],zero=stage == "zero")
    for index,row in enumerate(trace,1):
        validate_phase_trace(provenance["trace"][:index],row.get("work",{}))
    saved=provenance.get("solver_checkpoint",{})
    require(saved == metrics.get("solver_checkpoint") and saved.get("passed") is True
            and saved.get("resumable_by_original_adam") is False and saved.get("all_seven_model_hashes_match") is True
            and saved.get("frequency_t") == frequency_t and saved.get("local_factor") == local_factor
            and saved.get("cg_steps") == 96 and saved.get("true_residual_diagnostics") is True,
            "actual solver artifact proof missing/mismatched")
    require(all(saved.get(k) == v for k,v in phase_protocol().items()),"saved phase solver proof differs")
    checkpoint=Path(saved["path"])
    require(checkpoint.resolve().parent == directory.resolve() and sha256(checkpoint) == saved["sha256"],"actual nonresumable checkpoint changed")
    if stage == "zero":
        checks = metrics.get("audit_checks", {})
        require(set(checks) == {"pre", "post"} and all(checks[scan].get("passed") is True for scan in checks)
                and provenance.get("before") == provenance.get("after"), "clinical zero-step audit failed")
        require(provenance.get("zero_export_check") == metrics.get("zero_export_check")
                and provenance.get("zero_export_check",{}).get("passed") is True
                and all(math.isfinite(checks[s].get("source_array_relative_l2",float("nan")))
                        and 0 <= checks[s]["source_array_relative_l2"] <= 1e-5 for s in ("pre","post")),
                "zero-source complex/gauge reproduction failed")
    else:
        require(provenance.get("accepted_outer_steps", 0) > 0 and provenance.get("solver_status") == "complete",
                "solve did not complete an accepted update")
    actual_files=(provenance_path,metrics_path,checkpoint,directory/"actual_config.yaml",directory/"outer_trace.jsonl",
                  *[directory/f"candidate_{s}_complex.npy" for s in ("pre","post")],
                  *[directory/f"scores_{s}.json" for s in ("pre","post")])
    return {**{str(path):sha256(path) for path in actual_files},**frequency_files}


def _command(args, launch_path):
    strength_protocol(args.frequency_t,args.local_factor,args.wavelet_mode,args.wavelet_shift)
    if args.stage == "tiny":
        return [str(PYTHON), "-X", "utf8", "-u", str(ROOT / "gpu_preflight.py"), "--output", str(ROOT / "outputs" / args.exp),
                "--review-gate", str(args.review_gate), "--launch-manifest", str(launch_path),
                "--wavelet-mode", args.wavelet_mode,"--frequency-t",str(args.frequency_t),"--local-factor",str(args.local_factor),
                "--cg-steps",str(args.cg_steps),
                "--true-residual-diagnostics",
                "--phase-curvature-mode",args.phase_curvature_mode,
                *( ("--wavelet-shift", *(str(v) for v in args.wavelet_shift)) if args.wavelet_shift is not None else () )]
    command = [str(PYTHON), "-X", "utf8", "-u", str(ROOT / "runner.py"), "--mode", "zero" if args.stage == "zero" else "frequency_mm",
               "--exp", args.exp, "--review-gate", str(args.review_gate), "--cuda-operator-gate", str(args.cuda_operator_gate)]
    if args.zero_audit_gate is not None:
        command.extend(("--zero-audit-gate", str(args.zero_audit_gate)))
    for name in ("outer_steps", "cg_steps", "max_vjp_pairs", "max_render_pairs", "damping_rel", "local_factor"):
        command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    command.extend(("--wavelet-mode", args.wavelet_mode))
    command.extend(("--frequency-t",str(args.frequency_t)))
    command.append("--true-residual-diagnostics")
    command.extend(("--phase-curvature-mode",args.phase_curvature_mode))
    if args.wavelet_shift is not None:
        command.extend(("--wavelet-shift", *(str(value) for value in args.wavelet_shift)))
    return command


def execute(args, full_argv):
    runtime=parent_runtime()
    hashes = source_hashes()
    invariants = snapshot_invariants()
    validate_stage_gates(args, hashes, invariants)
    bound=control_bindings(args)
    require(not (ROOT / "outputs" / args.exp).exists(), "fresh output is required; preserve previous run")
    selection_path = VERIFY / f"{args.exp}_selection.json"
    campaign = load_campaign()
    if args.reserved_gpu is None:
        require(args.claim_fd is None and args.reservation_token is None, "internal reservation options cannot be supplied initially")
        with claim_lock(CLAIM_LOCK) as descriptor:
            selected = select_resource(campaign, preferred=args.preferred_gpu, environment=os.environ)
            args.reservation_token = uuid.uuid4().hex
            selection = {"parent_pid": os.getpid(), "process_start": process_start(),
                         **strength_protocol(args.frequency_t,args.local_factor,args.wavelet_mode,args.wavelet_shift),
                         "reservation_token": args.reservation_token, "timestamp": time.time(),
                         "stage": args.stage, "request": _request(args), "resource": selected,
                         "code_sha256": hashes, "invariants": invariants}
            write_new(selection_path, selection)
            environment = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(selected["gpu"])}
            os.execve(sys.executable, [sys.executable,"-X","utf8","-B","-u", str(Path(__file__).resolve()), *full_argv,
                       "--reserved-gpu", str(selected["gpu"]), "--reservation-token", args.reservation_token,
                       "--claim-fd", str(descriptor)], environment)
        raise RuntimeError("reexec unexpectedly returned")
    selection = read(selection_path)
    # Never close an arbitrary/stale descriptor until its ownership is verified.
    validate_reservation(args, selection, hashes, invariants)
    try:
        selected = select_resource(campaign, preferred=args.reserved_gpu, environment=os.environ)
        require(selected["gpu"] == args.reserved_gpu, "reserved GPU changed")
        if selection["resource"]["free"]:
            require(selected["free"], "previously idle GPU gained a workload; leave it untouched")
    finally:
        os.close(args.claim_fd)
        args.claim_fd = None
    # The lock can now be released: /proc exposes this parent's INITIAL CVD.
    # The parent continues to hold that loading lease until ALL child work ends.
    launch_path = VERIFY / f"{args.exp}_launch.json"
    command = _command(args, launch_path)
    launch = {"timestamp": time.time(), "parent_pid": os.getpid(), "stage": args.stage,
              **strength_protocol(args.frequency_t,args.local_factor,args.wavelet_mode, args.wavelet_shift),"control_bindings":bound,"parent_runtime":runtime,
              "parameters":_request(args),
              "reservation_token": args.reservation_token, "initial_visibility": initial_visibility(),
              "resource": selected, "initial_resource": selection["resource"], "command": command,
              "code_sha256": hashes, "invariants": invariants}
    require(source_hashes() == hashes, "density child source changed before launch")
    write_new(launch_path, launch)
    environment = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1",
                   "CUDA_CACHE_DISABLE": "1"}
    returncode, worker = None, None
    error = None
    with (VERIFY / f"{args.exp}_launch.log").open("x") as stream:
        try:
            worker = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            write_new(VERIFY / f"{args.exp}_started.json", {"parent_pid": os.getpid(), "worker_pid": worker.pid,
                      "reservation_token": args.reservation_token})
            returncode = wait_owned_worker(worker)
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        except BaseException as failure:
            error = failure
        finally:
            if worker is not None and worker.poll() is None:
                try:
                    wait_owned_worker(worker)
                except BaseException as failure:
                    if error is None:
                        error = failure
            if worker is not None:
                returncode = worker.returncode
            proof = {"status": "checking", "before": {"code_sha256": hashes, "invariants": invariants}, "after": {},
                     "control_bindings_before":bound,"control_bindings_after":{}}
            try:
                proof["after"] = {"code_sha256": source_hashes(), "invariants": snapshot_invariants()}
                require(proof["after"] == proof["before"], "source/input/checkpoint/extension state changed after child")
                proof["control_bindings_after"]={path:sha256(path) for path in bound}
                require(proof["control_bindings_after"] == bound,"review/control/recipe/incumbent changed during child")
                parent_runtime()
                proof["status"] = "verified"
            except BaseException as failure:
                proof.update(status="failed", error=repr(failure))
                if error is None:
                    error = failure
            write_new(VERIFY / f"{args.exp}_post_run_invariants.json", proof)
    if error is None:
        try:
            artifacts = validate_child_artifacts(args.stage, ROOT / "outputs" / args.exp, hashes, invariants,
                                                   **strength_protocol(args.frequency_t,args.local_factor,args.wavelet_mode, args.wavelet_shift))
        except BaseException as failure:
            error = failure
    if error is not None:
        error.worker_returncode = returncode
        error.worker_pid = worker.pid if worker is not None else None
        raise error
    proof_path = VERIFY / f"{args.exp}_post_run_invariants.json"
    write_new(VERIFY / f"{args.exp}_completed.json", {"passed": True, "status": "complete", "stage": args.stage,
              **strength_protocol(args.frequency_t,args.local_factor,args.wavelet_mode, args.wavelet_shift),"control_bindings":bound,"parent_runtime":runtime,
              "parameters":_request(args),
              "timestamp": time.time(), "parent_pid": os.getpid(), "worker_pid": worker.pid, "returncode": returncode,
              "code_sha256": hashes, "invariants": invariants, "artifacts": artifacts,
              "post_run_invariants": str(proof_path), "post_run_invariants_sha256": sha256(proof_path)})


def record_failure(args, error):
    if not args.reservation_token:
        return
    try:
        selection = read(VERIFY / f"{args.exp}_selection.json")
    except (OSError, ValueError):
        return
    if (selection.get("parent_pid"), selection.get("process_start"), selection.get("reservation_token")) != \
            (os.getpid(), process_start(), args.reservation_token):
        return
    destination = VERIFY / f"{args.exp}_failure.json"
    if destination.exists():
        return
    path = ROOT / "outputs" / args.exp / ("preflight.json" if args.stage == "tiny" else "provenance.json")
    artifact, artifact_error = {}, None
    try:
        if path.exists():
            artifact = read(path)
            require(isinstance(artifact, dict), "partial child artifact is not a JSON object")
    except (OSError, ValueError) as failure:
        artifact, artifact_error = {}, repr(failure)
    write_new(destination, {"status": "failed", "stage": args.stage, "timestamp": time.time(),
               "parent_pid": os.getpid(), "reservation_token": args.reservation_token, "error": repr(error),
               "returncode": getattr(error, "worker_returncode", getattr(error, "returncode", None)),
               "worker_pid": getattr(error, "worker_pid", None), "child_artifact": str(path),
               "artifact_status": artifact.get("status"), "artifact_read_error": artifact_error})


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--describe-runtime"]:
        print(json.dumps(parent_runtime()));return
    args = parser().parse_args(argv)
    try:
        execute(args, argv)
    except BaseException as error:
        try:
            record_failure(args, error)
        except Exception as failure:
            print(f"Failure evidence could not be written: {failure!r}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
