"""Synthetic temporary CPU queue fixtures, never clinical gates or MRI arrays."""
import copy
import json
import math
from pathlib import Path
import signal
import subprocess
import sys

import pytest

import proxdc_loop as queue


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def objective(score=-.12):
    heads,comparisons = {},{}
    for scan in ("pre","post"):
        heads[scan] = {}
        for metric,candidate in (("psnr",30.),("ssim",.9),("nmse",.01)):
            best = (candidate-score*10/math.log(10) if metric == "psnr" else
                    1-(1-candidate)*math.exp(score) if metric == "ssim" else candidate*math.exp(score))
            heads[scan][metric] = candidate
            comparisons[f"{scan}:{metric}"] = dict(candidate=candidate,best_opponent=best,
                margin=score,strict_win=score>0)
    return dict(synthetic_fixture=True,metrics=["psnr","ssim","nmse"],
        primary_objective="whole_head_min_six_normalized_margin",score=score,strict_success=score>0,
        whole_head_metrics=heads,whole_head_comparisons=comparisons)


@pytest.fixture
def synthetic(tmp_path,monkeypatch):
    """Only the observed-child validator/processes are replaced; real public
    parsing, stage-gate validation, journal and score-artifact checks execute.
    Every serialized mock gate is confined to pytest's temporary directory.
    """
    monkeypatch.setattr(queue,"ROOT",tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(queue.helpers,"ROOT",tmp_path)
    monkeypatch.setattr(queue.helpers,"source_commit",lambda:"synthetic_commit")
    hashes,invariants = {"synthetic_source":"not_a_clinical_hash"},{"synthetic_fixture":True}
    monkeypatch.setattr(queue.protocol,"code_hashes",lambda:hashes)
    monkeypatch.setattr(queue.protocol,"snapshot_invariants",lambda:invariants)
    keeper=tmp_path/"synthetic_keeper.json";save(keeper,{"synthetic_fixture":True})
    prior={**queue.PRIOR,"source":queue.s.descriptor(keeper)}
    monkeypatch.setattr(queue,"PRIOR",prior)
    baseline=tmp_path/"synthetic_baseline.json";save(baseline,objective(-.11778703065863405))
    monkeypatch.setattr(queue,"BASELINE",queue.s.descriptor(baseline))
    review=tmp_path/"synthetic_review.json"
    report=tmp_path/"synthetic_review.md";report.write_text("SYNTHETIC fixture, not an actual independent clinical review")
    save(review,dict(synthetic_fixture=True,schema_version=1,status="passed",passed=True,round=1,
        actionable_findings=0,unresolved_findings=0,route=queue.protocol.ROUTE,
        review_report=queue.s.descriptor(report),code_sha256=hashes,invariants=invariants))
    cli=["--review-gate",str(review)]
    for stage,exp in queue.STAGES:
        cli += [f"--{stage}-recipe",str(tmp_path/f"synthetic_{stage}_recipe.json")]
    args=queue.parser().parse_args(cli)
    for stage,exp in queue.STAGES:
        parsed=queue.protocol.parser().parse_args(queue.supervisor_command(args,stage,exp)[4:])
        save(getattr(args,stage+"_recipe"),dict(synthetic_fixture=True,schema_version=1,status="selected",
            route=queue.protocol.ROUTE,parameters=queue.protocol.parameters(parsed),code_sha256=hashes,invariants=invariants))
    events=[]
    real_validate=queue.protocol.validate_stage_gates

    def validate(parsed,actual_hashes,actual_invariants):
        events.append(("validate",parsed.stage))
        return real_validate(parsed,actual_hashes,actual_invariants)

    def observed(path,stage,actual_hashes,actual_invariants):
        value=queue.s.read(path)
        assert value["synthetic_fixture"] is True
        assert value["stage"] == stage and value["code_sha256"] == actual_hashes
        assert value["invariants"] == actual_invariants
        assert all(queue.s.sha256(p)==digest for p,digest in value["artifacts"].items())
        return value

    def run(command,log,state,kind,cpu=False):
        events.append(("run",kind if kind == "scorer" else state["active"]["stage"]))
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("x") as stream:stream.write("SYNTHETIC CPU process replacement, no GPU or MRI data\n")
        exp=state["active"]["exp"];directory=tmp_path/"outputs"/exp
        if kind == "supervisor":
            parsed=queue.protocol.parser().parse_args(command[4:])
            controls=real_validate(parsed,hashes,invariants)["control_bindings"]
            directory.mkdir(parents=True)
            metrics=dict(synthetic_fixture=True,training_seconds=.01,peak_memory_gb=.02,
                steps_observed=0,work_total={"synthetic_normal_calls":2},scans={"synthetic_fixture":True})
            save(directory/"metrics.json",metrics)
            artifacts={str(directory/"metrics.json"):queue.s.sha256(directory/"metrics.json")}
            if parsed.stage == "prox":
                for scan in ("pre","post"):
                    path=directory/f"candidate_{scan}_complex.npy"
                    path.write_bytes(b"SYNTHETIC bytes, not MRI array "+scan.encode())
                    artifacts[str(path)]=queue.s.sha256(path)
            save(directory/"completed.json",dict(synthetic_fixture=True,stage=parsed.stage,
                parameters=queue.protocol.parameters(parsed),run_dir=str(directory),artifacts=artifacts,
                control_bindings=controls,code_sha256=hashes,invariants=invariants))
        else:
            assert cpu is True and command[3] == str(tmp_path/"evaluate.py")
            out=tmp_path/"scores"/exp;out.mkdir(parents=True)
            value=objective();value.update(name=exp,run_dir=str(directory),all_inputs_unchanged=True,
                candidate_sha256={},input_hashes=dict(candidate={},files={}))
            for scan in ("pre","post"):
                path=directory/f"candidate_{scan}_complex.npy";digest=queue.s.sha256(path)
                value["candidate_sha256"][scan]=digest
                value["input_hashes"]["candidate"][scan]=dict(path=str(path),sha256=digest)
                value["input_hashes"]["files"][str(path)]=digest
            for name in ("whole_head_contract.json","lesion_planes.json","enhancement.json","input_hashes.json"):
                save(out/name,{"synthetic_fixture":True})
            (out/"whole_head.csv").write_text("synthetic_fixture\n")
            save(out/"objective.json",value)
        return dict(synthetic_fixture=True,returncode=0,wall_seconds=.01)

    monkeypatch.setattr(queue.protocol,"validate_stage_gates",validate)
    monkeypatch.setattr(queue.protocol,"validate_observed_gate",observed)
    monkeypatch.setattr(queue.helpers,"run_owned",run)
    return args,cli,events,tmp_path


def test_exact_public_commands_are_parsed_and_fixed():
    args=queue.parser().parse_args(["--gpu","3"])
    for stage,exp in queue.STAGES:
        command=queue.supervisor_command(args,stage,exp)
        parsed=queue.protocol.parser().parse_args(command[4:])
        assert parsed.preferred_gpu==3 and parsed.exp==exp and parsed.stage==stage
        assert queue.protocol.parameters(parsed)==dict(route="g2_raw_proxdc_v1",stage=stage,
            cg_max_matvecs=32,cg_rtol=1e-5,lambda_ratio=.1)
        assert bool(parsed.tiny_gate)==(stage!="tiny")
        assert bool(parsed.canonical_gate)==(stage=="prox")
    with pytest.raises(SystemExit):queue.parser().parse_args(["--lambda-ratio",".2"])


def test_public_import_remains_stdlib_only():
    completed=subprocess.run([sys.executable,"-B","-c",
        "import proxdc_loop,sys; assert 'torch' not in sys.modules; proxdc_loop.parser().parse_args(['--help'])"],
        capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr


def test_actual_gate_validation_precedes_each_child_and_only_prox_is_scored(synthetic):
    args,_,events,root=synthetic
    state=queue.execute(args)
    assert events==[("validate","tiny"),("run","tiny"),("validate","canonical"),("run","canonical"),
                   ("validate","prox"),("run","prox"),("run","scorer")]
    assert state["status"]=="candidate_complete_goal_unmet" and state["success"] is False
    assert state["prior_incumbent"]["score"]==-.11118128532897047
    records=queue.helpers.Ledger(root).records
    assert [row["event"] for row in records]==["baseline_context","stage_complete","stage_complete","stage_complete","scored"]
    result=records[-1]
    assert result["status"]=="discard" and result["scientific_rejection"] is True
    assert result["formal_output_corrections"]==0 and result["optimizer_updates"]==0
    assert result["formal_output_correction_id"]=="g2_proxdc_candidate_v1"
    assert sum(row["formal_output_corrections"] for row in records)==1
    assert result["work_total"]=={"synthetic_normal_calls":2}
    assert all(row["formal_output_corrections"]==0 for row in records[:3])
    assert (root/"results.tsv").read_text().splitlines()[0].split("\t")==list(queue.helpers.RESULT_COLUMNS)
    assert len((root/"results.tsv").read_text().splitlines())==3


def test_completed_reuse_revalidates_and_never_duplicates_or_reruns(synthetic):
    args,_,events,root=synthetic;queue.execute(args)
    before=(root/"autoresearch.jsonl").read_bytes();events.clear()
    queue.execute(args)
    assert events==[("validate","tiny"),("validate","canonical"),("validate","prox")]
    assert (root/"autoresearch.jsonl").read_bytes()==before


def test_incomplete_run_is_preserved_and_stops_without_retry(synthetic):
    _,cli,events,root=synthetic
    directory=root/"outputs/g2_proxdc_tiny_v1";directory.mkdir(parents=True)
    marker=directory/"failed_evidence";marker.write_text("preserve")
    assert queue.main(cli)==1
    assert queue.s.read(root/"loop_state.json")["status"]=="failed_setup"
    assert queue.helpers.Ledger(root).records[-1]["scientific_rejection"] is False
    before=(root/"autoresearch.jsonl").read_bytes();events.clear()
    assert queue.main(cli)==1 and not events
    assert marker.read_text()=="preserve" and (root/"autoresearch.jsonl").read_bytes()==before


def test_incomplete_score_is_refused_after_controls_without_scientific_rejection(synthetic):
    _,cli,events,root=synthetic;(root/"scores/g2_proxdc_candidate_v1").mkdir(parents=True)
    assert queue.main(cli)==1
    assert ("run","scorer") not in events
    state=queue.s.read(root/"loop_state.json")
    assert state["status"]=="failed_setup" and state["failure"]["scientific_rejection"] is False


def test_real_gate_validator_refuses_changed_recipe_before_child(synthetic):
    args,cli,events,root=synthetic
    recipe=queue.s.read(args.tiny_recipe);recipe["parameters"]["lambda_ratio"]=.2;save(args.tiny_recipe,recipe)
    assert queue.main(cli)==1 and events==[("validate","tiny")]
    assert queue.s.read(root/"loop_state.json")["status"]=="failed_setup"


@pytest.mark.parametrize("status",["failed_setup","crash","cancelled"])
def test_saved_failure_or_cancel_is_never_retried(synthetic,status):
    args,_,events,root=synthetic
    save(root/"loop_state.json",dict(family=queue.FAMILY,root=str(root),status=status))
    original=(root/"loop_state.json").read_bytes()
    with pytest.raises(ValueError,match="no automatic retry"):queue.execute(args)
    assert not events and (root/"loop_state.json").read_bytes()==original


def test_mutable_saved_keeper_cannot_weaken_the_historical_anchor(synthetic):
    args,_,events,root=synthetic;queue.execute(args)
    state=queue.s.read(root/"loop_state.json");state["prior_incumbent"]["score"]=-.2;save(root/"loop_state.json",state)
    original=(root/"loop_state.json").read_bytes();events.clear()
    with pytest.raises(ValueError,match="immutable historical keeper"):queue.execute(args)
    assert not events and (root/"loop_state.json").read_bytes()==original


def test_foreign_queue_state_is_refused_without_relabeling(synthetic):
    _,cli,events,root=synthetic;save(root/"loop_state.json",dict(family="foreign",root=str(root)))
    original=(root/"loop_state.json").read_bytes()
    assert queue.main(cli)==1 and not events
    assert (root/"loop_state.json").read_bytes()==original and not (root/"autoresearch.jsonl").exists()


def test_foreign_journal_binding_is_refused_without_modification(synthetic):
    args,_,events,root=synthetic;queue.execute(args)
    path=root/"autoresearch.jsonl";rows=queue.helpers.Ledger._read(path.read_text())
    rows[0]["bindings"]["family"]="foreign";path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    original=path.read_bytes();state=(root/"loop_state.json").read_bytes();events.clear()
    with pytest.raises(ValueError,match="foreign journal"):queue.execute(args)
    assert not events and path.read_bytes()==original and (root/"loop_state.json").read_bytes()==state


@pytest.mark.parametrize("field",["parameters","control_bindings"])
def test_completed_reuse_requires_exact_parameters_and_current_control_descriptors(synthetic,field):
    args,_,events,root=synthetic;queue.execute(args)
    path=root/"outputs/g2_proxdc_tiny_v1/completed.json";gate=queue.s.read(path)
    gate[field]={"synthetic_changed":True};save(path,gate);events.clear()
    with pytest.raises(ValueError,match="parameters/path/controls"):queue.execute(args)
    assert events==[("validate","tiny")]


def test_pending_cancellation_after_prox_does_not_start_the_scorer(synthetic,monkeypatch):
    _,cli,events,root=synthetic;checks=[]
    def cancel_at_score_boundary():
        checks.append(True)
        if len(checks)==4:
            raise queue.s.SupervisorCancelled(signal.SIGTERM)
    monkeypatch.setattr(queue.s,"check_cancellation",cancel_at_score_boundary)
    assert queue.main(cli)==143
    assert ("run","scorer") not in events
    assert queue.s.read(root/"loop_state.json")["status"]=="cancelled"


@pytest.mark.parametrize("cancelled",[False,True])
def test_process_failure_and_cancellation_are_not_scientific_rejections(synthetic,monkeypatch,cancelled):
    _,cli,_,root=synthetic
    def fail(*args,**kwargs):
        error=queue.s.SupervisorCancelled(signal.SIGTERM) if cancelled else RuntimeError("synthetic child failure")
        error.loop_runtime={"pid":12345,"synthetic_fixture":True,"cleanup":{"cleanup_confirmed":True}}
        raise error
    monkeypatch.setattr(queue.helpers,"run_owned",fail)
    assert queue.main(cli)==(143 if cancelled else 1)
    state=queue.s.read(root/"loop_state.json")
    assert state["status"]==("cancelled" if cancelled else "crash")
    assert state["failure"]["scientific_rejection"] is False


def test_keeper_and_provisional_quality_decisions_use_reused_measured_rules():
    assert queue.helpers.classify_objective(objective(-.115),queue.PRIOR["score"],queue.THRESHOLD)=="discard"
    assert queue.helpers.classify_objective(objective(-.10),queue.PRIOR["score"],queue.THRESHOLD)=="keep"
    assert queue.helpers.classify_objective(objective(.01),queue.PRIOR["score"],queue.THRESHOLD)=="provisional_win"
