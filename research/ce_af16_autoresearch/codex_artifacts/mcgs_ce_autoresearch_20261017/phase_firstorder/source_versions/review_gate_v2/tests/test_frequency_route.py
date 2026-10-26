"""Frequency route contracts and saved-array regression checks; CPU only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import types

import numpy as np
import pytest
import torch

import runner
import run_gpu as launch
from density_operator import DensityLinearOperator
from test_frequency_operator import frequency_trainer
from test_run_gpu import supervisor,own_selection,fake_worker


@pytest.mark.parametrize("t",[0.,.5,1.])
def test_public_frequency_profile_and_actual_child_command(t,tmp_path):
    assert runner.frequency_protocol(t) == launch.frequency_protocol(t)
    for stage in ("tiny","zero","solve"):
        args=launch.parser().parse_args(["--stage",stage,"--exp","fixture","--frequency-t",str(t),"--review-gate",str(tmp_path/"review")])
        if t != 1.:
            with pytest.raises(ValueError,match="fixedt1"):
                launch._command(args,tmp_path/"launch")
            continue
        command=launch._command(args,tmp_path/"launch")
        assert command[command.index("--frequency-t")+1] == str(t)
        assert command[command.index("--wavelet-mode")+1] == "uniform8"
        assert launch._request(args)["frequency_t"] == t
        assert len(launch.PRODUCTION) == 9 and "frequency_weights.py" in launch.PRODUCTION


@pytest.mark.parametrize("t",[-1.,.25,float("nan"),True])
def test_undeclared_frequency_parameters_are_rejected(t):
    with pytest.raises(ValueError): runner.frequency_protocol(t)
    with pytest.raises(ValueError): launch.frequency_protocol(t)


@pytest.mark.parametrize("mode,shift",[("single",None),("uniform8",[1,1,1])])
def test_frequency_route_cannot_change_phase_protocol(mode,shift):
    with pytest.raises(ValueError): runner.frequency_protocol(.5,mode,shift)
    with pytest.raises(ValueError): launch.frequency_protocol(.5,mode,shift)


@pytest.mark.parametrize("t",[0.,.5,1.])
def test_independent_saved_six_array_reader_checks_formula_and_source(t,tmp_path):
    operator=DensityLinearOperator(frequency_trainer(),frequency_t=t)
    evidence=operator.save_frequency_evidence(tmp_path)
    result=runner.validate_saved_frequency_evidence(tmp_path,evidence,t,operator)
    assert result["passed"] is True and result["frequency_t"] == t
    assert result["scans"]["pre"]["acquired_count"] == 3
    assert result["scans"]["post"]["acquired_count"] == 4
    assert all(row["effective_identity"] == (t == 0.) and row["interpolation_max_relative_error"] <= row["interpolation_relative_tolerance"]
               for row in result["scans"].values())


def rewrite_array(evidence,scan,name,value):
    info=evidence["scans"][scan][name];path=Path(info["path"])
    np.save(path,value,allow_pickle=False)
    info.update(file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),byte_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                shape=list(value.shape),dtype="torch."+str(value.dtype))
    (path.parent/"manifest.json").write_text(json.dumps(evidence))


def test_metadata_consistent_unflattened_array_still_fails_full_formula(tmp_path):
    operator=DensityLinearOperator(frequency_trainer(),frequency_t=.5)
    evidence=operator.save_frequency_evidence(tmp_path)
    source=np.load(evidence["scans"]["pre"]["source_weight"]["path"],allow_pickle=False)
    row=evidence["scans"]["pre"]
    row["effective_mean_squared_weight"]=row["source_mean_squared_weight"];row["relative_mass_error"]=0.
    rewrite_array(evidence,"pre","effective_weight",source)
    with pytest.raises(ValueError,match="interpolation/RMS"):
        runner.validate_saved_frequency_evidence(tmp_path,evidence,.5,operator)


def test_saved_fit_mask_cannot_replace_original_acquired_mask(tmp_path):
    operator=DensityLinearOperator(frequency_trainer(),frequency_t=.5)
    evidence=operator.save_frequency_evidence(tmp_path)
    fit=operator.trainer.mask_fit_p.detach().cpu().numpy()
    rewrite_array(evidence,"pre","acquired_mask",fit)
    with pytest.raises(ValueError,match="original acquired masks"):
        runner.validate_saved_frequency_evidence(tmp_path,evidence,.5,operator)


def test_saved_array_tampering_is_rejected_before_numeric_claim(tmp_path):
    operator=DensityLinearOperator(frequency_trainer(),frequency_t=1.)
    evidence=operator.save_frequency_evidence(tmp_path)
    Path(evidence["scans"]["post"]["effective_weight"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="path/hash"):
        runner.validate_saved_frequency_evidence(tmp_path,evidence,1.,operator)


def test_t0_checkpoint_stays_nonresumable_and_binds_frequency_metadata(tmp_path):
    operator=DensityLinearOperator(frequency_trainer(),frequency_t=0.)
    result=runner.optimize_density(operator,prior_kwargs={},wavelet_mode="uniform8",config=runner.SolverConfig(outer_steps=0))
    evidence=operator.save_frequency_evidence(tmp_path)
    execution={"declared_parameters":runner.frequency_protocol(0.),"child_command":["fixture"],"source_code_sha256":{}}
    path=tmp_path/"solver.pth"
    runner.write_solver_checkpoint(path,operator.model,result,source_checkpoint="fixture_source",source_checkpoint_sha256="fixture_sha",
                                   frequency_t=0.,frequency_evidence=evidence,execution=execution)
    after={"parameter_sha256":{n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                               for n,p in operator.model.named_parameters()}}
    assert runner.validate_saved_solver(path,after,result,0.,evidence,execution)["passed"] is True
    saved=torch.load(path,map_location="cpu",weights_only=True)
    assert "iter" not in saved and "optim_state" not in saved and saved["frequency_t"] == 0.
    saved["frequency_t"]=.5;torch.save(saved,path)
    with pytest.raises(ValueError,match="schema/parameters"):
        runner.validate_saved_solver(path,after,result,0.,evidence,execution)


def test_zero_source_complex_reproduction_bound_is_fixed():
    checks={s:dict(passed=True,source_array_relative_l2=0.) for s in ("pre","post")}
    assert runner.validate_zero_exports(checks)["exact_equal"] == {"pre":True,"post":True}
    checks["post"]["source_array_relative_l2"]=1.01e-5
    with pytest.raises(RuntimeError,match="fixed1e-5"):
        runner.validate_zero_exports(checks)


def test_actual_NVGEN_parent_remains_stdlib_only_and_numeric_child_rejects_it():
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1","OMP_NUM_THREADS":"1"}
    result=subprocess.run([str(launch.CPU_PYTHON),"-X","utf8","-B",str(launch.ROOT/"run_gpu.py"),"--describe-runtime"],capture_output=True,text=True,env=env)
    assert result.returncode == 0,result.stderr
    value=json.loads(result.stdout)
    assert value["torch_imported"] is False and value["weights_imported"] is False
    code="import sys;sys.path.insert(0,"+repr(str(launch.ROOT))+");import runner;runner.numerical_runtime()"
    wrong=subprocess.run([str(launch.CPU_PYTHON),"-X","utf8","-B","-c",code],capture_output=True,text=True,env=env)
    assert wrong.returncode != 0 and "actual GSTMI" in wrong.stderr


def test_actual_tiny_gate_cannot_be_reused_for_another_frequency(supervisor,monkeypatch):
    own_selection(supervisor);fake_worker(monkeypatch,supervisor)
    launch.execute(supervisor.args,[])
    for t in (0.,.5):
        with pytest.raises(ValueError,match="protocol mismatch|fixedt1"):
            launch.validate_observed_gate(supervisor.verify/"trial_completed.json","tiny",supervisor.hashes,supervisor.invariants,frequency_t=t)


@pytest.mark.parametrize("key,value",[("frequency_t",0.),("outer_steps",4),("cg_steps",79),("local_factor",.25)])
def test_formal_frequency_solve_enforces_declared_recipe_before_launch(supervisor,monkeypatch,key,value):
    supervisor.args.stage="solve";supervisor.args.frequency_t=1.;supervisor.args.local_factor=1.
    supervisor.args.cuda_operator_gate=Path("fixture_tiny")
    supervisor.args.zero_audit_gate=Path("fixture_zero")
    setattr(supervisor.args,key,value)
    monkeypatch.setattr(launch,"validate_observed_gate",lambda *a,**k:None)
    with pytest.raises(ValueError,match="fixed5x96|fixedt1|cg_steps96"):
        launch.validate_stage_gates(supervisor.args,supervisor.hashes,supervisor.invariants)


def test_selected_recipe_uses_latest_dynamic_prior_and_incumbent(tmp_path,monkeypatch):
    from test_lowreg_protocol import quality_recipe
    path,hashes,invariants=quality_recipe(tmp_path,monkeypatch)
    assert launch.validate_recipe(path,hashes,invariants)["incumbent"]["score"] == -.123
    document=tmp_path/"development_incumbent.json";current=json.loads(document.read_text())
    current["score"]=-.02;document.write_text(json.dumps(current))
    with pytest.raises(ValueError,match="incumbent bytes"):
        launch.validate_recipe(path,hashes,invariants)
