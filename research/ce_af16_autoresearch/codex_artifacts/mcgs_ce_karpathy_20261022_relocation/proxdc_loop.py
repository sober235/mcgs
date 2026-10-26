#!/usr/bin/env python3
"""One fixed proxDC candidate: observed controls, measured score, keep/discard."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import sys

import autoresearch_loop as helpers
import run_proxdc as protocol

s = protocol.base
ROOT = Path(__file__).resolve().parent
FAMILY = "g2_raw_proxdc_queue_v1"
STAGES = (("tiny", "g2_proxdc_tiny_v1"), ("canonical", "g2_proxdc_canonical_v1"),
          ("prox", "g2_proxdc_candidate_v1"))
PRIOR = {"name": "lowreg_f075_solve_v1", "score": -0.11118128532897047,
    "source": {"path": str(s.OLD/"frequency_lowreg_mm/outputs/lowreg_f075_solve_v1/metrics.json"),
               "sha256": "8ceeeee1f0d44aff6c080f572714be9382fc7f4789c6f052510bc1f32bd26805"}}
BASELINE = {"path": str(ROOT.parent/"mcgs_ce_karpathy_20261019/scores/family1_g2_baseline_v1/objective.json"),
            "sha256": "55fcbfc170e27d8d1be6c6b0013743037228f237baf25ccc6032da42edbcc0c2"}
THRESHOLD = .0001
STOPPED = ("failed_setup", "crash", "cancelled")


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--review-gate",type=Path,default=Path("verification/proxdc_review_v1.json"))
    for stage,exp in STAGES:
        p.add_argument(f"--{stage}-recipe",type=Path,default=Path(f"recipes/{exp}.json"))
    return p


def supervisor_command(args, stage, exp):
    command = [sys.executable,"-B","-u",str(ROOT/"run_proxdc.py"),
        "--route",protocol.ROUTE,"--stage",stage,"--exp",exp,
        "--review-gate",str(args.review_gate.resolve()),"--recipe",str(getattr(args,stage+"_recipe").resolve()),
        "--cg-max-matvecs","32","--cg-rtol","1e-5","--lambda-ratio","0.1",
        "--preferred-gpu",str(args.gpu),"--timeout","3600"]
    if stage != "tiny":
        command += ["--tiny-gate",str(ROOT/"outputs/g2_proxdc_tiny_v1/completed.json")]
    if stage == "prox":
        command += ["--canonical-gate",str(ROOT/"outputs/g2_proxdc_canonical_v1/completed.json")]
    return command


def existing_run(exp, parsed, hashes, invariants, controls):
    directory = ROOT/"outputs"/exp
    if not directory.exists():
        return None
    s.require((directory/"completed.json").is_file(), f"existing incomplete run refused: {directory}")
    gate = protocol.validate_observed_gate(directory/"completed.json",parsed.stage,hashes,invariants)
    s.require(gate["parameters"] == protocol.parameters(parsed) and Path(gate["run_dir"]).resolve() == directory
        and gate["control_bindings"] == controls, "completed run parameters/path/controls differ")
    return gate


def bindings(args, hashes, invariants):
    return dict(family=FAMILY,root=str(ROOT),queue_sha256=s.sha256(__file__),
        helper_sha256=s.sha256(helpers.__file__),code_sha256=hashes,invariants=invariants,
        review_gate=s.descriptor(args.review_gate),recipes={stage:s.descriptor(getattr(args,stage+"_recipe"))
            for stage,_ in STAGES},baseline=BASELINE,prior_incumbent=PRIOR,minimum_keep_improvement=THRESHOLD)


def execute(args):
    s.require(type(args.gpu) is int and args.gpu >= 0, "GPU index must be nonnegative")
    saved = s.read(ROOT/"loop_state.json") if (ROOT/"loop_state.json").exists() else None
    s.require(saved is None or (saved.get("family") == FAMILY and saved.get("root") == str(ROOT)),
              "foreign queue state refused without modification")
    s.require(saved is None or saved.get("status") not in STOPPED, "queue stopped; no automatic retry")
    path = ROOT/"autoresearch.jsonl"
    rows = helpers.Ledger._read(path.read_text() if path.exists() else "")
    s.require(not rows or saved is not None, "journal without queue state refused")
    state = saved or dict(family=FAMILY,root=str(ROOT),status="active",active=None,stages={},results={},
        source_commit=helpers.source_commit(),prior_incumbent=PRIOR,minimum_keep_improvement=THRESHOLD)
    try:
        hashes,invariants = protocol.code_hashes(),protocol.snapshot_invariants()
        prior,threshold = helpers.load_prior({"previous_whole_head_incumbent":PRIOR,
            "minimum_keep_improvement":THRESHOLD},saved)
        s.require(prior == PRIOR and threshold == THRESHOLD, "immutable historical keeper differs")
        baseline = s.read(s.verify_descriptor(BASELINE,"existing scored G2 baseline"))
        helpers.classify_objective(baseline,prior["score"],threshold)
        pins = bindings(args,hashes,invariants)
        s.require(saved is None or saved.get("bindings") == pins, "existing queue source/control bindings differ")
        s.require(all(row.get("bindings") == pins for row in rows), "foreign journal bindings refused")
    except BaseException as error:
        # Foreign/stale saved state and journals are preserved exactly. A fresh
        # setup refusal can be recorded without claiming complete input evidence.
        if saved is None and not rows:
            error.loop_state = state
        raise
    state["bindings"] = pins
    ledger = helpers.Ledger(ROOT)
    ledger._project()  # Refuse a foreign TSV before modifying queue state.
    helpers.atomic_state(state)
    common = dict(family=FAMILY,bindings=pins,source_commit=state["source_commit"])
    try:
        if ledger.get(FAMILY+":baseline_context") is None:
            ledger.append(dict(record_id=FAMILY+":baseline_context",event="baseline_context",**common,
                objective=BASELINE,formal_output_corrections=0,optimizer_updates=0,
                result_row=dict(commit=state["source_commit"],whole_head_margin=baseline["score"],memory_gb="",
                    status="baseline",description="Existing scored G2 context; no new stage or training")))
        for stage,exp in STAGES:
            state.update(active=dict(exp=exp,stage=stage,pid=None,process_kind="setup"),status="active")
            helpers.atomic_state(state)
            s.check_cancellation()
            s.require(bindings(args,protocol.code_hashes(),protocol.snapshot_invariants()) == pins,
                      "queue source, frozen input or control bytes changed")
            command = supervisor_command(args,stage,exp)
            parsed = protocol.parser().parse_args(command[4:])
            controls = protocol.validate_stage_gates(parsed,hashes,invariants)["control_bindings"]
            gate = existing_run(exp,parsed,hashes,invariants,controls)
            runtime = {"reused_completed":True}
            if gate is None:
                runtime = helpers.run_owned(command,ROOT/"loop_logs"/f"{exp}_supervisor.log",state,"supervisor")
                gate = existing_run(exp,parsed,hashes,invariants,controls)
                s.require(gate is not None, "successful supervisor omitted its completed gate")
            state["active"].update(pid=None,process_kind="setup")
            descriptor = s.descriptor(ROOT/"outputs"/exp/"completed.json")
            key = FAMILY+":"+stage
            previous = ledger.get(key)
            if previous is not None:
                s.require(previous["completed_gate"] == descriptor, "recorded completed gate changed")
            else:
                ledger.append(dict(record_id=key,event="stage_complete",stage=stage,exp=exp,**common,
                    completed_gate=descriptor,runtime=runtime,formal_output_corrections=int(stage == "prox"),optimizer_updates=0))
            state["stages"][exp] = dict(completed_gate=descriptor,parameters=protocol.parameters(parsed),runtime=runtime)
            helpers.atomic_state(state)
        # Only prox is scored. Tiny/canonical remain actual observed controls.
        state["active"].update(pid=None,process_kind="score_setup")
        s.check_cancellation()
        value,objective,scoring,reports = helpers.score_run(exp,gate,state,hashes)
        s.require(bindings(args,protocol.code_hashes(),protocol.snapshot_invariants()) == pins,
                  "queue source, frozen input or control bytes changed during scoring")
        s.check_cancellation()
        status = helpers.classify_objective(value,prior["score"],threshold)
        metrics = s.read(Path(gate["run_dir"])/"metrics.json")
        record = dict(record_id=FAMILY+":scored",event="scored",**common,exp=exp,status=status,
            completed_gate=descriptor,objective=objective,score_artifacts=reports,score=value["score"],
            strict_success=value["strict_success"],whole_head_metrics=value["whole_head_metrics"],
            incumbent_before=prior["score"],scientific_rejection=status == "discard",formal_output_corrections=0,
            formal_output_correction_id=exp,
            optimizer_updates=0,work_total=metrics["work_total"],scans=metrics["scans"],runtime=dict(scoring=scoring,
                numerical_seconds=metrics["training_seconds"],peak_memory_gb=metrics["peak_memory_gb"]),
            result_row=dict(commit=state["source_commit"],whole_head_margin=value["score"],
                memory_gb=metrics["peak_memory_gb"],status=status,
                description=f"{exp}; raw G2 proxDC; 0 optimizer updates; actual CG work in JSONL"))
        previous = ledger.get(record["record_id"])
        if previous is None:
            ledger.append(record)
        else:
            s.require(all(previous[k] == record[k] for k in ("completed_gate","objective","score_artifacts","score","status")),
                      "recorded scientific evidence changed")
        state.update(status="provisional_win_wait_confirmation" if status == "provisional_win" else "candidate_complete_goal_unmet",
            active=None,success=False,results={exp:{k:record[k] for k in ("status","score","objective")}},
            best_whole_head_margin=max(prior["score"],value["score"] if status in ("keep","provisional_win") else prior["score"]),
            next_step="independent repeat and matched GS-single required" if status == "provisional_win" else "root selects next hypothesis")
        helpers.atomic_state(state)
        return state
    except BaseException as error:
        error.loop_state = state
        raise


def record_stop(error):
    state = getattr(error,"loop_state",None)
    if state is None:
        return  # Preserve foreign state/journal rather than relabeling it.
    runtime = getattr(error,"loop_runtime",{})
    status = "cancelled" if isinstance(error,s.SupervisorCancelled) else (
        "crash" if runtime.get("pid") or (state.get("active") or {}).get("pid") else "failed_setup")
    failure = dict(status=status,error=repr(error),runtime=runtime,scientific_rejection=False,requires_root_diagnosis=True)
    state.update(status=status,failure=failure)
    helpers.atomic_state(state)
    ledger = helpers.Ledger(ROOT)
    key = FAMILY+":failure"
    if ledger.get(key) is None:
        ledger.append(dict(record_id=key,event="queue_stopped",family=FAMILY,bindings=state.get("bindings"),**failure,
            result_row=dict(commit=state["source_commit"],whole_head_margin="",memory_gb="",status=status,
                            description="Retain evidence; no scientific rejection; no automatic retry")))


def main(argv=None):
    os.chdir(ROOT)
    args = parser().parse_args(argv)
    with (ROOT/".proxdc_queue.lock").open("a+") as lock,s.cancellation_signals():
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another proxDC queue owns this worktree.",file=sys.stderr)
            return 1
        try:
            result = execute(args)
        except BaseException as error:
            record_stop(error)
            print(f"proxDC queue stopped: {error}",file=sys.stderr)
            return 128+error.signum if isinstance(error,s.SupervisorCancelled) else 1
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
