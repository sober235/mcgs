"""TEST fixtures for reporting; synthetic values never enter a real campaign report."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def report():
    path = ROOT / "report.py"
    assert path.exists(), "completed-trial report implementation is missing"
    spec = importlib.util.spec_from_file_location("campaign_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ledger_snapshot_ignores_unscored_events_but_rejects_truncated_tail(report, tmp_path):
    path = tmp_path / "TEST_ledger.jsonl"
    path.write_text(json.dumps({"event":"trial_started","exp":"B"}) + "\n")
    rows, evidence = report.read_ledger_snapshot(path)
    assert rows == [{"event":"trial_started","exp":"B"}]
    assert evidence["record_count"] == 1
    path.write_text('{"event":"trial_scored"')
    with pytest.raises(ValueError, match="incomplete"):
        report.read_ledger_snapshot(path)


def test_ledger_snapshot_rejects_nonfinite_numbers(report, tmp_path):
    path = tmp_path / "TEST_nan.jsonl"
    path.write_text('{"event":"trial_scored","score":NaN}\n')
    with pytest.raises(ValueError, match="nonfinite"):
        report.read_ledger_snapshot(path)


def test_plane_counts_preserve_all_59_and_separate_scans_from_joint_success(report):
    planes = []
    for index in range(59):
        comparisons = {f"{scan}:{metric}":{"strict_win":False}
                       for scan in ("pre","post") for metric in ("psnr","ssim","nmse","hfen")}
        if index in (0,1):
            for metric in ("psnr","ssim","nmse","hfen"):
                comparisons[f"pre:{metric}"]["strict_win"] = True
        if index in (1,2):
            for metric in ("psnr","ssim","nmse","hfen"):
                comparisons[f"post:{metric}"]["strict_win"] = True
        planes.append({"axis":0,"index":index,"core":150,"comparisons":comparisons,
                       "strict_success":index == 1})
    result = report.plane_counts({"planes":planes})
    assert result["eligible_planes"] == 59
    assert result["all_four_by_scan"] == {"pre":2,"post":2}
    assert result["all_eight_both_scans"] == 1
    assert result["per_condition"]["pre:psnr"] == 2
    with pytest.raises(ValueError, match="59"):
        report.plane_counts({"planes":planes[:-1]})


def test_enhancement_uses_own_scan_wm_units_all_six_regions_and_post_background(report):
    shape = (3,3,3)
    masks = {}
    for index, name in enumerate(report.REGIONS):
        mask = np.zeros(shape,bool)
        mask.flat[index] = True
        masks[name] = mask
    pre = np.ones(shape,np.float32) * 2
    post = np.ones(shape,np.float32) * 4
    # WM index2 is 2 and4, while core and rim change independently.
    post.flat[0] = 8
    pre.flat[1] = 4
    background = np.zeros(shape,bool)
    background[-1] = True
    row = report.enhancement_row("TEST", "TEST fixture", {"pre":pre,"post":post}, masks, background)
    assert row["lesion_core"] == 1.
    assert row["rim_0_3mm"] == -1.
    assert row["white_matter"] == 0.
    assert row["background_post"] == 1.
    assert all(name in row for name in report.REGIONS)
    scaled = report.enhancement_row("TEST", "TEST scaled", {"pre":pre*3,"post":post*.2}, masks, background)
    for name in (*report.REGIONS,"background_post"):
        assert scaled[name] == pytest.approx(row[name], abs=2e-7)


def test_enhancement_rejects_empty_roi_zero_wm_or_nonfinite_input(report):
    shape=(2,2,2)
    masks={name:np.ones(shape,bool) for name in report.REGIONS}
    pair={"pre":np.ones(shape,np.float32),"post":np.ones(shape,np.float32)}
    background=np.ones(shape,bool)
    masks["lesion_core"] = np.zeros(shape,bool)
    with pytest.raises(ValueError,match="empty"):
        report.enhancement_row("TEST","TEST",pair,masks,background)
    masks["lesion_core"][:] = True
    with pytest.raises(ValueError,match="median"):
        report.enhancement_row("TEST","TEST",{**pair,"pre":pair["pre"]*0},masks,background)
    pair["post"][0,0,0]=np.nan
    with pytest.raises(ValueError,match="finite"):
        report.enhancement_row("TEST","TEST",pair,masks,background)


def test_background_matches_reference_threshold_and_six_dilations(report):
    from scipy import ndimage
    gt = np.zeros((19,19,19),np.float32)
    gt[7:12,7:12,7:12] = 2
    expected=~ndimage.binary_dilation(gt > .05*float(np.quantile(gt,.999)),iterations=6)
    assert np.array_equal(report.background_mask(gt),expected)


def test_head_metrics_require_all_four_finite_volume_values(report):
    row={"head_psnr":35.,"head_ssim":.94,"head_nmse":.01,"head_hfen":.12,"psnr":99.}
    assert report.head_metrics(row)=={"psnr":35.,"ssim":.94,"nmse":.01,"hfen":.12}
    with pytest.raises(ValueError,match="head_hfen"):
        report.head_metrics({**row,"head_hfen":float("inf")})
    del row["head_ssim"]
    with pytest.raises(ValueError,match="head_ssim"):
        report.head_metrics(row)


def test_output_reservation_requires_fresh_campaign_reports_directory(report, tmp_path, monkeypatch):
    monkeypatch.setattr(report,"ROOT",tmp_path / "TEST_campaign")
    output=report.ROOT / "reports" / "TEST_report"
    assert report.reserve_output(output)==output
    with pytest.raises(FileExistsError):
        report.reserve_output(output)
    with pytest.raises(ValueError):
        report.reserve_output(tmp_path / "escape")


def test_completed_trial_rejects_test_placeholders_before_loading_arrays(report):
    with pytest.raises(ValueError,match="TEST|placeholder"):
        report.completed_trials([{"event":"trial_scored","exp":"TEST_fake","verification_placeholders":True}],[],{})


def scored_fixture(report, tmp_path):
    """Small TEST-only files exercise the real objective/run reader, never a preview."""
    baseline=[]
    for index in range(59):
        scans={}
        for scan in report.SCANS:
            methods={label:{"psnr":30.,"ssim":.9,"nmse":.1,"hfen":.2} for label in report.LABELS}
            methods["MC-GS"]={"psnr":20.,"ssim":.8,"nmse":.2,"hfen":.3}
            scans[scan]={"metrics":methods}
        baseline.append({"axis":0,"index":index,"core":150,"scans":scans})
    candidate=copy.deepcopy(baseline)
    for row in candidate:
        for scan in report.SCANS:
            row["scans"][scan]["metrics"]["MC-GS"]={"psnr":31.,"ssim":.91,"nmse":.09,"hfen":.18}
    run=tmp_path / "TEST_control_run"
    run.mkdir()
    checkpoint=tmp_path / "TEST_source.pth"
    config=tmp_path / "TEST_source.yaml"
    config.write_text("loss:\n  tv_weight: 0.1\n  lambda_wav: 0.1\n  lambda_wav_plain: 0.12\n")
    manifest={"source_checkpoint":str(checkpoint),"source_config":str(config),"files":{str(checkpoint):"TEST_hash"}}
    head={scan:{"head_psnr":35.,"head_ssim":.94,"head_nmse":.01,"head_hfen":.12} for scan in report.SCANS}
    nlr={"all_gradients_finite":True,"sigma_rel":.015,"start":4001,"refresh_every":250,
         "denoiser":"patch_lowrank","coefficients":{"pre":0.,"post":0.},"enabled":False}
    overrides={"local_factor":1.,"nlambda":0.,"sigma_rel":.015}
    metrics={"mode":"all","source_iteration":4000,"end_iteration":4500,"nlr":nlr,"overrides":overrides,
             "forward_backward_evaluations":500,"rows":head,"training_elapsed_seconds":1.}
    provenance={"status":"complete","source_iteration":4000,"end_iteration":4500,
                "source_checkpoint":str(checkpoint),"source_checkpoint_sha256":"TEST_hash","nlr":nlr,"overrides":overrides}
    import yaml
    (run / "actual_config.yaml").write_text(yaml.safe_dump({"loss":{"tv_weight":.1,"lambda_wav":.1,"lambda_wav_plain":.12,
        "lambda_nl":0.,"lambda_nl_plain":0.,"lambda_nl_contrast":0.,"nl_start":4001,"nl_refresh":250,
        "nl_sigma_rel":.015,"nl_denoiser":"patch_lowrank"},"training":{"resume_offset":4000,"max_iterations":4500}}))
    (run / "metrics.json").write_text(json.dumps(metrics))
    (run / "provenance.json").write_text(json.dumps(provenance))
    for scan in report.SCANS:
        np.save(run / f"candidate_{scan}_complex.npy",np.ones((2,2,2),np.complex64))
    scored=report.score(candidate,baseline,head)
    scored["candidate_sha256"]={scan:report.sha256_file(run / f"candidate_{scan}_complex.npy") for scan in report.SCANS}
    scored["source_checkpoint_sha256"]="TEST_hash"
    directory=tmp_path / "TEST_scores"
    directory.mkdir()
    (directory / "objective.json").write_text(json.dumps(scored))
    (directory / "lesion_allmetrics.json").write_text(json.dumps(candidate))
    parity=tmp_path / "TEST_parity.json"
    parity.write_text('{"passed":true}')
    record={"event":"trial_scored","exp":"B","candidate":{"name":"B","local_factor":1.,"nlambda":0.,"sigma_rel":.015},
            "run_dir":str(run),"objective":str(directory / "objective.json"),"control_parity":str(parity),
            "score":scored["score"],"strict_success":True,"accepted":True,
            "incumbent_before":report.score(baseline,baseline)["score"],"whole_head_metrics":head}
    return baseline,manifest,record


@pytest.mark.parametrize("parity",[{"passed":False},{"passed":True,"source_checkpoint_sha256":"wrong"},
                                   {"passed":True,"source_checkpoint_sha256":"TEST_hash","control_run_dir":"/wrong"}])
def test_control_parity_is_validated_before_trial_can_be_accepted(report,tmp_path,parity):
    baseline,manifest,record=scored_fixture(report,tmp_path)
    record["exp"]=record["candidate"]["name"]="A_control"
    Path(record["control_parity"]).write_text(json.dumps(parity))
    with pytest.raises(ValueError):
        report.completed_trials([record],baseline,manifest)


def test_reported_control_run_must_equal_the_validated_gate_run(report,tmp_path,monkeypatch):
    baseline,manifest,record=scored_fixture(report,tmp_path)
    record["exp"]=record["candidate"]["name"]="A_control"
    # The validator owns gate internals; this isolates the report's run-binding boundary.
    monkeypatch.setattr(report,"parity_control",lambda path,frozen:tmp_path/"different_canonical_control")
    with pytest.raises(ValueError,match="reported control run differs"):
        report.completed_trials([record],baseline,manifest)


def reference_fixture(report,tmp_path):
    topic=tmp_path / "TEST_CMP"
    (topic / "sim").mkdir(parents=True)
    (topic / "recon").mkdir()
    pair={scan:np.ones((3,3,3),np.complex64)*(index+1) for index,scan in enumerate(report.SCANS)}
    for scan in report.SCANS:
        np.save(topic / f"recon/gt_{scan}_complex.npy",pair[scan])
        np.save(topic / f"recon/gs_joint_{scan}_complex.npy",pair[scan])
    np.savez(topic / "sim/gt.npz",pre=pair["pre"]*3,post=pair["post"]*4,scale_pre=3.,scale_post=4.)
    roi=tmp_path / "TEST_rois.npz"
    np.savez(roi,names=np.array(list(report.ROI_KEYS)),**{f"roi{i}":np.ones((3,3,3),bool) for i in range(6)})
    manifest={"version":1,"files":{str(path):report.sha256_file(path) for path in [roi,*sorted((topic/"recon").glob("gt_*.npy"))]}}
    manifest_path=tmp_path / "TEST_loop_input_freeze.json"
    manifest_path.write_text(json.dumps(manifest))
    refs={"MC-GS":{"key":"gs_joint","volumes":{scan:str(topic/f"recon/gs_joint_{scan}_complex.npy") for scan in report.SCANS}}}
    return topic,roi,refs,manifest,manifest_path,tmp_path / "TEST_report_reference_freeze.json"


def test_reference_freeze_proves_gt_gauge_equivalence_and_binds_baseline(report,tmp_path):
    topic,roi,refs,manifest,mp,fp=reference_fixture(report,tmp_path)
    frozen=report.report_reference_freeze(topic,roi,refs,manifest,manifest_path=mp,freeze_path=fp)
    assert frozen["loop_input_freeze_sha256"]==report.sha256_file(mp)
    assert frozen["files"][str(roi)]==manifest["files"][str(roi)]
    assert frozen["gt_equivalence"]["pre"]["complex_exact_after_scale"]
    assert frozen["gt_equivalence"]["post"]["magnitude_exact_after_scale"]
    assert frozen["files"][refs["MC-GS"]["volumes"]["pre"]]==report.sha256_file(refs["MC-GS"]["volumes"]["pre"])


def test_existing_reference_manifest_is_never_refrozen_after_drift(report,tmp_path):
    topic,roi,refs,manifest,mp,fp=reference_fixture(report,tmp_path)
    report.report_reference_freeze(topic,roi,refs,manifest,manifest_path=mp,freeze_path=fp)
    original=fp.read_bytes()
    with Path(refs["MC-GS"]["volumes"]["post"]).open("ab") as stream:
        stream.write(b"changed baseline")
    with pytest.raises(ValueError,match="hash changed"):
        report.report_reference_freeze(topic,roi,refs,manifest,manifest_path=mp,freeze_path=fp)
    assert fp.read_bytes()==original


@pytest.mark.parametrize("changed",["roi","gt","baseline"])
def test_reference_drift_is_rejected_before_enhancement_loading(report,tmp_path,monkeypatch,changed):
    topic,roi,refs,manifest,mp,fp=reference_fixture(report,tmp_path)
    frozen=report.report_reference_freeze(topic,roi,refs,manifest,manifest_path=mp,freeze_path=fp)
    path={"roi":roi,"gt":topic/"sim/gt.npz","baseline":Path(refs["MC-GS"]["volumes"]["pre"])}[changed]
    with path.open("ab") as stream:
        stream.write(b"drift")
    monkeypatch.setattr(report.np,"load",lambda *args,**kwargs:pytest.fail("drift reached enhancement loader"))
    with pytest.raises(ValueError,match="hash|changed"):
        report.enhancement_bundle(topic,roi,refs,[],manifest,frozen)


@pytest.mark.parametrize("changed",["roi","gt_gauge"])
def test_first_reference_freeze_rejects_changed_roi_or_gt_gauge(report,tmp_path,changed):
    topic,roi,refs,manifest,mp,fp=reference_fixture(report,tmp_path)
    if changed=="roi":
        with roi.open("ab") as stream:
            stream.write(b"drift")
    else:
        np.savez(topic/"sim/gt.npz",pre=np.ones((3,3,3),np.complex64)*7,
                 post=np.ones((3,3,3),np.complex64)*8,scale_pre=3.,scale_post=4.)
    with pytest.raises(ValueError,match="ROI|hash|gauge"):
        report.report_reference_freeze(topic,roi,refs,manifest,manifest_path=mp,freeze_path=fp)
    assert not fp.exists()


def test_reference_changed_during_enhancement_is_rejected_after_loading(report,tmp_path,monkeypatch):
    topic,roi,refs,manifest,mp,fp=reference_fixture(report,tmp_path)
    frozen=report.report_reference_freeze(topic,roi,refs,manifest,manifest_path=mp,freeze_path=fp)
    monkeypatch.setattr(report,"background_mask",lambda gt:np.ones(gt.shape,bool))
    original=report.enhancement_row
    observed=[]
    def changed_after_loading(*args,**kwargs):
        observed.append(args[0])
        if args[0]=="truth":
            with (topic/"sim/gt.npz").open("ab") as stream:
                stream.write(b"drift after loading")
        return original(*args,**kwargs)
    monkeypatch.setattr(report,"enhancement_row",changed_after_loading)
    with pytest.raises(ValueError,match="hash changed"):
        report.enhancement_bundle(topic,roi,refs,[],manifest,frozen)
    assert observed==["truth","gs_joint"]


def test_recovered_scored_trial_is_not_terminal_failure(report):
    rows=[{"event":"trial_started","exp":"B"},{"event":"trial_failed","exp":"B","phase":"scoring"},
          {"event":"trial_scored","exp":"B"},{"event":"trial_started","exp":"C"},
          {"event":"trial_failed","exp":"C"},{"event":"trial_started","exp":"D"}]
    status=report.trial_status(rows,[{"name":"B","accepted":True}])
    assert status["status_counts"]=={"completed":1,"accepted":1,"rejected":0,"failed":1,"pending":1,"failure_events":2}
    assert status["terminal_failed_trials"]==["C"]
    assert status["pending_trials"]==["D"]


def test_completed_trial_recomputes_all59_objective_and_keeps_four_head_metrics(report,tmp_path):
    baseline,manifest,record=scored_fixture(report,tmp_path)
    trials=report.completed_trials([{"event":"trial_started","exp":"B"},record],baseline,manifest)
    assert len(trials)==1
    assert trials[0]["decision"]=="accepted"
    assert trials[0]["score"] > 0
    assert trials[0]["plane_counts"]["all_eight_both_scans"]==59
    assert trials[0]["whole_head_metrics"]["pre"]=={"psnr":35.,"ssim":.94,"nmse":.01,"hfen":.12}


@pytest.mark.parametrize("mutation",["score","acceptance","volume","duplicate","incomplete","missing_plane"])
def test_completed_trial_rejects_changed_evidence_or_acceptance(report,tmp_path,mutation):
    baseline,manifest,record=scored_fixture(report,tmp_path)
    records=[record]
    if mutation=="score":
        record["score"]+=.01
    elif mutation=="acceptance":
        record["accepted"]=False
    elif mutation=="volume":
        np.save(Path(record["run_dir"]) / "candidate_pre_complex.npy",np.ones((2,2,2),np.complex64)*2)
    elif mutation=="duplicate":
        records.append(copy.deepcopy(record))
    elif mutation=="incomplete":
        path=Path(record["run_dir"]) / "provenance.json"
        document=json.loads(path.read_text())
        document["status"]="running"
        path.write_text(json.dumps(document))
    else:
        path=Path(record["objective"]).parent / "lesion_allmetrics.json"
        path.write_text(json.dumps(json.loads(path.read_text())[:-1]))
    with pytest.raises(ValueError):
        report.completed_trials(records,baseline,manifest)


def test_plot_exports_real_baseline_without_inventing_a_completed_trial(report,tmp_path):
    base={"name":"Registered MC-GS i1","score":-.1,
          "whole_head_metrics":{"pre":{"psnr":35.,"ssim":.94,"nmse":.01,"hfen":.12},
                                "post":{"psnr":37.,"ssim":.95,"nmse":.009,"hfen":.11}},
          "plane_counts":{"all_eight_both_scans":0}}
    summary={"baseline":base,"trials":[],"references":{"ZS-SSDU":{"whole_head_metrics":base["whole_head_metrics"]}},
             "status_counts":{"completed":0,"failed":0,"pending":1}}
    output=tmp_path / "TEST_exports"
    output.mkdir()
    report.plot_report(summary,output)
    assert (output / "campaign.png").stat().st_size > 1000
    assert (output / "campaign.pdf").read_bytes().startswith(b"%PDF")
    assert summary["trials"] == []
