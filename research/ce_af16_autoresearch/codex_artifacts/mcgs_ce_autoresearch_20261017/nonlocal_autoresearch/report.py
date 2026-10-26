#!/usr/bin/env python3
"""CPU-only snapshot report of completed real CE AF16 autoresearch trials."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True  # Imported /data0 scoring modules remain read-only.

from objective import (BASELINE_SHA256, LABELS, METRICS, SCANS, load_baseline,
                       read_json, score, sha256_file)
from autoresearch import is_improvement, parity_control, validate_run, write_new

ROOT = Path(__file__).resolve().parent
CMP = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois_cmp")
SCRIPTS = Path("/data0/congcong/code/GS/mcgs_ce_mask_wt/ce_retro_cssense_af16/scripts")
ROI = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz")
REGIONS = ("lesion_core", "rim_0_3mm", "white_matter", "cortex", "vessels", "plexus")
ROI_KEYS = {"病灶核心":"lesion_core", "核心外 0-3 mm":"rim_0_3mm", "正常白质":"white_matter",
            "正常皮层":"cortex", "血管/静脉窦":"vessels", "脉络丛":"plexus"}


def read_ledger_snapshot(path):
    """Read one immutable byte snapshot; running/failed events never become results."""
    path = Path(path)
    raw = path.read_bytes() if path.exists() else b""
    if raw and not raw.endswith(b"\n"):
        raise ValueError("incomplete ledger tail; retry after the writer completes its record")
    def invalid(value):
        raise ValueError(f"nonfinite ledger JSON: {value}")
    records = [json.loads(line, parse_constant=invalid) for line in raw.decode().splitlines()]
    if any(not isinstance(row, dict) for row in records):
        raise ValueError("ledger records must be JSON objects")
    return records, {"path":str(path.resolve()), "sha256":hashlib.sha256(raw).hexdigest(),
                     "byte_count":len(raw), "record_count":len(records), "existed":path.exists()}


def head_metrics(row):
    values = {}
    for metric in METRICS:
        key = "head_" + metric
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"requires finite {key}; whole-volume fallback is forbidden")
        values[metric] = value
    return values


def plane_counts(objective):
    planes = objective["planes"]
    if len(planes) != 59 or len({(p["axis"], p["index"]) for p in planes}) != 59:
        raise ValueError("exactly 59 distinct lesion-containing planes required")
    result = {"eligible_planes":59, "all_four_by_scan":{}, "per_condition":{},
              "all_eight_both_scans":0,
              "region":"head mask on each predeclared lesion-containing plane; not lesion ROI"}
    for scan in SCANS:
        for metric in METRICS:
            key = f"{scan}:{metric}"
            flags = [p["comparisons"][key]["strict_win"] for p in planes]
            if any(type(flag) is not bool for flag in flags):
                raise ValueError("strict-win observations must be booleans")
            result["per_condition"][key] = sum(flags)
        result["all_four_by_scan"][scan] = sum(all(p["comparisons"][f"{scan}:{metric}"]["strict_win"]
                                                   for metric in METRICS) for p in planes)
    for plane in planes:
        joint = all(plane["comparisons"][f"{scan}:{metric}"]["strict_win"] for scan in SCANS for metric in METRICS)
        if type(plane["strict_success"]) is not bool or joint != plane["strict_success"]:
            raise ValueError("joint strict success conflicts with raw per-condition comparisons")
        result["all_eight_both_scans"] += joint
    return result


def completed_trials(records, baseline, manifest):
    scored = [row for row in records if row.get("event") == "trial_scored"]
    for row in scored:
        if (row.get("verification_placeholders") or row.get("test_fixture")
                or "TEST" in str(row.get("exp", "")).upper()):
            raise ValueError("TEST/placeholder results are forbidden in a real report")
    if not scored:
        return []
    incumbent = score(baseline, baseline)["score"]
    trials, seen = [], set()
    for record in scored:
        name = record["exp"]
        if name in seen:
            raise ValueError(f"duplicate scored trial {name}; refusing double-counted evidence")
        seen.add(name)
        candidate = record["candidate"]
        if candidate.get("name") != name:
            raise ValueError("ledger trial/candidate name mismatch")
        run = Path(record["run_dir"]).resolve(strict=True)
        head, provenance = validate_run(run, candidate, manifest, control=name == "A_control")
        metrics = read_json(run / "metrics.json")
        if metrics.get("verification_placeholders") or metrics.get("test_fixture"):
            raise ValueError("placeholder run metrics are forbidden")
        observed_head = {scan:head_metrics(head[scan]) for scan in SCANS}
        path = Path(record["objective"]).resolve(strict=True)
        stored = read_json(path)
        raw = read_json(path.parent / "lesion_allmetrics.json")
        recomputed = score(raw, baseline, head)
        for key in ("score", "strict_success", "planes", "best_plane", "baseline_sha256"):
            if stored.get(key) != recomputed[key]:
                raise ValueError(f"stored objective {key} disagrees with the unchanged all59 scorer")
        if (record["score"] != recomputed["score"] or record["strict_success"] != recomputed["strict_success"]
                or record.get("whole_head_metrics") != head or stored.get("whole_head_metrics") != head):
            raise ValueError("ledger, objective and completed run metrics disagree")
        if record.get("incumbent_before") != incumbent or type(record.get("accepted")) is not bool:
            raise ValueError("recorded incumbent/acceptance is inconsistent")
        accepted = is_improvement(recomputed["score"], incumbent)
        if record["accepted"] != accepted:
            raise ValueError("accepted flag conflicts with the recorded 1e-4 improvement threshold")
        if not math.isfinite(recomputed["score"]):
            raise ValueError("nonfinite objective")
        for scan in SCANS:
            path_volume = run / f"candidate_{scan}_complex.npy"
            if stored.get("candidate_sha256", {}).get(scan) != sha256_file(path_volume):
                raise ValueError("candidate reconstruction changed since scoring")
        if name == "A_control":
            parity = Path(record["control_parity"]).resolve(strict=True)
            if parity_control(parity, manifest) != run:
                raise ValueError("reported control run differs from the code/checkpoint/two-step/drift-bound parity control")
            parity_hash = sha256_file(parity)
        else:
            parity, parity_hash = None, None
        trial = {"name":name, "completed_index":len(trials) + 1, "candidate":candidate,
                 "score":recomputed["score"], "strict_success":recomputed["strict_success"],
                 "accepted":accepted, "decision":"accepted" if accepted else "rejected",
                 "incumbent_before":incumbent, "best_plane":recomputed["best_plane"],
                 "plane_counts":plane_counts(recomputed), "whole_head_metrics":observed_head,
                 "run_dir":str(run), "objective":str(path), "objective_sha256":sha256_file(path),
                 "metrics_sha256":sha256_file(run / "metrics.json"),
                 "source_checkpoint_sha256":provenance["source_checkpoint_sha256"],
                 "training_elapsed_seconds":metrics.get("training_elapsed_seconds"),
                 "nlr":metrics.get("nlr"), "control_parity":str(parity) if parity else None,
                 "control_parity_sha256":parity_hash}
        trials.append(trial)
        if accepted:
            incumbent = recomputed["score"]
    return trials


def background_mask(gt_post):
    from scipy import ndimage
    gt_post = np.abs(gt_post).astype(np.float32)
    if gt_post.ndim != 3 or not np.isfinite(gt_post).all():
        raise ValueError("GT post must be a finite 3-D array")
    p999 = float(np.quantile(gt_post, .999))
    return ~ndimage.binary_dilation(gt_post > .05 * p999, iterations=6)


def enhancement_row(method, label, pair, regions, background):
    """Exact WT float32 magnitude / own-WM-median post-minus-pre scoring formula."""
    if set(regions) != set(REGIONS):
        raise ValueError("all six reference enhancement ROIs required")
    magnitudes = {scan:np.abs(pair[scan]).astype(np.float32) for scan in SCANS}
    shape = magnitudes["pre"].shape
    if len(shape) != 3 or magnitudes["post"].shape != shape:
        raise ValueError("enhancement requires matching 3-D pre/post volumes")
    for name, mask in {**regions, "background":background}.items():
        if mask.shape != shape or mask.dtype != np.bool_ or not mask.any():
            raise ValueError(f"empty or invalid enhancement region {name}")
    for scan, magnitude in magnitudes.items():
        if not np.isfinite(magnitude).all():
            raise ValueError(f"enhancement {scan} must be finite")
        median = np.median(magnitude[regions["white_matter"]])
        if not np.isfinite(median) or median <= 0:
            raise ValueError(f"invalid {scan} white-matter median")
    # Match the original arithmetic order and float32 reductions exactly.
    pre = magnitudes["pre"] / np.median(magnitudes["pre"][regions["white_matter"]])
    post = magnitudes["post"] / np.median(magnitudes["post"][regions["white_matter"]])
    difference = post - pre
    return {"method":method, "label":label,
            **{name:float(difference[regions[name]].mean()) for name in REGIONS},
            "background_post":float(post[background].mean())}


def load_complex_pair(paths, shape=None):
    pair, evidence = {}, {}
    for scan in SCANS:
        path = Path(paths[scan])
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
        if volume.ndim != 3 or not np.iscomplexobj(volume) or (shape is not None and tuple(volume.shape) != tuple(shape)):
            raise ValueError(f"expected matching complex 3-D {scan} reconstruction: {path}")
        pair[scan] = volume
        evidence[scan] = {"path":str(path.resolve()), "sha256":sha256_file(path),
                          "shape":list(volume.shape), "dtype":str(volume.dtype)}
    return pair, evidence


def load_references(topic, manifest):
    summary_path = topic / "logs/summary.json"
    expected = manifest["files"].get(str(summary_path))
    if expected != sha256_file(summary_path):
        raise ValueError("CMP summary differs from frozen audited comparison")
    rows = read_json(summary_path)
    spec = importlib.util.spec_from_file_location("report_frozen_arms", SCRIPTS / "arms.py")
    arms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arms)
    arms.R = topic
    references = {}
    for scan in SCANS:
        registered = arms.arm_files(scan, rows)
        if [label for _, label, _ in registered] != list(LABELS):
            raise ValueError("CMP must retain all eleven audited methods in frozen order")
        for key, label, volume in registered:
            matches = [row for row in rows if row.get("method") == key and row.get("contrast") == scan]
            if len(matches) != 1:
                raise ValueError(f"expected exactly one CMP summary row for {label}/{scan}")
            reference = references.setdefault(label, {"key":key,"whole_head_metrics":{},"volumes":{}})
            reference["whole_head_metrics"][scan] = head_metrics(matches[0])
            reference["volumes"][scan] = str(volume)
    audited = {scan:read_json(topic / f"logs/zsssdu_{scan}.json") for scan in SCANS}
    for scan, document in audited.items():
        row = next(row for row in document["rows"] if row.get("method") == "zsssdu")
        if row.get("blocks") != 15 or row.get("improved_from") != "zsssdu_z2_b15_cg10_e400":
            raise ValueError("ZS-SSDU must retain the audited 15-block recipe")
        if head_metrics(row) != references["ZS-SSDU"]["whole_head_metrics"][scan]:
            raise ValueError("audited ZS-SSDU log/summary discrepancy")
    return references


def _verify_reference_inputs(frozen, manifest, topic, roi_path, references):
    """Check immutable bytes before loading and again after enhancement scoring."""
    source = Path(frozen["loop_input_freeze_path"])
    if (read_json(source) != manifest or sha256_file(source) != frozen["loop_input_freeze_sha256"]):
        raise ValueError("original loop input manifest hash changed")
    required = {str(roi_path), str(topic / "sim/gt.npz"),
                *[str(topic / f"recon/gt_{scan}_complex.npy") for scan in SCANS],
                *references["MC-GS"]["volumes"].values()}
    if set(frozen.get("files", {})) != required:
        raise ValueError("report reference freeze omits required ROI/GT/baseline inputs")
    if frozen["files"].get(str(roi_path)) != manifest["files"].get(str(roi_path)):
        raise ValueError("ROI expected hash differs from the original loop manifest")
    for scan in SCANS:
        gt = str(topic / f"recon/gt_{scan}_complex.npy")
        if frozen["files"][gt] != manifest["files"].get(gt):
            raise ValueError("GT reconstruction hash differs from the original loop manifest")
    for path, digest in frozen["files"].items():
        if sha256_file(path) != digest:
            raise ValueError(f"frozen report reference hash changed: {path}")


def report_reference_freeze(topic, roi_path, references, manifest, *, manifest_path=None,
                            freeze_path=None, records=()):
    """Extend the immutable v1 audit for report-only inputs before formal trials."""
    manifest_path = Path(manifest_path or ROOT / "verification/loop_input_freeze.json").resolve()
    freeze_path = Path(freeze_path or ROOT / "verification/report_reference_freeze.json").resolve()
    if read_json(manifest_path) != manifest:
        raise ValueError("loop input manifest does not match the supplied frozen audit")
    if freeze_path.exists():
        frozen = read_json(freeze_path)
        _verify_reference_inputs(frozen, manifest, topic, roi_path, references)
        return frozen
    if any(row.get("event") in ("trial_started", "trial_scored") and row.get("exp") != "A_control" for row in records):
        raise ValueError("report reference freeze must be created before formal trials")
    if manifest["files"].get(str(roi_path)) != sha256_file(roi_path):
        raise ValueError("ROI hash changed before first report reference freeze")
    paths = {str(roi_path), str(topic / "sim/gt.npz"),
             *[str(topic / f"recon/gt_{scan}_complex.npy") for scan in SCANS],
             *references["MC-GS"]["volumes"].values()}
    frozen = {"version":1, "created_utc":datetime.now(timezone.utc).isoformat(),
              "created_before_formal_trials":True, "loop_input_freeze_path":str(manifest_path),
              "loop_input_freeze_sha256":sha256_file(manifest_path),
              "files":{path:sha256_file(path) for path in sorted(paths)}, "gt_equivalence":{}}
    _verify_reference_inputs(frozen, manifest, topic, roi_path, references)
    with np.load(topic / "sim/gt.npz", allow_pickle=False) as archive:
        for scan in SCANS:
            scale = float(archive[f"scale_{scan}"])
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("GT simulation gauge scale must be finite and positive")
            reference = np.load(topic / f"recon/gt_{scan}_complex.npy", mmap_mode="r", allow_pickle=False)
            normalized = archive[scan] / np.float32(scale)
            exact = bool(np.array_equal(normalized, reference))
            magnitude_exact = bool(np.array_equal(np.abs(normalized).astype(np.float32), np.abs(reference).astype(np.float32)))
            if not exact or not magnitude_exact:
                raise ValueError(f"GT simulation {scan} is not gauge-equivalent to the frozen reconstruction")
            frozen["gt_equivalence"][scan] = {"scale_source":f"sim/gt.npz:scale_{scan}",
                                              "scale_float64":scale, "division_scale_float32":float(np.float32(scale)),
                                              "complex_exact_after_scale":exact,
                                              "magnitude_exact_after_scale":magnitude_exact,
                                              "normalization":"sim complex / float32(stored scale); no fitted scale or tolerance"}
    _verify_reference_inputs(frozen, manifest, topic, roi_path, references)
    write_new(freeze_path, frozen)
    return frozen


def enhancement_bundle(topic, roi_path, references, trials, manifest, reference_freeze):
    _verify_reference_inputs(reference_freeze, manifest, topic, roi_path, references)
    with np.load(roi_path, allow_pickle=False) as archive:
        names = [str(name) for name in archive["names"]]
        if len(names) != 6 or set(names) != set(ROI_KEYS):
            raise ValueError("expected exactly the six unchanged CS-SENSE reference ROIs")
        regions = {ROI_KEYS[name]:archive[f"roi{index}"].astype(bool) for index,name in enumerate(names)}
    gt_path = topic / "sim/gt.npz"
    with np.load(gt_path, allow_pickle=False) as archive:
        gt = {scan:np.abs(archive[scan]).astype(np.float32) for scan in SCANS}
    background = background_mask(gt["post"])
    truth = enhancement_row("truth", "truth (CS-SENSE)", gt, regions, background)
    rows, evidence = [truth], {}
    shape = gt["pre"].shape
    del gt
    for label, reference in references.items():
        pair, inputs = load_complex_pair(reference["volumes"], shape)
        for scan in SCANS:
            original = Path(reference["volumes"][scan])
            frozen = (reference_freeze if label == "MC-GS" else manifest)["files"].get(str(original))
            if inputs[scan]["sha256"] != frozen:
                raise ValueError(f"frozen opponent reconstruction changed: {label}/{scan}")
        rows.append(enhancement_row(reference["key"], label, pair, regions, background))
        evidence[reference["key"]] = inputs
        del pair
    for trial in trials:
        paths = {scan:Path(trial["run_dir"]) / f"candidate_{scan}_complex.npy" for scan in SCANS}
        pair, inputs = load_complex_pair(paths, shape)
        rows.append(enhancement_row(trial["name"], f"Trial {trial['name']}", pair, regions, background))
        evidence[trial["name"]] = inputs
        del pair
    for row in rows[1:]:
        for key in (*REGIONS, "background_post"):
            row["err_" + key] = row[key] - truth[key]
    _verify_reference_inputs(reference_freeze, manifest, topic, roi_path, references)
    return {"units":"white-matter units (each scan / its own WM median); post - pre",
            "regions":list(REGIONS), "background":"post magnitude outside GTpost >0.05*P99.9 dilated six voxels",
            "gt_role":"reference and ROI inputs used for scoring only; never denoiser targets or optimization",
            "formula_source":str(SCRIPTS / "enhancement_fidelity.py"),
            "formula_source_sha256":sha256_file(SCRIPTS / "enhancement_fidelity.py"),
            "roi_source":str(roi_path), "roi_sha256":sha256_file(roi_path),
            "gt_source":str(gt_path), "gt_sha256":sha256_file(gt_path),
            "roi_voxel_counts":{name:int(mask.sum()) for name,mask in regions.items()},
            "background_voxel_count":int(background.sum()), "rows":rows,"inputs":evidence}


def trial_status(records, trials):
    completed = {trial["name"] for trial in trials}
    failures = [row for row in records if row.get("event") == "trial_failed"]
    failed_names = {row["exp"] for row in failures} - completed
    started = {row["exp"] for row in records if row.get("event") == "trial_started"}
    pending = sorted(started - completed - failed_names)
    return {"status_counts":{"completed":len(trials), "accepted":sum(t["accepted"] for t in trials),
                             "rejected":sum(not t["accepted"] for t in trials), "failed":len(failed_names),
                             "pending":len(pending), "failure_events":len(failures)},
            "pending_trials":pending, "terminal_failed_trials":sorted(failed_names),
            "failed_trials":failures}


def reserve_output(output):
    output = Path(output).expanduser().resolve()
    parent = (ROOT / "reports").resolve()
    if output == parent or not output.is_relative_to(parent):
        raise ValueError("output must be a fresh subdirectory inside this campaign's reports/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    return output


def plot_report(summary, output):
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "pdf.fonttype":42,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "axes.linewidth":.7, "legend.frameon":False})
    fig = plt.figure(figsize=(9.2,6.4), layout="constrained")
    grid = fig.add_gridspec(2,2,height_ratios=(1.3,1))
    hero, psnr, hfen = fig.add_subplot(grid[0,:]), fig.add_subplot(grid[1,0]), fig.add_subplot(grid[1,1])
    trials = summary["trials"]
    base = summary["baseline"]
    series = [base, *trials]
    labels = ["i1 (4000)", *[trial["name"] for trial in trials]]
    xs = np.arange(len(series))
    values = [item["score"] for item in series]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("figure objective must be finite; no clipping or placeholder values")
    hero.axhline(0,color="#994c47",ls="--",lw=1,label="Strict goal: S > 0")
    hero.plot(0,base["score"],marker="D",color="#65676b",ms=6,label="Registered i1")
    incumbent = base["score"]
    best = [incumbent]
    for x,trial in enumerate(trials,1):
        accepted = trial["accepted"]
        hero.plot(x,trial["score"],marker="o" if accepted else "x",ms=6,
                  color="#4c7892" if accepted else "#b88043",ls="none")
        if accepted:
            incumbent = trial["score"]
        best.append(incumbent)
    hero.step(xs,best,where="post",lw=1,color="#4c7892",alpha=.6,label="Accepted incumbent")
    for x,item in enumerate(series):
        count = item["plane_counts"]["all_eight_both_scans"]
        hero.annotate(f"{count}/59",(x,item["score"]),xytext=(0,7),textcoords="offset points",ha="center",fontsize=7)
    low,high = min(values+[0]),max(values+[0])
    pad = max(.012,(high-low)*.16)
    hero.set_ylim(low-pad,high+pad*1.6)
    hero.set_xlim(-.35,max(.65,len(series)-.65))
    hero.set_ylabel("S: worst of eight margins, best of 59 planes")
    hero.set_title("a   Strict comparison objective across completed trials",loc="left",fontweight="bold")
    hero.legend(loc="lower right",fontsize=7,ncols=3)
    if not trials:
        hero.text(.55,.38,"No completed trials at this ledger snapshot",transform=hero.transAxes,ha="center",fontsize=9)
    for axis,metric,title in ((psnr,"psnr","b   Whole-head PSNR (higher is better)"),
                              (hfen,"hfen","c   Whole-head HFEN (lower is better)")):
        for scan,color,marker in (("pre","#4c7892","o"),("post","#8a8ba6","s")):
            ys=[item["whole_head_metrics"][scan][metric] for item in series]
            axis.plot(xs,ys,color=color,marker=marker,lw=1,ms=4,label=scan.upper())
            rivals = [entry["whole_head_metrics"][scan][metric] for label,entry in summary["references"].items() if label!="MC-GS"]
            envelope = (max if metric == "psnr" else min)(rivals)
            axis.axhline(envelope,color=color,ls="--",lw=.8,alpha=.8)
        axis.set_title(title,loc="left",fontweight="bold")
        axis.set_ylabel("PSNR (dB)" if metric=="psnr" else "HFEN")
        axis.legend(loc="best",fontsize=7)
        axis.margins(x=.12,y=.18)
    for axis in (hero,psnr,hfen):
        axis.set_xticks(xs,labels,rotation=25 if len(labels)>6 else 0)
        axis.set_xlabel("Completed trial order (each independently resumes iteration 4000)")
        axis.grid(axis="y",color="#e7e8ea",lw=.5)
    counts = summary["status_counts"]
    fig.suptitle("CE AF16 nonlocal autoresearch | one subject, one seed\n"
                 f"Completed {counts['completed']} · failed {counts['failed']} · pending {counts['pending']} | "
                 "plane counts are head-masked lesion-containing planes, not lesion ROIs",fontsize=10)
    fig.supxlabel("Dashed head-metric lines: best of all audited opponents per scan. Full metrics and six-ROI enhancement are in the source tables.\n"
                  "Dots/crosses in a: accepted/rejected by >1e-4 improvement. No replicate error bars or generalization claim.",fontsize=7)
    fig.savefig(output / "campaign.png",dpi=220)
    fig.savefig(output / "campaign.pdf")
    plt.close(fig)


def write_csv(path, rows, fields):
    with path.open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_tables(summary, output):
    methods = [{"name":"Registered i1",**summary["baseline"]},*summary["trials"]]
    write_csv(output / "trials.csv",[
        {"name":item["name"],"score":item["score"],"decision":item.get("decision","baseline"),
         "strict_success":item["strict_success"],"all_eight_both_scans":item["plane_counts"]["all_eight_both_scans"],
         "pre_all_four":item["plane_counts"]["all_four_by_scan"]["pre"],
         "post_all_four":item["plane_counts"]["all_four_by_scan"]["post"],
         **item.get("candidate",{})} for item in methods],
        ["name","score","decision","strict_success","all_eight_both_scans","pre_all_four","post_all_four",
         "local_factor","nlambda","sigma_rel"])
    heads = [{"method":label,"scan":scan,**entry["whole_head_metrics"][scan]}
             for label,entry in summary["references"].items() for scan in SCANS]
    heads.extend({"method":item["name"],"scan":scan,**item["whole_head_metrics"][scan]}
                 for item in summary["trials"] for scan in SCANS)
    write_csv(output / "whole_head_metrics.csv",heads,["method","scan",*METRICS])
    keys=[*REGIONS,"background_post"]
    write_csv(output / "enhancement_fidelity.csv",summary["enhancement_fidelity"]["rows"],
              ["method","label",*keys,*["err_"+key for key in keys]])
    lines=["# CE AF16 campaign snapshot", "", summary["limitations"], "",
           "S is the best of 59 head-masked lesion-containing planes, using the worst of eight strict pre/post metric margins. "
           "The goal requires S > 0. These plane metrics are distinct from the six enhancement ROIs.", "",
           "| Trial | Decision | S | Pre 4/4 | Post 4/4 | Both scans 8/8 |", "|---|---|---:|---:|---:|---:|"]
    for item in methods:
        count=item["plane_counts"]
        lines.append(f"| {item['name']} | {item.get('decision','baseline')} | {item['score']:.6f} | "
                     f"{count['all_four_by_scan']['pre']}/59 | {count['all_four_by_scan']['post']}/59 | {count['all_eight_both_scans']}/59 |")
    if not summary["trials"]:
        lines.extend(["","No completed trial results were available at this snapshot; reference values above are measured baseline evidence."])
    lines.extend(["","All four whole-head metrics for both scans are in `whole_head_metrics.csv`. "
                  "All six unchanged reference ROIs, post background and signed errors versus truth are in `enhancement_fidelity.csv`.",
                  "","Acceptance means an objective improvement greater than 1e-4; it does not imply the strict goal was reached. "
                  "Failed/pending trials have no imputed numerical results."])
    (output / "report.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger",type=Path,default=ROOT / "verification/loop_results.jsonl")
    parser.add_argument("--output",type=Path,default=ROOT / "reports" / datetime.now(timezone.utc).strftime("snapshot_%Y%m%dT%H%M%SZ"))
    args=parser.parse_args()
    output=None
    try:
        baseline=load_baseline(ROOT.parent / "baseline_lesion_allmetrics.json")
        manifest=read_json(ROOT / "verification/loop_input_freeze.json")
        records,ledger=read_ledger_snapshot(args.ledger)
        trials=completed_trials(records,baseline,manifest)
        references=load_references(CMP,manifest)
        base_objective=score(baseline,baseline)
        reference_path=ROOT / "verification/report_reference_freeze.json"
        reference_freeze=report_reference_freeze(CMP,ROI,references,manifest,records=records)
        summary={"status":"completed_real_trial_snapshot", "created_utc":datetime.now(timezone.utc).isoformat(),
                 "baseline_sha256":BASELINE_SHA256, "ledger_snapshot":ledger,
                 "figure_contract":{"claim":"Measured objective and head-metric changes across completed development trials; no unobserved results.",
                                    "archetype":"quantitative grid", "backend":"Python/matplotlib", "size_inches":[9.2,6.4],
                                    "statistics":"one subject, one seed; no independent replicates, tests or error bars",
                                    "regions":"objective: 59 head-masked lesion-containing planes; head metrics: whole-head volume; enhancement: six reference ROIs"},
                 "baseline":{"name":"Registered MC-GS i1", "score":base_objective["score"],
                             "strict_success":base_objective["strict_success"], "best_plane":base_objective["best_plane"],
                             "plane_counts":plane_counts(base_objective),
                             "whole_head_metrics":references["MC-GS"]["whole_head_metrics"]},
                 "references":references, "trials":trials,
                 **trial_status(records,trials),
                 "loop_input_freeze_sha256":reference_freeze["loop_input_freeze_sha256"],
                 "report_reference_freeze":{"path":str(reference_path),"sha256":sha256_file(reference_path),
                                              "manifest":reference_freeze},
                 "goal":{"strict_boundary":0,"criterion":"S > 0 with raw strict comparisons authoritative", "improvement_threshold":1e-4},
                 "limitations":"One paired subject and one seed; adaptive development comparisons are not independent replicates. "
                               "No significance or generalization claim. Historical retrospective scalar normalization is retained. "
                               "GT and six CS-SENSE reference ROIs are scoring inputs only.",
                 "report_script_sha256":sha256_file(__file__)}
        summary["enhancement_fidelity"]=enhancement_bundle(CMP,ROI,references,trials,manifest,reference_freeze)
        output=reserve_output(args.output)
        # A reserved incomplete directory is deliberately retained if export fails.
        (output / "BUILDING").write_text("Incomplete until summary.json is present.\n")
        plot_report(summary,output)
        export_tables(summary,output)
        summary["artifacts"]={path.name:{"sha256":sha256_file(path),"bytes":path.stat().st_size}
                              for path in output.iterdir() if path.is_file() and path.name!="BUILDING"}
        with (output / "summary.json").open("x") as stream:
            json.dump(summary,stream,indent=2,allow_nan=False)
            stream.write("\n")
        (output / "BUILDING").unlink()
        print(json.dumps({"output":str(output),"completed_trials":len(trials),"baseline_score":base_objective["score"]}))
    except (OSError,ValueError,KeyError,TypeError) as error:
        parser.exit(2,f"error: {error}"+(f"; incomplete output retained at {output}" if output else "")+"\n")


if __name__=="__main__":
    main()
