#!/usr/bin/env python3
"""CPU comparison mosaics for a completed, scored CE AF16 candidate.

The display follows the pinned original three-metric layout. It reads saved
arrays/scores only and writes a fresh figure directory, never a training run.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "mcgs_ce_autoresearch_20261017"
TOPIC = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois_cmp")
LOOP = OLD / "nonlocal_autoresearch/verification/loop_input_freeze.json"
REPORT = OLD / "nonlocal_autoresearch/verification/report_reference_freeze.json"
SUMMARY = OLD / "phase_firstorder/scoring/combined_phase_firstorder_v1/summary.json"
CONTRACT = OLD / "reassessment_without_hfen_20261019/metric_contract_v1.json"
BASELINE = OLD / "baseline_lesion_allmetrics.json"
LAYOUT = OLD / "qualitative_g2_three_metrics_20261019/original_layout_reference.py"
STYLE = OLD / "campaign_summary/qualitative_q20_20261018/academic_local.mplstyle"
ORIGINAL_RENDERER = OLD / "qualitative_g2_three_metrics_20261019/v2/render_source.py"
PINS = {
    LOOP: "fecc6fba1edc03fe06377717f45c8c51923c6b8656b123f7ed878f5288a358d7",
    REPORT: "6ebe48fa543af773274f7d0c10bc0c21593669b9305784a8382fab7e9896e634",
    SUMMARY: "ce12805b84710cfe119fabfc063bf62931aa7e797f28174a627df000d2422413",
    CONTRACT: "7404413f695bce3b59886de802815011615f7473a0a59626542ffd4dbe60cd13",
    LAYOUT: "84b75d6752aff660dd328f52a9a2f9c65b965b32608f16ddde153ca876e41957",
    STYLE: "83998dc5538ed9366b1015bd6e549ca5d39f9d7d2e6cb75f0fa9254f8c1fbb21",
    ORIGINAL_RENDERER: "50ab25b2ed21907671cf1c96c12b644bb456307ff58ab41930e2bcf98d19bd9e",
}
SCANS = ("pre", "post")
METRICS = ("psnr", "ssim", "nmse")
OPPONENTS = ("ZF", "CG-SENSE", "CS-SENSE", "Joint SENSE", "L1-SPIRiT", "ZS-SSDU",
             "3D-DIP", "3D-INR", "MC-INR", "GS-single")
PLANES = ((0, 204, "transverse"), (1, 118, "coronal"), (2, 44, "sagittal"))
HISTORICAL = (1, 129, "coronal")
BOXES = {(0, 204): (81, 133, 21, 73), (1, 118): (180, 222, 24, 66),
         (2, 44): (182, 222, 106, 146), (1, 129): (184, 224, 27, 67)}
WINDOWS = {"pre": [0., .5555555292632828], "post": [0., .5555555590324888]}
ORANGE = "#DAB25C"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value {value} in {path}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def save(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_counts(planes, objective, baseline):
    """Check counts against all 59 actual raw comparisons and frozen identities."""
    rows = planes["planes"]
    require(len(rows) == 59 and planes["eligible_planes"] == 59, "59 planes required")
    require([(p["axis"], p["index"], p["core"]) for p in rows] ==
            [(p["axis"], p["index"], p["core"]) for p in baseline], "Frozen plane identities changed")
    conditions = tuple(f"{s}:{m}" for s in SCANS for m in METRICS)
    counts = dict.fromkeys(conditions, 0)
    by_scan, paired = dict.fromkeys(SCANS, 0), 0
    for row, frozen in zip(rows, baseline):
        compared = row["comparisons"]
        require(set(compared) == set(conditions), "Unexpected scoring conditions")
        for condition in conditions:
            scan, metric = condition.split(":")
            c = compared[condition]
            require(set(c["opponents"]) == set(OPPONENTS), "Ten complete opponents required")
            require(all(math.isfinite(v) and not isinstance(v, bool)
                        for v in [c["candidate"], *c["opponents"].values()]), "Nonfinite plane metric")
            require(c["opponents"] == {label: frozen["scans"][scan]["metrics"][label][metric]
                                       for label in OPPONENTS}, "Frozen opponent plane scores changed")
            best = (min if metric == "nmse" else max)(c["opponents"].values())
            win = c["candidate"] < best if metric == "nmse" else c["candidate"] > best
            require(c["best_opponent"] == best and c["strict_win"] is win, "Inconsistent raw plane decision")
            counts[condition] += win
        wins = {s: all(compared[f"{s}:{m}"]["strict_win"] for m in METRICS) for s in SCANS}
        for scan in SCANS:
            by_scan[scan] += wins[scan]
        both = all(wins.values())
        require(row["strict_success"] is both, "Inconsistent paired plane decision")
        paired += both
    require(planes["per_condition_passing_planes"] == counts and
            planes["passing_planes_by_scan"] == by_scan and planes["passing_planes_both_scans"] == paired and
            planes["strict_success"] is (paired > 0), "59-plane summary counts disagree")
    require(objective["lesion_planes"] == {k: v for k, v in planes.items() if k != "planes"},
            "Objective and saved plane report disagree")
    return {"eligible_planes": 59, "passing_planes_both_scans": paired,
            "passing_planes_by_scan": by_scan, "per_condition_passing_planes": counts}


def prepare(run_dir, score_dir, name):
    """Resolve score/run/frozen-reference bindings before any figure is written."""
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha(path)
        require(expected is None or digest == expected, f"SHA256 mismatch: {path}")
        require(str(path) not in inputs or inputs[str(path)] == digest, f"Conflicting input hash: {path}")
        inputs[str(path)] = digest
        return path

    for path, digest in PINS.items():
        bind(path, digest)
    manifest, report, summary, contract = map(read, (LOOP, REPORT, SUMMARY, CONTRACT))
    require(report["loop_input_freeze_sha256"] == PINS[LOOP], "Freeze descriptors disagree")
    require(contract["metrics"] == list(METRICS) and contract["opponents"] == list(OPPONENTS), "Metric contract changed")
    baseline = read(bind(BASELINE, manifest["files"][str(BASELINE)]))
    run_dir, score_dir = Path(run_dir).resolve(strict=True), Path(score_dir).resolve(strict=True)
    objective_path, completed_path = bind(score_dir/"objective.json"), bind(run_dir/"completed.json")
    objective, completed = read(objective_path), read(completed_path)
    score_inputs = read(bind(score_dir/"input_hashes.json"))
    plane_path = bind(score_dir/"lesion_planes.json")
    planes = read(plane_path)
    saved_contract = read(bind(score_dir/"whole_head_contract.json"))
    require(name and Path(name).name == name and name not in (".", "..") and objective["name"] == name,
            "--name must equal the actual scored candidate name")
    require(objective["run_dir"] == str(run_dir) and completed["run_dir"] == str(run_dir), "Run directory binding mismatch")
    require(completed["status"] == "complete" and completed["passed"] is True and completed["returncode"] == 0
            and completed["stage"] == "trial" and completed["steps"] > 0, "A completed numerical trial is required")
    require(objective["metrics"] == list(METRICS) and objective["all_inputs_unchanged"] is True,
            "A completed three-metric score is required")
    require(score_inputs == objective["input_hashes"] and saved_contract == objective["whole_head_contract"],
            "Score manifests disagree with objective")
    require(objective["slice_contract"]["path"] == str(CONTRACT) and
            objective["slice_contract"]["sha256"] == PINS[CONTRACT], "Frozen slice contract mismatch")
    require(list(saved_contract["opponents"]) == list(OPPONENTS), "Opponent order or membership changed")
    paths = {label: {} for label in ("Reference", *OPPONENTS, "MC-GS")}
    for scan in SCANS:
        candidate = run_dir/f"candidate_{scan}_complex.npy"
        recorded = score_inputs["candidate"][scan]
        require(recorded["path"] == str(candidate) and recorded["shape"] == [256, 232, 176]
                and recorded["dtype"] == "complex64", "Scored candidate descriptor mismatch")
        expected = objective["candidate_sha256"][scan]
        require(recorded["sha256"] == expected == score_inputs["files"][str(candidate)] ==
                completed["artifacts"][str(candidate)], "Candidate export/score/completion hashes disagree")
        paths["MC-GS"][scan] = bind(candidate, expected)
        reference = TOPIC/f"recon/gt_{scan}_complex.npy"
        require(score_inputs["files"][str(reference)] == report["files"][str(reference)], "Reference score hash mismatch")
        paths["Reference"][scan] = bind(reference, report["files"][str(reference)])
        for label in OPPONENTS:
            descriptor = saved_contract["opponents"][label]["volumes"][scan]
            path = Path(summary["references"][label]["volumes"][scan])
            expected = manifest["files"][str(path)]
            require(descriptor == {"path": str(path), "sha256": expected} and
                    score_inputs["files"][str(path)] == expected, f"Frozen volume binding mismatch: {label}/{scan}")
            paths[label][scan] = bind(path, expected)
    # Both files are completion-bound, rather than assuming the claimed update budget.
    metrics_path = bind(run_dir/"metrics.json", completed["artifacts"][str(run_dir/"metrics.json")])
    launch_path = bind(run_dir/"launch.json", completed["artifacts"][str(run_dir/"launch.json")])
    metrics, launch = read(metrics_path), read(launch_path)
    require(all(launch.get(key) == value and metrics.get(key) == value
                for key,value in completed["parameters"].items()), "Launch/metrics/completion recipe mismatch")
    require(metrics["status"] == "complete" and metrics["steps_observed"] == completed["steps"]
            and metrics["gradients_finite"] is True, "Completed numerical update count mismatch")
    counts = validate_counts(planes, objective, baseline)
    mask3 = next(Path(p) for p in manifest["files"] if p.endswith("mask_af16_3d.h5"))
    bind(mask3, manifest["files"][str(mask3)])
    mask_pe = bind(TOPIC/"sim/mask/mask_af16_pe.npy")
    bind(Path(__file__))
    return {"inputs": inputs, "paths": paths, "objective": objective, "completed": completed,
            "planes": planes, "counts": counts, "mask3": mask3, "mask_pe": mask_pe,
            "objective_path": objective_path, "completed_path": completed_path, "plane_path": plane_path,
            "metrics": metrics, "launch": launch}


def orientation_helpers():
    # Importing the old script would run unrelated imports. Execute only its
    # two pinned, verbatim orientation helper definitions, as the old renderer did.
    nodes = [n for n in ast.parse(LAYOUT.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in ("plane", "mask_plane")]
    require(len(nodes) == 2, "Missing original orientation helpers")
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(LAYOUT), "exec"), namespace)
    return namespace["plane"], namespace["mask_plane"]


def render(context, out, name, include_historical=False):
    import h5py
    from skimage.metrics import structural_similarity
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    plane, mask_plane = orientation_helpers()
    labels = ("Reference", *OPPONENTS, "MC-GS")
    m2 = np.load(context["mask_pe"], allow_pickle=False) > 0
    with h5py.File(context["mask3"], "r") as archive:
        require(np.array_equal(np.asarray(archive["mask"]) > 0, np.broadcast_to(m2, (256,232,176))), "PE display mask differs from frozen mask")
    data = {}
    for scan in SCANS:
        vols = {}
        for label in labels:
            value = np.load(context["paths"][label][scan], mmap_mode="r", allow_pickle=False)
            require(value.shape == (256,232,176) and value.dtype == np.complex64 and np.isfinite(value).all(),
                    f"Invalid displayed volume: {label}/{scan}")
            vols[label] = np.abs(value)
        ref = vols["Reference"]
        p999, gmax = float(np.quantile(ref,.999)), float(ref.max())
        require(abs(p999/1.8-WINDOWS[scan][1]) < 1e-12, "Frozen display brightness changed")
        data[scan] = {"vols": vols, "head": ref > .05*p999, "gmax": gmax}
    selected_planes = (*PLANES, HISTORICAL) if include_historical else PLANES
    saved = {(row["axis"], row["index"]): row for row in context["planes"]["planes"]}
    validated_scores, differences = {}, []
    for axis, index, pname in selected_planes:
        for scan in SCANS:
            d, scores = data[scan], {}
            gt = np.take(d["vols"]["Reference"], index, axis=axis).astype(np.float64)
            head = np.take(d["head"], index, axis=axis)
            for label in labels[1:]:
                value = np.take(d["vols"][label], index, axis=axis).astype(np.float64)
                error = value[head]-gt[head]
                _, smap = structural_similarity(value, gt, data_range=d["gmax"], full=True)
                observed = {"psnr": float(20*np.log10(d["gmax"])-10*np.log10(np.mean(error**2))),
                            "ssim": float(smap[head].mean()), "nmse": float(np.sum(error**2)/np.sum(gt[head]**2))}
                authority = {m: (saved[axis,index]["comparisons"][f"{scan}:{m}"]["candidate"] if label == "MC-GS"
                                   else saved[axis,index]["comparisons"][f"{scan}:{m}"]["opponents"][label]) for m in METRICS}
                for metric in METRICS:
                    diff = abs(observed[metric]-authority[metric])
                    require(diff <= 1e-10, f"Displayed score parity failed: {scan}/{axis}/{index}/{label}/{metric}: {diff}")
                    differences.append(diff)
                scores[label] = authority
            validated_scores[f"{scan}:axis{axis}:{index}"] = scores
    out.mkdir(parents=True, exist_ok=False)
    save(out/"input_hashes_before.json", context["inputs"])
    plt.style.use(STYLE)
    plt.rcParams.update({"axes.grid":False, "savefig.bbox":None, "text.color":"#29322F", "pdf.fonttype":42})
    error_cmap = LinearSegmentedColormap.from_list("muted_blue_error", ["#070A0C","#243B53","#527B96","#A5BCC9","#EEE9DF"])
    sets = [("comparison_t204_c118_s44", PLANES, "Original example planes; no candidate-based plane selection")]
    if include_historical:
        sets.append(("historical_coronal129", (HISTORICAL,), "Historical G2-selected coronal129; not selected using this candidate"))
    figures, clipping = [], {}
    count = context["counts"]
    whole_success = context["objective"]["strict_success"]
    for stem_name, selected, purpose in sets:
        rows = [(kind,s,a,i,pn) for a,i,pn in selected for s in SCANS for kind in ("img","zoom")]
        ratios = [plane(data[s]["head"],a,i).shape[0]/plane(data[s]["head"],a,i).shape[1] if k == "img" else .5
                  for k,s,a,i,pn in rows]
        cw = 1.65
        width, height = cw*len(labels), cw*sum(ratios)
        fig = plt.figure(figsize=(width+1.,height+2.1))
        gs = fig.add_gridspec(len(rows),len(labels),height_ratios=ratios,left=.95/(width+1),right=.985,
                              top=1-1.25/(height+2.1),bottom=.85/(height+2.1),wspace=.03,hspace=.04)
        for ri,(kind,scan,axis,index,pname) in enumerate(rows):
            d = data[scan]
            gt, head = plane(d["vols"]["Reference"],axis,index), plane(d["head"],axis,index)
            y0,y1,x0,x1 = BOXES[axis,index]
            scores = validated_scores[f"{scan}:axis{axis}:{index}"]
            best = {m:(max if m != "nmse" else min)(v[m] for v in scores.values()) for m in METRICS}
            for ci,label in enumerate(labels):
                axh = fig.add_subplot(gs[ri,ci])
                axh.set_xticks([]); axh.set_yticks([]); axh.set_facecolor("black")
                value = plane(d["vols"][label],axis,index)
                if kind == "img":
                    vmax = WINDOWS[scan][1]
                    axh.imshow(value,cmap="gray",vmin=0,vmax=vmax,interpolation="nearest",origin="lower",aspect="equal")
                    axh.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor=ORANGE,lw=.9))
                    if ri == 0:
                        title = "Ref. AF5.33" if label == "Reference" else ("MC-GS candidate" if label == "MC-GS" else label)
                        axh.set_title(title,fontsize=10.5,loc="center",pad=9,weight="bold" if label == "MC-GS" else "normal")
                    if label == "Reference":
                        axh.text(.025,.98,f"Ref. ({scan})",color="#E5C984",fontsize=8.2,weight="bold",va="top",transform=axh.transAxes)
                    else:
                        values = scores[label]
                        text = (f"{values['psnr']:.2f} dB",f"SSIM {values['ssim']:.4f}",f"NMSE {values['nmse']:.5f}")
                        for k,(metric,line) in enumerate(zip(METRICS,text)):
                            axh.text(.025,.98-.087*k/max(ratios[ri],.6),line,color="#E5C984",fontsize=8.1,
                                     va="top",transform=axh.transAxes,weight="bold" if values[metric] == best[metric] else "normal")
                    clipping[f"{stem_name}:{scan}:{axis}:{index}:{label}:magnitude"] = float(np.mean(value > vmax))
                else:
                    for sp in axh.spines.values(): sp.set_visible(False)
                    left,right = axh.inset_axes([0.,0.,.495,1.]),axh.inset_axes([.505,0.,.495,1.])
                    for child in (left,right):
                        child.set_xticks([]); child.set_yticks([])
                        for sp in child.spines.values(): sp.set_visible(True); sp.set_edgecolor(ORANGE); sp.set_linewidth(.8)
                    left.imshow(value[y0:y1,x0:x1],cmap="gray",vmin=0,vmax=WINDOWS[scan][1],origin="lower",interpolation="nearest",aspect="auto")
                    if label == "Reference":
                        right.imshow(mask_plane(m2,axis,(256,232,176)),cmap="gray",origin="lower",interpolation="nearest",aspect="auto")
                    else:
                        error = np.where(head,np.abs(value.astype(np.float64)-gt.astype(np.float64))/d["gmax"],0.)[y0:y1,x0:x1]
                        right.imshow(error,cmap=error_cmap,vmin=0,vmax=.1,origin="lower",interpolation="nearest",aspect="auto")
                        clipping[f"{stem_name}:{scan}:{axis}:{index}:{label}:error"] = float(np.mean(error > .1))
                if ci == 0:
                    if kind == "img": axh.set_ylabel(f"{scan}  {pname} {index}",fontsize=9.5)
                    else: axh.text(-.04,.5,"zoom | error",rotation=90,ha="right",va="center",fontsize=8,transform=axh.transAxes)
        fig.suptitle(f"CE AF16 | measured reconstructions | MC-GS candidate: {name}\n"
                     "Per plane: plain (pre), then enhanced (post); full slice, lesion zoom (left) and absolute magnitude error (right).\n"
                     "PSNR / SSIM / NMSE: head-masked slice scores; bold = best among the 11 displayed reconstruction methods.\n"
                     "Reference = CS-SENSE AF5.33 reconstruction; fixed common gray windows. Reference zoom-right = sampling mask.",
                     fontsize=10.3,y=.993,linespacing=1.3)
        cax = fig.add_axes([.78,.58/(height+2.1),.19,.11/(height+2.1)])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,.1),cmap=error_cmap),cax=cax,orientation="horizontal")
        cb.set_label("Absolute magnitude error / max(reference)",fontsize=8.5)
        cb.ax.tick_params(labelsize=8,length=2)
        fig.text(.95/(width+1),.39/(height+2.1),
                 f"{purpose}. Paired three-metric wins: {count['passing_planes_both_scans']}/59; "
                 f"pre {count['passing_planes_by_scan']['pre']}/59, post {count['passing_planes_by_scan']['post']}/59.\n"
                 f"Whole-head six-metric target {'met' if whole_success else 'unmet'}. One case / one seed; "
                 "fixed original orientation, lesion boxes and display windows; no fitted method scaling.",
                 fontsize=8.5,va="center",linespacing=1.4)
        png,pdf = out/f"{stem_name}.png",out/f"{stem_name}.pdf"
        fig.savefig(png,dpi=300,facecolor="white")
        fig.savefig(pdf,dpi=200,facecolor="white")
        plt.close(fig)
        figures.append({"png":str(png),"pdf":str(pdf),"png_sha256":sha(png),"pdf_sha256":sha(pdf),
                        "planes":[list(p) for p in selected],"purpose":purpose})
    after = {p:sha(p) for p in context["inputs"]}
    require(after == context["inputs"], "An input changed during rendering")
    save(out/"input_hashes_after.json",after)
    save(out/"slice_scores_three_metrics.json",validated_scores)
    provenance = {"status":"rendered_pending_visual_QA","surface_class":"internal_review",
        "created_utc":datetime.now(timezone.utc).isoformat(),"name":name,"run_dir":context["objective"]["run_dir"],
        "source_objective":{"path":str(context["objective_path"]),"sha256":context["inputs"][str(context["objective_path"])]},
        "source_completed":{"path":str(context["completed_path"]),"sha256":context["inputs"][str(context["completed_path"])]},
        "candidate_sha256":context["objective"]["candidate_sha256"],"trial_steps":context["completed"]["steps"],
        "trial_mode":context["completed"]["mode"],"whole_head_score":context["objective"]["score"],
        "whole_head_strict_success":whole_success,"counts":count,"metrics":list(METRICS),
        "reference":"Frozen CS-SENSE AF5.33 reconstruction, not fully sampled ground truth",
        "score_authority":"Actual saved lesion_planes.json; displayed values independently recomputed from loaded volumes",
        "metric_parity_checks":len(differences),"maximum_absolute_metric_difference":max(differences),
        "display_windows":WINDOWS,"error_window":[0.,.1],"display_clipping_fractions":clipping,
        "zoom_boxes":{f"axis{a}:index{i}":list(BOXES[a,i]) for a,i,_ in selected_planes},
        "orientation":"Pinned original plane/mask_plane helper definitions, origin lower; no alternate flips",
        "error":"abs(abs(rec)-abs(reference))/global max(abs(reference)), head masked",
        "sampling_mask":"Original PE mask verified equal to the frozen acquired 3-D mask broadcast along readout",
        "input_sha256_before":context["inputs"],"input_sha256_after":after,"all_inputs_unchanged":True,
        "generator":str(Path(__file__).resolve()),"generator_sha256":context["inputs"][str(Path(__file__).resolve())],
        "style_source":str(ORIGINAL_RENDERER),"style_source_sha256":PINS[ORIGINAL_RENDERER],"figures":figures,
        "formal_mri_experiment_count_increment":0,
        "limits":"One case/seed adaptive development; fixed historical examples, unequal method budgets; figures do not establish superiority or generalization"}
    save(out/"provenance.json",provenance)
    return provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir",required=True,type=Path)
    ap.add_argument("--score-dir",required=True,type=Path)
    ap.add_argument("--out-dir",required=True,type=Path)
    ap.add_argument("--name",required=True)
    ap.add_argument("--include-historical-coronal129",action="store_true",
                    help="Add a separately labeled historical G2-selected coronal129 figure")
    args = ap.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("OMP_NUM_THREADS") == "1",
            "Use CUDA_VISIBLE_DEVICES= and OMP_NUM_THREADS=1 for this CPU-only renderer")
    out = args.out_dir.resolve()
    require(ROOT/"figures" in out.parents and not out.exists(), "--out-dir must be a fresh directory below this campaign's figures/")
    context = prepare(args.run_dir,args.score_dir,args.name)
    result = render(context,out,args.name,args.include_historical_coronal129)
    print(json.dumps({"output":str(out),"name":args.name,"metric_checks":result["metric_parity_checks"],
                      "max_metric_diff":result["maximum_absolute_metric_difference"],
                      "all_inputs_unchanged":True,"figures":result["figures"]},indent=2))


if __name__ == "__main__":
    main()
