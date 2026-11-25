#!/usr/bin/env python3
"""Evaluation of the second version (runs on amax, CPU).

    mevaluate.py --feas          the feasibility check (design section 10 step 1): DTa under the interleaved order
    mevaluate.py --dev TAG       report of a development round
    mevaluate.py --all           the formal matrix: criteria S1-S4, written to <root>/eval/summary.json

Scores. The frozen scorer inserts every sampled line (ledger head_psnr). An arm that weights the echo trains
(M2 adaptive or truth) is scored on the image with only the kept lines inserted (w >= 0.5, reference state):
head_psnr_wdc; that is its head_psnr_eval. Arms without weights keep the scorer's value. head_psnr_raw = raw render.
No clock value or duration is read, printed or written.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import weights
from mcommon import ARMS2, CONDITIONS2, DATA2, FORMAL, LEDGER2, MATRIX2, REG2, RES2, ROOT2, THRESH2, c1, exp_name

UNMOVED_MM = 0.05          # a clean echo train "did not move" at or below this true mean brain displacement (report only)

log = lambda *a: print(*a, flush=True)


def load_ledger(path=None):
    p = Path(path or LEDGER2)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def weighted(arm):
    return ARMS2[arm]["m2"] in ("adaptive", "truth")


def cached_scores(cache, key, size, compute):
    """Scores of one array of a result file, computed once: kept in the JSON file `cache` together with the size of
    the result file they were computed from; another size (a rerun) discards everything kept. compute(key) -> dict."""
    cache = Path(cache)
    kept = json.loads(cache.read_text()) if cache.exists() else {}
    if kept.get("canon_bytes") != int(size):
        kept = dict(canon_bytes=int(size), scores={})
    if key not in kept["scores"]:
        kept["scores"][key] = compute(key)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(kept))
        os.replace(tmp, cache)
    return kept["scores"][key]


def _file_scores(exp, key, single=False):
    """Head metrics of one array of <exp>_canon.npz; None when the file or the array is missing."""
    canon, vols = RES2 / f"{exp}_canon.npz", RES2 / f"{exp}_volumes.npz"
    if not (canon.exists() and vols.exists()):
        return None
    with np.load(canon) as z:
        if key not in z.files:
            return None
    import mrun
    return cached_scores(RES2 / f"{exp}_evalscores.json", key, canon.stat().st_size,
                         lambda k: mrun.head_scores(exp, k, single=single))


def _score_job(args):
    exp, single = args
    return exp, {k: _file_scores(exp, k, single) for k in ("rec_raw", "rec_w_dc")}


def fill_scores(ledger, workers=8):
    """Score the result files of all ledger rows in parallel processes (each file once, see cached_scores)."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = sorted({(r["exp"], ARMS2[r["arm"]]["trainer"] == "single") for r in ledger})
    if not jobs:
        return {}
    with ProcessPoolExecutor(max_workers=min(int(workers), len(jobs))) as pool:
        return dict(pool.map(_score_job, jobs))


def table(ledger, scores=None):
    """One row per experiment (the last ledger line of an experiment wins): exp, af, order, cond, arm, seed, tag,
    T1, T2. T2 (the only contrast of a single-contrast arm) carries head_psnr (scorer), head_psnr_raw,
    head_psnr_wdc (when the kept-lines image exists) and head_psnr_eval."""
    scores = scores or _file_scores
    rows = {}
    for r in ledger:
        single = ARMS2[r["arm"]]["trainer"] == "single"
        t2 = dict(r["plain"] if single else r["contrast"])
        raw, wdc = scores(r["exp"], "rec_raw", single), scores(r["exp"], "rec_w_dc", single)
        if raw:
            t2.update(head_psnr_raw=raw["head_psnr"], head_ssim_raw=raw["head_ssim"])
        if wdc:
            t2.update(head_psnr_wdc=wdc["head_psnr"], head_ssim_wdc=wdc["head_ssim"], head_hfen_wdc=wdc.get("head_hfen"))
        t2["head_psnr_eval"] = t2.get("head_psnr_wdc") if weighted(r["arm"]) else t2["head_psnr"]
        rows[r["exp"]] = dict(exp=r["exp"], af=r["af"], order=r["order"], cond=r["cond"], arm=r["arm"],
                              seed=r["seed"], tag=r.get("tag", ""), T1=None if single else dict(r["plain"]), T2=t2)
    return list(rows.values())


def val(tab, order, cond, arm, c="T2", key="head_psnr_eval", tag="", seed=42):
    """One number of the table; None when the run or the contrast is missing. T1 has no weighted variant, so its
    evaluation score is the scorer's head_psnr."""
    if c == "T1" and key == "head_psnr_eval":
        key = "head_psnr"
    for r in tab:
        if (r["order"], r["cond"], r["arm"], r["tag"], r["seed"]) == (order, cond, arm, tag, seed):
            return None if r[c] is None else r[c].get(key)
    return None


def recovered(tab, order, cond, arm, tag="", base="B0", ref="C0"):
    """Share of the T2 loss that an arm takes back: loss = base(ref) - base(cond); value = (arm - base(cond)) / loss.
    base = B0 (the unchanged joint model) or I-T2 for a single-contrast arm; ref = the condition without motion
    (C0, as registered; E2 for the report-only reading of E2Tb). The baselines are looked up with the arm's tag
    first, then untagged. value is None when the loss is below min_loss_db. None if a run is missing."""
    pick = lambda o, c: next((v for v in (val(tab, o, c, base, tag=tag), val(tab, o, c, base)) if v is not None), None)
    b0_c0, b0, a = pick("il", ref), pick(order, cond), val(tab, order, cond, arm, tag=tag)
    if b0_c0 is None or b0 is None or a is None:
        return None
    loss = b0_c0 - b0
    return dict(loss=loss, value=(a - b0) / loss if loss >= THRESH2["min_loss_db"] else None, b0_c0=b0_c0, b0=b0,
                arm_psnr=a, base=base)


def weight_detection(mm, truth):
    """A train counts as handled when its weight is below 0.5 or (M3) it was put into a state other than the
    reference state. recall = share of the contaminated trains that are handled; false_rate = share of the clean
    trains that are handled; frac_w1 = share of all trains with w = 1; auc of the last statistic z for the
    contaminated trains.
    Reported only, never judged (added after the formal rows were seen, 2026-11-23): false_rate_unmoved = the same
    share over the clean trains that did not move at all (true mean brain displacement <= UNMOVED_MM), so that a
    state opened for a real lasting shift below CONTAM_MM (Tb's step at 0.74) is not counted as a false detection;
    n_clean_moved / n_clean_moved_handled = the clean trains that did move, and how many of them were handled.
    Without a per-train displacement in the truth every clean train counts as unmoved."""
    w = np.asarray(mm["m2"]["w"], dtype=np.float64)
    cont = np.asarray(truth.get("contaminated", [False] * w.size), dtype=bool)   # no per-train truth: all clean
    disp = np.asarray(truth.get("train_brain_disp_mm", [0.0] * w.size), dtype=np.float64)
    handled = w < 0.5
    if mm.get("m3") and mm["m3"].get("n", 1) > 1:
        handled = handled | (np.asarray(mm["m3"]["state_of_train"]) != 0)
    z = mm["m2"].get("z")
    a = weights.auc(np.asarray(z, dtype=np.float64), cont) if z is not None else float("nan")
    unmoved, moved = ~cont & (disp <= UNMOVED_MM), ~cont & (disp > UNMOVED_MM)
    return dict(recall=float(handled[cont].mean()) if cont.any() else None,
                false_rate=float(handled[~cont].mean()) if (~cont).any() else None,
                false_rate_unmoved=float(handled[unmoved].mean()) if unmoved.any() else None,
                n_clean_moved=int(moved.sum()), n_clean_moved_handled=int(handled[moved].sum()),
                frac_w1=float((w == 1.0).mean()), auc=None if np.isnan(a) else a,
                n_contaminated=int(cont.sum()), n_down=int((w < 1.0).sum()))


def state_errors(mm, truth, brain, poses="final_poses"):
    """M3 against the truth: number of states found and true, the largest distance (in echo trains) between a found
    and a true boundary, and the largest pose error of a state (first version's evaluate.segment_errors). The
    boundary and pose errors are None when the numbers of states differ. None when the run has no M3.
    poses: which poses are compared -- "final_poses", "init_poses" (as set at the detection), or "identity"
    (no state pose at all: the size of the true shift)."""
    m3 = mm.get("m3")
    if not m3:
        return None
    import evaluate as ev1
    import geom
    raw = np.asarray(truth.get("state_of_train", [0] * len(m3["state_of_train"])), dtype=np.int64)
    true_b = [int(i) for i in np.flatnonzero(np.diff(raw)) + 1]
    n_true = len(true_b) + 1
    out = dict(n_states=int(m3["n"]), n_true=n_true, boundary_err_trains=None, rot_err_deg_max=None,
               t_err_mm_max=None, tre_mean_mm_max=None)
    if out["n_states"] != n_true:
        return out
    out["boundary_err_trains"] = max([abs(a - b) for a, b in zip(sorted(m3["boundaries"]), true_b)], default=0)
    zero = dict(theta_deg=[0.0, 0.0, 0.0], t_mm=[0.0, 0.0, 0.0])
    est = [zero] * n_true if poses == "identity" else m3.get(poses)
    if n_true > 1 and est:
        ref = int(truth["ref_state"])
        order = [ref] + [k for k in range(n_true) if k != ref]          # the run's numbering: reference first
        true_poses = [geom.Rigid.from_json(truth["state_poses"][k]) for k in order]
        e = ev1.segment_errors(est[:n_true], true_poses, 0, np.asarray(brain, dtype=bool))
        out.update(rot_err_deg_max=e["seg_rot_err_deg_max"], t_err_mm_max=e["seg_t_err_mm_max"],
                   tre_mean_mm_max=e["seg_tre_mean_mm_max"])
    return out


def inter_errors(st, truth_inter, disp, pre, brain):
    """Error of an adapter's inter-scan transform against the simulated one, inside the brain.
    st: the adapter's saved state; truth_inter: geom.Rigid of a rigid-only truth, or None when `disp` is given;
    disp: the truth's total displacement field (*shape, 3) in mm when it has a non-rigid part; pre: geom.Rigid, the
    transform found without simulated motion (the registration of C0), so the truth is (simulated motion) o pre
    -- the differential basis of the first version. Returns, against the whole true field, the errors of the rigid
    part alone (`rigid`) and of rigid + control grid (`full`); `rigid_part` = the estimated rigid part against the
    truth's rigid part alone (None when truth_inter is not given); and the largest control-grid displacement in the
    brain."""
    import evaluate as ev1
    import geom
    import warp
    brain = np.asarray(brain, dtype=bool)
    r = geom.grid_mm(brain.shape)
    x0 = pre.apply(r)
    y = x0 + warp._interp_field(np.asarray(disp, dtype=np.float64), x0) if disp is not None else truth_inter.apply(x0)
    d_true = y - r
    d_rigid = ev1.rigid_of(st).apply(r) - r
    part = None if truth_inter is None else ev1.field_errors(d_rigid, truth_inter.apply(x0) - r, brain)
    d_full = ev1.est_field(st, brain.shape)
    u = np.linalg.norm(ev1.est_field(st, brain.shape, part="nonrigid"), axis=-1)[brain]
    return dict(rigid_part=part, rigid=ev1.field_errors(d_rigid, d_true, brain),
                full=ev1.field_errors(d_full, d_true, brain), nonrigid_brain_max_mm=float(u.max()))


def inter_report(exp, order, cond, static=False):
    """inter_errors of one finished run, read from the files (None when the run has no adapter or the condition
    has no simulated inter-scan transform). static=True: only conditions WITHOUT a simulated inter-scan transform,
    whose truth is then the C0 registration itself (report only: how far the pose M1 found on motion-corrupted
    T2W data is from the pose without motion)."""
    import geom
    import reg
    p = RES2 / f"{exp}_motion.json"
    cdir = _cdir(order, cond)
    truth = json.loads((cdir / "motion_truth.json").read_text()) if (cdir / "motion_truth.json").exists() else {}
    if not p.exists() or bool(truth.get("inter")) == bool(static):
        return None
    if static:
        truth = dict(truth, inter=geom.Rigid.identity().to_json())
    has_disp = not static and (cdir / "disp_T2.npy").exists()       # a static truth has no inter-scan field
    disp = np.load(cdir / "disp_T2.npy").astype(np.float64) if has_disp else None
    brain = np.load(cdir / "brain.npy").astype(bool)
    af = next(t for t in exp.split("_") if t.startswith("af"))          # also for names with a prefix (smoke_)
    pre = reg.load_pose(REG2 / f"{af}_{order}_C0.json")
    return inter_errors(json.loads(p.read_text())["final"], geom.Rigid.from_json(truth["inter"]), disp, pre, brain)


def _load_mm(exp):
    p = RES2 / f"{exp}_mm.json"
    return json.loads(p.read_text()) if p.exists() else None


def _cdir(order, cond):
    """Data directory of a condition (the first version's for the conditions taken from it)."""
    v1 = CONDITIONS2[cond]["v1_data"]
    return c1.DATA / v1 if v1 else DATA2 / order / cond


def _truth(order, cond):
    return json.loads((_cdir(order, cond) / "motion_truth.json").read_text())


def feasibility(tab, mm=None, truth=None):
    """Design section 10 step 1, on DTa under the interleaved order: (1) the statistic of the unchanged model (B0,
    monitor) separates the contaminated echo trains, AUC >= feas_auc at the last computation; (2) truth weights (Ow)
    take back at least feas_recover of the loss."""
    mm = mm or _load_mm("mm_af12_il_DTa_B0")
    truth = truth or _truth("il", "DTa")
    cont = np.asarray(truth["contaminated"], dtype=bool)
    hist = [dict(it=h["it"], auc=weights.auc(np.asarray(h["z"], dtype=np.float64), cont)) for h in mm["m2"]["history"]]
    auc = hist[-1]["auc"] if hist else None
    r = recovered(tab, "il", "DTa", "Ow") or dict(loss=None, value=None)
    out = dict(auc=auc, auc_history=hist, loss=r["loss"], recover=r["value"], b0_c0=r.get("b0_c0"), b0=r.get("b0"),
               ow=r.get("arm_psnr"), n_contaminated=int(cont.sum()), n_trains=int(cont.size),
               pass_auc=bool(auc is not None and auc >= THRESH2["feas_auc"]),
               pass_recover=bool(r["value"] is not None and r["value"] >= THRESH2["feas_recover"]))
    if r["loss"] is not None and r["loss"] < THRESH2["min_loss_db"]:
        out["decision"] = "stop: motion too small to measure"
    else:
        out["decision"] = "go" if out["pass_auc"] and out["pass_recover"] else "stop"
    return out


def dev_report(tab, tag):
    """Rows of one development round (and the untagged rows they are compared with)."""
    out = []
    for r in tab:
        if r["tag"] != tag:
            continue
        d = dict(exp=r["exp"], T1=None if r["T1"] is None else r["T1"]["head_psnr"], T2_all=r["T2"]["head_psnr"],
                 T2_raw=r["T2"].get("head_psnr_raw"), T2_eval=r["T2"]["head_psnr_eval"])
        base = "I-T2" if ARMS2[r["arm"]]["trainer"] == "single" else "B0"
        d["recovered"] = recovered(tab, r["order"], r["cond"], r["arm"], tag=tag, base=base)
        d["vs_C0_B0"] = None if val(tab, "il", "C0", "B0") is None or d["T2_eval"] is None \
            else d["T2_eval"] - val(tab, "il", "C0", "B0")
        mm = _load_mm(r["exp"])
        if mm is not None:
            truth = _truth(r["order"], r["cond"])
            d["weights"] = weight_detection(mm, truth)
            if mm.get("m3"):
                brain = np.load(_cdir(r["order"], r["cond"]) / "brain.npy")
                d["states"] = state_errors(mm, truth, brain)
                d["states_init"] = state_errors(mm, truth, brain, poses="init_poses")
                d["states_identity"] = state_errors(mm, truth, brain, poses="identity")
                d["states"].update(detected_at=mm["m3"].get("detected_at"), boundaries=mm["m3"].get("boundaries"),
                                   init_poses=mm["m3"].get("init_poses"), final_poses=mm["m3"].get("final_poses"))
            if mm.get("align"):
                d["align"] = mm["align"]
        d["inter"] = inter_report(r["exp"], r["order"], r["cond"])
        out.append(d)
    return out


# ------------------------------------------------------------------------------------------------ criteria
def _item(value, threshold, sense, **extra):
    """One checked number. pass is None when the value is missing, never True."""
    if value is None:
        ok = None
    else:
        ok = bool(value >= threshold if sense == ">=" else value <= threshold if sense == "<=" else value == threshold)
    return dict(value=value, threshold=threshold, sense=sense, **{"pass": ok}, **extra)


def _all(parts):
    """None while any part is undecided, else whether all hold."""
    parts = list(parts)
    return None if (not parts or any(p is None for p in parts)) else all(parts)


def _diff(tab, a, b, c):
    """value(a) - value(b) for contrast c; a, b = (order, cond, arm); None when one is missing."""
    x, y = val(tab, *a, c=c), val(tab, *b, c=c)
    return None if x is None or y is None else x - y


def _s1(s):
    """Zero motion (C0, both orders): every arm against the unchanged model of its kind."""
    tab, T, arms = s["table"], THRESH2, {}
    for order, cond, arm in MATRIX2["gate"]:
        single = ARMS2[arm]["trainer"] == "single"
        base = ("il", "C0", "I-T2" if single else "B0")
        exp = exp_name("af12", order, cond, arm)
        st = s["states"].get(exp)
        n_states = st["n_states"] if st else (None if ARMS2[arm]["m3"] else 1)
        it = dict(dT1=_item(None if single else _diff(tab, (order, cond, arm), base, "T1"), T["s1_psnr"], ">="),
                  dT2=_item(_diff(tab, (order, cond, arm), base, "T2"), T["s1_psnr"], ">="),
                  frac_w1=_item((s["weights"].get(exp) or {}).get("frac_w1"), T["s1_w1_frac"], ">="),
                  n_states=_item(n_states, 1, "<="))
        it["pass"] = _all(v["pass"] for k, v in it.items() if not (single and k == "dT1"))
        arms[f"{order}:{arm}"] = it
    out = dict(arms=arms)
    out["pass"] = _all(a["pass"] for a in arms.values())
    r = (s.get("reg") or {}).get("il", {}).get("C0")
    if r:                                   # reported only: what the registration finds on the unmoved head
        import geom
        out["registration"] = dict(
            t_minus_native_mm=float(np.linalg.norm(np.asarray(r["t_mm"]) - np.asarray(c1.NATIVE_SHIFT_MM))),
            rot_deg=float(np.rad2deg(np.linalg.norm(geom.axis_angle(geom.rotmat(np.deg2rad(r["theta_deg"])))))))
    return out


def _s2(s):
    """Inter-scan (E2): registration error on the differential basis and the normalised change of PSNR.
    Normalised delta = PSNR(arm, E2) - PSNR(arm, C0); for T2 minus the same difference of the independent model."""
    tab, T, arms = s["table"], THRESH2, {}
    d_ind = _diff(tab, ("il", "E2", "I-T2"), ("il", "C0", "I-T2"), "T2")
    for arm in ("B0", "B1", "B123"):
        e = s["inter"].get(exp_name("af12", "il", "E2", arm)) or {}
        tre = lambda k: (e.get(k) or {}).get("tre_mean_mm")
        d1 = _diff(tab, ("il", "E2", arm), ("il", "C0", arm), "T1")
        d2 = _diff(tab, ("il", "E2", arm), ("il", "C0", arm), "T2")
        it = dict(tre_rigid=_item(tre("rigid_part"), T["s2_tre_rigid"], "<="),
                  tre_full=_item(tre("full"), T["s2_tre_nonrigid"], "<=", rigid_alone=tre("rigid"),
                                 nonrigid_brain_max_mm=e.get("nonrigid_brain_max_mm")),
                  delta_T1=_item(d1, T["s2_dT1"], ">="),
                  delta_T2=_item(None if d2 is None or d_ind is None else d2 - d_ind, T["s2_dT2"], ">=", raw=d2,
                                 independent=d_ind))
        if arm == "B0":                     # the unchanged model has no transform: reported, not judged
            for v in it.values():
                v["pass"] = None
            it["pass"] = None
        else:
            it["pass"] = _all(v["pass"] for v in it.values())
        arms[arm] = it
    return dict(arms=arms, **{"pass": _all(arms[a]["pass"] for a in ("B1", "B123"))})


def _s3(s):
    """Intra-scan, interleaved order: the full arm on Ta, Tb, Tc and E2Tb."""
    tab, T, conds = s["table"], THRESH2, {}
    for cond in ("Ta", "Tb", "Tc", "E2Tb"):
        exp = exp_name("af12", "il", cond, "B123")
        r = recovered(tab, "il", cond, "B123") or {}
        w = s["weights"].get(exp) or {}
        it = dict(recovered=_item(r.get("value"), T["s3_recover"], ">=", loss=r.get("loss"), b0_c0=r.get("b0_c0"),
                                  b0=r.get("b0"), arm_psnr=r.get("arm_psnr")),
                  recall=_item(w.get("recall"), T["s3_recall"], ">="),
                  false_rate=_item(w.get("false_rate"), T["s3_false"], "<="))
        if cond == "Tc":
            st = s["states"].get(exp)
            if st:
                found = st["n_states"] == st["n_true"]
                it["n_states"] = _item(st["n_states"], st["n_true"], "==")
                it["pose_deg"] = _item(st["rot_err_deg_max"], T["s3_pose_deg"], "<=")
                it["pose_mm"] = _item(st["t_err_mm_max"], T["s3_pose_mm"], "<=", boundary_err_trains=st["boundary_err_trains"])
                if not found:               # no pose to compare: the shift was not found as one state
                    it["pose_deg"]["pass"] = it["pose_mm"]["pass"] = False
            else:
                it["n_states"], it["pose_deg"], it["pose_mm"] = (_item(None, 2, "=="), _item(None, T["s3_pose_deg"], "<="),
                                                                  _item(None, T["s3_pose_mm"], "<="))
        skipped = r.get("loss") is not None and r["loss"] < T["min_loss_db"]
        if skipped:
            for v in it.values():
                v["pass"] = None
            it["note"] = (f"B0 loses only {r['loss']:.3f} dB against C0: the problem is not significant at this "
                          f"amplitude; not counted")
        parts = [v["pass"] for v in it.values() if isinstance(v, dict)]
        it["pass"], it["skipped"] = (None if skipped else _all(parts)), skipped
        it["others"] = {a: (recovered(tab, "il", cond, a) or {}).get("value") for a in ("B2", "B12", "Ow", "Op")
                        if val(tab, "il", cond, a) is not None}
        conds[cond] = it
    return dict(conds=conds, **{"pass": _all(c["pass"] for c in conds.values() if not c["skipped"])})


def _s4(s):
    """Lesions (TbL against Tb): contrast recovery of the full arm against the unchanged model, and crosstalk."""
    b0, b = (s.get("lesions") or {}).get("B0"), (s.get("lesions") or {}).get("B123")
    if not b0 or not b:
        return {"pass": None}
    ref = {x["label"]: x["contrast_recovery"] for x in b0}
    per = [dict(label=x["label"], contrast=x["contrast"], b123=x["contrast_recovery"], b0=ref.get(x["label"]))
           for x in b if x["contrast_recovery"] is not None and ref.get(x["label"]) is not None]
    xt = [abs(x["crosstalk"]) - x["crosstalk_tol"] for x in b if x.get("crosstalk") is not None]
    out = dict(recovery_vs_B0=_item(min((p["b123"] - p["b0"] for p in per), default=None), -THRESH2["s4_recovery_drop"],
                                    ">=", per_lesion=per),
               crosstalk=_item(max(xt, default=None), 0.0, "<="))
    out["pass"] = _all(v["pass"] for v in out.values())
    alt = {k: _s4(dict(lesions=v)).get("recovery_vs_B0") for k, v in (s.get("lesions_alt") or {}).items() if v}
    if alt:      # report only: the same T2 image rule for B0 and B123 (all measured lines, or no data consistency)
        out["report_only"] = {k: dict(value=v["value"], per_lesion=v.get("per_lesion")) for k, v in alt.items() if v}
    return out


def _kz(s):
    """Linear order: reported only. no_harm = the full arm is not more than kz_harm below the unchanged model."""
    tab, out = s["table"], {}
    for cond in ("Ta", "Tc"):
        v = _diff(tab, ("kz", cond, "B123"), ("kz", cond, "B0"), "T2")
        out[cond] = dict(value=v, threshold=THRESH2["kz_harm"], sense=">=", no_harm=None if v is None else bool(v >= THRESH2["kz_harm"]),
                         loss=_diff(tab, ("il", "C0", "B0"), ("kz", cond, "B0"), "T2"),
                         others={a: _diff(tab, ("kz", cond, a), ("kz", cond, "B0"), "T2") for a in ("Ow", "Op")
                                 if val(tab, "kz", cond, a) is not None}, **{"pass": None})
    return out


def criteria(summary):
    """S1-S4 of design section 11 and the linear-order report. summary: dict(table, weights, states, inter, lesions,
    reg). Every checked number is dict(value, threshold, sense, pass); a missing input gives pass None, never True."""
    return dict(S1=_s1(summary), S2=_s2(summary), S3=_s3(summary), S4=_s4(summary), kz=_kz(summary))


CLOCK_WORDS = ("time", "elapsed", "timestamp", "date")


def clock_keys(obj, path=""):
    """Paths of all keys whose name contains a clock word (the summary must have none)."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if any(w in str(k).lower() for w in CLOCK_WORDS):
                out.append(p)
            out += clock_keys(v, p)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += clock_keys(v, f"{path}[{i}]")
    return out


# ------------------------------------------------------------------------------------------------ summary
def _vols2(exp, arm, t2="eval"):
    """{"T1": (rec, gt), "T2": (rec, gt)} of one run. t2 = "eval": T2 of an arm with echo-train weights is its
    evaluated image (only the kept lines inserted), as in head_psnr_eval -- the image of S4; "all": every arm's T2 is
    the trainer's own image (all measured lines inserted); "raw": every arm's T2 is the raw render without data
    consistency. "all" and "raw" are report only: they apply the same rule to every arm. None if a file is
    missing."""
    p = RES2 / f"{exp}_volumes.npz"
    if not p.exists():
        return None
    with np.load(p) as v:
        out = {"T2": (v["rec"], v["gt"])} if ARMS2[arm]["trainer"] == "single" else \
            {"T1": (v["recT1"], v["gtT1"]), "T2": (v["recFL"], v["gtFL"])}
    canon = RES2 / f"{exp}_canon.npz"
    if t2 == "raw":
        if not canon.exists():
            return None
        with np.load(canon) as z:
            if "rec_raw" not in z.files:
                return None
            out["T2"] = (np.abs(z["rec_raw"]).astype(np.float32), out["T2"][1])
    elif t2 == "eval" and weighted(arm) and canon.exists():
        with np.load(canon) as z:
            if "rec_w_dc" in z.files:
                out["T2"] = (np.abs(z["rec_w_dc"]).astype(np.float32), out["T2"][1])
    return out


def lesion_report(order="il", cond="TbL", ref_cond="Tb", arms=("B0", "B123", "I-T2"), t2="eval"):
    """Lesion numbers (first version's evaluate.lesion_rows) of the runs on `cond`, with the same arm's run on
    `ref_cond` as the run without lesions; t2 = which T2 image (see _vols2). {} when the condition was not built."""
    import evaluate as ev1
    cdir = _cdir(order, cond)
    if not (cdir / "lesion_mask_T2.npy").exists():
        return {}
    truth = json.loads((cdir / "motion_truth.json").read_text())
    masks = dict(T1=np.load(cdir / "lesion_mask_canon.npy"), T2=np.load(cdir / "lesion_mask_T2.npy"))
    out = {}
    for arm in arms:
        v = _vols2(exp_name("af12", order, cond, arm), arm, t2)
        if v is not None:
            out[arm] = ev1.lesion_rows(v, _vols2(exp_name("af12", order, ref_cond, arm), arm, t2) or {}, truth, masks)
    return out


def build_summary():
    """Everything the result tables need, from the formal runs (untagged, seed 42; the development conditions are
    not in it). No clock key anywhere (asserted)."""
    formal = [exp_name("af12", *row) for row in FORMAL]
    tab = [r for r in table(load_ledger()) if r["exp"] in set(formal)]
    weights_, states, inter, inter_static = {}, {}, {}, {}
    for r in tab:
        mm = _load_mm(r["exp"])
        if mm is None:
            continue
        truth = _truth(r["order"], r["cond"])
        weights_[r["exp"]] = weight_detection(mm, truth)
        if mm.get("m3"):
            brain = np.load(_cdir(r["order"], r["cond"]) / "brain.npy")
            states[r["exp"]] = dict(state_errors(mm, truth, brain), boundaries=mm["m3"].get("boundaries"),
                                    init=state_errors(mm, truth, brain, poses="init_poses"),
                                    identity=state_errors(mm, truth, brain, poses="identity"),
                                    final_poses=mm["m3"].get("final_poses"))
        e = inter_report(r["exp"], r["order"], r["cond"])
        if e is not None:
            inter[r["exp"]] = e
        e = inter_report(r["exp"], r["order"], r["cond"], static=True)
        if e is not None:
            inter_static[r["exp"]] = e
    reg = {}
    for p in sorted(REG2.glob("af12_*.json")):
        _, order, cond = p.stem.split("_", 2)
        d = json.loads(p.read_text())
        reg.setdefault(order, {})[cond] = {k: d[k] for k in ("theta_deg", "t_mm", "cost", "check") if k in d}
    rec = {}
    for r in tab:
        if r["cond"] != "C0":
            base = "I-T2" if ARMS2[r["arm"]]["trainer"] == "single" else "B0"
            rec.setdefault(f"{r['order']}:{r['cond']}", {})[r["arm"]] = recovered(tab, r["order"], r["cond"], r["arm"],
                                                                                 base=base)
    s = dict(af="af12", n_formal=len(formal), n_runs=len(tab), missing=[e for e in formal if e not in {r["exp"] for r in tab}],
             thresholds=dict(THRESH2), table=tab, weights=weights_, states=states, inter=inter, reg=reg, recovered=rec,
             lesions=lesion_report(), inter_static=inter_static,
             lesions_alt={k: lesion_report(t2=k) for k in ("all", "raw")},
             recovered_ref_E2={a: recovered(tab, "il", "E2Tb", a, ref="E2")
                               for a in sorted({r["arm"] for r in tab if (r["order"], r["cond"]) == ("il", "E2Tb")})})
    s["criteria"] = criteria(s)
    bad = clock_keys(s)
    assert not bad, f"clock keys in the summary: {bad}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feas", action="store_true")
    ap.add_argument("--dev", metavar="TAG", help="development round, e.g. d1; '' = the untagged rows")
    ap.add_argument("--all", action="store_true", help="formal matrix -> <root>/eval/summary.json")
    ap.add_argument("--workers", type=int, default=8, help="processes that score result files")
    args = ap.parse_args()
    ledger = load_ledger()
    if args.dev is not None:                # a development report scores its own rows and the baselines only
        ledger = [r for r in ledger if r.get("tag", "") in (args.dev, "")]
    fill_scores(ledger, args.workers)
    tab = table(ledger)
    if args.feas:
        f = feasibility(tab)
        for name in ("mm_af12_il_C0_B0", "mm_af12_il_DTa_B0", "mm_af12_il_DTa_Ow"):
            r = next((x for x in tab if x["exp"] == name), None)
            if r:
                log(f"{name}: T1 {r['T1']['head_psnr']:.3f}  T2 all lines {r['T2']['head_psnr']:.3f}  "
                    f"raw {r['T2'].get('head_psnr_raw')}  eval {r['T2']['head_psnr_eval']}")
        log(json.dumps(f, indent=1))
        mm = _load_mm("mm_af12_il_DTa_B0")
        log("FEAS z of the last computation, per echo train (C = contaminated): "
            + " ".join(f"{z:.1f}{'C' if c else ''}" for z, c in zip(mm["m2"]["z"], _truth("il", "DTa")["contaminated"])))
        log(f"FEASIBILITY decision: {f['decision']}")
    elif args.dev is not None:
        log(json.dumps(dev_report(tab, args.dev), indent=1))
    elif args.all:
        s = build_summary()
        out = ROOT2 / "eval" / "summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(s, indent=1) + "\n")
        c = s["criteria"]
        log(f"summary: {s['n_runs']} of {s['n_formal']} formal runs; missing {s['missing']}")
        for k in ("S1", "S2", "S3", "S4"):
            log(f"{k}: pass = {c[k]['pass']}")
        log("kz: " + json.dumps({k: (v["value"], v["no_harm"]) for k, v in c["kz"].items()}))
        log(f"wrote {out}")
    else:
        ap.error("nothing to do")


if __name__ == "__main__":
    main()
