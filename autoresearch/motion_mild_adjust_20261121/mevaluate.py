#!/usr/bin/env python3
"""Evaluation of the second version (runs on amax, CPU).

    mevaluate.py --feas          the feasibility check (design section 10 step 1): DTa under the interleaved order
    mevaluate.py --dev TAG       report of a development round

Scores. The frozen scorer inserts every sampled line (ledger head_psnr). An arm that weights the echo trains
(M2 adaptive or truth) is scored on the image with only the kept lines inserted (w >= 0.5, reference state):
head_psnr_wdc; that is its head_psnr_eval. Arms without weights keep the scorer's value. head_psnr_raw = raw render.
No clock value or duration is read, printed or written.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import weights
from mcommon import ARMS2, CONDITIONS2, DATA2, LEDGER2, REG2, RES2, THRESH2, c1

log = lambda *a: print(*a, flush=True)


def load_ledger(path=None):
    p = Path(path or LEDGER2)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def weighted(arm):
    return ARMS2[arm]["m2"] in ("adaptive", "truth")


def _file_scores(exp, key, single=False):
    """Head metrics of one array of <exp>_canon.npz; None when the file or the array is missing."""
    canon, vols = RES2 / f"{exp}_canon.npz", RES2 / f"{exp}_volumes.npz"
    if not (canon.exists() and vols.exists()):
        return None
    with np.load(canon) as z:
        if key not in z.files:
            return None
    import mrun
    return mrun.head_scores(exp, key, single=single)


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


def recovered(tab, order, cond, arm, tag="", base="B0"):
    """Share of the T2 loss that an arm takes back: loss = base(C0) - base(cond); value = (arm - base(cond)) / loss.
    base = B0 (the unchanged joint model) or I-T2 for a single-contrast arm. The baselines are looked up with the
    arm's tag first, then untagged. value is None when the loss is below min_loss_db. None if a run is missing."""
    pick = lambda o, c: next((v for v in (val(tab, o, c, base, tag=tag), val(tab, o, c, base)) if v is not None), None)
    b0_c0, b0, a = pick("il", "C0"), pick(order, cond), val(tab, order, cond, arm, tag=tag)
    if b0_c0 is None or b0 is None or a is None:
        return None
    loss = b0_c0 - b0
    return dict(loss=loss, value=(a - b0) / loss if loss >= THRESH2["min_loss_db"] else None, b0_c0=b0_c0, b0=b0,
                arm_psnr=a, base=base)


def weight_detection(mm, truth):
    """A train counts as handled when its weight is below 0.5 or (M3) it was put into a state other than the
    reference state. recall = share of the contaminated trains that are handled; false_rate = share of the clean
    trains that are handled; frac_w1 = share of all trains with w = 1; auc of the last statistic z for the
    contaminated trains."""
    w = np.asarray(mm["m2"]["w"], dtype=np.float64)
    cont = np.asarray(truth["contaminated"], dtype=bool)
    handled = w < 0.5
    if mm.get("m3") and mm["m3"].get("n", 1) > 1:
        handled = handled | (np.asarray(mm["m3"]["state_of_train"]) != 0)
    z = mm["m2"].get("z")
    a = weights.auc(np.asarray(z, dtype=np.float64), cont) if z is not None else float("nan")
    return dict(recall=float(handled[cont].mean()) if cont.any() else None,
                false_rate=float(handled[~cont].mean()) if (~cont).any() else None,
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
    -- the differential basis of the first version. Returns the errors of the rigid part alone, of rigid + control
    grid, and the largest control-grid displacement in the brain."""
    import evaluate as ev1
    import geom
    import warp
    brain = np.asarray(brain, dtype=bool)
    r = geom.grid_mm(brain.shape)
    x0 = pre.apply(r)
    y = x0 + warp._interp_field(np.asarray(disp, dtype=np.float64), x0) if disp is not None else truth_inter.apply(x0)
    d_true = y - r
    d_rigid = ev1.rigid_of(st).apply(r) - r
    d_full = ev1.est_field(st, brain.shape)
    u = np.linalg.norm(ev1.est_field(st, brain.shape, part="nonrigid"), axis=-1)[brain]
    return dict(rigid=ev1.field_errors(d_rigid, d_true, brain), full=ev1.field_errors(d_full, d_true, brain),
                nonrigid_brain_max_mm=float(u.max()))


def inter_report(exp, order, cond):
    """inter_errors of one finished run, read from the files (None when the run has no adapter or the condition
    has no simulated inter-scan transform)."""
    import geom
    import reg
    spec = CONDITIONS2[cond]
    p = RES2 / f"{exp}_motion.json"
    cdir = c1.DATA / spec["v1_data"] if spec["v1_data"] else DATA2 / order / cond
    truth = json.loads((cdir / "motion_truth.json").read_text()) if (cdir / "motion_truth.json").exists() else {}
    if not p.exists() or not truth.get("inter"):
        return None
    disp = np.load(cdir / "disp_T2.npy").astype(np.float64) if (cdir / "disp_T2.npy").exists() else None
    brain = np.load(cdir / "brain.npy").astype(bool)
    af = next(t for t in exp.split("_") if t.startswith("af"))          # also for names with a prefix (smoke_)
    pre = reg.load_pose(REG2 / f"{af}_{order}_C0.json")
    return inter_errors(json.loads(p.read_text())["final"], geom.Rigid.from_json(truth["inter"]), disp, pre, brain)


def _load_mm(exp):
    p = RES2 / f"{exp}_mm.json"
    return json.loads(p.read_text()) if p.exists() else None


def _truth(order, cond):
    return json.loads((DATA2 / order / cond / "motion_truth.json").read_text())


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
                brain = np.load(DATA2 / r["order"] / r["cond"] / "brain.npy")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feas", action="store_true")
    ap.add_argument("--dev", metavar="TAG", help="development round, e.g. d1; '' = the untagged rows")
    args = ap.parse_args()
    tab = table(load_ledger())
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
    else:
        ap.error("nothing to do")


if __name__ == "__main__":
    main()
