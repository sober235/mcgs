#!/usr/bin/env python3
"""Evaluation of the motion study (design sections 11 and 12). Runs on amax, CPU only.

    evaluate.py --all      image table, normalised deltas, motion errors, lesion metrics, zero-motion checks and the
                           criteria S1-S4 for every AF with ledger rows -> ROOT/eval/summary.json (+ artifact copy)
    evaluate.py --dev      motion errors of the development rows and the zero-motion check of C0 (Task 8)

Image metrics come from the frozen scorer (ledger rows written by run.py). Where the design asks for a different data
consistency (arms that model intra-scan motion: only the reference segment's lines are inserted), the frozen
head-only scorer is applied to that volume. Nothing here feeds back into training.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

import geom
import warp
from common import ARMS, CONDITIONS, DATA, LEDGER, MASKS, N, NATIVE_SHIFT_MM, RES, ROOT, TR, exp_name

THRESH = dict(s1_psnr=-0.05, s1_t_mm=0.3, s1_rot_deg=0.2, s1_nonrigid_mm=0.3, s1_seg_deg=0.2, s1_seg_mm=0.2,
              s2_t2=-0.1, s2_t1=-0.05, s2_tre_rigid=0.2, s2_tre_nonrigid=0.3,
              s3_recovery=0.9, s3_centroid_mm=0.5, s4_recovered=0.5, s4_seg_deg=0.3, s4_seg_mm=0.3,
              s4_min_loss_db=0.05)
log = lambda *a: print(*a, flush=True)


# ------------------------------------------------------------------------------------------------ images
def load_ledger(path=None):
    """Last ledger row of every experiment."""
    rows = {}
    for line in Path(path or LEDGER).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["exp"]] = r
    return list(rows.values())


def image_table(ledger):
    """One row per run with the metrics of T1 and T2. Joint runs: plain = T1, contrast = T2; single runs have one."""
    out = []
    for r in ledger:
        row = dict(af=r["af"], cond=r["cond"], arm=r["arm"], seed=r["seed"], exp=r["exp"], tag=r.get("tag", ""),
                   T1=None, T2=None)
        if ARMS[r["arm"]]["trainer"] == "single":
            row["T1" if ARMS[r["arm"]]["slot"] == "plain" else "T2"] = dict(r["plain"])
        else:
            row["T1"], row["T2"] = dict(r["plain"]), dict(r["contrast"])
        for c in ("T1", "T2"):
            if row[c] is not None:
                row[c]["head_psnr_eval"] = row[c]["head_psnr"]
        out.append(row)
    return out


def models_intra(arm):
    return bool(ARMS[arm]["segments"] or ARMS[arm]["oracle"])


def attach_alt_scores(table):
    """Adds to T2 (or the single contrast) of patched runs: the raw render's head metrics and, for arms that model
    intra-scan motion on an intra-scan condition, the metrics with only the reference segment's lines inserted;
    there head_psnr_eval becomes that value (design 11.1)."""
    sys.path.insert(0, str(TR))
    from diag_artefact.headonly import score_headonly
    for row in table:
        canon, vols = RES / f"{row['exp']}_canon.npz", RES / f"{row['exp']}_volumes.npz"
        if not (canon.exists() and vols.exists()):
            continue
        single = ARMS[row["arm"]]["trainer"] == "single"
        c = ("T1" if ARMS[row["arm"]]["slot"] == "plain" else "T2") if single else "T2"
        with np.load(vols) as v:
            gt = v["gt"] if single else v["gtFL"]
        with np.load(canon) as z:
            raw = score_headonly(np.abs(z["rec_raw"]), gt)
            row[c].update(head_psnr_raw=raw["head_psnr"], head_ssim_raw=raw["head_ssim"])
            if "rec_ref_dc" in z.files and CONDITIONS[row["cond"]]["intra"] and models_intra(row["arm"]):
                s = score_headonly(np.abs(z["rec_ref_dc"]), gt)
                row[c].update(head_psnr_refdc=s["head_psnr"], head_ssim_refdc=s["head_ssim"],
                              head_hfen_refdc=s["head_hfen"], head_psnr_eval=s["head_psnr"])
    return table


def _val(table, af, cond, arm, c, key="head_psnr_eval", seed=42, tag=""):
    for r in table:
        if (r["af"], r["cond"], r["arm"], r["seed"], r.get("tag", "")) == (af, cond, arm, seed, tag) and r.get(c):
            return r[c].get(key)
    return None


def normalised_delta(table, af, cond, arm):
    """Delta = PSNR(arm, cond) - PSNR(arm, C0), minus the same difference of the independent single-contrast model."""
    out, notes = {}, []
    for c, indep in (("T1", "I-T1"), ("T2", "I-T2")):
        x = _val(table, af, cond, arm, c)
        base = _val(table, af, "C0", arm, c)
        if x is not None and base is None:
            base = _val(table, af, "C0", "A0", c)
            notes.append(f"{c}: no C0 run of {arm}; the C0 value of A0 is used")
        if x is None or base is None:
            out[c] = None
            continue
        d = x - base
        ix, ib = _val(table, af, cond, indep, c), _val(table, af, "C0", indep, c)
        if ix is not None and ib is not None:
            d -= ix - ib
        else:
            notes.append(f"{c}: not normalised (no {indep} run for {cond})")
        out[c] = float(d)
    out["notes"] = notes
    return out


# ------------------------------------------------------------------------------------------------ motion
def rigid_of(state):
    if not state or state.get("theta_deg") is None:
        return geom.Rigid.identity()
    return geom.Rigid(geom.rotmat(np.deg2rad(state["theta_deg"])), np.asarray(state["t_mm"], dtype=np.float64))


def est_field(state, shape, seg=None, part="all"):
    """T(r) - r of a saved adapter state on the voxel grid (numpy copy of MotionAdapter.transform).
    part="nonrigid": the control-grid displacement u(r) alone."""
    r = geom.grid_mm(shape)
    u = np.zeros_like(r)
    if state.get("ctrl_mm") is not None:
        ctrl = np.asarray(state["ctrl_mm"], dtype=np.float64)
        g = ctrl.shape[0]
        half = np.asarray(shape, dtype=np.float64) / 2.0
        idx = np.clip(np.moveaxis((r / half + 1.0) / 2.0 * (g - 1), -1, 0), 0, g - 1)
        u = np.stack([ndi.map_coordinates(ctrl[..., k], idx, order=1, mode="nearest") for k in range(3)], axis=-1)
    if part == "nonrigid":
        return u
    x = rigid_of(state).apply(r + u)
    segs = state.get("segments") or []
    if seg is not None and segs and int(seg) != int(state["ref_segment"]):
        x = rigid_of(segs[int(seg)]).apply(x)
    return x - r


def _angle_deg(R):
    return float(np.rad2deg(np.linalg.norm(geom.axis_angle(R))))


def _kabsch_residual_rms(P, Q):
    """RMS distance between point sets after the best rigid fit of P onto Q."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return float(np.sqrt(((((P - pc) @ R.T + qc) - Q) ** 2).sum(-1).mean()))


def field_errors(d_est, d_true, brain):
    t = geom.tre(d_est, d_true, brain)
    r = geom.grid_mm(brain.shape)[brain][::7]
    return dict(tre_mean_mm=t["mean_mm"], tre_max_mm=t["max_mm"],
                nonrigid_rmse_mm=_kabsch_residual_rms(r + d_est[brain][::7], r + d_true[brain][::7]))


def rigid_motion_errors(est, truth, c0_est, brain):
    """est / truth / c0_est: geom.Rigid. Absolute truth = truth o (assumed native shift); differential truth =
    truth o (the same arm's estimate on C0)."""
    r = geom.grid_mm(brain.shape)
    d_est = est.apply(r) - r
    out = dict(rot_err_deg=_angle_deg(est.R @ truth.R.T))
    for tag, pre in (("absolute", geom.Rigid(np.eye(3), np.asarray(NATIVE_SHIFT_MM, dtype=np.float64))),
                     ("differential", c0_est)):
        if pre is None:
            continue
        e = field_errors(d_est, truth.compose(pre).apply(r) - r, brain)
        out.update({f"tre_mean_mm_{tag}": e["tre_mean_mm"], f"tre_max_mm_{tag}": e["tre_max_mm"]})
    return out


def segment_errors(est_segments, true_mean, ref_seg, brain):
    """Per-segment pose errors. est_segments: [{theta_deg, t_mm}] relative to the reference segment;
    true_mean: geom.Rigid mean pose of every segment (relative to the reference echo train)."""
    r = geom.grid_mm(brain.shape)[brain][::7]
    ref_inv = true_mean[ref_seg].inverse()
    rot, tr, tre = [], [], []
    for k, e in enumerate(est_segments):
        want, got = true_mean[k].compose(ref_inv), rigid_of(e)
        rot.append(_angle_deg(got.R @ want.R.T))
        tr.append(float(np.linalg.norm(got.t - want.t)))
        tre.append(float(np.linalg.norm(got.apply(r) - want.apply(r), axis=-1).mean()))
    return dict(seg_rot_err_deg=rot, seg_t_err_mm=tr, seg_tre_mean_mm=tre, seg_rot_err_deg_max=max(rot),
                seg_t_err_mm_max=max(tr), seg_tre_mean_mm_max=max(tre))


def _brain(cond):
    p = DATA / cond / "brain.npy"
    return np.load(p if p.exists() else DATA / "C0/brain.npy").astype(bool)


def _state(exp):
    p = RES / f"{exp}_motion.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["final"]


def motion_errors(exp, cond, c0_exp=None):
    """Errors of the estimated transform of one run against the simulated truth (None if the run has no adapter)."""
    st = _state(exp)
    if st is None:
        return None
    spec, brain, r = CONDITIONS[cond], _brain(cond), geom.grid_mm(N)
    truth = json.loads((DATA / cond / "motion_truth.json").read_text()) if cond != "C0" else {}
    g = geom.Rigid.from_json(truth["inter"]) if truth.get("inter") else geom.Rigid.identity()
    disp = np.load(DATA / cond / "disp_T2.npy").astype(np.float64) if spec["nonrigid"] else None
    seg_mean = [geom.Rigid.from_json(p) for p in truth["segment_mean_poses"]] if spec["intra"] else None
    ref = seg_mean[int(truth["ref_seg"])] if seg_mean else geom.Rigid.identity()

    def t_true(x):                                      # (reference-segment pose) o (simulated inter-scan motion)
        y = x + warp._interp_field(disp, x) if disp is not None else g.apply(x)
        return ref.apply(y)

    d_est = est_field(st, N)
    out = dict(exp=exp, cond=cond, oracle_fit_mm=st.get("oracle_fit_mm"),
               rot_err_deg=_angle_deg(rigid_of(st).R @ (ref.R @ g.R).T),
               est_theta_deg=st.get("theta_deg"), est_t_mm=st.get("t_mm"))
    pres = [("absolute", r + np.asarray(NATIVE_SHIFT_MM, dtype=np.float64))]
    c0 = _state(c0_exp) if c0_exp else None
    if c0 is not None:
        pres.append(("differential", r + est_field(c0, N)))
        out["differential_against"] = c0_exp
    for tag, x0 in pres:
        e = field_errors(d_est, t_true(x0) - r, brain)
        out.update({f"{k}_{tag}": v for k, v in e.items()})
    if st.get("ctrl_mm") is not None:
        out["nonrigid_brain_max_mm"] = float(np.linalg.norm(est_field(st, N, part="nonrigid"), axis=-1)[brain].max())
    if st.get("segments"):
        true_mean = seg_mean or [geom.Rigid.identity()] * len(st["segments"])
        out.update(segment_errors(st["segments"], true_mean, int(st["ref_segment"]), brain))
    return out


def zero_motion_checks(af, table, tag=""):
    """C0: every adapter arm against A0, the estimated pose against the image-based native shift."""
    out = {}
    for row in table:
        if (row["af"] != af or row["cond"] != "C0" or row["seed"] != 42 or row["arm"] in ("A0", "O")
                or row.get("tag", "") != tag):
            continue
        st = _state(row["exp"])
        if st is None:
            continue
        brain = _brain("C0")
        d = dict(dpsnr_T1=None, dpsnr_T2=None, t_minus_native_mm=None, rot_deg=None, nonrigid_brain_max_mm=None,
                 seg_max_deg=None, seg_max_mm=None)
        for c in ("T1", "T2"):
            a = _val(table, af, "C0", row["arm"], c, "head_psnr", tag=tag)
            b = _val(table, af, "C0", "A0", c, "head_psnr")
            if a is not None and b is not None:
                d[f"dpsnr_{c}"] = float(a - b)
        if st.get("theta_deg") is not None:
            d["t_minus_native_mm"] = float(np.linalg.norm(np.asarray(st["t_mm"]) - np.asarray(NATIVE_SHIFT_MM)))
            d["rot_deg"] = float(np.linalg.norm(st["theta_deg"]))
            d["t_mm"], d["theta_deg"] = st["t_mm"], st["theta_deg"]
        if st.get("ctrl_mm") is not None:
            d["nonrigid_brain_max_mm"] = float(np.linalg.norm(est_field(st, N, part="nonrigid"), axis=-1)[brain].max())
        if st.get("segments"):
            d["seg_max_deg"] = float(max(np.linalg.norm(s["theta_deg"]) for s in st["segments"]))
            d["seg_max_mm"] = float(max(np.linalg.norm(s["t_mm"]) for s in st["segments"]))
        out[row["arm"]] = d
    return out


# ------------------------------------------------------------------------------------------------ lesions
def _shell(mask):
    """3 voxels thick, 1 voxel away from the mask."""
    return ndi.binary_dilation(mask, iterations=4) & ~ndi.binary_dilation(mask, iterations=1)


def contrast_recovery(rec, gt, mask):
    """(lesion mean - shell mean) of the reconstruction over the same quantity of the reference."""
    sh = _shell(mask)
    return float((rec[mask].mean() - rec[sh].mean()) / (gt[mask].mean() - gt[sh].mean()))


def centroid_shift_mm(rec, gt, mask):
    """Distance between the centroids of the lesion's excess (or deficit) signal in reconstruction and reference."""
    region, sh = ndi.binary_dilation(mask, iterations=2), _shell(mask)
    sign = np.sign(gt[mask].mean() - gt[sh].mean())
    idx = np.argwhere(region).astype(np.float64)

    def cen(v):
        w = np.clip(sign * (v[region] - v[sh].mean()), 0.0, None)
        return (idx * w[:, None]).sum(0) / max(float(w.sum()), 1e-12)
    return float(np.linalg.norm(cen(rec) - cen(gt)))


def _vols(exp, arm):
    p = RES / f"{exp}_volumes.npz"
    if not p.exists():
        return None
    with np.load(p) as v:
        if ARMS[arm]["trainer"] == "single":
            c = "T1" if ARMS[arm]["slot"] == "plain" else "T2"
            return {c: (v["rec"], v["gt"])}
        return {"T1": (v["recT1"], v["gtT1"]), "T2": (v["recFL"], v["gtFL"])}


def lesion_metrics(af, arm, cond="E3", ref_cond="E2"):
    """Per lesion: contrast recovery and centroid shift in its own contrast; crosstalk = mean (rec - ref) of the OTHER
    contrast inside the lesion mask, relative to the surrounding signal, minus the same number of the run without
    lesions (ref_cond); tolerance = 3 standard errors of that difference from the shells' voxel scatter."""
    v = _vols(exp_name(af, cond, arm), arm)
    if v is None:
        return None
    v0 = _vols(exp_name(af, ref_cond, arm), arm) or {}
    truth = json.loads((DATA / cond / "motion_truth.json").read_text())
    masks = dict(T1=np.load(DATA / cond / "lesion_mask_canon.npy"), T2=np.load(DATA / cond / "lesion_mask_T2.npy"))
    out = []
    for k, site in enumerate(truth["lesion_sites"], start=1):
        own = site["contrast"]
        other = "T1" if own == "T2" else "T2"
        d = dict(label=k, contrast=own, diameter_mm=site["diameter_mm"], gain=site["gain"],
                 contrast_recovery=None, centroid_shift_mm=None, crosstalk=None, crosstalk_tol=None)
        if own in v:
            rec, gt = v[own]
            m = masks[own] == k
            d.update(contrast_recovery=contrast_recovery(rec, gt, m), centroid_shift_mm=centroid_shift_mm(rec, gt, m))
        if other in v:
            rec, gt = v[other]
            m = masks[other] == k
            sh = _shell(m)
            scale = float(gt[sh].mean())
            x = float((rec[m] - gt[m]).mean()) / scale
            d["other_bias"] = x
            if other in v0:
                rec0, gt0 = v0[other]
                x0 = float((rec0[m] - gt0[m]).mean()) / float(gt0[sh].mean())
                se = np.sqrt(((rec - gt)[sh].var() + (rec0 - gt0)[sh].var())) / scale / np.sqrt(m.sum())
                d.update(crosstalk=x - x0, crosstalk_tol=float(3.0 * se), other_bias_ref=x0)
        out.append(d)
    return out


# ------------------------------------------------------------------------------------------------ criteria
def _item(value, threshold, sense, **extra):
    ok = None if value is None else bool(value >= threshold if sense == ">=" else value <= threshold)
    return dict(value=value, threshold=threshold, sense=sense, **{"pass": ok}, **extra)


def _worst(values, pick):
    vals = [v for v in values if v is not None]
    return pick(vals) if vals else None


def criteria(summary, af="af12"):
    """S1-S4 of design section 12. A missing input gives pass=None, never True."""
    s = summary[af]
    table, zero, delta, motion, lesions = s["table"], s["zero"], s["delta"], s["motion"], s["lesions"]
    T = THRESH
    out = {}
    z = list(zero.values())
    out["S1"] = dict(
        psnr=_item(_worst([a[k] for a in z for k in ("dpsnr_T1", "dpsnr_T2")], min), T["s1_psnr"], ">="),
        translation=_item(_worst([a["t_minus_native_mm"] for a in z], max), T["s1_t_mm"], "<="),
        rotation=_item(_worst([a["rot_deg"] for a in z], max), T["s1_rot_deg"], "<="),
        nonrigid=_item(_worst([a["nonrigid_brain_max_mm"] for a in z], max), T["s1_nonrigid_mm"], "<="),
        segments_deg=_item(_worst([a["seg_max_deg"] for a in z], max), T["s1_seg_deg"], "<="),
        segments_mm=_item(_worst([a["seg_max_mm"] for a in z], max), T["s1_seg_mm"], "<="))
    out["S2"] = {}
    for cond in ("E1a", "E1b", "E2"):
        dc = delta.get(cond, {})
        arm = "A2" if (cond == "E2" or "A1" not in dc) else "A1"
        d = dc.get(arm, {})
        m = motion.get(exp_name(af, cond, arm)) or {}
        basis = "differential" if m.get("tre_mean_mm_differential") is not None else "absolute"
        thr = T["s2_tre_nonrigid"] if CONDITIONS[cond]["nonrigid"] else T["s2_tre_rigid"]
        out["S2"][cond] = dict(arm=arm, delta_T2=_item(d.get("T2"), T["s2_t2"], ">="),
                               delta_T1=_item(d.get("T1"), T["s2_t1"], ">="),
                               tre=_item(m.get(f"tre_mean_mm_{basis}"), thr, "<=", basis=basis,
                                         absolute=m.get("tre_mean_mm_absolute")),
                               a0_delta_T2=dc.get("A0", {}).get("T2"))
    les = lesions.get("A2")
    if not les:
        out["S3"] = {"pass": None}
    else:
        own = [x for x in les if x["contrast_recovery"] is not None]
        xt = [x for x in les if x["crosstalk"] is not None]
        out["S3"] = dict(
            recovery=_item(_worst([x["contrast_recovery"] for x in own], min), T["s3_recovery"], ">="),
            centroid=_item(_worst([x["centroid_shift_mm"] for x in own], max), T["s3_centroid_mm"], "<="),
            crosstalk=_item(_worst([abs(x["crosstalk"]) - x["crosstalk_tol"] for x in xt], max), 0.0, "<="))
        parts = [out["S3"][k]["pass"] for k in ("recovery", "centroid", "crosstalk")]
        out["S3"]["pass"] = None if any(p is None for p in parts) else all(parts)
    out["S4"] = {}
    for cond in ("E4a", "E4b", "E4c"):
        arm = "A23" if cond == "E4c" else "A3"
        a0c, a0x, ax = (_val(table, af, "C0", "A0", "T2"), _val(table, af, cond, "A0", "T2"),
                        _val(table, af, cond, arm, "T2"))
        rec, note = None, None
        if None not in (a0c, a0x, ax):
            loss = a0c - a0x
            if loss < T["s4_min_loss_db"]:
                note = f"A0 loses only {loss:.3f} dB against C0; the problem is not significant at this amplitude"
            else:
                rec = float((ax - a0x) / loss)
        m = motion.get(exp_name(af, cond, arm)) or {}
        seg = [m.get("seg_rot_err_deg_max"), m.get("seg_t_err_mm_max")]
        seg_ok = None if None in seg else bool(seg[0] <= T["s4_seg_deg"] and seg[1] <= T["s4_seg_mm"])
        out["S4"][cond] = dict(arm=arm, recovered=_item(rec, T["s4_recovered"], ">=", note=note,
                                                        a0_c0=a0c, a0=a0x, arm_psnr=ax),
                               segments={"rot_err_deg_max": seg[0], "t_err_mm_max": seg[1],
                                         "threshold": [T["s4_seg_deg"], T["s4_seg_mm"]], "pass": seg_ok})
    return out


# ------------------------------------------------------------------------------------------------ CLI
def _c0_reference(af, arm, exps, tag=""):
    """The C0 run whose estimate stands for the native offset in the differential error: same arm, else A2, A1."""
    sfx = f"_{tag}" if tag else ""
    for a in (arm, "A2", "A1"):
        if exp_name(af, "C0", a) + sfx in exps and ARMS[a]["rigid"]:
            return exp_name(af, "C0", a) + sfx
    return None


def build_summary():
    table = [r for r in attach_alt_scores(image_table(load_ledger())) if not r["tag"]]
    exps = {r["exp"] for r in table}
    summary = {}
    for af in sorted(MASKS):
        rows = [r for r in table if r["af"] == af]
        if not rows:
            continue
        delta, motion, lesions = {}, {}, {}
        for r in rows:
            if r["seed"] != 42:
                continue
            if r["cond"] != "C0":
                delta.setdefault(r["cond"], {})[r["arm"]] = normalised_delta(table, af, r["cond"], r["arm"])
            m = motion_errors(r["exp"], r["cond"], _c0_reference(af, r["arm"], exps) if r["cond"] != "C0" else None)
            if m is not None:
                motion[r["exp"]] = m
            if r["cond"] == "E3":
                lesions[r["arm"]] = lesion_metrics(af, r["arm"])
        summary[af] = dict(table=rows, zero=zero_motion_checks(af, table), delta=delta, motion=motion, lesions=lesions)
        summary[af]["criteria"] = criteria(summary, af)
    return summary


def dev_report(tag):
    """One development round: motion errors on D1-D3 and the zero-motion check of the round's C0 runs."""
    table = image_table(load_ledger())
    exps = {r["exp"] for r in table}
    out = dict(tag=tag, zero=zero_motion_checks("af12", table, tag), motion={}, images={})
    for r in table:
        if r["tag"] != tag:
            continue
        out["images"][r["exp"]] = {c: r[c]["head_psnr"] for c in ("T1", "T2") if r[c]}
        if r["cond"].startswith("D"):
            out["motion"][r["exp"]] = motion_errors(r["exp"], r["cond"], _c0_reference("af12", r["arm"], exps, tag))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dev", metavar="TAG", help="development round, e.g. r1")
    args = ap.parse_args()
    assert args.all != bool(args.dev), "choose --all or --dev TAG"
    res, name = (build_summary(), "summary.json") if args.all else (dev_report(args.dev), f"dev_{args.dev}.json")
    out = ROOT / "eval"
    out.mkdir(exist_ok=True)
    tmp = out / f"{name}.tmp{os.getpid()}"
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, out / name)
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if art:
        shutil.copy(out / name, Path(art) / name)
    if args.dev:
        log(json.dumps(res, indent=1))
    else:
        for af, s in res.items():
            log(f"== {af}: {len(s['table'])} runs")
            log(json.dumps(s["criteria"], indent=1))


if __name__ == "__main__":
    main()
