#!/usr/bin/env python3
"""Numbers of finished runs of the third version (amax, CPU; reads only).

    neval.py --runs DNc:B0 DNc:N4:d1 DNc:On [--order il]

Per run: head PSNR of T1W and T2W (the scorer's all-lines image and, for a run with weights or states, the image
whose hard data consistency inserts only the kept lines of the reference state), the recovered share of the T2W loss
against the unchanged model on the same condition, the states found, and the error of every state's transform
against the truth inside the brain:  | pose_k(inter(r + u_k(r))) - inter(r + a_k u_true(r)) |,  a_k = the median
true amplitude of the state's trains. "rigid only" is the same error without the estimated field, "nothing" the
error of leaving the state where the reference is (= the size of the truth). No clock value is read or printed.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from ncommon import DATA3, ROOT3

import mrun  # noqa: E402  (second version, on this study's root)
import geom  # noqa: E402
import nadapter  # noqa: E402
import simulate  # noqa: E402
import warp  # noqa: E402

log = lambda *a: print(*a, flush=True)
C0_B0 = dict(T1=35.834, T2=36.306)                           # second version, il C0 / B0: the level without motion


def rigid(d):
    return geom.Rigid(geom.rotmat(np.deg2rad(np.asarray(d["theta_deg"], dtype=np.float64))),
                      np.asarray(d["t_mm"], dtype=np.float64))


def state_errors(final, state_of_train, truth, u_true, brain):
    """Per state k >= 1 of the run: errors (mm, brain mean / max) of the full estimate, of the rigid pose alone and
    of no correction, with the truth amplitude of the state."""
    shape = brain.shape
    r = geom.grid_mm(shape)[brain]
    inter = rigid(final) if final.get("theta_deg") is not None else geom.Rigid.identity()
    amp = np.asarray(truth.get("soft_amp") or [0.0] * len(state_of_train))
    st = np.asarray(state_of_train)
    out = {}
    for k in sorted(set(st.tolist()) - {0}):
        a = float(np.median(amp[st == k]) - np.median(amp[st == 0]))
        want = inter.apply(r + a * u_true[brain])
        pose = rigid(final["segments"][k])
        ctrl = (final.get("field") or {}).get("ctrl_mm")
        u = nadapter.dense_field(ctrl[k], shape)[brain] if ctrl is not None else np.zeros_like(r)
        e = lambda got: np.linalg.norm(got - want, axis=-1)
        full, rig, none = e(pose.apply(inter.apply(r + u))), e(pose.apply(inter.apply(r))), e(inter.apply(r))
        out[k] = dict(true_amp=a, n_trains=int((st == k).sum()),
                      full=[float(full.mean()), float(full.max())], rigid_only=[float(rig.mean()), float(rig.max())],
                      nothing=[float(none.mean()), float(none.max())],
                      field_peak_mm=float(np.linalg.norm(u, axis=-1).max()) if ctrl is not None else None,
                      pose=final["segments"][k])
    return out


def one(order, cond, arm, tag=""):
    exp = mrun.exp_name("af12", order, cond, arm, tag=tag)
    res = mrun.RES2
    sf = res / f"{exp}_score.json"
    if not sf.exists():
        return dict(exp=exp, done=False)
    sc = json.loads(sf.read_text())
    mm = json.loads((res / f"{exp}_mm.json").read_text())
    out = dict(exp=exp, done=True, T1=sc["plain"]["head_psnr"], T2_all=sc["contrast"]["head_psnr"],
               n_down=int(sum(1 for w in mm["m2"]["w"] if w is not None and w < 1.0)))
    with np.load(res / f"{exp}_canon.npz") as z:
        has_dc = "rec_w_dc" in z.files
    out["T2"] = mrun.head_scores(exp, "rec_w_dc")["head_psnr"] if has_dc else out["T2_all"]
    m3 = mm.get("m3")
    if m3:
        out.update(n_states=m3["n"], boundaries=m3["boundaries"], detected_at=m3["detected_at"])
        mf = res / f"{exp}_motion.json"
        cdir = DATA3 / order / cond
        truth = json.loads((cdir / "motion_truth.json").read_text())
        if m3["n"] > 1 and mf.exists() and truth.get("field"):
            brain = np.load(cdir / "brain.npy").astype(bool)
            f = truth["field"]
            u = warp.rbf_field(f["seed"], f["peak_mm"], brain, f["brain_max_mm"], shape=brain.shape,
                               sigma_mm=simulate.FULL["sigma_mm"])
            out["states"] = state_errors(json.loads(mf.read_text())["final"], m3["state_of_train"], truth, u, brain)
        out["true_boundaries"] = [int(i) + 1 for i in np.flatnonzero(np.diff(truth.get("nr_state_of_train", [0])))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, metavar="COND:ARM[:TAG]")
    ap.add_argument("--order", default="il")
    ap.add_argument("--name", default="neval")
    args = ap.parse_args()
    rows = [one(args.order, *r.split(":")) for r in args.runs]
    base = {r["exp"].split("_")[3]: r for r in rows if r["done"] and r["exp"].split("_")[4] == "B0"}
    for r in rows:
        if not r["done"]:
            log(f"{r['exp']}: not finished")
            continue
        cond = r["exp"].split("_")[3]
        b = base.get(cond)
        if b is not None and C0_B0["T2"] - b["T2"] >= 0.1:
            r["recovered"] = (r["T2"] - b["T2"]) / (C0_B0["T2"] - b["T2"])
            r["dT1"] = r["T1"] - b["T1"]
        rec = f", recovered {r['recovered']:.3f}, dT1 {r['dT1']:+.3f}" if "recovered" in r else ""
        log(f"{r['exp']}: T1 {r['T1']:.3f}, T2 {r['T2']:.3f} (all lines {r['T2_all']:.3f}){rec}; down-weighted trains "
            f"{r['n_down']}" + (f"; states {r['n_states']}, boundaries {r['boundaries']} (truth {r['true_boundaries']})"
                                if "n_states" in r else ""))
        for k, s in (r.get("states") or {}).items():
            log(f"  state {k} ({s['n_trains']} trains, true amplitude {s['true_amp']:+.2f}): error in the brain mean / max "
                f"{s['full'][0]:.3f} / {s['full'][1]:.3f} mm; rigid pose alone {s['rigid_only'][0]:.3f} / "
                f"{s['rigid_only'][1]:.3f}; no correction {s['nothing'][0]:.3f} / {s['nothing'][1]:.3f}; "
                f"largest field in the brain {s['field_peak_mm']}")
    (ROOT3 / "eval").mkdir(parents=True, exist_ok=True)
    out = ROOT3 / "eval" / f"{args.name}.json"
    out.write_text(json.dumps(dict(no_motion=C0_B0, runs=rows), indent=1))
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if art:
        (Path(art) / out.name).write_text(out.read_text())
    log(f"wrote {out.name}")


if __name__ == "__main__":
    main()
