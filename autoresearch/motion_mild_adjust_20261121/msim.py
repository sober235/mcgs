#!/usr/bin/env python3
"""Data of the second version: one condition of mcommon.CONDITIONS2 under one acquisition order.

    msim.py --cond Ta --order il            build into ROOT2/data/il/Ta/ (refuses to overwrite)
    msim.py --cond Ta --order il --check    re-hash the files against the manifest and print the QC record

The simulator is the first version's (simulate.build_custom); this file adds the choice of the acquisition order, the
soft-tissue field that accompanies a movement and the side files of conditions without intra-scan motion (C0, and the
conditions whose data are the first version's files: those are referenced by path and sha256, never copied).
No clock value or duration is printed or written anywhere.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from mcommon import CONDITIONS2, CONTAM_MM, DATA2, ORDER_NAME, ORDERS, SOFT, c1

import geom  # noqa: E402  (first version's library, on sys.path through mcommon)
import lesions  # noqa: E402
import schedule  # noqa: E402
import simulate  # noqa: E402
import warp  # noqa: E402

log = simulate.log


def _static(src, order, opts, brain):
    """Schedule and all-clean truth of a condition without intra-scan motion."""
    support = ~np.all(src["ksp_T2"] == 0, axis=0).all(axis=0)
    sch = schedule.build(support, k_seg=opts["k_seg"], etl=opts["etl"], order=ORDER_NAME[order])
    n = sch["n_trains"]
    truth = dict(order=ORDER_NAME[order], intra=None, n_trains=n, ref_train=sch["ref_train"], etl=opts["etl"],
                 soft_amp=[0.0] * n, train_brain_disp_mm=[0.0] * n, contaminated=[False] * n,
                 train_resid_disp_mm=[0.0] * n, contaminated_in_state=[False] * n, state_of_train=[0] * n,
                 state_poses=[geom.Rigid.identity().to_json()], ref_state=0, contam_mm=CONTAM_MM,
                 shape=list(src["ksp_T2"].shape[1:]))
    return dict(schedule=sch, truth=truth, brain=brain, qc={})


def build(cond, order, src, small=False):
    """One condition under one order. src: arrays ksp_T1, ksp_T2, csm_T1, csm_T2 (files' layout)."""
    assert order in ORDERS, order
    spec = CONDITIONS2[cond]
    opts = simulate.SMALL if small else simulate.FULL
    src = dict(src)
    t1_mag = np.abs(simulate.combine(src["ksp_T1"], src["csm_T1"], opts["fft_workers"]))
    src["brain"] = lesions.brain_mask(t1_mag, erode=opts["brain_erode"], cut_below=opts["brain_cut"])
    if not spec["intra"]:
        assert not (spec["inter"] or spec["nonrigid"] or spec["lesions"]), "static conditions are C0 or first-version data"
        out = _static(src, order, opts, src["brain"])
    else:
        inter = geom.rigid_from_pivot(spec["inter"]["theta_deg"], spec["inter"]["t_sim_mm"]) if spec["inter"] else None
        u, meta = None, {}
        if spec["nonrigid"]:
            nr = spec["nonrigid"]
            u = warp.rbf_field(nr["seed"], nr["peak_mm"], src["brain"], nr["brain_max_mm"], shape=t1_mag.shape,
                               sigma_mm=opts["sigma_mm"])
            meta["nonrigid"] = dict(nr, sigma_mm=list(opts["sigma_mm"]), basis="4 Gaussian RBFs, see warp.rbf_field")
        soft_u = None
        if any(e.get("soft") for e in spec["intra"]):
            peak = min(SOFT["peak_mm"], 1.5) if small else SOFT["peak_mm"]     # the test phantom has no room for 2 mm
            soft_u = warp.rbf_field(SOFT["seed"], peak, src["brain"], SOFT["brain_max_mm"],
                                    shape=t1_mag.shape, sigma_mm=opts["sigma_mm"])
        sites = lesions.pick_sites(t1_mag, src["brain"], min_sep=opts["min_sep"]) if spec["lesions"] else None
        out = simulate.build_custom(src, inter=inter, u=u, intra_events=spec["intra"], lesion_sites=sites,
                                    small=small, meta=meta, order=ORDER_NAME[order], soft_u=soft_u,
                                    contam_mm=CONTAM_MM)
        out["brain"] = src["brain"]
        out["truth"]["soft"] = dict(SOFT, peak_mm=peak, sigma_mm=list(opts["sigma_mm"])) if soft_u is not None else None
        f_ref = float(out["schedule"]["train_time_frac"][out["schedule"]["ref_train"]])
        for e in spec["intra"]:
            assert not (e["kind"] == "pulse" and e["at"] <= f_ref < e["at"] + e["dur"]), \
                f"{cond}/{order}: the reference train lies inside a pulse"
    out["truth"].update(condition=cond, order_code=order, v1_data=spec["v1_data"])
    return out


def _v1_files(name):
    """Paths and sha256 of the first version's files of one condition (read-only reference)."""
    d = c1.DATA / name
    files = {f: d / f for f in ("ksp_T2.h5", "disp_T2.npy", "brain.npy", "motion_truth.json") if (d / f).exists()}
    assert "ksp_T2.h5" in files, f"{d}: the first version's data are missing"
    return {f: dict(path=str(p), sha256=simulate.sha(p)) for f, p in files.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=sorted(CONDITIONS2))
    ap.add_argument("--order", required=True, choices=ORDERS)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out_dir = DATA2 / args.order / args.cond
    if args.check:
        man = json.loads((out_dir / "manifest.json").read_text())
        bad = [f for f, h in man["files"].items() if simulate.sha(out_dir / f) != h]
        bad += [f"v1:{f}" for f, d in (man.get("v1_files") or {}).items() if simulate.sha(d["path"]) != d["sha256"]]
        t = json.loads((out_dir / "motion_truth.json").read_text())
        log(f"[check {args.order}/{args.cond}] files {len(man['files'])} mismatching {bad}")
        log(json.dumps(dict(json.loads((out_dir / "qc.json").read_text()), n_trains=t["n_trains"],
                            ref_train=t["ref_train"], n_contaminated=int(sum(t["contaminated"])),
                            n_states=len(t["state_poses"]), ref_state=t["ref_state"],
                            brain_disp_mm_max=max(t["train_brain_disp_mm"])), indent=1))
        sys.exit(1 if bad else 0)
    assert not out_dir.exists(), f"{out_dir} exists; conditions are never overwritten"
    HYW = c1.HYW
    paths = {"ksp_T1": HYW / "ksp_T1.h5", "ksp_T2": HYW / "ksp_T2.h5",
             "csm_T1": HYW / "csm_espirit_T1.h5", "csm_T2": HYW / "csm_espirit_T2.h5"}
    src = {k: simulate._read(p, ("kspace", "ksp") if k.startswith("ksp") else ("map", "csm")) for k, p in paths.items()}
    log(f"[msim {args.order}/{args.cond}] loaded {src['ksp_T2'].shape}")
    out = build(args.cond, args.order, src)
    tmp = out_dir.parent / f".{args.cond}.tmp{os.getpid()}"
    tmp.mkdir(parents=True)
    truth = out["truth"]
    if "ksp_T2" in out and out["ksp_T2"] is not src["ksp_T2"]:
        simulate._write_ksp(tmp / "ksp_T2.h5", out["ksp_T2"])
    if "ksp_T1" in out:
        simulate._write_ksp(tmp / "ksp_T1.h5", out["ksp_T1"])
    if "gt_ksp_T2" in out:
        if out["gt_ksp_T2"] is src["ksp_T2"]:
            truth["gt_ksp_path"] = str(paths["ksp_T2"])                 # reference pose = the original scan
        else:
            simulate._write_ksp(tmp / "gt_ksp_T2.h5", out["gt_ksp_T2"])
            truth["gt_ksp_path"] = str(out_dir / "gt_ksp_T2.h5")
    sch = out["schedule"]
    np.savez_compressed(tmp / "schedule.npz", train_of=sch["train_of"], seg_of=sch["seg_of"],
                        seg_of_train=sch["seg_of_train"], train_time_frac=sch["train_time_frac"],
                        n_trains=sch["n_trains"], ref_train=sch["ref_train"])
    for key in ("disp_T2", "lesion_mask_canon", "lesion_mask_T2", "brain"):
        if key in out:
            np.save(tmp / f"{key}.npy", out[key])
    (tmp / "motion_truth.json").write_text(json.dumps(truth, indent=1))
    (tmp / "qc.json").write_text(json.dumps(out["qc"], indent=1))
    snap = Path(".mcgs-manifest.json")
    spec = CONDITIONS2[args.cond]
    man = dict(condition=args.cond, order=args.order, files={f.name: simulate.sha(f) for f in sorted(tmp.iterdir())},
               v1_files=_v1_files(spec["v1_data"]) if spec["v1_data"] else None,
               sources={k: dict(path=str(p), sha256=simulate.sha(p)) for k, p in paths.items()},
               source_digest=json.loads(snap.read_text()).get("source_digest") if snap.exists() else None)
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1))
    tmp.rename(out_dir)
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if art:
        for f in ("qc.json", "motion_truth.json", "manifest.json"):
            shutil.copy(out_dir / f, Path(art) / f"{args.order}_{args.cond}_{f}")
    log(f"[msim {args.order}/{args.cond}] wrote {sorted(man['files'])}")
    log(json.dumps(dict(out["qc"], n_trains=truth["n_trains"], ref_train=truth["ref_train"],
                        n_contaminated=int(sum(truth["contaminated"])), n_states=len(truth["state_poses"])), indent=1))


if __name__ == "__main__":
    main()
