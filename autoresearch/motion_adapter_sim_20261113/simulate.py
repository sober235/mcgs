#!/usr/bin/env python3
"""Simulator of the motion study (design section 7): hyw multi-coil k-space + known T2W motion (+ lesions).

Per coil c:  y_c = shutter * F[ x_c + S_c (W(m') - m) ]  =  shutter * ( y_c(original) + F[ S_c (W(m') - m) ] )
with x_c the original coil image, S_c the topic's ESPIRiT map (fixed to the scanner), m = sum_c conj(S_c) x_c,
m' = m with lesions, W the motion warp (warp.py). Zero motion returns the input array itself; the real noise stays.
Intra-scan motion: every echo train takes its phase-encode lines from the simulation at that train's pose.

    simulate.py --cond E2            build one condition into ROOT/data/E2/ (refuses to overwrite)
    simulate.py --cond E2 --check    re-hash the files against the manifest and print the QC record

No clock value or duration is printed or written anywhere.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.fft as sfft

import geom
import lesions
import schedule
import warp
from common import CONDITIONS, DATA, ETL, HYW, K_SEG, NATIVE_SHIFT_MM, PIVOT_OFFSET_MM

FULL = dict(etl=ETL, k_seg=K_SEG, brain_erode=12, brain_cut=35, sigma_mm=(30.0, 50.0), min_sep=25.0, procs=12,
            fft_workers=16)
SMALL = dict(etl=64, k_seg=K_SEG, brain_erode=3, brain_cut=14, sigma_mm=(8.0, 14.0), min_sep=13.0, procs=1,
             fft_workers=1)
AX = (-3, -2, -1)
log = lambda *a: print(*a, flush=True)


def fft3c(x, workers=None):
    """The trainer's centred orthonormal FFT (paired_gs/data/transforms.py fft3c)."""
    return sfft.fftshift(sfft.fftn(sfft.ifftshift(x, axes=AX), axes=AX, norm="ortho", workers=workers), axes=AX)


def ifft3c(x, workers=None):
    return sfft.fftshift(sfft.ifftn(sfft.ifftshift(x, axes=AX), axes=AX, norm="ortho", workers=workers), axes=AX)


def combine(ksp, csm, workers=None):
    """m = sum_c conj(S_c) ifft(y_c), coil by coil (complex128)."""
    m = np.zeros(ksp.shape[1:], dtype=np.complex128)
    for c in range(ksp.shape[0]):
        m += np.conj(csm[c]) * ifft3c(ksp[c], workers)
    return m


def _is_identity(g):
    return bool(np.array_equal(g.R, np.eye(3)) and not np.any(g.t))


_G = {}                                    # read-only state shared with forked workers


def _moved(pose):
    """W_T(m') for T = pose o (id + u)."""
    if _G["u"] is None:
        return _G["m_les"] if _is_identity(pose) else warp.rigid(_G["m_les"], pose)
    return warp.nonrigid_then_rigid(_G["m_les"], _G["u"], pose)


def _delta_lines(pose, sel2d):
    """F[S_c (W_T(m') - m)] on the phase-encode points sel2d, all readout samples: (C, N0, n_selected) complex64."""
    d = _moved(pose) - _G["m"]
    csm, w = _G["csm"], _G["opts"]["fft_workers"]
    out = np.empty((csm.shape[0], csm.shape[1], int(sel2d.sum())), dtype=np.complex64)
    for c in range(csm.shape[0]):
        out[c] = fft3c(csm[c] * d, w)[:, sel2d]
    return out


def _delta_full(d, csm, workers):
    out = np.empty(csm.shape, dtype=np.complex64)
    for c in range(csm.shape[0]):
        out[c] = fft3c(csm[c] * d, workers)
    return out


def _pose_job(args):
    key, pose_json, trains = args
    sel = np.isin(_G["train_of"], trains)
    return key, _delta_lines(geom.Rigid.from_json(pose_json), sel)


def build_custom(src, inter=None, u=None, intra_events=None, lesion_sites=None, small=False, meta=None):
    """Core of the simulator. src: dict of arrays ksp_T1, ksp_T2, csm_T1, csm_T2 in the files' layout (C, N0, N1, N2).
    inter: geom.Rigid or None; u: displacement field (*N, 3) or None; intra_events: list of events or None;
    lesion_sites: output of lesions.pick_sites or None."""
    opts = SMALL if small else FULL
    y2, S2 = src["ksp_T2"], src["csm_T2"]
    shape = y2.shape[1:]
    zero2 = np.all(y2 == 0, axis=0)                           # the vendor shutter: every coil exactly zero
    w = opts["fft_workers"]
    m = combine(y2, S2, w)
    out, qc = {}, {}
    g = inter if inter is not None else geom.Rigid.identity()
    has_inter = inter is not None or u is not None
    m_les, lab = m, None
    if lesion_sites:
        m_les, lab = lesions.insert(m, lesion_sites, "T2")
    _G.update(m=m, m_les=m_les, u=u, csm=S2, opts=opts)

    # ---- reference pose (the inter-scan pose): this is the whole simulation when there is no intra-scan motion
    d_ref = _moved(g) - m
    if not np.any(d_ref):
        ref_ksp = y2
    else:
        ref_ksp = y2 + _delta_full(d_ref, S2, w)
        ref_ksp[:, zero2] = 0
        ss = (np.abs(S2) ** 2).sum(0)
        e = np.abs(d_ref) ** 2
        qc["delta_energy_fraction_outside_csm_support"] = float(e[ss < 0.5].sum() / max(e.sum(), 1e-30))
    truth = dict(convention="T maps a material point r (mm, FOV-centre origin, axes S-I/A-P/L-R) of the canonical "
                            "(T1) frame to its position in the T2W image: r' = R (r + u(r)) + t; image = m o T^-1",
                 inter=g.to_json() if inter is not None else None, pivot_offset_mm=list(PIVOT_OFFSET_MM),
                 nonrigid=(meta or {}).get("nonrigid"), intra=intra_events, native_shift_mm=list(NATIVE_SHIFT_MM),
                 lesion_sites=lesion_sites, shape=list(shape))

    if not intra_events:
        out["ksp_T2"] = ref_ksp
    else:
        support = ~zero2.all(axis=0)
        sch = schedule.build(support, k_seg=opts["k_seg"], etl=opts["etl"])
        rel = schedule.poses(intra_events, sch["train_time_frac"], sch["ref_train"])
        groups = {}
        for n, p in enumerate(rel):
            if _is_identity(p):
                continue                                   # reference pose: lines already in ref_ksp
            key = json.dumps(p.to_json(), sort_keys=True)
            groups.setdefault(key, []).append(n)
        ksp = ref_ksp.copy()
        _G.update(train_of=sch["train_of"])
        jobs = [(k, geom.Rigid.from_json(json.loads(k)).compose(g).to_json(), tr) for k, tr in groups.items()]
        if opts["procs"] > 1 and len(jobs) > 4:
            with multiprocessing.get_context("fork").Pool(opts["procs"]) as pool:
                it = pool.imap_unordered(_pose_job, jobs)
                for i, (key, lines) in enumerate(it):
                    sel = np.isin(sch["train_of"], groups[key])
                    ksp[:, :, sel] = y2[:, :, sel] + lines
                    if (i + 1) % 10 == 0:
                        log(f"  poses done {i + 1}/{len(jobs)}")
        else:
            for job in jobs:
                key, lines = _pose_job(job)
                sel = np.isin(sch["train_of"], groups[key])
                ksp[:, :, sel] = y2[:, :, sel] + lines
        ksp[:, zero2] = 0
        out.update(ksp_T2=ksp, gt_ksp_T2=ref_ksp, schedule=sch)
        seg_mean = schedule.segment_mean_poses(rel, sch["seg_of_train"], opts["k_seg"])
        truth.update(train_poses=[p.to_json() for p in rel], segment_mean_poses=[p.to_json() for p in seg_mean],
                     seg_of_train=[int(x) for x in sch["seg_of_train"]], ref_train=sch["ref_train"],
                     ref_seg=sch["ref_seg"], n_trains=sch["n_trains"], etl=opts["etl"], k_seg=opts["k_seg"],
                     n_distinct_poses=len(groups) + 1)
        sel_ref = sch["train_of"] == sch["ref_train"]
        qc["reference_train_equals_reference_pose"] = bool(np.array_equal(ksp[:, :, sel_ref], ref_ksp[:, :, sel_ref]))

    qc["shutter_zero_set_identical"] = bool(np.array_equal(np.all(out["ksp_T2"] == 0, axis=0), zero2))
    if has_inter:
        disp = warp.displacement(u if u is not None else np.zeros((*shape, 3)), g, shape=shape).astype(np.float32)
        out["disp_T2"] = disp
        if "brain" in src:
            n = np.linalg.norm(disp, axis=-1)[src["brain"]]
            qc["brain_displacement_mean_mm"], qc["brain_displacement_max_mm"] = float(n.mean()), float(n.max())
        if u is None and "brain" in src:
            back = warp.rigid(_moved(g), g.inverse())
            b = src["brain"]
            qc["rigid_roundtrip_rel_rms_brain"] = float(np.sqrt((np.abs(back - m_les)[b] ** 2).mean())
                                                        / np.sqrt((np.abs(m_les)[b] ** 2).mean()))
    if lesion_sites:
        y1, S1 = src["ksp_T1"], src["csm_T1"]
        zero1 = np.all(y1 == 0, axis=0)
        m1 = combine(y1, S1, w)
        m1_les, _ = lesions.insert(m1, lesion_sites, "T1")
        k1 = y1 + _delta_full(m1_les - m1, S1, w)
        k1[:, zero1] = 0
        lab2 = np.zeros(shape, dtype=np.uint8)
        _G.update(m_les=None)
        for k, s in enumerate(lesion_sites, start=1):
            _G["m_les"] = lesions.weight(shape, s).astype(np.complex128)
            lab2[np.real(_moved(g)) > 0.5] = k
        _G["m_les"] = m_les
        out.update(ksp_T1=k1, lesion_mask_canon=lab, lesion_mask_T2=lab2)
    out.update(truth=truth, qc=qc)
    return out


def build(cond, src, small=False):
    """Build one named condition of common.CONDITIONS."""
    spec = CONDITIONS[cond]
    opts = SMALL if small else FULL
    src = dict(src)
    need_brain = spec["inter"] or spec["nonrigid"] or spec["lesions"]
    t1_mag = None
    if need_brain:
        t1_mag = np.abs(combine(src["ksp_T1"], src["csm_T1"], opts["fft_workers"]))
        src["brain"] = lesions.brain_mask(t1_mag, erode=opts["brain_erode"], cut_below=opts["brain_cut"])
    inter = geom.rigid_from_pivot(spec["inter"]["theta_deg"], spec["inter"]["t_sim_mm"]) if spec["inter"] else None
    u, meta = None, {}
    if spec["nonrigid"]:
        nr = spec["nonrigid"]
        u = warp.rbf_field(nr["seed"], nr["peak_mm"], src["brain"], nr["brain_max_mm"], shape=t1_mag.shape,
                           sigma_mm=opts["sigma_mm"])
        meta["nonrigid"] = dict(nr, sigma_mm=list(opts["sigma_mm"]), basis="4 Gaussian RBFs, see warp.rbf_field")
    sites = lesions.pick_sites(t1_mag, src["brain"], min_sep=opts["min_sep"]) if spec["lesions"] else None
    out = build_custom(src, inter=inter, u=u, intra_events=spec["intra"], lesion_sites=sites, small=small, meta=meta)
    out["truth"]["condition"] = cond
    if need_brain:
        out["brain"] = src["brain"]
    return out


# ------------------------------------------------------------------------------------------------ files
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _read(path, keys):
    import h5py
    with h5py.File(path, "r") as h:
        key = next(k for k in keys if k in h)
        a = h[key][:]
    if a.dtype.names:
        a = a["real"] + 1j * a["imag"]
    assert np.iscomplexobj(a) and a.ndim == 4 and a.shape[0] < a.shape[1], f"{path}: unexpected layout {a.shape}"
    return np.ascontiguousarray(a.astype(np.complex64))


def _write_ksp(path, arr):
    import h5py
    with h5py.File(path, "w") as h:
        h.create_dataset("kspace", data=np.ascontiguousarray(arr, dtype=np.complex64))


def _qc_png(path, cond, out, src):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = FULL["fft_workers"]
    a = np.abs(combine(src["ksp_T2"], src["csm_T2"], w))
    b = np.abs(combine(out["ksp_T2"], src["csm_T2"], w))
    c = [n // 2 for n in a.shape]
    vmax = float(np.quantile(a, 0.999))
    fig, axs = plt.subplots(3, 3, figsize=(12, 11))
    for row, (vol, title, kw) in enumerate(((a, "original T2", dict(cmap="gray", vmin=0, vmax=vmax)),
                                            (b, f"{cond} T2", dict(cmap="gray", vmin=0, vmax=vmax)),
                                            (b - a, "difference", dict(cmap="RdBu_r", vmin=-0.2 * vmax, vmax=0.2 * vmax)))):
        for col, sl in enumerate((vol[:, :, c[2]], vol[:, c[1], :], vol[c[0], :, :])):
            axs[row, col].imshow(sl, **kw)
            axs[row, col].set_title(f"{title}, plane through axis {2 - col} centre", fontsize=9)
            axs[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=[c for c in CONDITIONS if c != "C0"])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out_dir = DATA / args.cond
    if args.check:
        man = json.loads((out_dir / "manifest.json").read_text())
        bad = [f for f, h in man["files"].items() if sha(out_dir / f) != h]
        log(f"[check {args.cond}] files {len(man['files'])} mismatching {bad}")
        log(json.dumps(json.loads((out_dir / "qc.json").read_text()), indent=1))
        sys.exit(1 if bad else 0)
    assert not out_dir.exists(), f"{out_dir} exists; conditions are never overwritten"
    paths = {"ksp_T1": HYW / "ksp_T1.h5", "ksp_T2": HYW / "ksp_T2.h5",
             "csm_T1": HYW / "csm_espirit_T1.h5", "csm_T2": HYW / "csm_espirit_T2.h5"}
    src = {k: _read(p, ("kspace", "ksp") if k.startswith("ksp") else ("map", "csm")) for k, p in paths.items()}
    log(f"[simulate {args.cond}] loaded {src['ksp_T2'].shape}")
    out = build(args.cond, src)
    tmp = DATA / f".{args.cond}.tmp{os.getpid()}"
    tmp.mkdir(parents=True)
    truth = out["truth"]
    if out["ksp_T2"] is not src["ksp_T2"]:
        _write_ksp(tmp / "ksp_T2.h5", out["ksp_T2"])
    if "ksp_T1" in out:
        _write_ksp(tmp / "ksp_T1.h5", out["ksp_T1"])
    if "gt_ksp_T2" in out:
        if out["gt_ksp_T2"] is src["ksp_T2"]:
            truth["gt_ksp_path"] = str(paths["ksp_T2"])                 # reference pose = the original scan
        else:
            _write_ksp(tmp / "gt_ksp_T2.h5", out["gt_ksp_T2"])
            truth["gt_ksp_path"] = str(out_dir / "gt_ksp_T2.h5")
        sch = out["schedule"]
        np.savez_compressed(tmp / "schedule.npz", train_of=sch["train_of"], seg_of=sch["seg_of"],
                            seg_of_train=sch["seg_of_train"], train_time_frac=sch["train_time_frac"])
    for key in ("disp_T2", "lesion_mask_canon", "lesion_mask_T2", "brain"):
        if key in out:
            np.save(tmp / f"{key}.npy", out[key])
    (tmp / "motion_truth.json").write_text(json.dumps(truth, indent=1))
    (tmp / "qc.json").write_text(json.dumps(out["qc"], indent=1))
    _qc_png(tmp / f"qc_{args.cond}.png", args.cond, out, src)
    snap = Path(".mcgs-manifest.json")
    man = dict(condition=args.cond, files={f.name: sha(f) for f in sorted(tmp.iterdir())},
               sources={k: dict(path=str(p), sha256=sha(p)) for k, p in paths.items()},
               source_digest=json.loads(snap.read_text()).get("source_digest") if snap.exists() else None)
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1))
    tmp.rename(out_dir)
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if art:
        for f in (f"qc_{args.cond}.png", "qc.json", "motion_truth.json", "manifest.json"):
            shutil.copy(out_dir / f, Path(art) / (f if f.startswith("qc_") else f"{args.cond}_{f}"))
    log(f"[simulate {args.cond}] wrote {sorted(man['files'])}")
    log(json.dumps(out["qc"], indent=1))


if __name__ == "__main__":
    main()
