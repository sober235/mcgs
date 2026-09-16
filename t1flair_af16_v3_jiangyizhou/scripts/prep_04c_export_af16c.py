#!/usr/bin/env python3
"""prep_04c_export_af16c.py -- phase-3 acquisition probe, step 2: the FLAIR probe mask "af16c" from the draw pool.

Spec: mcgs_af16_wt/docs/superpowers/specs/2026-09-12-af16-phase3-nonlocal-design.md, section 5 (Track B, step 1).
Pool: data/probe_c_draws/draw_NN.mat from scripts/prep_02c_mask_af16c.m (vdPoisMex, accely 4, calib 24, ellipse 0,
pp 1, accelx swept 3.0-3.9; every draw carries its accelx). vdPoisMex seeds from time(0), so identical draws are
possible in principle and are dropped here.

Steps
  1. FLAIR's acquired support on the (ky,kz) plane exactly as prep_04_export.py computes it: readout-line energy of
     the CFL source, every 8th readout line, max over coils > 0 (expected 34139 of 45056 = 0.7577). Cross-checked
     against coil 0 of the h5 over every readout line (the fig_mask_support.py convention).
  2. Per draw, the points inside the support. Pick the draw whose in-support count is closest to TARGET = 2966 from
     above (ties: the lowest draw index).
  3. Drop samples from the outermost radius (Euclidean distance from the k-space centre (128, 88) in index units,
     random tie-break with a fixed seed) until exactly 2966 remain inside the support; everything outside the
     support stays 0.
  4. Write data/mask_vdpois_256x176_af16c_flair.mat (key mask, float32 (256,176)), broadcast it along the readout
     into data/mask_af16c_flair_3d.h5 (dataset mask, uint8 (256,256,176), gzip 1 -- the layout of
     data/mask_af16_3d.h5), the statistics into logs/prep_04c_export_af16c.json, and the figure
     figures/probe_c_mask_T1_vs_FLAIRc.png (T1 = the current mask, the new FLAIR mask, their union), mirrored to
     /home/congcongliu/figs/t1flair_v3_af16/.
T1 keeps data/mask_af16_3d.h5. Nothing registered in this topic is read for writing or touched.
--fig_only re-renders the figure from the written mask and json without touching the mask files.
"""
import glob, hashlib, json, os, sys
import h5py, numpy as np, scipy.io
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt

SRC = "/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL/vol3_jiangyizhou/flair_train_jiangyizhou_cc"
R = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou"
FIGS = "/home/congcongliu/figs/t1flair_v3_af16"
TARGET = 2966                      # the current mask's budget: the probe changes where FLAIR samples, not how much
CENTRE = (128, 88)                 # k-space DC on (ky, kz), prep_01_inspect.py
CAL = (slice(128 - 12, 128 + 12), slice(88 - 12, 88 + 12))   # the 24x24 calibration block
SEED = 0
CUR_MAT = f"{R}/data/mask_vdpois_256x176_af16.mat"
CUR_H5 = f"{R}/data/mask_af16_3d.h5"
OUT_MAT = f"{R}/data/mask_vdpois_256x176_af16c_flair.mat"
OUT_H5 = f"{R}/data/mask_af16c_flair_3d.h5"
OUT_JSON = f"{R}/logs/prep_04c_export_af16c.json"
OUT_FIG = "probe_c_mask_T1_vs_FLAIRc.png"


def cfl(base):                     # verbatim from prep_04_export.py
    with open(base + ".hdr") as f:
        dims = [int(x) for x in f.read().split("\n")[1].split()]
    n = int(np.prod(dims))
    return np.memmap(base + ".cfl", dtype=np.complex64, mode="r", shape=(n,)).reshape(dims, order="F")[..., 0]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    # 1. FLAIR support, exactly as prep_04_export.py
    k = cfl(SRC)
    nx, ny, nz, nc = k.shape
    sup = np.zeros((ny, nz), dtype=bool)
    for x in range(0, nx, 8):
        sup |= (np.abs(np.asarray(k[x])).max(axis=-1) > 0)
    del k
    with h5py.File(f"{R}/data/ksp_FLAIR.h5", "r") as f:          # symlink into the AF12 topic, read only
        sup_h5 = np.any(np.abs(np.array(f["kspace"][0])) > 0, axis=0)
    n_sup = int(sup.sum())
    print(f"[support] FLAIR readout-line energy: {n_sup} of {sup.size} = {sup.mean():.4f}; "
          f"h5 coil-0 all lines: {int(sup_h5.sum())}; identical={bool((sup == sup_h5).all())}", flush=True)
    assert n_sup == 34139, n_sup

    cur = scipy.io.loadmat(CUR_MAT)["mask"].astype(bool)
    with h5py.File(CUR_H5, "r") as f:
        cur3 = f["mask"][:]
    assert cur.shape == (ny, nz) and (cur3 == cur3[0:1]).all() and (cur3[0].astype(bool) == cur).all()
    calib = np.zeros((ny, nz), dtype=bool); calib[CAL] = True
    outside = ~calib

    # 2. the pool
    pool, seen = [], {}
    for p in sorted(glob.glob(f"{R}/data/probe_c_draws/draw_*.mat")):
        d = scipy.io.loadmat(p)
        m = d["mask"].astype(bool)
        assert m.shape == (ny, nz), (p, m.shape)
        key = hashlib.md5(np.packbits(m).tobytes()).hexdigest()
        rec = dict(draw=os.path.basename(p), accelx=round(float(d["accelx"].squeeze()), 2),
                   accely=int(d["accely"].squeeze()), calib=int(d["calib"].squeeze()),
                   nnz=int(m.sum()), in_support=int((m & sup).sum()), duplicate_of=seen.get(key))
        seen.setdefault(key, rec["draw"])
        pool.append((rec, m))
    cands = [(r, m) for r, m in pool if r["duplicate_of"] is None and r["in_support"] >= TARGET]
    assert cands, "no draw has >= TARGET points inside the support; extend the accelx sweep downward"
    rec, m = min(cands, key=lambda rm: (rm[0]["in_support"], rm[0]["draw"]))
    print(f"[pick] {rec['draw']} accelx {rec['accelx']} nnz {rec['nnz']} in_support {rec['in_support']} "
          f"(excess {rec['in_support'] - TARGET})", flush=True)

    # 3. trim from the outermost radius
    m_in = m & sup
    ky, kz = np.nonzero(m_in)
    r = np.hypot(ky - CENTRE[0], kz - CENTRE[1])
    n_excess = int(m_in.sum()) - TARGET
    rng = np.random.default_rng(SEED)
    order = np.lexsort((rng.random(r.size), -r))            # largest radius first, random among ties
    drop = order[:n_excess]
    new = m_in.copy()
    new[ky[drop], kz[drop]] = False
    assert int(new.sum()) == TARGET and not (new & ~sup).any()
    r_keep = np.hypot(*(np.nonzero(new)[i] - CENTRE[i] for i in (0, 1)))
    trim = dict(n_trimmed=n_excess, seed=SEED,
                trimmed_radius_min=float(r[drop].min()) if n_excess else None,
                trimmed_radius_max=float(r[drop].max()) if n_excess else None,
                kept_radius_max=float(r_keep.max()), kept_radius_median=float(np.median(r_keep)),
                current_mask_radius_max=float(np.hypot(*(np.nonzero(cur)[i] - CENTRE[i] for i in (0, 1))).max()))

    # 4. outputs
    m2 = new.astype(np.float32)
    scipy.io.savemat(OUT_MAT, {"mask": m2}, format="5")
    mask3d = np.broadcast_to(m2[None, :, :], (256, 256, 176)).astype(np.float32)
    with h5py.File(OUT_H5, "w") as h:
        h.create_dataset("mask", data=mask3d.astype(np.uint8), compression="gzip", compression_opts=1)

    n_new, n_cur = int(new.sum()), int(cur.sum())
    union = new | cur
    n_out_plane = int(outside.sum())
    stats = {
        "target_samples": TARGET, "support_points": n_sup, "plane_points": int(sup.size),
        "support_frac": float(sup.mean()), "support_h5_coil0_points": int(sup_h5.sum()),
        "samples": n_new, "in_support_samples": int((new & sup).sum()),
        "overlap_with_af16": int((new & cur).sum()), "overlap_with_af16_in_support": int((new & cur & sup).sum()),
        "af16_samples": n_cur, "af16_in_support": int((cur & sup).sum()),
        "calib24_samples": int(new[CAL].sum()), "calib24_samples_af16": int(cur[CAL].sum()), "calib24_points": 576,
        "sampling_frac_of_plane": n_new / sup.size, "AF_vs_full_grid": sup.size / n_new,
        "AF_vs_flair_support": n_sup / n_new, "AF_vs_flair_support_af16": n_sup / int((cur & sup).sum()),
        "outside_calib": {
            "plane_points": n_out_plane, "support_points": int((sup & outside).sum()),
            "af16_samples": int((cur & outside).sum()), "af16c_samples": int((new & outside).sum()),
            "union_samples": int((union & outside).sum()), "overlap_samples": int((new & cur & outside).sum()),
            "union_coverage_of_plane": int((union & outside).sum()) / n_out_plane,
            "af16_coverage_of_plane": int((cur & outside).sum()) / n_out_plane,
            "union_coverage_of_flair_support": int((union & outside & sup).sum()) / int((sup & outside).sum()),
        },
        "chosen_draw": rec, "trim": trim,
        "pool": [r_ for r_, _ in pool], "pool_size": len(pool),
        "pool_unique": sum(1 for r_, _ in pool if r_["duplicate_of"] is None),
        "generator": "vdPoisMex(256,176,256,176, accelx, 4, 24, 0, 1), scripts/prep_02c_mask_af16c.m",
        "mask2d_path": OUT_MAT, "mask3d_path": OUT_H5, "current_mask3d_path": CUR_H5,
        "md5": {"mask2d": md5(OUT_MAT), "mask3d": md5(OUT_H5), "current_mask3d": md5(CUR_H5)},
    }
    json.dump(stats, open(OUT_JSON, "w"), indent=2)

    figure(new, cur, sup, stats)
    print(json.dumps({k: v for k, v in stats.items() if k != "pool"}, indent=2))
    print(f"wrote {OUT_MAT}\n      {OUT_H5}\n      {OUT_JSON}\n      {R}/figures/{OUT_FIG} (+ {FIGS})")


def figure(new, cur, sup, stats):
    """Three panels: T1 on the current mask, the new FLAIR mask, their union; saved to figures/ and the served dir."""
    ny, nz = new.shape
    rec, oc, n_new, n_cur, n_sup = stats["chosen_draw"], stats["outside_calib"], int(new.sum()), int(cur.sum()), int(sup.sum())
    union = new | cur
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7))
    axes[0].imshow(cur, cmap="gray", interpolation="nearest")
    axes[0].contour(sup, levels=[0.5], colors="r", linewidths=1.2)
    axes[0].set_title(f"T1: current AF16 mask (mask_af16_3d.h5)\n{n_cur} pts, {int((cur & ~sup).sum())} outside the "
                      f"FLAIR shutter (red edge)\nAF {sup.size / n_cur:.2f} on the full plane", fontsize=9.5)
    rgb = np.zeros((ny, nz, 3)); rgb[sup] = 0.18; rgb[new] = 1.0
    axes[1].imshow(rgb, interpolation="nearest")
    axes[1].set_title(f"FLAIR: probe mask af16c (mask_af16c_flair_3d.h5)\n{n_new} pts, all in the shutter; "
                      f"{rec['draw'].replace('.mat', '')}, accelx {rec['accelx']}, {stats['trim']['n_trimmed']} trimmed\n"
                      f"AF {n_sup / n_new:.2f} on its data ({stats['AF_vs_flair_support_af16']:.2f} today), "
                      f"{sup.size / n_new:.2f} on the plane", fontsize=9.5)
    rgb = np.zeros((ny, nz, 3)); rgb[sup] = 0.18
    rgb[cur & ~new] = (0.25, 0.55, 1.0); rgb[new & ~cur] = (1.0, 0.6, 0.1); rgb[new & cur] = 1.0
    axes[2].imshow(rgb, interpolation="nearest")
    axes[2].set_title(f"Union: {int(union.sum())} pts, {int((new & cur).sum())} shared (white)\nT1-only blue, "
                      f"FLAIR-only orange\noutside the 24x24 calib block: {100 * oc['union_coverage_of_plane']:.1f}% "
                      f"of the plane (one mask {100 * oc['af16_coverage_of_plane']:.1f}%)", fontsize=9.5)
    for a in axes:
        a.set_xlabel("kz (176)"); a.set_ylabel("ky (256)"); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("jiangyizhou AF16, phase-3 acquisition probe: T1 on the current mask, FLAIR on an in-shutter draw",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(w_pad=2.5)
    os.makedirs(FIGS, exist_ok=True)
    for d in (f"{R}/figures", FIGS):
        fig.savefig(os.path.join(d, OUT_FIG), dpi=600)
        fig.savefig(os.path.join(d, OUT_FIG.replace(".png", ".pdf")))


def fig_only():
    new = scipy.io.loadmat(OUT_MAT)["mask"].astype(bool)
    cur = scipy.io.loadmat(CUR_MAT)["mask"].astype(bool)
    with h5py.File(f"{R}/data/ksp_FLAIR.h5", "r") as f:
        sup = np.any(np.abs(np.array(f["kspace"][0])) > 0, axis=0)      # identical to the CFL support (json)
    figure(new, cur, sup, json.load(open(OUT_JSON)))
    print(f"re-rendered {R}/figures/{OUT_FIG} (+ {FIGS})")


if __name__ == "__main__":
    fig_only() if "--fig_only" in sys.argv[1:] else main()
