"""Shared helpers for the phase-1 drivers (spec 5.5-5.7): spoke sets, BART inputs, reconstruction,
scoring and figures. prepare_inputs() writes every input before any job starts, so parallel jobs never
write the same file."""
from __future__ import annotations

import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import DERIVED_ROOT, FIGS, GRID, KEEP, RECON_GRID, TOPIC  # noqa: E402
from cardiac_fr import bart  # noqa: E402
from cardiac_fr.gauge import image_metrics, kspace_nrmse, norm_scale  # noqa: E402

TRAJ_R = DERIVED_ROOT / "common" / "traj_r.npy"
LOGS = TOPIC / "logs"
LOOK_SETS = ("r0", "low", "high")   # GRID (112^3, 220 mm), BART's own scaling: anatomy, orientation, polarity


def grid_of(name: str) -> int:
    """Spec 5.6/5.7 correction 5 (2026-09-11): REF, the held-out spokes and the gauge subsets are reconstructed on
    RECON_GRID (2x FOV, same voxels), so the signal outside the 220 mm FOV is modelled instead of folded in; the
    stored .npy and every metric use the central GRID^3 (the protocol FOV)."""
    return GRID if name in LOOK_SETS else RECON_GRID


def sens_of(subject: str, name: str) -> Path:
    """ESPIRiT maps for the set's grid: sens (112^3, s05) or sens2x (224^3, s05b)."""
    return DERIVED_ROOT / subject / ("sens" if grid_of(name) == GRID else "sens2x")


def work_dir(subject: str) -> Path:
    d = DERIVED_ROOT / subject / "bart"
    d.mkdir(parents=True, exist_ok=True)
    return d


def recon_dir(subject: str) -> Path:
    d = TOPIC / "recon" / subject
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_ids(subject: str, name: str) -> np.ndarray:
    d = DERIVED_ROOT / subject
    if name in ("r0", "low", "high"):
        return np.load(d / "spokes.npz")[f"{name}_ids"]
    if name == "ref":
        return np.load(d / "ref_ids.npy")
    return np.load(d / "gauge_split.npz")[name]


def prepare_inputs(subject: str, name: str, echoes) -> None:
    ids = set_ids(subject, name)
    k = np.load(TRAJ_R, mmap_mode="r")[:, KEEP][:, :, ids] * grid_of(name)
    bart.write_cfl(work_dir(subject) / f"traj_{name}", bart.bart_traj(np.asarray(k)))
    with h5py.File(DERIVED_ROOT / subject / "raw.h5", "r") as h:
        for e in echoes:
            y = h["ksp"][e][ids][:, :, KEEP]
            bart.write_cfl(work_dir(subject) / f"ksp_{name}_e{e}", bart.bart_ksp(y))


def recon(subject: str, name: str, echo: int, reg: str, lam: float, iters: int, tag: str,
          w: float | None = None, debug: bool = False, lam_rel: float | None = None) -> dict:
    wd = work_dir(subject)
    out = wd / f"rec_{tag}_e{echo}"
    info = bart.pics(wd / f"ksp_{name}_e{echo}", sens_of(subject, name), out, wd / f"traj_{name}",
                     reg=reg, lam=lam, iters=iters, w=w, debug=debug,
                     log=LOGS / f"bart_{subject}_{tag}_e{echo}.log")
    x = bart.read_cfl(out)
    if grid_of(name) != GRID:
        x = bart.crop_center(x, GRID)                     # the protocol FOV; the full volume stays in rec_<tag> (cfl)
    np.save(recon_dir(subject) / f"{tag}_e{echo}.npy", x.astype(np.complex64))
    info.update(subject=subject, set=name, echo=echo, reg=reg, lam=lam, lam_rel=lam_rel, iters=iters, w=w, tag=tag,
                n_spokes=int(set_ids(subject, name).size), grid=grid_of(name))
    with open(LOGS / f"recon_{subject}.jsonl", "a") as fh:
        fh.write(json.dumps(info) + "\n")
    return info


def ref_lmax(subject: str) -> float:
    """Largest eigenvalue of A^H A for the subject's REF spoke set (echo-independent), cached in
    recon/<subject>/ref_lmax.json. Spec 5.6 (revised 2026-09-11): lambda = lambda_rel x this value."""
    p = recon_dir(subject) / "ref_lmax.json"
    if p.exists():
        d = json.loads(p.read_text())
        if d.get("grid", GRID) == grid_of("ref"):         # the value depends on the grid and the maps
            return float(d["lmax"])
    wd = work_dir(subject)
    if not (wd / "ksp_ref_e0.cfl").exists():
        prepare_inputs(subject, "ref", [0])
    lmax = bart.max_eigenvalue(wd / "ksp_ref_e0", sens_of(subject, "ref"), wd / "traj_ref", wd,
                               log=LOGS / f"bart_{subject}_maxeig.log")
    p.write_text(json.dumps({"lmax": lmax, "grid": grid_of("ref"), "set": "ref",
                             "n_spokes": int(set_ids(subject, "ref").size)}, indent=1))
    return lmax


def ref_scaling(subject: str, echo: int) -> float:
    """BART's data scaling for the REF spokes of this echo (-d5 'Inverse scaling of the data'), shared as -w by every
    subset of the echo (spec 5.6). The REF run (s06 --debug) caches all echoes in recon/<subject>/ref_scaling.json;
    before it, a one-step probe measures it (the value depends on the data and the operator, not on the iterations
    or on lambda)."""
    p = recon_dir(subject) / "ref_scaling.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    if str(echo) not in d:
        wd = work_dir(subject)
        if not (wd / f"ksp_ref_e{echo}.cfl").exists():
            prepare_inputs(subject, "ref", [echo])
        info = bart.pics(wd / f"ksp_ref_e{echo}", sens_of(subject, "ref"), wd / "scaling_probe", wd / "traj_ref",
                         reg="l2", lam=1e-3, iters=1, debug=True, log=LOGS / f"bart_{subject}_scaling_probe.log")
        d[str(echo)] = info["scaling"]
        p.write_text(json.dumps(d, indent=1))
    return float(d[str(echo)])


def run_many(jobs: list, n_parallel: int = 1) -> list:
    with ThreadPoolExecutor(max_workers=n_parallel) as ex:
        return list(ex.map(lambda j: recon(**j), jobs))


def heldout_nrmse(subject: str, tag: str, echo: int) -> float:
    """||A_ho x - y_ho|| / ||y_ho|| with A_ho = coil maps x BART NUFFT on the held-out spokes
    (prepare_inputs(subject, "heldout", ...) must have run first)."""
    wd = work_dir(subject)
    pred = wd / f"pred_{tag}_e{echo}"
    bart.forward(wd / "traj_heldout", wd / f"rec_{tag}_e{echo}", sens_of(subject, "heldout"), pred, wd)
    return kspace_nrmse(bart.read_cfl(pred), bart.read_cfl(wd / f"ksp_heldout_e{echo}"))


def score(subject: str, tag: str, echo: int, heart_box=None) -> dict:
    ref = np.abs(np.load(recon_dir(subject) / f"ref_e{echo}.npy"))
    sc = norm_scale(ref)
    rec = np.abs(np.load(recon_dir(subject) / f"{tag}_e{echo}.npy"))
    m = image_metrics(rec / sc, ref / sc, heart_box=heart_box)
    m["heldout_nrmse"] = heldout_nrmse(subject, tag, echo)
    return m


def save_fig(fig, name: str) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    (TOPIC / "figures").mkdir(exist_ok=True)
    p = FIGS / name
    fig.savefig(p, dpi=110)
    shutil.copy2(p, TOPIC / "figures" / name)
    print(f"http://localhost:8765/cardiac_fr/{name}\n{p}", flush=True)
    return p


def three_plane(vols: dict, name: str, center=None, title: str = "", box=None) -> Path:
    """One row per volume, three orthogonal planes through `center`, grey window [0, P99.5] per row;
    optional box ((x0, x1), (y0, y1), (z0, z1)) drawn in yellow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    first = np.abs(next(iter(vols.values())))
    c = center or tuple(n // 2 for n in first.shape)
    fig, ax = plt.subplots(len(vols), 3, figsize=(9, 3 * len(vols)), squeeze=False)
    for i, (label, v) in enumerate(vols.items()):
        v = np.abs(v)
        vmax = float(np.quantile(v, 0.995))
        for j, (sl, (a, b)) in enumerate(((v[c[0]], (1, 2)), (v[:, c[1]], (0, 2)), (v[:, :, c[2]], (0, 1)))):
            ax[i, j].imshow(sl.T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if box is not None:
                ax[i, j].add_patch(Rectangle((box[a][0], box[b][0]), box[a][1] - box[a][0],
                                             box[b][1] - box[b][0], fill=False, ec="yellow", lw=1))
        ax[i, 0].set_ylabel(label, fontsize=8)
    for j, t in enumerate((f"axis0={c[0]}  (x: axis1, y: axis2)", f"axis1={c[1]}  (x: axis0, y: axis2)",
                           f"axis2={c[2]}  (x: axis0, y: axis1)")):
        ax[0, j].set_title(t, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    p = save_fig(fig, name)
    plt.close(fig)
    return p
