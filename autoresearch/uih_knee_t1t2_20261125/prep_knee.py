#!/usr/bin/env python3
"""prep_knee.py -- build an MC-GS topic from one subject of the UIH 3D knee set: T1 + T2, retrospective AF12.

User 2026-11-25: "executing using uih dataset covering t1 and t2". Modelled on the brain T1+T2 topic
(autoresearch/t1t2_aligned_subject_20261113/prep_tangbing.py); what differs is stated here.

  layout    files hold kspace as (n0, coils, n2, n3) with the readout along n2. Written as (C, X, Y, Z) with
            X = readout (n2), Y = n0, Z = n3, so readout_axis 0 / phase_axes [1, 2] as in the brain topics.
  support   the positions the scanner did not acquire are not zero in these exports: they hold low values, two
            to four decades below the median line energy, with no clean gap (audit 2026-11-25). The acquired
            support on the PE plane is taken by the audit's rule (outer-readout energy of the line at least 1 %
            of the median), restricted to the region connected to the DC, holes filled; every other line is
            set to exactly zero and the removed energy share is recorded. T1: elliptical corners. T2: corners plus the one-sided partial-Fourier band (and
            the few leakage lines beyond it, which are not connected to the DC region). The support maps are
            written for the trainer's `known_zero_support`, so those lines are treated as measured zeros.
            Consequence: the T2 reference is the partial-Fourier zero-filled image, not a fully sampled one.
  shift     T2 is moved onto T1 by the integer voxel shift measured here (NMI / NGF search of the brain
            screening), applied as an exact linear phase on the k-space before anything else; the residual
            must be <= 0.3 voxel on every axis by both metrics.
  mask      vd-Poisson, the house generator (BART poisson, seed 0, 24 x 24 ACS, no shutter; functions imported
            from the September topic's make_mask_af16.py, plane and target replaced): round(NY*NZ/12) points
            on the whole NY x NZ plane, the same mask for both contrasts.
  maps      house ESPIRiT recipe on the prepared k-space (the files' own s_maps are not used: unchecked).
  config    the tangbing T1+T2 config with the data block replaced; known_zero on both contrasts.

    gpu.py run --cpu -- @python autoresearch/uih_knee_t1t2_20261125/prep_knee.py SUBJECT_NUMBER

Subjects are numbered S01..S16 in sorted path order; folder names (personal names) are never printed or stored.
Everything is computed and checked in memory first; the topic directory is created last and never overwritten.
"""
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import scipy.fft as sfft
from scipy import ndimage as nd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "t1t2_aligned_subject_20261113"))
from screen_t1t2 import offset  # noqa: E402

SRC_ROOT = "/data2/congcong/data/3D/3D_data20261112/kspace_csm/knee"
RES = Path("/data0/congcong/code/GS/Results_20260908")
TB_CFG = RES / "T1T2_AF12_v4_tangbing/configs/t1t2_af12_tangbing.yaml"
SEPT = RES / "CE_retro_cssense_af16"
BART = "/home/congcongliu/src/bart"
ECALIB = ["ecalib", "-t", "0.001", "-c", "0.0", "-r", "24", "-m", "1"]
AF = 12
SI = int(sys.argv[1])
SID = "S%02d" % SI
T = RES / f"KNEE_UIH_T1T2_AF{AF}_{SID}"
ART = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
log = lambda *a: print(*a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_cfl(base):
    dims = [int(x) for x in open(base + ".hdr").read().split("\n")[1].split()]
    return np.memmap(base + ".cfl", dtype=np.complex64, mode="r", shape=(int(np.prod(dims)),)).reshape(dims, order="F")


def write_cfl(base, arr):
    with open(base + ".hdr", "w") as f:
        f.write("# Dimensions\n" + " ".join(str(d) for d in arr.shape) + "\n")
    arr.astype(np.complex64).ravel(order="F").tofile(base + ".cfl")


def c_ifft(x, axes):
    return sfft.fftshift(sfft.ifftn(sfft.ifftshift(x, axes=axes), axes=axes, norm="ortho", workers=8), axes=axes)


def c_fft(x, axes):
    return sfft.fftshift(sfft.fftn(sfft.ifftshift(x, axes=axes), axes=axes, norm="ortho", workers=8), axes=axes)


def roll_phase(n, s):
    """k-space factor (centred FFT convention) that rolls the image by s voxels; checked to be exactly that."""
    p = c_fft(np.roll(c_ifft(np.ones(n, np.complex128), (0,)), s), (0,))
    v = np.random.default_rng(0).standard_normal(n) + 1j * np.random.default_rng(1).standard_normal(n)
    assert np.allclose(np.abs(p), 1, atol=1e-12)
    assert np.allclose(c_fft(np.roll(c_ifft(v, (0,)), s), (0,)), p * v, atol=1e-10)
    return p


def rss(k):                                     # (X, Y, Z, C) -> (X, Y, Z)
    acc = np.zeros(k.shape[:3])
    for c in range(k.shape[3]):
        acc += np.abs(c_ifft(np.asarray(k[..., c], dtype=np.complex128), (0, 1, 2))) ** 2
    return np.sqrt(acc)


def load(tag):
    subs = sorted({os.path.dirname(p) for p in glob.glob(SRC_ROOT + "/**/*.mat", recursive=True)})
    p = glob.glob(subs[SI - 1] + "/*_" + tag + "_mx3d*.mat")[0]
    with h5py.File(p, "r") as f:
        k = f["kspace"][()]                     # (n0, C, n2, n3)
    return np.ascontiguousarray(np.transpose(k, (2, 0, 3, 1))), sha(p)   # (X = n2, Y = n0, Z = n3, C)


EMPTY_REL = 0.01      # the audit's rule (audit_sampling.py): a PE line is not acquired when its outer-readout
                      # energy is below 1 % of the median over the plane


def support_of(k):
    """Acquired support on the (Y, Z) plane by the audit's rule, restricted to the region connected to the DC,
    holes filled (an isolated low line inside the acquired region stays data, it is not zeroed)."""
    nx = k.shape[0]
    P = (np.abs(k).astype(np.float64) ** 2).sum(axis=3)                     # (X, Y, Z)
    cx = int(np.argmax(P.sum(axis=(1, 2))))
    Q = P[np.abs(np.arange(nx) - cx) > 0.25 * nx].mean(axis=0)              # outer-readout energy per PE line
    E = P.sum(axis=0)
    above = Q >= EMPTY_REL * np.median(Q)
    lab, _ = nd.label(above)
    dc = np.unravel_index(int(np.argmax(E)), E.shape)
    conn = lab == lab[dc]
    sup = nd.binary_fill_holes(conn)
    le = np.log10(np.maximum(Q, 1e-300) / np.median(Q))
    info = {"rule": "outer-readout energy >= %.2f * median, connected to DC, holes filled" % EMPTY_REL,
            "readout_centre": cx, "dc_yz": [int(d) for d in dc],
            "lines_above_threshold": int(above.sum()), "support_lines": int(sup.sum()),
            "dropped_unconnected_lines": int((above & ~sup).sum()), "filled_hole_lines": int((sup & ~conn).sum()),
            "zeroed_energy_frac": float(E[~sup].sum() / E.sum()), "support_frac": float(sup.mean()),
            "log10_outer_energy_rel_median_percentiles_unacquired_50_99_max":
                [float(v) for v in np.percentile(le[~sup], [50, 99, 100])] if (~sup).any() else None,
            "log10_outer_energy_rel_median_min_acquired": float(le[sup].min())}
    return sup, info


def main():
    assert not T.exists(), f"{T} exists; refusing to touch it"
    rep = {"subject": SID, "af": AF, "layout": "(C, X=readout n2, Y=n0, Z=n3)", "contrasts": {}}
    ks, sups = {}, {}
    for tag in ("T1", "T2"):
        k, h = load(tag.lower())
        sup, info = support_of(k)
        assert info["zeroed_energy_frac"] < 2e-3, f"{tag}: zeroing would remove signal: {info}"
        assert 0.6 < info["support_frac"] < 0.9, f"{tag}: support far from the audit's 0.76-0.82: {info}"
        k[:, ~sup, :] = 0
        ks[tag], sups[tag] = k, sup
        rep["contrasts"][tag] = {"source_sha256": h, "dims": list(k.shape), **info}
        log(f"[{tag}] dims {k.shape}; {json.dumps(info)}")
    assert ks["T1"].shape == ks["T2"].shape
    nx, ny, nz, nc = ks["T1"].shape

    vols = {"T1": rss(ks["T1"])}
    v2 = rss(ks["T2"])
    before = {f"ax{a}": offset(vols["T1"], v2, a, -2.5, 2.5) for a in range(3)}
    shift = [-int(round((before[f"ax{a}"]["nmi"] + before[f"ax{a}"]["ngf"]) / 2)) for a in range(3)]
    # offset() returns the shift s that, applied to T2 (fshift by s), best matches T1; fshift(v, s) moves the
    # image by +s voxels, so the roll to apply is +s. Determine the sign empirically instead of trusting this:
    cand = {}
    for sgn in (+1, -1):
        r = v2
        for a in range(3):
            r = np.roll(r, -sgn * shift[a], axis=a)
        cand[sgn] = {f"ax{a}": offset(vols["T1"], r, a, -1.5, 1.5) for a in range(3)}
    worst = {s: max(abs(v) for ax in c.values() for v in ax.values()) for s, c in cand.items()}
    sgn = min(worst, key=worst.get)
    roll = [-sgn * s for s in shift]
    rep["offset_before_T1_T2"], rep["roll_applied_to_T2_xyz"] = before, roll
    log("offset before (voxels):", json.dumps(before), "| roll applied to T2 (X, Y, Z):", roll)
    k = ks["T2"]
    r0 = v2
    for a in range(3):
        if roll[a]:
            shp = [1, 1, 1, 1]
            shp[a] = k.shape[a]
            k = k * roll_phase(k.shape[a], roll[a]).astype(np.complex64).reshape(shp)
            wrap = np.take(r0, range(r0.shape[a] - roll[a], r0.shape[a]) if roll[a] > 0 else range(0, -roll[a]), axis=a)
            wf = float((wrap ** 2).sum() / (r0 ** 2).sum())
            rep["contrasts"]["T2"][f"wrapped_energy_frac_ax{a}"] = wf
            assert wf < 5e-3, f"the roll on axis {a} would wrap signal across the FOV edge: {wf}"
            r0 = np.roll(r0, roll[a], axis=a)
    assert np.array_equal(np.abs(k).max(axis=(0, 3)) > 0, sups["T2"]), "the shift changed the support"
    ks["T2"] = np.ascontiguousarray(k.astype(np.complex64))
    vols["T2"] = rss(ks["T2"])
    err = float(np.abs(vols["T2"] - r0).max() / r0.max())
    assert err < 1e-4, f"shift is not an exact roll: {err}"
    rep["contrasts"]["T2"]["roll_check_max_rel_err"] = err
    after = {f"ax{a}": offset(vols["T1"], vols["T2"], a, -1.5, 1.5) for a in range(3)}
    rep["offset_after_T1_T2"] = after
    log("offset after (voxels):", json.dumps(after))
    assert all(abs(v) <= 0.3 for ax in after.values() for v in ax.values()), "pair still offset after the shift"

    # mask: the house vd-Poisson generator on this plane
    spec = importlib.util.spec_from_file_location("mask16", SEPT / "scripts/make_mask_af16.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    mk.NX, mk.NY, mk.NZ = nx, ny, nz
    mk.TARGET = round(ny * nz / AF)
    m, minfo = mk.find_mask()
    assert m.shape == (ny, nz) and int(m.sum()) == mk.TARGET and mk.acs_block(m).all()
    minfo = {k_: v for k_, v in minfo.items() if k_ != "visited"}
    minfo.update(plane=[ny, nz], target=int(mk.TARGET), acs=int(mk.ACS),
                 generator_sha256=sha(SEPT / "scripts/make_mask_af16.py"))
    rep["mask"] = minfo
    for tag in ("T1", "T2"):
        eff = m & sups[tag]
        assert sups[tag][mk.acs_slice()].all(), f"{tag}: the ACS block is not fully inside the acquired support"
        rep["contrasts"][tag].update(sampled_inside_support=int(eff.sum()), sampled_outside_support=int((m & ~sups[tag]).sum()),
                                     AF_whole_plane=float(ny * nz / m.sum()),
                                     AF_vs_own_support=float(sups[tag].sum() / eff.sum()))
        log(f"[{tag}] AF whole plane {ny * nz / m.sum():.2f}, vs own support {sups[tag].sum() / eff.sum():.2f}")

    # only now: the topic
    T.mkdir()
    for d in ("data", "logs", "configs", "scripts", "recon", "figures"):
        (T / d).mkdir()
    with h5py.File(T / f"data/mask_af{AF}_3d.h5", "w") as f:
        f.create_dataset("mask", data=np.broadcast_to(m[None], (nx, ny, nz)).astype(np.uint8))
    np.save(T / f"data/mask_af{AF}_pe.npy", m.astype(np.uint8))
    for tag in ("T1", "T2"):
        np.save(T / f"data/support_{tag}.npy", sups[tag])
        write_cfl(str(T / f"data/ksp_{tag}"), ks[tag][..., None])
        with h5py.File(T / f"data/ksp_{tag}.h5", "w") as h:
            h.create_dataset("kspace", data=np.ascontiguousarray(np.moveaxis(ks[tag], 3, 0)), chunks=(1, 32, ny, nz))
        np.save(T / f"recon/rss_{tag}.npy", vols[tag].astype(np.float32))
        log(f"[{tag}] ksp written")
    del ks

    env = dict(os.environ, TOOLBOX_PATH=BART, PATH=f"{BART}:{os.environ.get('PATH', '')}", OMP_NUM_THREADS="16")

    def ecalib(tag):
        r = subprocess.run([f"{BART}/bart", *ECALIB, str(T / f"data/ksp_{tag}"), str(T / f"data/csm_espirit_{tag}")],
                           env=env, capture_output=True, text=True)
        return tag, r.returncode, (r.stdout + r.stderr)[-800:]

    with ThreadPoolExecutor(2) as ex:
        for tag, rc, tail in ex.map(ecalib, ("T1", "T2")):
            log(f"[{tag}] ecalib rc {rc}\n{tail}")
            assert rc == 0, f"ecalib failed on {tag}"
    sos_mid = {}
    for tag in ("T1", "T2"):
        s = np.squeeze(read_cfl(str(T / f"data/csm_espirit_{tag}")))
        arr = np.ascontiguousarray(np.moveaxis(np.asarray(s), 3, 0))
        assert arr.shape == (nc, nx, ny, nz), arr.shape
        sos = np.sqrt((np.abs(arr) ** 2).sum(0))
        with h5py.File(T / f"data/csm_espirit_{tag}.h5", "w") as h:
            h.create_dataset("map", data=arr, chunks=(1, 32, ny, nz))
        rep["contrasts"][tag]["csm_sos_within_5pct_of_1"] = float((np.abs(sos - 1) < 0.05).mean())
        sos_mid[tag] = sos[nx // 2]
        for ext in (".cfl", ".hdr"):
            os.remove(T / f"data/ksp_{tag}{ext}")
            os.remove(T / f"data/csm_espirit_{tag}{ext}")
        for f in (f"ksp_{tag}.h5", f"csm_espirit_{tag}.h5", f"support_{tag}.npy"):
            rep[f"sha256_{f}"] = sha(T / "data" / f)
        log(f"[{tag}] csm written; SoS within 5 % of 1 on {rep['contrasts'][tag]['csm_sos_within_5pct_of_1']:.4f}")
    rep[f"sha256_mask_af{AF}_3d.h5"] = sha(T / f"data/mask_af{AF}_3d.h5")

    # config: the tangbing T1+T2 config, data block replaced
    import yaml
    cfg = yaml.safe_load(open(TB_CFG))
    for slot, tag in (("plain", "T1"), ("contrast", "T2")):
        b = cfg["data"][slot]
        assert b["name"] == tag
        b.update(data_path=str(T / f"data/ksp_{tag}.h5"), csm_path=str(T / f"data/csm_espirit_{tag}.h5"),
                 mask_path=str(T / f"data/mask_af{AF}_3d.h5"), acceleration_factor=AF, known_zero=True,
                 known_zero_support=str(T / f"data/support_{tag}.npy"))
    name = f"knee_uih_t1t2_af{AF}_{SID.lower()}_joint"
    cfg["output"] = {"exp_name": name, "output_dir": f"results_arc/{name}"}
    head = (f"# knee_uih_t1t2_af{AF}.yaml -- UIH 3D knee, subject {SID}, T1 + T2, retrospective AF{AF}.\n"
            f"# Written by mcgs autoresearch/uih_knee_t1t2_20261125/prep_knee.py from the tangbing T1+T2 config\n"
            f"# ({TB_CFG}); only the data block and the names differ. known_zero on both contrasts with explicit\n"
            f"# support maps (T1: elliptical corners; T2: corners + partial-Fourier band).\n")
    (T / f"configs/knee_uih_t1t2_af{AF}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False))
    rep["config"] = str(T / f"configs/knee_uih_t1t2_af{AF}.yaml")

    # QC picture
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(14, 7), layout="compressed")
    ax[0, 0].imshow(sups["T1"].T, cmap="gray", origin="lower"); ax[0, 0].set_title(f"T1 support {sups['T1'].mean():.3f}")
    ax[0, 1].imshow(sups["T2"].T, cmap="gray", origin="lower"); ax[0, 1].set_title(f"T2 support {sups['T2'].mean():.3f}")
    ax[0, 2].imshow(m.T, cmap="gray", origin="lower"); ax[0, 2].set_title(f"mask AF{AF}: {int(m.sum())} points")
    ax[0, 3].imshow(sos_mid["T1"].T, cmap="viridis", origin="lower", vmin=0, vmax=1.1); ax[0, 3].set_title("T1 map SoS, mid X")
    for j, tag in enumerate(("T1", "T2")):
        v = vols[tag]
        vm = np.percentile(v, 99.5)
        ax[1, 2 * j].imshow(v[nx // 2].T, cmap="gray", origin="lower", vmin=0, vmax=vm); ax[1, 2 * j].set_title(f"{tag} RSS, mid X (Y-Z)")
        ax[1, 2 * j + 1].imshow(v[:, :, nz // 2].T, cmap="gray", origin="lower", vmin=0, vmax=vm); ax[1, 2 * j + 1].set_title(f"{tag} RSS, mid Z (X-Y)")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.savefig(T / "figures/prep_qc.png", dpi=150)
    shutil.copy2(T / "figures/prep_qc.png", ART / f"prep_qc_{SID}.png")

    json.dump(rep, open(T / "logs/prep_knee.json", "w"), indent=2)
    shutil.copy2(__file__, T / "scripts" / Path(__file__).name)
    json.dump(rep, open(ART / f"prep_knee_{SID}.json", "w"), indent=2)
    shutil.copy2(T / f"configs/knee_uih_t1t2_af{AF}.yaml", ART / f"knee_uih_t1t2_af{AF}_{SID}.yaml")
    log("PREP OK " + json.dumps(rep))


if __name__ == "__main__":
    main()
