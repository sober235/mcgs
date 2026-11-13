#!/usr/bin/env python3
"""prep_tangbing.py -- a T1 + T2 pair without a voxel offset, for the T1+T2 row of the framework ablation.

User 2026-11-13: replace the huangyuwei T1+T2 pair (T2 about one voxel off T1 along the readout) by a new
T1W / T2W pair with no voxel offset. screen_t1t2.py found no offset-free pair among the seven MC_Jia subjects;
the closest is vol4_tangbing (the pre-registered held-out subject): readout and left-right aligned (<= 0.1 voxel),
anterior-posterior off by exactly one voxel (NMI 1.00, NGF 1.05). The user approved (2026-11-13, "OK"): use
tangbing and remove that voxel by an integer shift of T2 while the data set is built, before any undersampling.

The shift is exact: an image roll of +1 along axis 1 (ky) is a unit-modulus linear phase on the k-space, so every
measured sample keeps its magnitude, measured zeros (the elliptical shutter) stay exactly zero, and the
retrospective mask applies unchanged. The coil maps are estimated from the shifted k-space with the house ESPIRiT
recipe, so they move with it. Everything else follows T1T2_AF12_retro (same grid, same AF12 mask file, same
k-space layout, same map recipe).

    gpu.py run --cpu -- @python autoresearch/t1t2_aligned_subject_20261113/prep_tangbing.py

Everything is computed and checked in memory first; only then is the new topic T created (never overwritten).
Checks: shift exactness, offset after the shift (<= 0.25 voxel on every axis by both metrics), ecalib rc,
map SoS, mask bytes identical to T1T2_AF12_retro.
"""
import hashlib
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_t1t2 import offset  # noqa: E402  (the screening's metrics)

SRCD = "/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL/vol4_tangbing"
FILES = {"T1": "t1_fsp_train_tangbing_cc", "T2": "t2_train_tangbing_cc"}
RES = Path("/data0/congcong/code/GS/Results_20260908")
REF = RES / "T1T2_AF12_retro"
T = RES / "T1T2_AF12_v4_tangbing"
SHIFT = {"T2": (1, +1)}          # contrast: (image axis, integer voxels), from screen_t1t2.py
BART = "/home/congcongliu/src/bart"
ECALIB = ["ecalib", "-t", "0.001", "-c", "0.0", "-r", "24", "-m", "1"]   # house recipe (prep_03_csm.sh)
log = lambda *a: print(*a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_cfl(base):
    dims = [int(x) for x in open(base + ".hdr").read().split("\n")[1].split()]
    n = int(np.prod(dims))
    return np.memmap(base + ".cfl", dtype=np.complex64, mode="r", shape=(n,)).reshape(dims, order="F")


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


def rss(k):                                     # (X, Y, Z, C) -> (X, Y, Z), in double precision
    acc = np.zeros(k.shape[:3])
    for c in range(k.shape[3]):
        acc += np.abs(c_ifft(np.asarray(k[..., c], dtype=np.complex128), (0, 1, 2))) ** 2
    return np.sqrt(acc)


def main():
    assert not T.exists(), f"{T} exists; refusing to touch it"
    rep = {"source": SRCD, "files": FILES, "shift": {k: list(v) for k, v in SHIFT.items()}, "contrasts": {}}
    m2 = np.asarray(h5py.File(REF / "data/mask_af12_3d.h5", "r")["mask"][0]).astype(bool)   # (ky, kz)

    ks, vols = {}, {}
    for tag, f in FILES.items():
        k = np.asarray(read_cfl(f"{SRCD}/{f}")[..., 0])            # (X, Y, Z, C) complex64
        nx, ny, nz, nc = k.shape
        sup = np.abs(k).max(axis=(0, 3)) > 0                       # acquired support on the PE plane
        rep["contrasts"][tag] = {}
        if tag in SHIFT:
            ax, s = SHIFT[tag]
            assert ax == 1, "only the ky axis is implemented"
            r0 = rss(k)
            wrap = r0[:, -s:, :] if s > 0 else r0[:, :-s, :]          # the rows the circular roll carries across the edge
            wrap_frac = float((wrap ** 2).sum() / (r0 ** 2).sum())
            rep["contrasts"][tag]["wrapped_rows_energy_frac"] = wrap_frac
            assert wrap_frac < 1e-3, f"the roll would wrap signal across the FOV edge: {wrap_frac}"
            p = roll_phase(ny, s).astype(np.complex64)
            k2 = k * p[None, :, None, None]
            assert np.array_equal(np.abs(k2) > 0, np.abs(k) > 0), "a measured zero changed"
            assert np.allclose(np.abs(k2), np.abs(k), rtol=1e-6, atol=0)
            k = k2
            del k2
            vols[tag] = rss(k)
            err = float(np.abs(vols[tag] - np.roll(r0, s, axis=ax)).max() / r0.max())
            assert err < 1e-5, f"shift is not an exact roll: {err}"
            rep["contrasts"][tag]["roll_check_max_rel_err"] = err
            del r0
        else:
            vols[tag] = rss(k)
        eff = m2 & sup
        rep["contrasts"][tag].update({
            "dims": [nx, ny, nz, nc], "support_frac": float(sup.mean()),
            "sampled_frac_of_plane": float(eff.mean()), "AF_vs_full_rect": float(1 / eff.mean()),
            "AF_vs_own_support": float(sup.mean() / eff.mean())})
        ks[tag] = k
        log(f"[{tag}] loaded; support {sup.mean():.4f}, AF vs own support {sup.mean() / eff.mean():.2f}")

    # offset after the shift, by the screening's metrics: every axis within a quarter voxel
    after = {f"ax{a}": offset(vols["T1"], vols["T2"], a, -2 if a == 0 else -1.5, 2 if a == 0 else 1.5) for a in range(3)}
    rep["offset_after_shift_T1_T2"] = after
    log("offset after shift (T2 vs T1, voxels):", json.dumps(after))
    assert all(abs(v) <= 0.25 for ax in after.values() for v in ax.values()), "pair still offset after the shift"

    # only now: the topic
    T.mkdir()
    for d in ("data", "logs", "configs", "scripts", "recon", "figures"):
        (T / d).mkdir()
    for f in ("mask_af12_3d.h5", "mask_vdpois_256x176_af12.mat"):     # byte copies (identical across MC_Jia topics)
        shutil.copy2(REF / "data" / f, T / "data" / f)
        assert sha(T / "data" / f) == sha(REF / "data" / f)
        rep[f"sha256_{f}"] = sha(T / "data" / f)
    for tag, k in ks.items():
        write_cfl(str(T / f"data/ksp_{tag}"), k[..., None])         # for ecalib (BART order X,Y,Z,C,1)
        with h5py.File(T / f"data/ksp_{tag}.h5", "w") as h:
            h.create_dataset("kspace", data=np.ascontiguousarray(np.moveaxis(k, 3, 0)), chunks=(1, 32, 256, 176))
        log(f"[{tag}] ksp written")
    del ks
    for tag, v in vols.items():                                       # coil-RSS references, as T1T2_AF12_retro/recon
        np.save(T / f"recon/rss_{tag}.npy", v.astype(np.float32))

    env = dict(os.environ, TOOLBOX_PATH=BART, PATH=f"{BART}:{os.environ.get('PATH', '')}", OMP_NUM_THREADS="16")

    def ecalib(tag):
        r = subprocess.run([f"{BART}/bart", *ECALIB, str(T / f"data/ksp_{tag}"), str(T / f"data/csm_espirit_{tag}")],
                           env=env, capture_output=True, text=True)
        return tag, r.returncode, (r.stdout + r.stderr)[-1500:]

    with ThreadPoolExecutor(2) as ex:                                 # both contrasts in parallel
        for tag, rc, tail in ex.map(ecalib, FILES):
            log(f"[{tag}] ecalib rc {rc}\n{tail}")
            assert rc == 0, f"ecalib failed on {tag}"
    for tag in FILES:
        s = np.squeeze(read_cfl(str(T / f"data/csm_espirit_{tag}")))   # (X, Y, Z, C)
        arr = np.ascontiguousarray(np.moveaxis(np.asarray(s), 3, 0))
        sos = np.sqrt((np.abs(arr) ** 2).sum(0))
        with h5py.File(T / f"data/csm_espirit_{tag}.h5", "w") as h:
            h.create_dataset("map", data=arr, chunks=(1, 32, 256, 176))
        rep["contrasts"][tag]["csm_sos_within_5pct_of_1"] = float((np.abs(sos - 1) < 0.05).mean())
        log(f"[{tag}] csm h5 written; SoS within 5 % of 1 on {rep['contrasts'][tag]['csm_sos_within_5pct_of_1']:.4f}")
        del arr, s, sos
    for tag in FILES:                                                   # CFL copies were only for ecalib
        for ext in (".cfl", ".hdr"):
            os.remove(T / f"data/ksp_{tag}{ext}")
            os.remove(T / f"data/csm_espirit_{tag}{ext}")
        for f in (f"ksp_{tag}.h5", f"csm_espirit_{tag}.h5"):
            rep[f"sha256_{f}"] = sha(T / "data" / f)

    json.dump(rep, open(T / "logs/prep_tangbing.json", "w"), indent=2)
    shutil.copy2(__file__, T / "scripts" / Path(__file__).name)
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if art:
        json.dump(rep, open(os.path.join(art, "prep_tangbing.json"), "w"), indent=2)
    log("PREP OK " + json.dumps(rep))


if __name__ == "__main__":
    main()
