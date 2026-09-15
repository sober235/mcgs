#!/usr/bin/env python3
"""make_masks_af16_diffmask.py -- two alternative mask patterns at the AF16 topic's acceleration (2026-09-15).

Diff-mask study on the jiangyizhou T1 + FLAIR pair: same k-space, ESPIRiT maps, ground truth, gauge, arms and the frozen
AF16 phase-3 MC-GS recipe as T1FLAIR_AF16_v3_jiangyizhou; only the mask PATTERN differs from that topic's house
vd-Poisson disc:
  gauss   2-D Gaussian variable-density random sampling      -> T1FLAIR_AF16_gauss_v3_jiangyizhou
  radial  Cartesian pseudo-radial spokes through DC          -> T1FLAIR_AF16_radial_v3_jiangyizhou
Both are labelled AF 16 (nominal, exactly as the AF16 topic is) and both sit at that topic's actual T1 acceleration
under the user's definition of 2026-09-14 (AF = PE-plane points / sampled points that carry that contrast's data):
T1 acquires the whole 256 x 176 plane, so T1 at 15.20x = exactly 2964 of 45056 points (the AF16 mask: 2966, 15.19x).
FLAIR shares the mask and, with its elliptical vendor shutter (34139 acquired lines), lands higher: reported, not tuned.

Common to both patterns
  - the house calibration block, fully sampled: the largest fully sampled centred rectangle of the AF16 vd-Poisson mask
    is 24 x 16 (ky x kz; vdPoisMex's calib=24 scaled by 176/256 in kz), the block recon_spirit.py's acs_box() finds.
    Kept identical so the L1-SPIRiT arm calibrates from the same ACS in every topic;
  - DC at (128, 88) (prep_01_inspect.py of the AF12 topic: no mask-registration roll), one shared mask per study;
  - deterministic: numpy Generator(seed 0), unlike vdPoisMex (time(0)) -- these masks CAN be regenerated;
  - rn = sqrt(((ky-128)/128)^2 + ((kz-88)/88)^2), the normalised elliptical radius, i.e. the coordinates in which
    vdPoisMex's variable density is defined.

gauss   outside the ACS, TARGET - 384 points are drawn without replacement with probability proportional to
        exp(-rn^2 / (2 s^2)). The width s is not a hand-picked number: it is chosen on a 0.005 grid so that the seeded
        draw's median rn over its sampled points matches the AF16 vd-Poisson mask's (the density curve is matched
        where a Gaussian can follow it; what differs is the point process, i.i.d. random vs Poisson disc).
        --gauss_s overrides the width (the inspection figure shows 0.7 s and 1.4 s beside it).
radial  N spokes through DC at angles i * 180 / N degrees, i = 0..N-1, in the normalised coordinates (a spoke reaches
        the plane edge in every direction: elliptical field of view, like the shutter); each spoke is rasterised by
        rounding 1024 points along it to the nearest grid point. N is the smallest spoke count whose union with the
        ACS reaches >= TARGET points; the excess is trimmed at the largest rn (random tie-break), so the spokes end
        on one common elliptical radius.

Per topic this writes data/mask_<kind>_256x176_af16.mat (key 'mask', float32 (256, 176), the AF24 topic's format),
data/mask_af16_3d.h5 ('mask', uint8 (256, 256, 176), broadcast along the readout, the prep_04_export.py layout) and
logs/make_masks_af16_diffmask.json (every number the README quotes). Refuses to overwrite an existing mask: every
result of a topic depends on it.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

RES = Path("/data0/congcong/code/GS/Results_20260908")
TOPICS = {"gauss": RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou",
          "radial": RES / "T1FLAIR_AF16_radial_v3_jiangyizhou"}
REF_MASK = RES / "T1FLAIR_AF16_v3_jiangyizhou/data/mask_vdpois_256x176_af16.mat"
KSP_FLAIR = RES / "T1FLAIR_AF12_v3_jiangyizhou/data/ksp_FLAIR.h5"         # every jyz topic's symlink target: READ ONLY
NX, NY, NZ = 256, 256, 176                                                # (readout, ky, kz)
CY, CZ = 128, 88
N_PLANE = NY * NZ
TARGET = 2964                                   # 45056 / 2964 = 15.201: T1 at 15.20x, the AF16 topic's actual T1 AF
ACS = (24, 16)                                  # ky x kz, the AF16 mask's fully sampled centred block
SEED = 0
S_GRID = np.round(np.arange(0.05, 1.0001, 0.005), 3)
KY, KZ = np.mgrid[0:NY, 0:NZ]
RN = np.sqrt(((KY - CY) / (NY / 2)) ** 2 + ((KZ - CZ) / (NZ / 2)) ** 2)


def acs_block():
    m = np.zeros((NY, NZ), dtype=bool)
    m[CY - ACS[0] // 2:CY + ACS[0] // 2, CZ - ACS[1] // 2:CZ + ACS[1] // 2] = True
    return m


def largest_full_rect(m):
    """largest (by area) fully sampled rectangle centred on DC, even extents (ky, kz)"""
    best = (0, 0)
    for hy in range(1, 64):
        if not m[CY - hy:CY + hy, CZ - 1:CZ + 1].all():
            break
        for hz in range(1, 64):
            if not m[CY - hy:CY + hy, CZ - hz:CZ + hz].all():
                break
            if 4 * hy * hz > best[0] * best[1]:
                best = (2 * hy, 2 * hz)
    return best


def flair_shutter():
    with h5py.File(KSP_FLAIR, "r") as f:
        sup = np.any(np.abs(np.array(f["kspace"][0])) > 0, axis=0)
    assert sup.shape == (NY, NZ) and int(sup.sum()) == 34139, (sup.shape, int(sup.sum()))
    return sup


def draw_gauss(s, seed=SEED):
    rng = np.random.default_rng(seed)
    m = acs_block()
    flat = m.reshape(-1)
    cand = np.flatnonzero(~flat)
    p = np.exp(-RN.reshape(-1)[cand] ** 2 / (2.0 * float(s) ** 2))
    p /= p.sum()
    flat[rng.choice(cand, size=TARGET - int(flat.sum()), replace=False, p=p)] = True
    assert int(m.sum()) == TARGET
    return m


def fit_gauss_width(ref):
    """the grid width whose seeded draw has the reference mask's median normalised radius"""
    target = float(np.median(RN[ref]))
    rows = [(float(s), float(np.median(RN[draw_gauss(float(s))]))) for s in S_GRID]
    s_best, med_best = min(rows, key=lambda r: (abs(r[1] - target), r[0]))
    return s_best, med_best, target, rows


def spokes(n):
    m = np.zeros((NY, NZ), dtype=bool)
    t = np.linspace(-1.0, 1.0, 4 * NY)
    for i in range(n):
        th = i * np.pi / n
        ky = np.rint(CY + t * (NY / 2) * np.cos(th)).astype(int)
        kz = np.rint(CZ + t * (NZ / 2) * np.sin(th)).astype(int)
        ok = (ky >= 0) & (ky < NY) & (kz >= 0) & (kz < NZ)
        m[ky[ok], kz[ok]] = True
    return m


def draw_radial(seed=SEED):
    rng = np.random.default_rng(seed)
    acs = acs_block()
    for n in range(1, 400):
        u = acs | spokes(n)
        if int(u.sum()) >= TARGET:
            break
    else:
        raise RuntimeError("no spoke count reaches the target")
    excess = int(u.sum()) - TARGET
    outside = np.flatnonzero((u & ~acs).reshape(-1))
    r = RN.reshape(-1)[outside]
    order = np.lexsort((rng.random(r.size), -r))           # largest normalised radius first, random tie-break
    m = u.copy()
    m.reshape(-1)[outside[order[:excess]]] = False
    assert int(m.sum()) == TARGET
    info = dict(spokes=n, angle_step_deg=180.0 / n, angles_deg=[round(i * 180.0 / n, 3) for i in range(n)],
                points_before_trim=int(u.sum()), trimmed=excess,
                rn_max_kept=float(RN[m & ~acs].max()),
                rn_trimmed_min=float(r[order[:excess]].min()) if excess else None,
                points_per_spoke_mean=float((int(u.sum()) - int(acs.sum())) / n))
    return m, info


def stats(m, sup):
    n, n_fl = int(m.sum()), int((m & sup).sum())
    return dict(points=n, plane_points=N_PLANE, T1_AF=N_PLANE / n,
                FLAIR_points_with_data=n_fl, FLAIR_points_outside_shutter=n - n_fl, FLAIR_AF=N_PLANE / n_fl,
                FLAIR_AF_vs_own_shutter_old_convention=int(sup.sum()) / n_fl,
                centre24x24=int(m[CY - 12:CY + 12, CZ - 12:CZ + 12].sum()),
                largest_full_centred_rect_ky_kz=list(largest_full_rect(m)),
                median_rn=float(np.median(RN[m])), mean_rn=float(RN[m].mean()))


def write(kind, m, extra):
    T = TOPICS[kind]
    mat, h5p = T / f"data/mask_{kind}_256x176_af16.mat", T / "data/mask_af16_3d.h5"
    for p in (mat, h5p):
        assert not p.exists(), f"{p} exists; refusing to overwrite a registered mask"
    (T / "logs").mkdir(exist_ok=True)
    sio.savemat(mat, {"mask": m.astype(np.float32)}, format="5")
    back = sio.loadmat(mat)["mask"]
    assert back.shape == (NY, NZ) and (back.astype(bool) == m).all()
    m3 = np.broadcast_to(m[None, :, :], (NX, NY, NZ)).astype(np.uint8)
    with h5py.File(h5p, "w") as h:
        h.create_dataset("mask", data=m3, compression="gzip", compression_opts=1)
    rep = dict(kind=kind, label="AF 16 (nominal); actual = the AF16 topic's own T1 acceleration",
               target_points=TARGET, seed=SEED, dc=[CY, CZ], acs_block_ky_kz=list(ACS), **extra,
               mask_mat=str(mat), mask_mat_md5=hashlib.md5(open(mat, "rb").read()).hexdigest(), mask3d=str(h5p))
    json.dump(rep, open(T / "logs/make_masks_af16_diffmask.json", "w"), indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="gauss,radial")
    ap.add_argument("--gauss_s", type=float, default=None, help="override the fitted Gaussian width (rn units)")
    a = ap.parse_args()
    sup = flair_shutter()
    ref = sio.loadmat(REF_MASK)["mask"].astype(bool)
    rs = stats(ref, sup)
    print(f"[ref] vd-Poisson AF16: {rs['points']} points, T1 {rs['T1_AF']:.3f}x, FLAIR {rs['FLAIR_AF']:.3f}x, "
          f"24x24 centre {rs['centre24x24']}, full rect {rs['largest_full_centred_rect_ky_kz']}, "
          f"median rn {rs['median_rn']:.4f}", flush=True)
    for kind in a.kinds.split(","):
        if kind == "gauss":
            if a.gauss_s is None:
                s, med, target, rows = fit_gauss_width(ref)
                fit = dict(rule="median rn of the sampled points matched to the AF16 vd-Poisson mask on a 0.005 grid",
                           s=s, median_rn=med, reference_median_rn=target, grid=rows)
            else:
                s, fit = a.gauss_s, dict(rule="user override", s=a.gauss_s)
            m = draw_gauss(s)
            extra = dict(pattern="2-D Gaussian variable density, i.i.d. without replacement outside the ACS",
                         gauss=dict(s=s, sigma_ky_lines=s * NY / 2, sigma_kz_lines=s * NZ / 2, fit=fit),
                         reference_vdpois_af16=dict(mask=str(REF_MASK), **rs), **stats(m, sup))
        elif kind == "radial":
            m, info = draw_radial()
            extra = dict(pattern="Cartesian pseudo-radial spokes, uniform angles, elliptical extent, "
                                 "outermost points trimmed to the budget",
                         radial=info, reference_vdpois_af16=dict(mask=str(REF_MASK), **rs), **stats(m, sup))
        else:
            raise SystemExit(f"unknown kind {kind}")
        rep = write(kind, m, extra)
        print(f"[{kind}] {rep['points']} points, T1 {rep['T1_AF']:.3f}x, FLAIR {rep['FLAIR_AF']:.3f}x "
              f"({rep['FLAIR_points_outside_shutter']} outside the shutter), 24x24 centre {rep['centre24x24']}, "
              f"full rect {rep['largest_full_centred_rect_ky_kz']}, median rn {rep['median_rn']:.4f}; "
              f"{'s=%.3f' % rep['gauss']['s'] if kind == 'gauss' else '%d spokes, trimmed %d' % (rep['radial']['spokes'], rep['radial']['trimmed'])}; "
              f"md5 {rep['mask_mat_md5']}", flush=True)


if __name__ == "__main__":
    main()
