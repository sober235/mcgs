#!/usr/bin/env python3
"""prep_03_pick_mask_af24.py -- choose the AF24 mask from the pool written by prep_02_mask_pool_af24.m (2026-09-14).

Rule, fixed before the pool was drawn. AF = PE-plane points / sampled points that carry that contrast's data (user
decision 2026-09-14). T1's acquired support is the whole plane, so T1 at exactly 24x = exactly 1877 of 45056 points.
  1. drop exact duplicates (the generator seeds from time(0));
  2. among the draws with exactly 1877 points take the lowest draw index -- the generator's own output, no trimming;
  3. only if no draw has exactly 1877: take the draw closest from above and drop points from the outermost radius
     (distance from the k-space centre (128, 88), random tie-break, seed 0) until 1877 remain -- the rule of
     T1FLAIR_AF16_v3_jiangyizhou/scripts/prep_04c_export_af16c.py.
FLAIR's acquired support (vendor shutter, 34139 of 45056 lines) is read from coil 0 of data/ksp_FLAIR.h5, the
fig_mask_support.py convention (identical to the prep_04_export.py rule, checked in prep_04c_export_af16c.py); it is
only reported here and plays no part in the choice.
Writes data/mask_vdpois_256x176_af24.mat (key 'mask', float32 (256,176)) and logs/prep_03_pick_mask_af24.json.
Refuses to overwrite an existing mask file: every AF24 result will depend on it.
"""
import glob, hashlib, json, os
import h5py, numpy as np, scipy.io

R = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou"
POOL = f"{R}/data/mask_pool_af24"
OUT_MAT = f"{R}/data/mask_vdpois_256x176_af24.mat"
OUT_JSON = f"{R}/logs/prep_03_pick_mask_af24.json"
NY, NZ = 256, 176
N_PLANE = NY * NZ
TARGET = 1877                                   # 45056 / 1877 = 24.004 -> T1 at 24x
CENTRE = (128, 88)
SEED = 0


def md5_file(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    assert not os.path.exists(OUT_MAT), f"{OUT_MAT} exists; refusing to overwrite the AF24 mask"
    with h5py.File(f"{R}/data/ksp_FLAIR.h5", "r") as f:          # symlink into the AF12 topic: READ ONLY
        sup = np.any(np.abs(np.array(f["kspace"][0])) > 0, axis=0)
    assert sup.shape == (NY, NZ) and int(sup.sum()) == 34139, (sup.shape, int(sup.sum()))

    pool, seen = [], {}
    for p in sorted(glob.glob(f"{POOL}/draw_*.mat")):
        d = scipy.io.loadmat(p)
        m = d["mask"].astype(bool)
        assert m.shape == (NY, NZ), (p, m.shape)
        key = hashlib.md5(np.packbits(m).tobytes()).hexdigest()
        rec = dict(draw=os.path.basename(p), stage=str(d["stage"].squeeze()),
                   accelx=round(float(d["accelx"].squeeze()), 4), accely=int(d["accely"].squeeze()),
                   calib=int(d["calib"].squeeze()), ellipse=int(d["ellipse"].squeeze()), pp=int(d["pp"].squeeze()),
                   nnz=int(m.sum()), flair_lines=int((m & sup).sum()), duplicate_of=seen.get(key))
        assert (rec["accely"], rec["calib"], rec["ellipse"], rec["pp"]) == (4, 24, 0, 1), rec
        seen.setdefault(key, rec["draw"])
        pool.append((rec, m))
    assert pool, f"no draws in {POOL}"
    uniq = [(r, m) for r, m in pool if r["duplicate_of"] is None]
    exact = [(r, m) for r, m in uniq if r["nnz"] == TARGET]
    nn = np.array([r["nnz"] for r, _ in uniq])
    print(f"[pool] {len(pool)} draws, {len(pool) - len(uniq)} duplicates, nnz {nn.min()}-{nn.max()}, "
          f"exactly {TARGET}: {len(exact)}", flush=True)

    trim = None
    if exact:
        rec, m = min(exact, key=lambda rm: rm[0]["draw"])
        new = m.copy()
        how = "exact draw, untrimmed"
    else:                                                           # fallback: closest from above, trimmed
        above = [(r, m) for r, m in uniq if r["nnz"] > TARGET]
        assert above, "no draw at or above the target; extend the pool to lower accelx"
        rec, m = min(above, key=lambda rm: (rm[0]["nnz"], rm[0]["draw"]))
        ky, kz = np.nonzero(m)
        r = np.hypot(ky - CENTRE[0], kz - CENTRE[1])
        n_excess = int(m.sum()) - TARGET
        order = np.lexsort((np.random.default_rng(SEED).random(r.size), -r))
        drop = order[:n_excess]
        new = m.copy()
        new[ky[drop], kz[drop]] = False
        trim = dict(n_trimmed=n_excess, seed=SEED, trimmed_radius_min=float(r[drop].min()),
                    trimmed_radius_max=float(r[drop].max()))
        how = f"closest from above ({rec['nnz']}), trimmed {n_excess} outermost points"
    assert int(new.sum()) == TARGET

    n_fl = int((new & sup).sum())
    cal = new[CENTRE[0] - 12:CENTRE[0] + 12, CENTRE[1] - 12:CENTRE[1] + 12]
    scipy.io.savemat(OUT_MAT, {"mask": new.astype(np.float32)}, format="5")
    back = scipy.io.loadmat(OUT_MAT)["mask"]
    assert back.shape == (NY, NZ) and (back.astype(bool) == new).all()
    exact_fl = sorted(r["flair_lines"] for r, _ in exact)
    rep = dict(rule="T1 exactly 24x under AF = PE-plane points / sampled points with data (user 2026-09-14): "
                    "lowest-index exact draw, else closest from above trimmed at the outermost radius",
               target_points=TARGET, plane_points=N_PLANE, chosen=rec, how=how, trim=trim,
               mask_points=TARGET, calib24_points=int(cal.sum()),
               T1=dict(sampled_points_with_data=TARGET, AF=N_PLANE / TARGET),
               FLAIR=dict(support_points=int(sup.sum()), sampled_points_with_data=n_fl,
                          points_outside_shutter=TARGET - n_fl, AF=N_PLANE / n_fl,
                          AF_against_own_shutter_old_convention=int(sup.sum()) / n_fl),
               pool=dict(draws=len(pool), duplicates=len(pool) - len(uniq), nnz_min=int(nn.min()),
                         nnz_max=int(nn.max()), exact_hits=len(exact),
                         exact_hits_flair_lines_min=exact_fl[0] if exact_fl else None,
                         exact_hits_flair_lines_max=exact_fl[-1] if exact_fl else None,
                         draws_list=[r for r, _ in pool]),
               mask_mat=OUT_MAT, mask_mat_md5=md5_file(OUT_MAT))
    json.dump(rep, open(OUT_JSON, "w"), indent=1)
    print(f"[pick] {rec['draw']} stage {rec['stage']} accelx {rec['accelx']} -> {how}", flush=True)
    print(f"[mask] {TARGET} of {N_PLANE} points, 24x24 centre {int(cal.sum())}/576; "
          f"T1 AF {N_PLANE / TARGET:.3f}; FLAIR {n_fl} points on data, {TARGET - n_fl} outside the shutter, "
          f"AF {N_PLANE / n_fl:.3f}; md5 {rep['mask_mat_md5']}", flush=True)
    if exact_fl:
        print(f"[pool] FLAIR points on data across the {len(exact)} exact draws: {exact_fl[0]}-{exact_fl[-1]} "
              f"(AF {N_PLANE / exact_fl[-1]:.2f}-{N_PLANE / exact_fl[0]:.2f})", flush=True)
    print("wrote", OUT_MAT, OUT_JSON)


if __name__ == "__main__":
    main()
