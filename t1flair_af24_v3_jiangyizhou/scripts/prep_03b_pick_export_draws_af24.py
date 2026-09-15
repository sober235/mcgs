#!/usr/bin/env python3
"""prep_03b_pick_export_draws_af24.py -- masks b, c, d for the AF24 mask-draw variance study (2026-09-15).

Pool: data/mask_pool_af24_draws/ (prep_02b_mask_pool_draws_af24.m). Rule = the registered mask's rule: exact duplicates
dropped, and so is any draw identical to draw a (data/mask_vdpois_256x176_af24.mat); the three lowest-index draws with
exactly 1877 points become b, c, d, untrimmed. Writes data/mask_vdpois_256x176_af24_d{b,c,d}.mat and, broadcast along
the readout, data/mask_af24_d{b,c,d}_3d.h5 (the layout of data/mask_af24_3d.h5), and logs/prep_03b_pick_export_draws_af24.json
with, for draws a-d: points, T1 / FLAIR AF under the user's definition (PE-plane points / sampled points that carry that
contrast's data), FLAIR points outside its shutter, the 24x24 centre, points shared with draw a. Refuses to overwrite.
"""
import glob, hashlib, json, os
import h5py, numpy as np, scipy.io

R = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou"
POOL = f"{R}/data/mask_pool_af24_draws"
TARGET, N_PLANE = 1877, 256 * 176
TAGS = ("b", "c", "d")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    outs = [f"{R}/data/mask_vdpois_256x176_af24_d{t}.mat" for t in TAGS] + [f"{R}/data/mask_af24_d{t}_3d.h5" for t in TAGS]
    assert not any(os.path.exists(p) for p in outs), "draw masks exist; refusing to overwrite"
    with h5py.File(f"{R}/data/ksp_FLAIR.h5", "r") as f:          # symlink into the AF12 topic: READ ONLY
        sup = np.any(np.abs(np.array(f["kspace"][0])) > 0, axis=0)
    assert int(sup.sum()) == 34139
    a = scipy.io.loadmat(f"{R}/data/mask_vdpois_256x176_af24.mat")["mask"].astype(bool)
    assert int(a.sum()) == TARGET
    key = lambda m: hashlib.md5(np.packbits(m).tobytes()).hexdigest()
    seen, exact, n_draws = {key(a): "registered draw a"}, [], 0
    for p in sorted(glob.glob(f"{POOL}/draw_*.mat")):
        n_draws += 1
        d = scipy.io.loadmat(p); m = d["mask"].astype(bool)
        assert m.shape == (256, 176)
        assert (int(d["accely"].squeeze()), int(d["calib"].squeeze()), int(d["ellipse"].squeeze()), int(d["pp"].squeeze())) == (4, 24, 0, 1)
        k = key(m)
        if k in seen:
            print(f"[pool] {os.path.basename(p)} duplicates {seen[k]}, dropped"); continue
        seen[k] = os.path.basename(p)
        if int(m.sum()) == TARGET:
            exact.append((os.path.basename(p), round(float(d["accelx"].squeeze()), 4), m))
    assert len(exact) >= 3, f"only {len(exact)} exact draws in the pool"
    rep = dict(rule="three lowest-index draws with exactly 1877 points, duplicates and copies of draw a dropped, untrimmed",
               pool=POOL, pool_draws=n_draws, exact_draws=[e[0] for e in exact], draws={})
    masks = {"a": ("mask_vdpois_256x176_af24.mat (registered)", None, a)}
    for t, (name, accelx, m) in zip(TAGS, exact[:3]):
        masks[t] = (name, accelx, m)
        mat = f"{R}/data/mask_vdpois_256x176_af24_d{t}.mat"
        scipy.io.savemat(mat, {"mask": m.astype(np.float32)}, format="5", do_compression=True)
        with h5py.File(f"{R}/data/mask_af24_d{t}_3d.h5", "x") as h:
            h.create_dataset("mask", data=np.broadcast_to(m[None], (256, 256, 176)).astype(np.uint8),
                             compression="gzip", compression_opts=1)
    for t, (name, accelx, m) in masks.items():
        n_fl = int((m & sup).sum())
        r = dict(source=name, accelx=accelx, points=int(m.sum()), T1_AF=N_PLANE / int(m.sum()), FLAIR_points_on_data=n_fl,
                 FLAIR_AF=N_PLANE / n_fl, FLAIR_outside_shutter=int(m.sum()) - n_fl,
                 centre24=int(m[116:140, 76:100].sum()), shared_with_a=int((m & a).sum()))
        if t != "a":
            r["mat_md5"] = md5(f"{R}/data/mask_vdpois_256x176_af24_d{t}.mat")
            r["h5_md5"] = md5(f"{R}/data/mask_af24_d{t}_3d.h5")
        rep["draws"][t] = r
        print(t, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
    json.dump(rep, open(f"{R}/logs/prep_03b_pick_export_draws_af24.json", "w"), indent=1)
    print("wrote", f"{R}/logs/prep_03b_pick_export_draws_af24.json")


if __name__ == "__main__":
    main()
