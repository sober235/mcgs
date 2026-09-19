#!/usr/bin/env python3
"""prep_mask.py -- the retrospective AF-12 mask of this topic, and every file the arms read.

AF is the user's definition: sampled points / whole matrix, ACS counted. 304 x 104 = 31616 and
AF 12 give the budget 2635, all inside the acquired support (19914 points), i.e. 7.56x against
the data that exists. Writes the sampling mask (Omega), the union mask (Omega | Z) that every
non-GS arm uses as its forward operator, the support map the GS arms get as known_zero_support,
and the BART pattern for `pics -p`.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt")
from uih_mtp.retro_mask import (broadcast_readout, calib_square, load_draw_pool, load_support,
                                select_draw, trim_to_budget, write_mask_h5, write_mask_mat,
                                write_pattern_cfl, write_support_h5)

R = Path("/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro")
CONTRASTS = ("T1w", "T2s")
SHAPE = (304, 104)
N_RO = 320
CENTRE = (152, 52)
CALIB = 24
AF = 12
SEED = 0


def main():
    sup, zero, acs = load_support(R / f"data/masks_{CONTRASTS[0]}.npz")
    for c in CONTRASTS[1:]:
        s2, z2, a2 = load_support(R / f"data/masks_{c}.npz")
        assert np.array_equal(sup, s2) and np.array_equal(zero, z2) and acs == a2, \
            "the two contrasts must share the acquisition support"
    assert sup.shape == SHAPE, sup.shape
    budget = int(round(sup.size / AF))
    cal = calib_square(SHAPE, CENTRE, CALIB)
    assert bool(sup[cal].all()), "the calibration square must lie inside the acquired support"

    pool = load_draw_pool(R / "data/mask_draws")
    assert pool, "no draws: run scripts/prep_mask_draws.m first"
    meta, draw = select_draw(pool, sup, budget)
    omega = trim_to_budget(draw, sup, budget, CENTRE, protect=cal, seed=SEED)
    n_trim = int((draw & sup).sum()) - budget
    assert int(omega.sum()) == budget and not (omega & ~sup).any()
    n_cal_draw = int(draw[cal].sum())
    assert int(omega[cal].sum()) == n_cal_draw, "the trim removed calibration points"
    if n_trim > 0.10 * budget:
        raise SystemExit(f"trim removed {n_trim} of {budget} points (> 10 %): refine the accel sweep "
                         f"around {meta.get('accelx')} in steps of 0.01 and rerun")

    union = omega | zero
    write_mask_mat(R / "data/mask_af12_2d.mat", omega)
    write_mask_h5(R / "data/mask_af12_3d.h5", broadcast_readout(omega, N_RO))
    write_mask_h5(R / "data/mask_af12_union_3d.h5", broadcast_readout(union, N_RO))
    write_support_h5(R / "data/support_2d.h5", sup)
    write_pattern_cfl(str(R / "data/pattern_union"), union, N_RO)

    r = np.hypot(*(np.nonzero(omega)[i] - CENTRE[i] for i in (0, 1)))
    stats = dict(
        af_definition="sampled points / whole PE matrix, ACS counted (user 2026-09-12)",
        budget=budget, plane_points=int(sup.size), n_support=int(sup.sum()), n_zero=int(zero.sum()),
        AF_vs_full_grid=sup.size / budget, AF_vs_support=float(sup.sum()) / budget,
        omega_points=int(omega.sum()), omega_frac=float(omega.mean()),
        union_points=int(union.sum()), union_frac=float(union.mean()),
        calib24_samples=int(omega[cal].sum()), calib24_points=int(cal.sum()),
        calib24_in_chosen_draw=n_cal_draw,
        # house convention: vdPoisMex's `calib` biases the density near DC, it does NOT fully
        # sample a calib x calib block. The adult topics' masks sample 472 / 482 of their 576
        # calibration positions (full only to 16x16); this plane's draw samples fewer because it
        # is 304 x 104 at an isotropic acceleration. Harmless here: every arm uses the ESPIRiT
        # maps computed from the complete reference, so no arm self-calibrates from this mask.
        acs_lines=list(acs), centre=list(CENTRE), n_readout=N_RO, seed=SEED,
        chosen_draw=meta, pool_size=len(pool),
        trim=dict(n_trimmed=n_trim, kept_radius_max=float(r.max()), kept_radius_median=float(np.median(r))),
        generator="vdPoisMex(304,104,304,104, a, a, 24, 0, 1), scripts/prep_mask_draws.m",
        files={k: str(R / v) for k, v in dict(
            mask2d="data/mask_af12_2d.mat", mask3d="data/mask_af12_3d.h5",
            union3d="data/mask_af12_union_3d.h5", support="data/support_2d.h5",
            pattern="data/pattern_union.cfl").items()},
    )
    json.dump(stats, open(R / "logs/prep_mask.json", "w"), indent=2)
    # compatibility shim: the copied figure scripts read the AF label from this file
    json.dump({"contrasts": {c: {"AF_vs_own_support": float(sup.sum()) / budget,
                                 "AF_vs_full_grid": sup.size / budget,
                                 "support_frac": float(sup.mean()), "samples": budget}
                             for c in CONTRASTS}},
              open(R / "logs/prep_04_export.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
