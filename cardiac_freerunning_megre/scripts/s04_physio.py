"""Spec 5.4: PhysioInfo <-> ADC mapping (must be exact), steady state, respiratory thirds, polarity
criterion (a), cardiac phase. Writes spokes.npz and physio.json per subject."""
import json
import sys

import h5py
import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import DERIVED_ROOT, N_ECHO, N_SEG, RAW_ROOT, SUBJECTS, T_STEADY_MS, TOPIC  # noqa: E402
from cardiac_fr.physio import (expiration_end_by_histogram, load_physio, spoke_table,  # noqa: E402
                               third_at_end, timestamp_mismatch)

ok_all = True
for s in sys.argv[1:] or SUBJECTS:
    ph = load_physio(RAW_ROOT / s / "PhysioInfo.mat")
    with h5py.File(DERIVED_ROOT / s / "raw.h5", "r") as h:
        ts_ticks, lin = h["ts_ticks"][:], h["lin"][:]
    mis = timestamp_mismatch(ph["TimeStamp_ms"], ts_ticks)
    tab = spoke_table(ph, lin, N_ECHO, N_SEG, T_STEADY_MS)
    use = tab["steady"] & ~tab["is_si"]
    r0, resp = tab["lin"][use], tab["resp"][use]
    low, high = r0[third_at_end(resp, "low")], r0[third_at_end(resp, "high")]
    np.savez(DERIVED_ROOT / s / "spokes.npz", **tab, r0_ids=r0, low_ids=low, high_ids=high)
    rep = {"subject": s, "timestamp_max_abs_diff_ms": mis, "n_spoke": int(lin.size),
           "n_si": int(tab["is_si"].sum()), "n_before_steady": int((~tab["steady"]).sum()),
           "n_r0": int(r0.size), "n_third": int(low.size),
           "expiration_end_by_histogram": expiration_end_by_histogram(resp),
           "resp_p2_p50_p98": [float(np.percentile(resp, q)) for q in (2, 50, 98)],
           "rr_ms_median": float(np.median(np.diff(ph["cardTriggers"]))),
           "card_phase_nan": int(np.isnan(tab["card_phase"]).sum())}
    txt = json.dumps(rep, indent=1)
    (DERIVED_ROOT / s / "physio.json").write_text(txt)
    (TOPIC / "logs" / f"physio_{s}.json").write_text(txt)
    print(json.dumps(rep), flush=True)
    ok_all &= mis == 0.0
sys.exit(0 if ok_all else 1)
