#!/usr/bin/env python
"""Noise floor of the k-space self-checks from the repeated first echo.

The raw file holds echo 0 twice per flip angle (set 0 and set 1, same lines, same readout). Half the
energy of their difference is the measurement noise, so for any set of lines
    floor = ||k_set1 - k_set0|| / (sqrt(2) ||k_set1||)
is the relative error a perfect predictor of those lines would still show. Reported for the whole
acquired set and for the lines hidden by the ACS hold-out check (odd lines 153..163), for both flip
angles. For the later echoes the noise is the same (same readout) while the signal decays, so their
floor is the echo-0 floor scaled by the energy ratio of the hidden lines.

    ./py noise_floor.py
"""
import json, os, sys, time

import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt")
from uih_mtp.uih_raw import read_lines   # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_mtp_grappa import INDEX, RAW, TOPIC, load_index   # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.sqrt(2.0) * np.linalg.norm(b)))


def main():
    idx = load_index(INDEX)
    acs = None
    out = {}
    for fa, u1 in ((16, 1), (4, 0)):
        t0 = time.time()
        k1 = read_lines(RAW, idx, set_=1, con=0, u1=u1, ro_crop=320)
        k0 = read_lines(RAW, idx, set_=0, con=0, u1=u1, ro_crop=320)
        m = idx.mask(set_=1, con=0, u1=u1)
        sel = idx.select(1, 0, u1)
        ref_pe = np.unique(idx.counters["pe"][sel & idx.ref]); acs = (int(ref_pe.min()), int(ref_pe.max()) + 1)
        mid = acs[0] + (acs[1] - acs[0]) // 2
        hidden = np.zeros_like(m)
        for y in range(mid, acs[1]):
            if y % 2 == 1:
                hidden[y] = m[y]
        res = dict(all_lines=rel(k1[:, :, m], k0[:, :, m]), hidden_acs_lines=rel(k1[:, :, hidden], k0[:, :, hidden]),
                   energy_hidden_echo0=float(np.linalg.norm(k1[:, :, hidden]) ** 2), seconds=time.time() - t0)
        # scale to the other echoes of the same flip angle: same noise, weaker signal on the hidden lines
        for echo in range(1, 5):
            ke = read_lines(RAW, idx, set_=1, con=echo, u1=u1, ro_crop=320)
            e = float(np.linalg.norm(ke[:, :, hidden]) ** 2)
            res[f"hidden_acs_lines_echo{echo}_scaled"] = res["hidden_acs_lines"] * np.sqrt(res["energy_hidden_echo0"] / e)
            del ke
        out[f"FA{fa}"] = res
        print(f"FA{fa}: floor all lines {res['all_lines']:.4f}, hidden ACS lines echo0 {res['hidden_acs_lines']:.4f}, "
              + ", ".join(f"echo{e} {res[f'hidden_acs_lines_echo{e}_scaled']:.4f}" for e in range(1, 5)), flush=True)
        del k0, k1
    out["acs"] = list(acs); out["hidden"] = "odd PE lines in the upper half of the ACS block"
    json.dump(out, open(os.path.join(TOPIC, "logs", "noise_floor.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
