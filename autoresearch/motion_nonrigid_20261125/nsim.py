#!/usr/bin/env python3
"""Data of the third version: one condition of ncommon.CONDITIONS3 (deformation only, no rigid part).

    nsim.py --cond DNa --order il            build into ROOT3/data/il/DNa/ (refuses to overwrite)
    nsim.py --cond DNa --order il --check

The builder is the second version's msim.py, run on this study's conditions, field and data root (ncommon); its
code is not changed. The truth gains the states of the deformation (a lasting change of the field's amplitude opens
a state; the second version's states only know rigid steps). No clock value or duration is printed or written.
"""
import sys

import numpy as np

from ncommon import CONDITIONS3, FIELDS

import msim  # noqa: E402
import schedule  # noqa: E402


def deformation_states(events, frac, ref_train):
    """(state_of_train, amplitude of every state): the states of the lasting part of the field's amplitude, relative
    to the reference train; state 0 is the reference train's."""
    steps = [e for e in (events or []) if e["kind"] == "step"]
    lvl = schedule.soft_amp(steps, frac) if steps else np.zeros(len(frac))
    lvl = np.round(lvl - lvl[ref_train], 9)
    amps = [0.0] + sorted(set(lvl.tolist()) - {0.0})
    return [amps.index(v) for v in lvl.tolist()], amps


def _build(cond, order, src, small=False):
    out = _orig(cond, order, src, small)
    t = out["truth"]
    st, amps = deformation_states(CONDITIONS3[cond]["intra"], out["schedule"]["train_time_frac"], t["ref_train"])
    f = CONDITIONS3[cond]["field"]
    t.update(nr_state_of_train=st, nr_state_amp=amps, field=FIELDS[f] if f else None)
    return out


def _cond_of(argv):
    return argv[argv.index("--cond") + 1] if "--cond" in argv[:-1] else None


if __name__ == "__main__":
    cond = _cond_of(sys.argv)
    _orig = msim.build
    msim.build = _build
    if cond in CONDITIONS3 and CONDITIONS3[cond]["field"]:
        msim.SOFT = dict(FIELDS[CONDITIONS3[cond]["field"]])
    msim.main()
