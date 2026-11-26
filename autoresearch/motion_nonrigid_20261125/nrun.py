#!/usr/bin/env python3
"""Launcher of the third version (runs on amax, inside a gpu.py job): the second version's mrun.py on this study's
root, conditions and arms, with npatch.py as the child's patch and the M4 settings in the motion config.

    nrun.py --af af12 --order il --cond DNc --arm N4 --gpu 3 --settings ../motion_nonrigid_20261125/settings_dev1.json [--smoke] [--tag d1]
    nrun.py --dry --all

The truth file a run reads is derived from the condition's: the states are the deformation's (the second version's
truth only knows rigid states), their poses are the identity, and a train is wrong within its state when its
amplitude differs from its state's.
"""
import json
from pathlib import Path

import ncommon
from ncommon import ARMS3

import mrun  # noqa: E402

HERE = Path(__file__).resolve().parent
_command, _motion_config = mrun.command, mrun.motion_config


def derived_truth(truth):
    """The truth with the deformation's states in the fields the second version reads."""
    if "nr_state_of_train" not in truth:
        return truth
    t = dict(truth)
    st, amps = truth["nr_state_of_train"], truth["nr_state_amp"]
    amp = truth.get("soft_amp") or [0.0] * len(st)
    ident = dict(truth["state_poses"][0])
    t.update(state_of_train=list(st), state_poses=[ident] * len(amps), ref_state=0,
             contaminated_in_state=[bool(abs(a - amps[s]) > 1e-9) for a, s in zip(amp, st)])
    return t


def motion_config(af, order, cond, arm, seed, settings, smoke=False, tag=""):
    cfg = _motion_config(af, order, cond, arm, seed, settings, smoke=smoke, tag=tag)
    mode = ARMS3[arm]["m4"]
    cfg["m4"] = dict(settings["m4"], mode=mode) if mode else None
    src = Path(cfg["truth_path"])
    out = mrun.motion_path(cfg["exp"]).with_name(f"{cfg['exp']}_truth.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(derived_truth(json.loads(src.read_text())), indent=1))
    cfg["truth_path"] = str(out)
    return cfg


def command(*a, **k):
    exp, cmd = _command(*a, **k)
    return exp, [str(HERE / "npatch.py") if c.endswith("mpatch.py") else c for c in cmd]


if __name__ == "__main__":
    mrun.command, mrun.motion_config = command, motion_config
    mrun.main()
