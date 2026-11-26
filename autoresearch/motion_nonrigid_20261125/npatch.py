#!/usr/bin/env python3
"""Runtime patch of the third version: the second version's mpatch.py (unchanged) with M4 added around it.

    npatch.py --motion FILE -- <cycle.py arguments>          (cwd = the trainer copy)

A run without the key m4 (B0, B2, B123, Ow) is the second version's run exactly. With m4:
  - the adapter gets one control-grid field per state (nadapter); the fields stay zero and frozen until M3 opens a
    state, then they learn together with the state's rigid pose (mode "learn"), or they are fixed to the truth's
    field from the first iteration (mode "truth");
  - for m4.freeze iterations after the detection the data term of the new states does not update the Gaussians
    (their render uses detached Gaussian parameters): only the state's pose and field learn, so the T2W density
    cannot absorb the deformation first. The reference state's data term trains the Gaussians as before;
  - a smoothness and an L2 term on the fields are added to the T2W data term.
Nothing here prints or stores a clock value or a duration.
"""
import json
import sys
from pathlib import Path

import numpy as np

import ncommon  # noqa: F401  (first: points the second version at this study)

import mpatch  # noqa: E402

M4_MODES = ("learn", "truth")
M4_KEYS = ("mode", "grid", "u_max_mm", "lr", "freeze", "l2", "smooth")


def validate(cfg):
    mpatch.validate(cfg)
    m4 = cfg.get("m4")
    if m4 is not None:
        assert cfg["m3"] is not None, "M4 lives on the states of M3"
        assert not [k for k in M4_KEYS if k not in m4], f"m4 needs {M4_KEYS}"
        assert m4["mode"] in M4_MODES and int(m4["grid"]) >= 2 and int(m4["freeze"]) >= 0
        assert (m4["mode"] == "truth") == (cfg["m3"]["mode"] == "truth"), "truth fields come with truth states"
    return cfg


class DetachedModel:
    """A view of the Gaussian model whose tensors (attributes and method results) are cut off from the graph."""

    def __init__(self, model):
        object.__setattr__(self, "_m", model)

    def __getattr__(self, name):
        import torch
        v = getattr(object.__getattribute__(self, "_m"), name)
        if torch.is_tensor(v):
            return v.detach()
        if callable(v):
            def call(*a, **k):
                r = v(*a, **k)
                return r.detach() if torch.is_tensor(r) else r
            return call
        return v


def frozen_now(st, m4):
    """True while the new states' data term must not update the Gaussians."""
    return m4["mode"] == "learn" and st.detected_at is not None and st.it <= st.detected_at + int(m4["freeze"])


def state_groups(m4):
    def groups(base_groups, adapter, lrs):
        if adapter is None or adapter.n_segments == 0:
            return base_groups
        full = dict(rigid_theta=0.0, rigid_t=0.0, nonrigid=0.0, seg_theta=lrs["seg_theta"], seg_t=lrs["seg_t"],
                    seg_ctrl=float(m4["lr"]))
        return list(base_groups) + [g for g in adapter.param_groups(full) if g["name"] in ("seg_theta", "seg_t", "seg_ctrl")]
    return groups


def make_kloss_factory(orig_make, m4, tag, log=print):
    def make_kloss(orig, tag_, fft, render_state=None, on_detect=None):
        def render_state4(tr, k):
            if not frozen_now(tr._mm, m4):
                return render_state(tr, k)
            real = tr.model
            tr.model = DetachedModel(real)
            try:
                return render_state(tr, k)
            finally:
                tr.model = real

        def on_detect4(tr, st, new):
            on_detect(tr, st, new)
            tr.motion.seg_ctrl.requires_grad_(True)
            log(f"[m4] fields of states {new} learn from iteration {st.it}; the Gaussians do not see these states' "
                f"data term until iteration {st.it + int(m4['freeze'])}", flush=True)

        kl = orig_make(orig, tag_, fft, render_state=render_state4, on_detect=on_detect4)

        def kloss(self, vol, csm, ksp_meas, mask, freq_weight, norm=None):
            loss = kl(self, vol, csm, ksp_meas, mask, freq_weight, norm)
            st, ad = getattr(self, "_mm", None), getattr(self, "motion", None)
            if (st is not None and ad is not None and st.n_states > 1 and ad.seg_ctrl.requires_grad
                    and ksp_meas is getattr(self, f"kspace_fit_{tag}", None)):
                loss = loss + ad.field_regulariser(float(m4["l2"]), float(m4["smooth"]), range(1, st.n_states))
            return loss
        return kloss
    return make_kloss


def truth_fields(cfg, shape):
    """{state: dense field (*shape, 3), mm} of the truth's states (amplitude of the state times the field)."""
    import simulate
    import warp
    truth = json.loads(Path(cfg["truth_path"]).read_text())
    f = truth["field"]
    brain = np.load(cfg["brain_path"]).astype(bool)
    u = warp.rbf_field(f["seed"], f["peak_mm"], brain, f["brain_max_mm"], shape=tuple(shape),
                       sigma_mm=simulate.FULL["sigma_mm"])
    return {k: a * u for k, a in enumerate(truth["nr_state_amp"]) if k > 0}


def install(cfg):
    m4 = cfg.get("m4")
    if m4 is None:
        return mpatch.install(cfg)
    import adapter
    import trainers.joint_trainer as JT

    import nadapter
    adapter.MotionAdapter = nadapter.field_adapter(int(m4["grid"]), float(m4["u_max_mm"]))
    tag = "c" if cfg["channel"] == "contrast" else "p"
    mpatch.state_groups = state_groups(m4)
    mpatch.make_kloss = make_kloss_factory(mpatch.make_kloss, m4, tag)
    orig = mpatch.install(cfg)
    J = JT.JointTrainer
    setup2 = J._setup_optimizer

    def setup(self):
        setup2(self)
        if m4["mode"] == "truth":
            for k, u in truth_fields(cfg, self.volume_shape).items():
                peak, mean = self.motion.set_field(k, u)
                print(f"[m4] truth field of state {k} on the {m4['grid']}^3 grid, largest control value {peak:.3f} mm, "
                      f"mean {[round(v, 3) for v in mean]} mm (in the state's pose), fixed", flush=True)
        print(f"[m4] mode={m4['mode']} grid={m4['grid']} u_max={m4['u_max_mm']} mm lr={m4['lr']} freeze={m4['freeze']} "
              f"l2={m4['l2']} smooth={m4['smooth']}", flush=True)

    J._setup_optimizer = setup
    return orig


def main():
    import os
    import runpy
    i = sys.argv.index("--")
    assert sys.argv[1] == "--motion", "usage: npatch.py --motion FILE -- <cycle.py arguments>"
    cfg = validate(json.loads(Path(sys.argv[2]).read_text()))
    cyc = sys.argv[i + 1:]
    tr_root = Path.cwd()
    assert (tr_root / "arc/cycle.py").is_file(), "run from the trainer copy"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cyc[cyc.index("--gpu") + 1]       # what cycle.py sets itself
    sys.path.insert(0, str(tr_root / "paired_gs"))
    install(cfg)
    sys.argv = [str(tr_root / "arc/cycle.py"), *cyc]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
