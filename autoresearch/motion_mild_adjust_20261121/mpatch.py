#!/usr/bin/env python3
"""Runtime patch of the second version, applied in a child process to the frozen trainer copy.

    mpatch.py --motion FILE -- <cycle.py arguments>          (cwd = the trainer copy)

Every run of the study goes through this file, B0 included: there the per-echo-train statistic is only recorded
(mode "monitor") and the data term receives the trainer's own mask object, so the training is MC-GS unchanged.
M2: the T2 data term sum_n r_n becomes sum_n w_n r_n. The trainer's loss is sum |diff * mask|^2 / norm, so the
weights enter as mask * sqrt(w) with the fixed normalisation left alone; w carries no gradient and adds no parameter.
Nothing here prints or stores a clock value or a duration.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

import mcommon  # noqa: F401  (puts the first version's directory on sys.path)
import weights

REQUIRED = ("exp", "arm", "cond", "order", "af", "seed", "channel", "out_dir", "schedule_path", "truth_path",
            "brain_path", "m1", "m2", "m3", "reg", "native_shift_mm")
M2_MODES = ("monitor", "adaptive", "truth")
M2_KEYS = ("mode", "z0", "z1", "cap", "warmup", "every", "n_shells")
_ACTIVE = {}


def validate(cfg):
    missing = [k for k in REQUIRED if k not in cfg]
    assert not missing, f"motion config lacks {missing}"
    assert cfg["channel"] in ("plain", "contrast"), cfg["channel"]
    m2 = cfg["m2"]
    assert m2 is not None and not [k for k in M2_KEYS if k not in m2], f"m2 needs {M2_KEYS}"
    assert m2["mode"] in M2_MODES, m2["mode"]
    assert cfg["schedule_path"], "the per-echo-train statistic needs the schedule"
    assert m2["mode"] != "truth" or cfg["truth_path"], "truth weights need the motion truth"
    assert int(m2["every"]) >= 1 and int(m2["warmup"]) >= 1 and float(m2["z1"]) > float(m2["z0"])
    return cfg


# ------------------------------------------------------------------------------------------- pure pieces
def weight_map(w, train_of):
    """Weight of every (ky, kz) point: w of its echo train; 1 outside the acquired support (train -1)."""
    import torch
    wt = torch.as_tensor(np.asarray(w, dtype=np.float64), device=train_of.device)
    return torch.where(train_of >= 0, wt[train_of.clamp_min(0)], torch.ones((), dtype=wt.dtype, device=wt.device))


def weighted_mask(mask, wmap):
    """mask * sqrt(w): the trainer squares diff * mask, so this weights every point's residual energy by w.
    All-one weights return the mask object itself (bit-identical loss)."""
    if wmap is None or bool((wmap == 1).all()):
        return mask
    return mask * wmap.to(mask.dtype).sqrt()


def dc_keep_map(w, train_of, state_of_train=None, ref_state=0):
    """(ky, kz) map of the sampled lines that the scoring's hard data consistency may insert: trains with w >= 0.5
    that belong to the reference state; 1 outside the acquired support."""
    keep = np.asarray(w, dtype=np.float64) >= 0.5
    if state_of_train is not None:
        keep = keep & (np.asarray(state_of_train) == ref_state)
    return weight_map(keep.astype(np.float64), train_of)


def residual_energy(vol, csm, ksp_meas, mask, fft, chunk=4):
    """Residual energy of every (ky, kz) point (summed over coils and readout) and the sampled points, as numpy."""
    import torch
    with torch.no_grad():
        m = mask if mask.ndim == csm.ndim else mask.unsqueeze(0)
        e = None
        for c in range(0, csm.shape[0], chunk):
            mc = m if m.shape[0] == 1 else m[c:c + chunk]
            d = (fft(vol.unsqueeze(0) * csm[c:c + chunk]) - ksp_meas[c:c + chunk]) * mc
            p = (d.real.pow(2) + d.imag.pow(2)).double()
            p = p.sum(dim=tuple(range(p.ndim - 2)))
            e = p if e is None else e + p
        sampled = m.reshape(-1, *m.shape[-2:]).amax(0) > 0
    return e.cpu().numpy(), sampled.cpu().numpy()


class MMState:
    """Per-echo-train state of one training: weights, the residual statistic and its history."""

    def __init__(self, cfg, train_of, n_trains, truth=None):
        self.m2 = cfg["m2"]
        self.mode = self.m2["mode"]
        assert self.mode in ("monitor", "truth"), f"m2 mode {self.mode!r} is not available yet"
        self.train_of = np.asarray(train_of)
        self.n = int(n_trains)
        self.shells = weights.shell_index(self.train_of.shape, int(self.m2["n_shells"]))
        self.w = np.ones(self.n)
        self.state_of_train, self.ref_state = np.zeros(self.n, dtype=np.int64), 0
        self.rho, self.z, self.history, self.it = None, None, [], 0
        self._wmap, self._wmap_for = None, None
        if self.mode == "truth":
            assert truth is not None and "contaminated" in truth, "truth weights need the truth's contaminated trains"
            self.w = 1.0 - np.asarray(truth["contaminated"], dtype=np.float64)
            assert self.w.shape == (self.n,)

    def due(self):
        w0, ev = int(self.m2["warmup"]), int(self.m2["every"])
        return self.it >= w0 and (self.it - w0) % ev == 0

    def step(self, e2d_fn):
        """Once per evaluation of the T2 data term. e2d_fn() -> (energy per (ky, kz), sampled points); it is only
        called when the statistic is due. Returns True when the weights changed."""
        self.it += 1
        if not self.due():
            return False
        e2d, sampled = e2d_fn()
        self.rho = weights.train_stat(e2d, sampled, self.train_of, self.n, self.shells)
        self.z = weights.robust_z(self.rho, self.state_of_train)
        self.history.append(dict(it=self.it, n_down=int((self.w < 1.0).sum()), w_sum=float(self.w.sum()),
                                 z=[round(float(v), 4) for v in self.z]))
        return False

    def wmap_for(self, mask):
        """Weight map on the mask's device, or None while every weight is 1."""
        if (self.w == 1.0).all():
            return None
        key = (self.w.tobytes(), str(mask.device))
        if self._wmap_for != key:
            import torch
            self._wmap = weight_map(self.w, torch.from_numpy(self.train_of).long().to(mask.device))
            self._wmap_for = key
        return self._wmap

    def record(self):
        lst = lambda a: None if a is None else [None if not np.isfinite(v) else float(v) for v in a]
        return dict(mode=self.mode, settings={k: self.m2[k] for k in M2_KEYS}, rho=lst(self.rho), z=lst(self.z),
                    w=lst(self.w), history=self.history)


def make_kloss(orig, tag, fft):
    """Hook of JointTrainer._kspace_loss. Only the call of this run's motion channel is touched (the same test as the
    first version: ksp_meas is the trainer's own fit tensor of that channel)."""
    def kloss(self, vol, csm, ksp_meas, mask, freq_weight, norm=None):
        st = getattr(self, "_mm", None)
        if st is None or ksp_meas is not getattr(self, f"kspace_fit_{tag}", None):
            return orig(self, vol, csm, ksp_meas, mask, freq_weight, norm)
        st.step(lambda: residual_energy(vol, csm, ksp_meas, mask, fft))
        return orig(self, vol, csm, ksp_meas, weighted_mask(mask, st.wmap_for(mask)), freq_weight, norm)
    return kloss


# ------------------------------------------------------------------------------------------- patches
def install(cfg):
    import torch
    import trainers.joint_trainer as JT
    import trainers.single_contrast_trainer as ST
    from data.transforms import fft3c, ifft3c

    import patch as v1patch                                   # building blocks of the first version

    J, S = JT.JointTrainer, ST.SingleContrastTrainer
    ch = cfg["channel"]
    tag = "c" if ch == "contrast" else "p"
    orig = dict(setup=J._setup_optimizer, kloss=J._kspace_loss, load=JT._load_dataset, jtrain=J.train, strain=S.train)

    def load(data_cfg):
        return v1patch.load_with_reference_gt(orig["load"], data_cfg)

    def setup(self):
        sch = np.load(cfg["schedule_path"])
        truth = json.loads(Path(cfg["truth_path"]).read_text()) if cfg["truth_path"] else None
        st = MMState(cfg, sch["train_of"], int(sch["n_trains"]), truth)
        mask = getattr(self, f"mask_fit_{tag}")
        assert tuple(mask.shape[-2:]) == st.train_of.shape, (tuple(mask.shape), st.train_of.shape)
        self._mm = st
        _ACTIVE["tr"] = self
        print(f"[mm] arm={cfg['arm']} cond={cfg['cond']} order={cfg['order']} channel={ch} m2={st.mode} "
              f"trains={st.n} down-weighted at start={int((st.w < 1.0).sum())}", flush=True)
        orig["setup"](self)

    @torch.no_grad()
    def save(self):
        out, st = Path(cfg["out_dir"]), self._mm
        rec = dict(exp=cfg["exp"], arm=cfg["arm"], cond=cfg["cond"], order=cfg["order"], channel=ch, n_trains=st.n,
                   iterations=st.it, m2=st.record(), m3=None, reg=None)
        (out / f"{cfg['exp']}_mm.json").write_text(json.dumps(rec))
        c64 = lambda t: t.detach().cpu().numpy().astype("complex64")
        x = self._render(ch)
        arrays = dict(rec_raw=c64(x))
        if st.mode != "monitor":                              # the scorer's hard data consistency, kept lines only
            csm, mf = getattr(self, f"csm_{tag}"), getattr(self, f"mask_fit_{tag}")
            mf = (mf if mf.ndim == 3 else mf[0]).float()
            keep = dc_keep_map(st.w, torch.from_numpy(st.train_of).long().to(mf.device), st.state_of_train, st.ref_state)
            m = (mf * keep.to(mf.dtype)).unsqueeze(0)
            k_model = fft3c(csm * x.unsqueeze(0))
            k_dc = m * getattr(self, f"kspace_dc_{tag}") + (1.0 - m) * k_model
            arrays["rec_w_dc"] = c64((ifft3c(k_dc) * csm.conj()).sum(0))
        np.savez_compressed(out / f"{cfg['exp']}_canon.npz", **arrays)
        print(f"[mm] saved {cfg['exp']}_mm.json and {cfg['exp']}_canon.npz ({sorted(arrays)}); "
              f"statistic computed {len(st.history)} times, down-weighted trains {int((st.w < 1.0).sum())}", flush=True)

    def wrap_train(fn):
        def train(self, *a, **k):
            res = fn(self, *a, **k)
            save(self)
            return res
        return train

    JT._load_dataset = ST._load_dataset = load
    J.train, S.train = wrap_train(orig["jtrain"]), wrap_train(orig["strain"])
    J._setup_optimizer = setup
    J._kspace_loss = make_kloss(orig["kloss"], tag, fft3c)
    return orig


def main():
    import runpy
    i = sys.argv.index("--")
    assert sys.argv[1] == "--motion", "usage: mpatch.py --motion FILE -- <cycle.py arguments>"
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
