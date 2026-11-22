import json

import numpy as np
import pytest
import torch

import mpatch

SHAPE = (8, 12, 10)                                         # readout, ky, kz
N_TRAINS = 6


def _toy(seed=0):
    g = torch.Generator().manual_seed(seed)
    ky, kz = np.meshgrid(np.arange(12), np.arange(10), indexing="ij")
    train_of = ((ky * 10 + kz) % N_TRAINS).astype(np.int64)
    train_of[(ky < 2) & (kz < 2)] = -1                      # outside the acquired support (known-zero corner)
    sampled = (torch.rand(12, 10, generator=g) < 0.5).double()
    sampled[0, 0] = 1.0                                     # a fit point outside the support
    mask = sampled.unsqueeze(0).expand(*SHAPE).contiguous()
    c = lambda *s: torch.complex(torch.randn(*s, generator=g, dtype=torch.float64),
                                 torch.randn(*s, generator=g, dtype=torch.float64))
    return dict(train_of=train_of, mask=mask, vol=c(*SHAPE), csm=c(3, *SHAPE), ksp=c(3, *SHAPE))


def _fft(x):
    return torch.fft.fftn(x, dim=(-3, -2, -1))


def _loss(vol, csm, ksp, mask, norm):                       # the trainer's formula (joint_trainer._kspace_loss)
    diff = (_fft(vol.unsqueeze(0) * csm) - ksp) * mask.unsqueeze(0)
    return (diff.real.pow(2) + diff.imag.pow(2)).sum() / norm


def _cfg(mode, **over):
    m2 = dict(mode=mode, z0=4.0, z1=8.0, cap=0.25, warmup=4, every=2, n_shells=4)
    return dict(exp="x", arm="B0", cond="DTa", order="il", af="af12", seed=42, channel="contrast", out_dir="/tmp",
                schedule_path="s.npz", truth_path="t.json", brain_path="b.npy", m1=None, m2=m2, m3=None,
                reg=dict(ctrl_l2=0.0, ctrl_smooth=0.0, seg_l2=0.0, seg_smooth=0.0), native_shift_mm=[0.5, 0.1, -0.14],
                **over)


def test_weighted_mask_identity_bitexact():
    t = _toy()
    assert mpatch.weighted_mask(t["mask"], None) is t["mask"]
    assert mpatch.weighted_mask(t["mask"], torch.ones(12, 10, dtype=torch.float64)) is t["mask"]
    wmap = mpatch.weight_map(np.ones(N_TRAINS), torch.from_numpy(t["train_of"]))
    assert mpatch.weighted_mask(t["mask"], wmap) is t["mask"]
    a = _loss(t["vol"], t["csm"], t["ksp"], t["mask"], 7.0)
    b = _loss(t["vol"], t["csm"], t["ksp"], mpatch.weighted_mask(t["mask"], wmap), 7.0)
    assert torch.equal(a, b)


def test_weighted_loss_equals_weighted_sum():
    t = _toy(1)
    w = np.array([0.0, 0.25, 1.0, 1.0, 0.25, 0.0])
    tr = torch.from_numpy(t["train_of"])
    wmap = mpatch.weight_map(w, tr)
    assert wmap.shape == (12, 10) and float(wmap[0, 0]) == 1.0          # outside the support: weight 1
    got = _loss(t["vol"], t["csm"], t["ksp"], mpatch.weighted_mask(t["mask"], wmap), 5.0)
    diff = (_fft(t["vol"].unsqueeze(0) * t["csm"]) - t["ksp"]) * t["mask"].unsqueeze(0)
    e2d = (diff.real.pow(2) + diff.imag.pow(2)).sum(dim=(0, 1))          # per (ky, kz)
    want = e2d[tr < 0].sum() + sum(w[n] * e2d[tr == n].sum() for n in range(N_TRAINS))
    assert abs(float(got) - float(want) / 5.0) <= 1e-9 * abs(float(want))


def test_residual_energy_matches_the_loss_per_point():
    t = _toy(2)
    e2d, sampled = mpatch.residual_energy(t["vol"], t["csm"], t["ksp"], t["mask"], _fft)
    assert e2d.shape == (12, 10) and sampled.dtype == bool and sampled.shape == (12, 10)
    assert np.array_equal(sampled, t["mask"][0].numpy() > 0)
    assert abs(e2d.sum() - float(_loss(t["vol"], t["csm"], t["ksp"], t["mask"], 1.0))) <= 1e-9 * e2d.sum()


def test_truth_weights_zero_contaminated():
    t = _toy()
    truth = dict(contaminated=[False, True, False, False, True, False])
    st = mpatch.MMState(_cfg("truth"), t["train_of"], N_TRAINS, truth)
    assert st.w.tolist() == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    with pytest.raises(AssertionError):
        mpatch.MMState(_cfg("truth"), t["train_of"], N_TRAINS, None)       # truth weights need the truth


def test_monitor_schedule_and_history():
    t = _toy()
    st = mpatch.MMState(_cfg("monitor"), t["train_of"], N_TRAINS, None)
    calls = []

    def e2d_fn():
        calls.append(st.it)
        return np.ones((12, 10)), t["mask"][0].numpy() > 0

    for _ in range(9):
        changed = st.step(e2d_fn)
        assert changed is False and (st.w == 1.0).all()
    assert calls == [4, 6, 8]                                # warmup 4, every 2
    assert [h["it"] for h in st.history] == [4, 6, 8]
    assert all(len(h["z"]) == N_TRAINS and h["n_down"] == 0 and h["w_sum"] == float(N_TRAINS) for h in st.history)
    rec = st.record()
    assert rec["mode"] == "monitor" and len(rec["rho"]) == N_TRAINS and len(rec["w"]) == N_TRAINS
    json.dumps(rec)


class _FakeTrainer:
    def __init__(self, t, st):
        self.kspace_fit_c, self.kspace_fit_p = t["ksp"], t["ksp"].clone()
        self._mm = st
        self.seen = []


def test_monitor_never_touches_the_loss():
    t = _toy()
    st = mpatch.MMState(_cfg("monitor"), t["train_of"], N_TRAINS, None)
    sentinel = torch.tensor(3.0)

    def orig(self, vol, csm, ksp_meas, mask, freq_weight, norm=None):
        self.seen.append(mask)
        return sentinel

    hook = mpatch.make_kloss(orig, "c", _fft)
    tr = _FakeTrainer(t, st)
    for i in range(7):
        out = hook(tr, t["vol"], t["csm"], tr.kspace_fit_c, t["mask"], None, norm=2.0)
        assert out is sentinel and tr.seen[-1] is t["mask"]
    assert st.it == 7 and len(st.history) == 2               # iterations 4 and 6
    hook(tr, t["vol"], t["csm"], tr.kspace_fit_p, t["mask"], None, norm=2.0)      # the other contrast's call
    assert st.it == 7 and tr.seen[-1] is t["mask"]
    tr._mm = None
    assert hook(tr, t["vol"], t["csm"], tr.kspace_fit_c, t["mask"], None, norm=2.0) is sentinel


def test_truth_hook_passes_the_weighted_mask():
    t = _toy()
    truth = dict(contaminated=[False, True, False, False, False, False])
    st = mpatch.MMState(_cfg("truth"), t["train_of"], N_TRAINS, truth)
    seen = []

    def orig(self, vol, csm, ksp_meas, mask, freq_weight, norm=None):
        seen.append(mask)
        return _loss(vol, csm, ksp_meas, mask, norm)

    tr = _FakeTrainer(t, st)
    got = mpatch.make_kloss(orig, "c", _fft)(tr, t["vol"], t["csm"], tr.kspace_fit_c, t["mask"], None, norm=2.0)
    tro = torch.from_numpy(t["train_of"])
    assert torch.equal(seen[0][:, tro == 1], torch.zeros_like(seen[0][:, tro == 1]))
    assert torch.equal(seen[0][:, tro != 1], t["mask"][:, tro != 1])
    assert float(got) < float(_loss(t["vol"], t["csm"], t["ksp"], t["mask"], 2.0))


def test_dc_keep_map():
    t = _toy()
    w = np.array([1.0, 0.4, 0.5, 1.0, 0.0, 1.0])
    keep = mpatch.dc_keep_map(w, torch.from_numpy(t["train_of"]))
    tro = t["train_of"]
    assert keep.shape == (12, 10) and float(keep[0, 0]) == 1.0
    for n, want in enumerate([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]):
        assert (keep.numpy()[tro == n] == want).all(), n
    st = np.array([0, 0, 1, 1, 0, 0])
    keep = mpatch.dc_keep_map(w, torch.from_numpy(tro), state_of_train=st, ref_state=0)
    for n, want in enumerate([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]):
        assert (keep.numpy()[tro == n] == want).all(), n


def test_validate_rejects_missing_schedule():
    mpatch.validate(_cfg("monitor"))
    with pytest.raises(AssertionError):
        mpatch.validate(_cfg("monitor", **{}) | dict(schedule_path=None))
    with pytest.raises(AssertionError):
        mpatch.validate({k: v for k, v in _cfg("monitor").items() if k != "m2"})
    bad = _cfg("monitor")
    bad["m2"]["mode"] = "sometimes"
    with pytest.raises(AssertionError):
        mpatch.validate(bad)
    with pytest.raises(AssertionError):
        mpatch.validate(_cfg("truth") | dict(truth_path=None))


def test_validate_m1():
    ok = dict(reg_path="r.json", align=dict(start=300, end=1500, lr=5e-3, weight=1.0))
    mpatch.validate(_cfg("monitor") | dict(m1=ok))
    mpatch.validate(_cfg("monitor") | dict(m1=dict(reg_path="r.json", align=None)))
    for bad in (dict(align=None), dict(reg_path=None, align=None), dict(reg_path="r.json", align=dict(start=5, end=5, lr=1e-3, weight=1.0)),
                dict(reg_path="r.json", align=dict(start=0, end=5, lr=1e-3))):
        with pytest.raises(AssertionError):
            mpatch.validate(_cfg("monitor") | dict(m1=bad))
    with pytest.raises(AssertionError):                                    # M1 is for the joint model's T2W channel
        mpatch.validate(_cfg("monitor") | dict(m1=ok, channel="plain"))
