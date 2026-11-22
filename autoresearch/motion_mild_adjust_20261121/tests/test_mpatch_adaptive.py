import numpy as np
import torch

import mpatch
import schedule

KY, KZ, N0 = 44, 30, 4


def _toy(seed=0):
    sup = np.ones((KY, KZ), dtype=bool)
    sch = schedule.build(sup, etl=12, order="interleaved")           # 1320 points, 110 trains of 12
    g = torch.Generator().manual_seed(seed)
    sampled = (torch.rand(KY, KZ, generator=g) < 0.5).double()
    mask = sampled.unsqueeze(0).expand(N0, KY, KZ).contiguous()
    c = lambda *s: torch.complex(torch.randn(*s, generator=g, dtype=torch.float64),
                                 torch.randn(*s, generator=g, dtype=torch.float64))
    vol, csm = c(N0, KY, KZ), c(2, N0, KY, KZ)
    ksp = _fft(vol.unsqueeze(0) * csm) + 0.05 * c(2, N0, KY, KZ)     # the model fits the data up to noise
    return dict(train_of=sch["train_of"], n=sch["n_trains"], mask=mask, vol=vol, csm=csm, ksp=ksp)


def _fft(x):
    return torch.fft.fftn(x, dim=(-3, -2, -1))


def _loss(self, vol, csm, ksp, mask, freq_weight, norm=None):
    diff = (_fft(vol.unsqueeze(0) * csm) - ksp) * mask.unsqueeze(0)
    return (diff.real.pow(2) + diff.imag.pow(2)).sum() / norm


def _cfg(warmup=600, every=100, mode="adaptive"):
    m2 = dict(mode=mode, z0=4.0, z1=8.0, cap=0.25, warmup=warmup, every=every, n_shells=6)
    return dict(m2=m2)


def _plant(t, trains, amount=3.0):
    """Offset the measured lines of some trains, as a movement during those trains would."""
    sel = torch.from_numpy(np.isin(t["train_of"], trains))
    ksp = t["ksp"].clone()
    ksp[:, :, sel] += amount
    return ksp


class _Tr:
    def __init__(self, st, ksp):
        self._mm, self.kspace_fit_c, self.kspace_fit_p = st, ksp, ksp


def test_adaptive_schedule():
    t = _toy()
    st = mpatch.MMState(_cfg(), t["train_of"], t["n"], None)
    ksp = _plant(t, [7, 8, 9])
    calls = []

    def e2d_fn():
        calls.append(st.it)
        return mpatch.residual_energy(t["vol"], t["csm"], ksp, t["mask"], _fft)

    changed = []
    for _ in range(1000):
        before = st.it
        assert before >= 600 or (st.w == 1.0).all()                  # nothing is down-weighted before the warm-up ends
        if st.step(e2d_fn):
            changed.append(st.it)
    assert calls == [600, 700, 800, 900, 1000]
    assert changed == [600]                                          # the same data give the same weights afterwards
    assert [h["it"] for h in st.history] == calls and st.history[0]["n_down"] == 3
    assert st.record()["mode"] == "adaptive"


def test_adaptive_downweights_planted_trains():
    t = _toy(1)
    bad = [20, 21, 22, 60, 61, 62]
    ksp = _plant(t, bad)
    st = mpatch.MMState(_cfg(warmup=2, every=2), t["train_of"], t["n"], None)
    hook = mpatch.make_kloss(_loss, "c", _fft)
    tr = _Tr(st, ksp)
    first = float(hook(tr, t["vol"], t["csm"], ksp, t["mask"], None, norm=1.0))        # iteration 1: all weights 1
    second = float(hook(tr, t["vol"], t["csm"], ksp, t["mask"], None, norm=1.0))       # iteration 2: recomputed, applied
    assert (st.w[bad] == 0.0).all() and (np.delete(st.w, bad) == 1.0).all()
    assert second < 0.5 * first


def test_weights_carry_no_gradient():
    t = _toy(2)
    ksp = _plant(t, [5, 6])
    st = mpatch.MMState(_cfg(warmup=1, every=1), t["train_of"], t["n"], None)
    vol = t["vol"].clone().requires_grad_(True)
    seen = []

    def orig(self, vol, csm, ksp_meas, mask, freq_weight, norm=None):
        seen.append(mask)
        return _loss(self, vol, csm, ksp_meas, mask, freq_weight, norm)

    loss = mpatch.make_kloss(orig, "c", _fft)(_Tr(st, ksp), vol, t["csm"], ksp, t["mask"], None, norm=1.0)
    assert (st.w < 1.0).any() and seen[0] is not t["mask"] and not seen[0].requires_grad
    loss.backward()
    assert vol.grad is not None and torch.isfinite(vol.grad.abs()).all() and float(vol.grad.abs().sum()) > 0.0


def test_single_trainer_channel():
    t = _toy(3)
    ksp_p, ksp_c = _plant(t, [30, 31]), t["ksp"]
    st = mpatch.MMState(_cfg(warmup=1, every=1), t["train_of"], t["n"], None)
    tr = _Tr(st, ksp_p)
    tr.kspace_fit_c = ksp_c
    hook = mpatch.make_kloss(_loss, "p", _fft)                        # the single-contrast trainer's only channel
    hook(tr, t["vol"], t["csm"], ksp_c, t["mask"], None, norm=1.0)
    assert st.it == 0
    hook(tr, t["vol"], t["csm"], ksp_p, t["mask"], None, norm=1.0)
    assert st.it == 1 and (st.w[[30, 31]] == 0.0).all()
