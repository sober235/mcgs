import numpy as np
import torch

import mpatch
import schedule

KY, KZ, N0, NT = 44, 30, 4, 110
M3 = dict(mode="detect", at=6, min_frac=0.10, cp_t=8.0, max_states=3, init="zero", lrs=dict(seg_theta=5e-4, seg_t=1e-2))


def _fft(x):
    return torch.fft.fftn(x, dim=(-3, -2, -1))


def _loss(self, vol, csm, ksp, mask, freq_weight, norm=None):
    diff = (_fft(vol.unsqueeze(0) * csm) - ksp) * mask.unsqueeze(0)
    return (diff.real.pow(2) + diff.imag.pow(2)).sum() / norm


def _toy(seed=0):
    sch = schedule.build(np.ones((KY, KZ), dtype=bool), etl=12, order="interleaved")
    assert sch["n_trains"] == NT
    g = torch.Generator().manual_seed(seed)
    mask = (torch.rand(KY, KZ, generator=g) < 0.5).double().unsqueeze(0).expand(N0, KY, KZ).contiguous()
    c = lambda *s: torch.complex(torch.randn(*s, generator=g, dtype=torch.float64),
                                 torch.randn(*s, generator=g, dtype=torch.float64))
    vol, csm = c(N0, KY, KZ), c(2, N0, KY, KZ)
    ksp = _fft(vol.unsqueeze(0) * csm) + 0.05 * c(2, N0, KY, KZ)
    return dict(train_of=sch["train_of"], mask=mask, vol=vol, csm=csm, ksp=ksp)


def _cfg(m3=M3, mode="monitor", warmup=2, every=2):
    return dict(m2=dict(mode=mode, z0=4.0, z1=8.0, cap=0.25, warmup=warmup, every=every, n_shells=6), m3=m3)


def _shifted(t, first, amount=0.3):
    """Measured data whose trains from `first` on do not fit the model any more (a lasting shift of the object):
    their lines get an extra random error, so the residual level of those trains is higher and scatters as it does
    in real data (a constant offset would give a scatter of zero inside the shifted state)."""
    sel = torch.from_numpy(t["train_of"] >= first)
    g = torch.Generator().manual_seed(99)
    ksp = t["ksp"].clone()
    n = int(sel.sum())
    ksp[:, :, sel] += amount * torch.complex(torch.randn(2, N0, n, generator=g, dtype=torch.float64),
                                             torch.randn(2, N0, n, generator=g, dtype=torch.float64))
    return ksp


class _Tr:
    def __init__(self, st, ksp):
        self._mm, self.kspace_fit_c, self.kspace_fit_p = st, ksp, ksp


def test_relabel_states_puts_the_reference_train_in_state_zero():
    s = mpatch.relabel_states(NT, [60], ref_train=13)
    assert (s[:60] == 0).all() and (s[60:] == 1).all()
    s = mpatch.relabel_states(NT, [60], ref_train=80)
    assert (s[:60] == 1).all() and (s[60:] == 0).all()
    s = mpatch.relabel_states(NT, [30, 70], ref_train=50)
    assert (s[:30] == 1).all() and (s[30:70] == 0).all() and (s[70:] == 2).all()
    assert (mpatch.relabel_states(NT, [], ref_train=5) == 0).all()


def _run(t, ksp, cfg, iters=8):
    st = mpatch.MMState(cfg, t["train_of"], NT, None, ref_train=13)
    calls, found = [], []
    hook = mpatch.make_kloss(_loss, "c", _fft,
                             render_state=lambda tr, s: calls.append((st.it + 1, s)) or t["vol"],
                             on_detect=lambda tr, state, new: found.append((state.it, list(new))))
    tr = _Tr(st, ksp)
    losses = [float(hook(tr, t["vol"], t["csm"], ksp, t["mask"], None, norm=1.0)) for _ in range(iters)]
    return st, calls, found, losses


def test_no_changepoint_means_no_states():
    t = _toy()
    st, calls, found, _ = _run(t, t["ksp"], _cfg())
    assert st.n_states == 1 and st.detected_at == 6 and st.boundaries == [] and calls == [] and found == []
    assert (st.state_of_train == 0).all() and st.record_m3()["n"] == 1


def test_detection_opens_a_state_and_renders_it_every_iteration():
    t = _toy(1)
    st, calls, found, _ = _run(t, _shifted(t, 60), _cfg())
    assert st.n_states == 2 and st.boundaries == [60] and found == [(6, [1])]
    assert (st.state_of_train[:60] == 0).all() and (st.state_of_train[60:] == 1).all()
    assert calls == [(7, 1), (8, 1)]                              # from the iteration after the detection on
    rec = st.record_m3()
    assert rec["mode"] == "detect" and rec["n"] == 2 and rec["detected_at"] == 6 and rec["boundaries"] == [60]
    assert rec["state_of_train"] == st.state_of_train.tolist()


def test_detection_only_happens_at_the_set_iteration():
    t = _toy(2)
    cfg = _cfg(dict(M3, at=5), warmup=2, every=2)                 # 5 is not an iteration of the weight schedule
    st, _, found, _ = _run(t, _shifted(t, 60), cfg)
    assert found == [(5, [1])] and st.detected_at == 5
    off = _cfg(None)
    st, calls, found, _ = _run(t, _shifted(t, 60), off)           # without M3 nothing is detected
    assert st.n_states == 1 and calls == [] and found == [] and st.record_m3() is None


def test_states_partition_the_mask():
    t = _toy(3)
    st, *_ = _run(t, _shifted(t, 60), _cfg(mode="adaptive"))
    st.w = np.where(np.arange(NT) % 9 == 0, 0.3, 1.0)             # some trains down-weighted in both states
    masks = st.state_masks(t["mask"])
    assert len(masks) == 2 and torch.equal(masks[0] + masks[1], t["mask"])
    wm = st.wmap_for(t["mask"])
    total = sum(mpatch.weighted_mask(m, wm) ** 2 for m in masks)
    assert torch.allclose(total, t["mask"] * wm, atol=1e-12)
    s2d = st.state_of_train[np.maximum(t["train_of"], 0)]
    assert not bool(masks[1][:, torch.from_numpy(s2d == 0)].any())


def test_state_loss_is_the_sum_over_states():
    t = _toy(4)
    ksp = _shifted(t, 60)
    st = mpatch.MMState(_cfg(dict(M3, at=1), warmup=1, every=1), t["train_of"], NT, None, ref_train=13)
    other = t["vol"] * 1.1                                         # what the model renders for state 1
    seen = []

    def orig(self, vol, csm, ksp_meas, mask, freq_weight, norm=None):
        seen.append((vol, mask))
        return _loss(self, vol, csm, ksp_meas, mask, freq_weight, norm)

    hook = mpatch.make_kloss(orig, "c", _fft, render_state=lambda tr, s: other)
    tr = _Tr(st, ksp)
    hook(tr, t["vol"], t["csm"], ksp, t["mask"], None, norm=1.0)   # iteration 1: detection, still one state
    assert st.n_states == 2
    seen.clear()
    got = hook(tr, t["vol"], t["csm"], ksp, t["mask"], None, norm=1.0)
    m = st.state_masks(t["mask"])
    want = _loss(None, t["vol"], t["csm"], ksp, m[0], None, 1.0) + _loss(None, other, t["csm"], ksp, m[1], None, 1.0)
    assert len(seen) == 2 and seen[0][0] is t["vol"] and seen[1][0] is other
    assert abs(float(got) - float(want)) < 1e-9 * float(want)


def test_statistic_uses_each_states_own_render():
    """After the states exist, a state whose render fits its data has a normal residual, so its trains are neither
    down-weighted nor reported as outliers."""
    t = _toy(5)
    ksp = _shifted(t, 60)
    st = mpatch.MMState(_cfg(dict(M3, at=1), mode="adaptive", warmup=1, every=1), t["train_of"], NT, None, ref_train=13)
    hook = mpatch.make_kloss(_loss, "c", _fft, render_state=lambda tr, s: t["vol"])
    tr = _Tr(st, ksp)
    hook(tr, t["vol"], t["csm"], ksp, t["mask"], None, norm=1.0)
    assert st.n_states == 2 and (st.w == 1.0).all()                # the weights start again when states open
    hook(tr, t["vol"], t["csm"], ksp, t["mask"], None, norm=1.0)
    in1 = st.state_of_train == 1
    assert int(in1.sum()) == 50 and np.abs(np.median(st.z[in1])) < 1.0          # z is taken within a state ...
    assert int((st.w[in1] < 1.0).sum()) <= 2                       # ... so the state is not down-weighted as a whole
    ungrouped = mpatch.weights.robust_z(st.rho)                    # (taken over all trains it would be: far above z1)
    assert np.median(ungrouped[in1]) > 8.0


def _truth():
    state = np.zeros(NT, dtype=int)
    state[66:] = 1                                                 # 40 % of the trains after a lasting shift
    cont = np.zeros(NT, dtype=bool)
    cont[66:] = True                                               # all of them moved relative to the reference
    resid = np.zeros(NT, dtype=bool)
    resid[66] = True                                               # one train was acquired during the change itself
    eye = dict(R=np.eye(3).tolist(), t=[0.0, 0.0, 0.0])
    moved = dict(R=np.eye(3).tolist(), t=[0.5, 0.0, 0.0])
    return dict(state_of_train=state.tolist(), state_poses=[eye, moved], ref_state=0, contaminated=cont.tolist(),
                contaminated_in_state=resid.tolist())


def test_truth_mode_uses_truth_states():
    t = _toy(6)
    cfg = _cfg(dict(M3, mode="truth"), mode="truth")
    st = mpatch.MMState(cfg, t["train_of"], NT, _truth(), ref_train=13)
    assert st.n_states == 2 and st.state_of_train.tolist() == _truth()["state_of_train"]
    assert st.w[66] == 0.0 and st.w.sum() == NT - 1                # only the train that is wrong within its state
    assert [p["t"] for p in st.truth_poses] == [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    calls = []
    hook = mpatch.make_kloss(_loss, "c", _fft, render_state=lambda tr, s: calls.append(s) or t["vol"])
    hook(_Tr(st, t["ksp"]), t["vol"], t["csm"], t["ksp"], t["mask"], None, norm=1.0)
    assert calls == [1]                                            # from the first iteration on
    # the reference state of the truth is not its state 0: the numbering follows the reference
    tr = _truth()
    tr["ref_state"] = 1
    inv = dict(R=np.eye(3).tolist(), t=[-0.5, 0.0, 0.0])
    tr["state_poses"] = [inv, dict(R=np.eye(3).tolist(), t=[0.0, 0.0, 0.0])]
    st = mpatch.MMState(cfg, t["train_of"], NT, tr, ref_train=80)
    assert (st.state_of_train[66:] == 0).all() and (st.state_of_train[:66] == 1).all()
    assert [p["t"] for p in st.truth_poses] == [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]]


def test_truth_weights_without_m3_stay_as_before():
    t = _toy(7)
    st = mpatch.MMState(_cfg(None, mode="truth"), t["train_of"], NT, _truth(), ref_train=13)
    assert st.n_states == 1 and st.w.sum() == NT - 44              # every contaminated train


def test_seg_groups_are_last_and_present_from_start():
    from adapter import MotionAdapter
    ad = MotionAdapter((32, 32, 24), True, True, 3, 0)
    for p in ad.parameters():
        p.requires_grad_(False)
    base = [dict(params=[torch.zeros(3, requires_grad=True)], lr=1e-3, name="positions")]
    groups = mpatch.state_groups(base, ad, dict(seg_theta=5e-4, seg_t=1e-2))
    assert [g["name"] for g in groups] == ["positions", "seg_theta", "seg_t"]
    assert groups[1]["params"][0] is ad.seg_theta and groups[1]["lr"] == 5e-4 and groups[2]["lr"] == 1e-2
    assert not ad.seg_theta.requires_grad                          # in the optimiser from the start, switched on later
    torch.optim.Adam(groups)                                       # the optimiser accepts the frozen groups
    assert mpatch.state_groups(base, None, {}) == base


# ---- the statistic of M2 and the detection statistic of M3 are settings
def _fftc(x):
    d = (-3, -2, -1)
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=d), dim=d), dim=d)


def _between(t, first, d=(0.0, 0.06, -0.04)):
    """Measured data of an object that moves by d voxels at train `first` and stays there, seen by a model that sits
    half-way between the two positions (what the training converges to): the residual energy of the two parts is
    then the same, only the direction of the misfit differs."""
    base = _fftc(t["vol"].unsqueeze(0) * t["csm"])
    k = [((torch.arange(n) - n // 2) / n).double() for n in (N0, KY, KZ)]
    kd = d[0] * k[0][:, None, None] + d[1] * k[1][None, :, None] + d[2] * k[2][None, None, :]
    late = torch.from_numpy(t["train_of"] >= first)
    g = torch.Generator().manual_seed(5)
    noise = 0.05 * torch.complex(torch.randn(2, N0, KY, KZ, generator=g, dtype=torch.float64),
                                 torch.randn(2, N0, KY, KZ, generator=g, dtype=torch.float64))
    return torch.where(late, base * torch.exp(-1j * np.pi * kd), base * torch.exp(1j * np.pi * kd)) + noise


def _run_c(t, ksp, cfg, iters=8):
    st = mpatch.MMState(cfg, t["train_of"], NT, None, ref_train=13)
    found = []
    hook = mpatch.make_kloss(_loss, "c", _fftc, render_state=lambda tr, s: t["vol"],
                             on_detect=lambda tr, state, new: found.append((state.it, list(new))))
    tr = _Tr(st, ksp)
    for _ in range(iters):
        hook(tr, t["vol"], t["csm"], ksp, t["mask"], None, norm=1.0)
    return st, found


def test_energy_statistic_is_blind_to_a_shift_the_model_sits_between():
    t = _toy(6)
    st, found = _run_c(t, _between(t, 60), _cfg())
    assert st.n_states == 1 and st.boundaries == [] and found == []
    assert st.record_m3()["settings"]["stat"] == "energy"                 # the default, as the runs before had it


def test_shift_statistic_finds_that_shift():
    t = _toy(6)
    st, found = _run_c(t, _between(t, 60), _cfg(dict(M3, stat="shift")))
    assert st.n_states == 2 and st.boundaries == [60] and found == [(6, [1])]
    rec = st.record_m3()
    assert rec["settings"]["stat"] == "shift" and np.shape(rec["shift"]) == (NT, 3)
    d = np.array(rec["shift"], dtype=float)
    step = np.nanmedian(d[60:], 0) - np.nanmedian(d[:60], 0)
    assert np.allclose(step, [0.0, 0.06, -0.04], atol=0.01), step


def test_shift_statistic_opens_nothing_without_motion():
    t = _toy(7)
    clean = _fftc(t["vol"].unsqueeze(0) * t["csm"]) + (t["ksp"] - _fft(t["vol"].unsqueeze(0) * t["csm"]))
    st, found = _run_c(t, clean, _cfg(dict(M3, stat="shift")))
    assert st.n_states == 1 and st.detected_at == 6 and found == [] and np.shape(st.record_m3()["shift"]) == (NT, 3)


def test_m2_statistic_kind_is_a_setting():
    t = _toy(8)
    out = {}
    for kind in (None, "sum", "median"):
        cfg = _cfg(None)
        if kind:
            cfg["m2"]["stat"] = kind
        st, _ = _run_c(t, _shifted(t, 60), cfg, iters=2)
        out[kind] = st
        assert st.record()["settings"]["stat"] == (kind or "sum")
    e2d, sampled = mpatch.residual_energy(t["vol"], t["csm"], _shifted(t, 60), t["mask"], _fftc)
    want = lambda kind: mpatch.weights.train_stat(e2d, sampled, t["train_of"], NT, out[None].shells, kind=kind)
    assert np.allclose(out[None].rho, want("sum")) and np.allclose(out["sum"].rho, want("sum"))
    assert np.allclose(out["median"].rho, want("median")) and not np.allclose(out["median"].rho, out["sum"].rho)
