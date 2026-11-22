"""The linearised shift of every echo train: M3's detection statistic.

For one echo train, the translation d (3 numbers, voxels) of the model volume that explains the train's residual
best in the least-squares sense, y ~ yhat * exp(-i 2 pi k.d / N) ~ yhat * (1 - i 2 pi k.d / N). A lasting shift of
the head moves d by the same vector for every train after it, whatever share of the scan it covers and wherever the
model sits between the two positions; the residual energy does not (it is equal for the two states when the model
sits half-way). d has the sign of the object's displacement: f(x - d).

Pure numpy except line_sums (torch, used inside the trainer). No third-party import at module level besides numpy.
"""
import numpy as np

import weights

KEYS = ("A0", "A1", "A2", "Q0", "Q1", "E")


def _kx(n):
    return (np.arange(n) - n // 2) / n


def line_sums_np(vol, csm, ksp_meas, mask, fft):
    """Per (ky, kz) line, summed over coils and the readout (first spatial axis), with yhat = mask * fft(csm * vol)
    and y = the measured data: A0 = sum |yhat|^2, A1 = sum kx |yhat|^2, A2 = sum kx^2 |yhat|^2,
    Q0 = sum Im(conj(yhat) y), Q1 = sum kx Im(conj(yhat) y), E = sum |y - yhat|^2. kx = k / N, centred.
    mask: the sampling mask (0 / 1), (ny, nz) or per coil (coils, nx, ny, nz)."""
    kx = _kx(vol.shape[0])[:, None, None]
    out = {k: 0.0 for k in KEYS}
    for c in range(csm.shape[0]):
        m = mask if np.ndim(mask) < 4 else mask[c]
        yh = fft(csm[c] * vol) * m
        y = ksp_meas[c] * m
        p = yh.real ** 2 + yh.imag ** 2
        q = np.imag(np.conj(yh) * y)
        r = y - yh
        out["A0"] = out["A0"] + p.sum(0)
        out["A1"] = out["A1"] + (kx * p).sum(0)
        out["A2"] = out["A2"] + (kx * kx * p).sum(0)
        out["Q0"] = out["Q0"] + q.sum(0)
        out["Q1"] = out["Q1"] + (kx * q).sum(0)
        out["E"] = out["E"] + (r.real ** 2 + r.imag ** 2).sum(0)
    return out


def line_sums(vol, csm, ksp_meas, mask, fft, chunk=4):
    """line_sums_np with torch tensors (no gradient), in coil chunks; returns numpy arrays (ny, nz).
    mask: the sampling mask (0 / 1), (ny, nz)-broadcastable or per coil (coils, nx, ny, nz) as the trainer holds
    it -- never the weighted mask of M2."""
    import torch
    with torch.no_grad():
        nx = vol.shape[0]
        kx = ((torch.arange(nx, device=vol.device) - nx // 2) / nx).to(torch.float64).reshape(1, nx, 1, 1)
        per_coil = mask.ndim == csm.ndim
        out = {k: None for k in KEYS}
        add = lambda k, v: out.__setitem__(k, v if out[k] is None else out[k] + v)
        for c in range(0, csm.shape[0], chunk):
            m = mask[c:c + chunk] if per_coil else mask
            yh = fft(vol.unsqueeze(0) * csm[c:c + chunk]) * m
            y = ksp_meas[c:c + chunk] * m
            p = (yh.real.pow(2) + yh.imag.pow(2)).double()
            q = (yh.real * y.imag - yh.imag * y.real).double()            # Im(conj(yhat) y)
            r = y - yh
            e = (r.real.pow(2) + r.imag.pow(2)).double()
            red = lambda t: t.sum(dim=(0, 1))
            add("A0", red(p)); add("A1", red(kx * p)); add("A2", red(kx * kx * p))
            add("Q0", red(q)); add("Q1", red(kx * q)); add("E", red(e))
    return {k: v.cpu().numpy() for k, v in out.items()}


def train_shift(sums, train_of, n_trains):
    """d[n_trains, 3] in voxels (axis order of the volume) from the line sums; NaN for a train without data.
    Normal equations of the linearised model: G d = b with G = (2 pi)^2 sum k k^T |yhat|^2 and
    b = -2 pi sum k Im(conj(yhat) y)."""
    train_of = np.asarray(train_of)
    ny, nz = train_of.shape
    ky = _kx(ny)[:, None] * np.ones((1, nz))
    kz = _kx(nz)[None, :] * np.ones((ny, 1))
    d = np.full((int(n_trains), 3), np.nan)
    for t in range(int(n_trains)):
        m = train_of == t
        a0 = sums["A0"][m]
        if not m.any() or a0.sum() <= 0.0:
            continue
        a1, a2, y, z = sums["A1"][m], sums["A2"][m], ky[m], kz[m]
        q0, q1 = sums["Q0"][m], sums["Q1"][m]
        G = np.array([[a2.sum(), (y * a1).sum(), (z * a1).sum()],
                      [(y * a1).sum(), (y * y * a0).sum(), (y * z * a0).sum()],
                      [(z * a1).sum(), (y * z * a0).sum(), (z * z * a0).sum()]]) * (2.0 * np.pi) ** 2
        b = -2.0 * np.pi * np.array([q1.sum(), (y * q0).sum(), (z * q0).sum()])
        try:
            d[t] = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            pass
    return d


def changepoints_vec(d, min_frac, cp_t, max_states):
    """Lasting changes of level of a vector series d[n, 3] over acquisition time: weights.changepoints's binary
    segmentation, where the split of a segment is the one of the component with the largest statistic.
    Returns the first train of every new state, ascending; at most max_states - 1."""
    d = np.asarray(d, dtype=np.float64)
    min_len = max(int(np.ceil(min_frac * d.shape[0])), 2)
    segments, found = [(0, d.shape[0])], []
    while len(found) < int(max_states) - 1:
        cands = []
        for a, b in segments:
            stat, k = max((weights._best_split(d[:, j], a, b, min_len) for j in range(d.shape[1])),
                          key=lambda sk: sk[0] if sk[1] is not None else -1.0)
            if k is not None and stat >= cp_t:
                cands.append((stat, k, a, b))
        if not cands:
            break
        stat, k, a, b = max(cands)
        found.append(int(k))
        segments.remove((a, b))
        segments += [(a, k), (k, b)]
    return sorted(found)


def state_shifts(d, state_of_train, n_states):
    """Translation of every state against the reference state (state 0), voxels: difference of the medians of d."""
    d, state = np.asarray(d, dtype=np.float64), np.asarray(state_of_train)
    ref = np.nanmedian(d[state == 0], axis=0)
    return np.stack([np.nanmedian(d[state == k], axis=0) - ref for k in range(int(n_states))])
