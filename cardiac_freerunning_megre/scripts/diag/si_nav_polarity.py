"""Diagnostic (2026-09-12): respiratory polarity from the SI self-navigation spokes, independent of the image
criterion (b). Per SI spoke (seg 0 of every interleave, echo 0) the 1D inverse FFT of the readout is the projection of
the object onto image axis2 (same direction as the image; superior = low index). The high-resp third's projection is
shifted against the low-resp third's per coil (gradient correlation, sub-voxel), and each shot's shift against the
mean projection is correlated with the PT respiratory value. Shift > 0 = towards the feet. If the high-resp third sits
towards the head, the high end of the PT signal is end-expiration.  Usage: si_nav_polarity.py <Vxx> [...]"""
import sys

import h5py
import numpy as np

D = "/data2/congcong/data/cardiac_3d_cine_public/derived/megre"
LAGS = np.arange(-10, 11)


def shift(a, b):                                                  # b(z) = a(z - d); returns d per row
    ga, gb = np.gradient(a, axis=-1), np.gradient(b, axis=-1)
    n = ga.shape[-1]
    c = np.array([np.sum(ga[..., max(0, -L):n - max(0, L)] * gb[..., max(0, L):n - max(0, -L)], axis=-1) for L in LAGS])
    i = np.argmax(c, axis=0)
    ii = np.clip(i, 1, len(LAGS) - 2)
    take = lambda k: np.take_along_axis(c, k[None], 0)[0]
    cm, c0, cp = take(ii - 1), take(ii), take(ii + 1)
    den = cm - 2 * c0 + cp
    f = np.where((i > 0) & (i < len(LAGS) - 1) & (den != 0), 0.5 * (cm - cp) / np.where(den == 0, 1, den), 0.0)
    return LAGS[i] + f


for s in sys.argv[1:]:
    sp = np.load(f"{D}/{s}/spokes.npz")
    sel = sp["is_si"] & sp["steady"]
    lin, resp = sp["lin"][sel], sp["resp"][sel]
    with h5py.File(f"{D}/{s}/raw.h5", "r") as h:
        y = h["ksp"][0][lin].copy()                               # (n, 24, 224)
    y[..., :3] = 0
    y[..., 221:] = 0
    proj = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(y, axes=-1), axis=-1), axes=-1))[..., 56:168]
    o = np.argsort(resp)
    n3 = resp.size // 3
    lo, hi = proj[o[:n3]].mean(0), proj[o[-n3:]].mean(0)          # (24, 112)
    d_third = shift(lo, hi)
    tmpl = proj.mean(0)
    d_shot = np.stack([shift(tmpl[c], proj[:, c, :]) for c in range(24)], axis=1)
    r = np.array([np.corrcoef(d_shot[:, c], resp)[0, 1] for c in range(24)])
    rss = np.sqrt((proj ** 2).sum(1))                             # (n, 112)
    d_rss = float(shift(rss[o[:n3]].mean(0), rss[o[-n3:]].mean(0)))
    r_rss = float(np.corrcoef(shift(rss.mean(0), rss), resp)[0, 1])
    strong = np.abs(r) > 0.5
    print(f"{s}: {resp.size} SI spokes. RSS projection: high third vs low third {d_rss:+.2f} vox (+ = feet), "
          f"per-shot shift vs PT resp r = {r_rss:+.3f}")
    print(f"   coils with |r| > 0.5: {int(strong.sum())}/24; among them median r {np.median(r[strong]) if strong.any() else float('nan'):+.3f}, "
          f"median third shift {np.median(d_third[strong]) if strong.any() else float('nan'):+.2f} vox, "
          f"signs of the third shift: {int((d_third[strong] < 0).sum())} head / {int((d_third[strong] > 0).sum())} feet")
    top = np.argsort(-np.abs(r))[:6]
    print("   strongest coils (coil, r, third shift):", ", ".join(f"({c}, {r[c]:+.2f}, {d_third[c]:+.2f})" for c in top))
    verdict = "high = end-expiration (superior)" if np.median(d_third[strong]) < 0 else "low = end-expiration"
    print(f"   -> navigator says: {verdict if strong.any() else 'undetermined'}", flush=True)
