"""PhysioInfo.mat (pilot tone) <-> imaging ADCs, steady state, respiratory gating (spec 5.4).

Row i of TimeStamp_ms / respSignal is imaging ADC i after the noise ADC is removed; rows are
spoke-major, echo-minor (verified on V07 against the MDH timestamps, 2.5 ms ticks)."""
from __future__ import annotations

import numpy as np
import scipy.io as sio


def load_physio(path) -> dict[str, np.ndarray]:
    m = sio.loadmat(str(path), squeeze_me=True)
    return {k: np.asarray(v, dtype=np.float64) for k, v in m.items() if not k.startswith("__")}


def timestamp_mismatch(ts_ms: np.ndarray, ts_ticks: np.ndarray) -> float:
    t = (np.asarray(ts_ticks, dtype=np.int64) - int(ts_ticks[0])) * 2.5
    if t.shape != np.shape(ts_ms):
        raise ValueError(f"row count differs: {np.shape(ts_ms)} vs {t.shape}")
    return float(np.max(np.abs(np.asarray(ts_ms) - t)))


def per_spoke(values: np.ndarray, n_echo: int) -> np.ndarray:
    return np.asarray(values).reshape(-1, n_echo).mean(axis=1)


def spoke_time_ms(ts_ms: np.ndarray, n_echo: int) -> np.ndarray:
    return np.asarray(ts_ms).reshape(-1, n_echo)[:, 0].copy()


def cardiac_phase(t_ms: np.ndarray, triggers_ms: np.ndarray) -> np.ndarray:
    trig = np.sort(np.asarray(triggers_ms, dtype=np.float64))
    k = np.searchsorted(trig, t_ms, side="right") - 1
    ph = np.full(np.shape(t_ms), np.nan)
    ok = (k >= 0) & (k < trig.size - 1)
    ph[ok] = (t_ms[ok] - trig[k[ok]]) / (trig[k[ok] + 1] - trig[k[ok]])
    return ph


def expiration_end_by_histogram(resp: np.ndarray, bins: int = 50) -> str:
    """Criterion (a): end-expiration is the longest plateau, so the histogram mode sits at that end.
    'high' if the mode lies above the robust mid-range (P2 + P98) / 2, else 'low'."""
    hist, edges = np.histogram(resp, bins=bins)
    k = int(np.argmax(hist))
    mode = 0.5 * (edges[k] + edges[k + 1])
    mid = 0.5 * (np.percentile(resp, 2) + np.percentile(resp, 98))
    return "high" if mode > mid else "low"


def dwell_asymmetry(resp: np.ndarray, frac: float = 0.2) -> float:
    """Strength of criterion (a): time spent in the top `frac` of the robust range (P2..P98) over the time spent in
    the bottom `frac`. End-expiration is a pause, so the ratio is well above 1 when the high end is end-expiration
    and well below 1 when the low end is. Spec 5.4 correction 7 (2026-09-12): when the image criterion (b) is
    below its 1-voxel resolution, (a) is adopted alone only if this ratio is >= 2 (high) or <= 0.5 (low)."""
    p2, p98 = np.percentile(resp, 2), np.percentile(resp, 98)
    rng = p98 - p2
    up = float(np.mean(resp > p98 - frac * rng))
    dn = float(np.mean(resp < p2 + frac * rng))
    return up / max(dn, 1e-12)


def third_at_end(resp: np.ndarray, end: str, frac: float = 1 / 3) -> np.ndarray:
    n_keep = int(round(frac * resp.size))
    order = np.argsort(resp, kind="stable")
    keep = order[-n_keep:] if end == "high" else order[:n_keep]
    m = np.zeros(resp.size, dtype=bool)
    m[keep] = True
    return m


def spoke_table(physio: dict, lin: np.ndarray, n_echo: int, n_seg: int, t_steady_ms: float) -> dict:
    t = spoke_time_ms(physio["TimeStamp_ms"], n_echo)
    lin = np.asarray(lin)
    return {"lin": lin, "shot": lin // n_seg, "seg_in_shot": lin % n_seg, "time_ms": t,
            "resp": per_spoke(physio["respSignal"], n_echo),
            "card_phase": cardiac_phase(t, physio["cardTriggers"]),
            "is_si": (lin % n_seg) == 0, "steady": t >= t_steady_ms}
