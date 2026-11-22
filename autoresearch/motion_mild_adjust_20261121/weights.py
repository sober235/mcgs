"""Per-echo-train residual statistic, robust weights and change points (numpy, pure functions).

M2 of the second version: an echo train whose residual is far above what its k-space positions normally show is
down-weighted in the T2 data term. M3: a lasting change of the statistic's level over acquisition time marks a
persistent shift (a second state). Nothing here knows about the trainer."""
import numpy as np

_K = 1.4826                    # MAD -> standard deviation of a normal distribution


def _radius(shape2d):
    n1, n2 = shape2d
    ky, kz = np.meshgrid(np.arange(n1) - n1 // 2, np.arange(n2) - n2 // 2, indexing="ij")
    return np.sqrt((ky / (n1 / 2.0)) ** 2 + (kz / (n2 / 2.0)) ** 2)


def shell_index(shape2d, n_shells):
    """Shell number of every (ky, kz) point: equal steps of the normalised radius, 0 at the centre."""
    r = _radius(shape2d)
    return np.minimum((r / r.max() * n_shells).astype(np.int64), n_shells - 1)


STAT_KINDS = ("sum", "median")


def train_stat(e2d, sampled2d, train_of, n_trains, shells, kind="sum"):
    """rho_n: the residual energy of train n against the energy its sampled points normally show. The normal energy
    of a point is the median over the sampled points of its shell, interpolated in log-energy over the radius so that
    a train lying on one side of a shell is not biased. Trains without a sampled point: NaN.
    kind "sum": (energy of the train) / (normal energy of its points) -- a few strong lines carry it;
    kind "median": the median over the train's lines of energy / normal energy -- it moves only when most lines of
    the train are raised, which is what motion during a train does."""
    if kind not in STAT_KINDS:
        raise ValueError(f"unknown statistic {kind!r}; known: {STAT_KINDS}")
    e2d = np.asarray(e2d, dtype=np.float64)
    ok = np.asarray(sampled2d, dtype=bool) & (train_of >= 0)
    r = _radius(e2d.shape)
    centres, logs = [], []
    for s in np.unique(shells[ok]):
        sel = ok & (shells == s)
        med = float(np.median(e2d[sel]))
        if med > 0.0:
            centres.append(float(r[sel].mean()))
            logs.append(np.log(med))
    rho = np.full(int(n_trains), np.nan)
    if not centres:
        return rho
    expected = np.exp(np.interp(r[ok], centres, logs))
    if kind == "median":
        ratio, tr = e2d[ok] / expected, train_of[ok]
        order = np.argsort(tr, kind="stable")
        starts = np.searchsorted(tr[order], np.arange(int(n_trains) + 1))
        for t in range(int(n_trains)):
            if starts[t + 1] > starts[t]:
                rho[t] = np.median(ratio[order[starts[t]:starts[t + 1]]])
        return rho
    num = np.bincount(train_of[ok], weights=e2d[ok], minlength=int(n_trains))
    den = np.bincount(train_of[ok], weights=expected, minlength=int(n_trains))
    has = den > 0.0
    rho[has] = num[has] / den[has]
    return rho


def robust_z(rho, groups=None):
    """(rho - median) / (1.4826 MAD), within each group (state). NaN entries and groups with MAD = 0 give z = 0."""
    rho = np.asarray(rho, dtype=np.float64)
    groups = np.zeros(rho.shape, dtype=np.int64) if groups is None else np.asarray(groups)
    z = np.zeros(rho.shape)
    for g in np.unique(groups):
        sel = (groups == g) & np.isfinite(rho)
        if sel.sum() < 2:
            continue
        med = np.median(rho[sel])
        mad = np.median(np.abs(rho[sel] - med))
        if mad > 0.0:
            z[sel] = (rho[sel] - med) / (_K * mad)
    return z


def weights_from_z(z, z0, z1, cap):
    """1 up to z0, linear down to 0 at z1. At most floor(cap * n) trains are down-weighted: the ones with the
    largest z; the others return to 1."""
    z = np.asarray(z, dtype=np.float64)
    w = np.clip((z1 - z) / (z1 - z0), 0.0, 1.0)
    n_max = int(np.floor(cap * z.size))
    down = np.flatnonzero(w < 1.0)
    if down.size > n_max:
        keep = down[np.argsort(-z[down], kind="stable")[:n_max]]
        w2 = np.ones_like(w)
        w2[keep] = w[keep]
        w = w2
    return w


def _best_split(x, a, b, min_len):
    """Best split of x[a:b] into two levels: (statistic, index of the first element of the right part).
    The position is the one that minimises the summed absolute deviation from the two medians, searched without the
    length limit, so a change that leaves a part shorter than min_len is refused instead of being moved to the
    nearest allowed position. The statistic is computed at that position."""
    best_cost, k_best = None, None
    for k in range(a + 2, b - 1):
        left, right = x[a:k], x[k:b]
        left, right = left[np.isfinite(left)], right[np.isfinite(right)]
        if left.size < 2 or right.size < 2:
            continue
        cost = np.abs(left - np.median(left)).sum() + np.abs(right - np.median(right)).sum()
        if best_cost is None or cost < best_cost:
            best_cost, k_best = cost, k
    if k_best is None or k_best - a < min_len or b - k_best < min_len:
        return 0.0, None
    left, right = x[a:k_best], x[k_best:b]
    left, right = left[np.isfinite(left)], right[np.isfinite(right)]
    ml, mr = np.median(left), np.median(right)
    mad = np.median(np.concatenate([np.abs(left - ml), np.abs(right - mr)]))
    if mad <= 0.0:
        return 0.0, None
    return float(abs(ml - mr) / (_K * mad * np.sqrt(1.0 / left.size + 1.0 / right.size))), k_best


def changepoints(rho, min_frac, cp_t, max_states):
    """Lasting changes of level of rho over acquisition time (binary segmentation on medians). A split counts when
    its statistic |difference of medians| / (1.4826 pooled MAD sqrt(1/n1 + 1/n2)) reaches cp_t and both parts hold
    at least min_frac of all trains. Returns the first train of every new state, ascending; at most max_states - 1."""
    x = np.asarray(rho, dtype=np.float64)
    min_len = max(int(np.ceil(min_frac * x.size)), 2)
    segments, found = [(0, x.size)], []
    while len(found) < int(max_states) - 1:
        cands = [(*_best_split(x, a, b, min_len), a, b) for a, b in segments]
        cands = [c for c in cands if c[1] is not None and c[0] >= cp_t]
        if not cands:
            break
        stat, k, a, b = max(cands)
        found.append(int(k))
        segments.remove((a, b))
        segments += [(a, k), (k, b)]
    return sorted(found)


def auc(score, truth):
    """Area under the ROC curve of score for truth (bool); ties count half. NaN when one class is empty."""
    score, truth = np.asarray(score, dtype=np.float64), np.asarray(truth, dtype=bool)
    pos, neg = score[truth], score[~truth]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))
