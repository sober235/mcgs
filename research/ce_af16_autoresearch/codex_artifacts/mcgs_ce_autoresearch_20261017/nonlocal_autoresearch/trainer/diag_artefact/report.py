# diag_artefact/report.py
"""Assemble readings into verdicts. "insufficient" is a legitimate outcome and must stay reachable —
a proposition with no clean reading is reported as such rather than propped up by a number whose
calibration stake did not pass.

`adjudicate()` and the two figure/JSON helpers below are the brief's own "minimal implementation"
(task-9-brief.md Step 3), kept verbatim — its numeric thresholds (2.0 / 0.2 / 0.1) are the
pre-registered P4 rule, not something invented here. `run_all()` is the piece the brief's Step 3
does not ship code for: it wires M1-M5 and the controls into the `readings` dict `adjudicate`
consumes, computing P1-P3's own verdicts (P4's verdict is computed directly by `adjudicate` from
the raw air_curve/psf_peak_sidelobe readings `run_all` reports).

Design discipline carried over from `diag_artefact.metrics`/`.controls` (do not invent thresholds):
P1/P2/P3 each read a NULL-PERMUTATION band off the actual data (the same technique `surrogate_band`
already uses for autocorrelation, generalised here to plain NCC) rather than a hard-coded cutoff.
The one exception is P1's own literal rule (autocorrelation inside the surrogate band at every lag
>= 1) — that IS the pre-registered spec rule, kept as written; its own false-positive behaviour on
pure white noise is measured and reported alongside it, not used to adjust the rule.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import io as sio

from .jsonio import write_json
from .metrics import (
    air_energy_curve,
    autocorr_profile,
    gradient_correlation,
    half_voxel_shift,
    ncc,
    psf_spread,
    selectivity,
    surrogate_band,
)
from .volumes import brain_mask, wm_mask, load_volumes

MAX_LAG = 8
N_SURROGATE = 64
SURROGATE_SEED = 0

DEFAULT_MASK_PATH = ("/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask/"
                      "mask3d_vdpois_af12_transverse_ALIGNED.mat")
DEFAULT_RC_LAM000 = "results_arc/rc_lam000_volumes.npz"


# ---------------------------------------------------------------------------
# shared statistical primitive: a permutation-null band for NCC, the same
# technique `surrogate_band` already uses for autocorrelation.
# ---------------------------------------------------------------------------

def _permutation_null_ncc(field: np.ndarray, ref: np.ndarray, mask: np.ndarray,
                          n: int = N_SURROGATE, seed: int = SURROGATE_SEED) -> np.ndarray:
    """`n` values of ncc(permutation_i(field), ref, mask) — what correlation against `ref` looks
    like for a field with the SAME histogram as `field` but no spatial relation to anything. This is
    the "no genuine relationship" reference used to decide whether an observed ncc(field, ref) is
    "high" or "low" without inventing a constant.

    Two call patterns exist and are NOT the same thing (fix round 1 review, I5 — noted here so the
    docstring and every call site agree): P1 and P2 call this SELF-referentially
    (`field is ref`, e.g. `_permutation_null_ncc(err_rec, err_rec, wm)`) to ask "how correlated can
    a within-mask reshuffle of X be with X itself" — the reference for judging a SEPARATE, already-
    computed statistic (`ncc(err_rec, err_zf)`) against. P3 calls it with a genuinely different
    `ref` (the GT gradient) to null-test `ncc(err_rec, grad(gt))` directly. Both are valid uses of
    the same primitive; only the self-referential form needs an exchangeability caveat, because a
    within-mask reshuffle assumes the field has no spatial structure of its own to destroy — see
    `_residual_is_exchangeable` and `_circular_shift_null_ncc` below, added after fix round 1 found
    P3's residual is NOT exchangeable on real data, which silently understates this null's spread.
    """
    rng = np.random.default_rng(seed)
    vals = field[mask].astype(np.float64)
    scratch = np.zeros_like(field, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        scratch[mask] = rng.permutation(vals)
        out[i] = ncc(scratch, ref, mask)
    return out


N_SHIFT_NULL = 200


def _circular_shift_draws(field: np.ndarray, ref: np.ndarray, mask: np.ndarray, n: int, seed: int,
                          within_mask: bool):
    """`n` draws of ncc(circular_shift(field), ref, support), and the support size of each draw.

    `within_mask=True` zeroes `field` outside `mask` before rolling and scores each draw on
    `mask & roll(mask)`, so only genuine within-mask values are used; `within_mask=False` rolls the
    whole field and scores on `mask`, so values from outside the mask slide into it. Draws whose
    support comes out empty contribute a 0 to `supports` and no value.
    """
    rng = np.random.default_rng(seed)
    axes = tuple(range(field.ndim))
    source = np.where(mask, field.astype(np.float64), 0.0) if within_mask else field.astype(np.float64)
    values, supports = [], []
    for _ in range(n):
        shift = tuple(int(rng.integers(1, s)) for s in field.shape)
        support = (mask & np.roll(mask, shift=shift, axis=axes)) if within_mask else mask
        k = int(support.sum())
        supports.append(k)
        if k == 0:
            continue
        values.append(ncc(np.roll(source, shift=shift, axis=axes), ref, support))
    return np.asarray(values, dtype=np.float64), np.asarray(supports, dtype=np.int64)


def _circular_shift_null_ncc(field: np.ndarray, ref: np.ndarray, mask: np.ndarray,
                             observed: float | None = None, n: int = N_SHIFT_NULL,
                             seed: int = SURROGATE_SEED) -> dict:
    """A null for ncc(field, ref) that stays valid when `field` is NOT spatially exchangeable
    (fix round 1, C1) — the case `_permutation_null_ncc`'s within-mask reshuffle cannot handle,
    because reshuffling destroys structure the field really has and so UNDERSTATES the spread.
    A circular shift relocates the same values instead, preserving `field`'s own autocorrelation
    exactly while destroying its alignment with `ref`'s anatomy.

    TWO VARIANTS, BOTH REPORTED, THE WIDER ONE GATES. The final review asked for the within-mask
    variant (rolling the whole field slides out-of-mask values into the mask, so the null is partly
    a statement about a population the real reading never touches). Measuring it revealed the
    opposite failure on a sparse mask: WM is 1.05% of the volume and spatially concentrated, so the
    MEDIAN overlap of the WM mask with a randomly-rolled copy of itself is exactly ZERO voxels, and
    the few draws that do overlap score |ncc| up to 0.35 off a handful of voxels. So each variant
    has a distinct defect — the whole-field one uses the wrong values, the within-mask one uses the
    right values on a support that can be a tiny fraction of the real reading's — and neither is
    uniformly better:

        rim mask   (2.46M voxels, 17% of the volume): within-mask 0.085, whole-field 0.067
        WM mask    (151k voxels, 1.05%):              within-mask 0.121, whole-field 0.016

    Rather than pick per call site (which would be choosing the null after seeing which way it
    pointed), both are computed and the WIDER 97.5th percentile gates. That is conservative by
    construction — it can only make a reading harder to call significant, never easier — which is
    the right direction of error for a diagnosis whose spec prefers "insufficient" over a verdict
    resting on an uncalibrated number. Every number from both variants, including the support
    statistics that expose the sparse-mask degeneracy, is recorded.

    `n` is 200, not the 48 of fix round 1: a 97.5th percentile estimated from 48 draws is itself
    noisy at roughly the level of the gap it adjudicates.
    """
    n_mask = int(mask.sum())
    out = {"n_draws": int(n), "mask_n_voxels": n_mask}
    hi = {}
    for name, within in (("within_mask", True), ("whole_field", False)):
        values, supports = _circular_shift_draws(field, ref, mask, n=n, seed=seed,
                                                 within_mask=within)
        entry = {
            "n_valid_draws": int(values.size),
            "median_support_fraction": float(np.median(supports) / n_mask) if n_mask else 0.0,
        }
        if values.size:
            entry["hi97_5"] = float(np.percentile(np.abs(values), 97.5))
            entry["max_abs"] = float(np.abs(values).max())
            if observed is not None:
                entry["empirical_p"] = float((np.abs(values) >= abs(observed)).mean())
            hi[name] = entry["hi97_5"]
        else:
            entry["hi97_5"] = None
            entry["note"] = ("every draw left an empty support -- this variant cannot be "
                             "estimated on a mask this sparse")
        out[name] = entry

    if not hi:
        raise ValueError("neither circular-shift variant produced a usable draw")
    gating = max(hi, key=lambda k: hi[k])
    out["gating_variant"] = gating
    out["hi97_5"] = hi[gating]
    out["empirical_p"] = out[gating].get("empirical_p")
    out["note"] = ("both variants are reported; the WIDER 97.5th percentile gates (conservative by "
                   "construction). See _circular_shift_null_ncc's docstring for why neither "
                   "variant is uniformly correct.")
    return out


def _residual_is_exchangeable(err: np.ndarray, mask: np.ndarray, max_lag: int = MAX_LAG) -> bool:
    """Is `_permutation_null_ncc`'s exchangeability assumption valid for `err` inside `mask`?

    Exactly the same all-lags-inside-the-surrogate-band test P1 applies to `rec-gt` itself and to
    `zf-gt` as P1's own stake (`diag_artefact.metrics.autocorr_profile`/`surrogate_band`) — reused
    here as the validity gate for the null P3 depends on, rather than invented fresh.
    """
    profile = autocorr_profile(err, mask, max_lag=max_lag)
    lo, hi = surrogate_band(err, mask, n=N_SURROGATE, seed=SURROGATE_SEED, max_lag=max_lag)
    return bool(all(lo[lag] <= profile[lag] <= hi[lag] for lag in range(1, max_lag + 1)))


def _p1_all_lags_false_positive_rate(shape=(32, 32, 32), n_realizations: int = 20,
                                     max_lag: int = MAX_LAG, n_surrogate: int = N_SURROGATE) -> float:
    """How often does PURE WHITE NOISE (genuinely no structure) satisfy P1's own literal rule
    (autocorrelation inside the surrogate band at every lag >= 1) by chance alone? The band is a
    2.5/97.5-percentile band, so even structureless data escapes it at any one lag ~5% of the time,
    and the 8-lag conjunction fails often. Reproduces (2026-08-11, independently re-verified here
    rather than taken on faith) 11/20 = 55% — the false-positive rate quoted for this rule.
    """
    m = np.ones(shape, dtype=bool)
    passed = 0
    for i in range(n_realizations):
        rng = np.random.default_rng(3 + i)
        white = rng.normal(0, 1, shape)
        lo, hi = surrogate_band(white, m, n=n_surrogate, seed=0, max_lag=max_lag)
        p = autocorr_profile(white, m, max_lag=max_lag)
        ok = all(bool(lo[lag] <= p[lag] <= hi[lag]) for lag in range(1, max_lag + 1))
        passed += int(ok)
    return passed / n_realizations


# ---------------------------------------------------------------------------
# P1 — WM speckle = residual incoherent undersampling aliasing
# ---------------------------------------------------------------------------

def _p1(err_rec: np.ndarray, err_zf: np.ndarray, wm: np.ndarray) -> dict:
    profile = autocorr_profile(err_rec, wm, max_lag=MAX_LAG)
    lo, hi = surrogate_band(err_rec, wm, n=N_SURROGATE, seed=SURROGATE_SEED, max_lag=MAX_LAG)
    inside_per_lag = [bool(lo[lag] <= profile[lag] <= hi[lag]) for lag in range(1, MAX_LAG + 1)]
    inside_all_lags = all(inside_per_lag)

    r_rec_zf = ncc(err_rec, err_zf, wm)
    null = _permutation_null_ncc(err_rec, err_rec, wm)
    null_hi = float(np.percentile(np.abs(null), 97.5))
    ncc_high = bool(abs(r_rec_zf) > null_hi)

    # Stake: does the SAME rule read the expected way on zf's OWN residual -- the purest possible
    # aliasing case (mask only, no prior)? Section 10 established this mask is blue-noise/diffuse,
    # so its own aliasing should ALSO be incoherent (inside the band). If it is not, "inside band"
    # cannot be trusted as "this mask's aliasing signature" and the reading is discarded.
    zf_profile = autocorr_profile(err_zf, wm, max_lag=MAX_LAG)
    zf_lo, zf_hi = surrogate_band(err_zf, wm, n=N_SURROGATE, seed=SURROGATE_SEED, max_lag=MAX_LAG)
    zf_inside_all_lags = all(bool(zf_lo[lag] <= zf_profile[lag] <= zf_hi[lag])
                             for lag in range(1, MAX_LAG + 1))

    fp_rate = _p1_all_lags_false_positive_rate()

    if not zf_inside_all_lags:
        verdict = "insufficient"
        reason = ("stake failed: zf's own residual (pure aliasing, no prior) does not read "
                  "'inside band' under this exact rule, so 'inside band' cannot be trusted as "
                  "this mask's aliasing signature")
    elif inside_all_lags and ncc_high:
        verdict = "true"
        reason = "no long-range structure AND same-family as zf's aliasing beyond chance"
    elif (not inside_all_lags) and (not ncc_high):
        verdict = "false"
        reason = "structure present in the autocorrelation AND not same-family as zf's aliasing"
    else:
        verdict = "insufficient"
        reason = "the two sub-conditions disagree (mixed signal)"

    # Final review (I2): what does spec section 2's rule, applied LITERALLY as written, yield from
    # these numbers? It is a biconditional ("true iff A AND B") with no "insufficient" branch, so
    # it always answers. It is recorded ALONGSIDE `verdict`, never instead of it: the implemented
    # rule's extra "insufficient" branches encode spec section 3's discard discipline, which the
    # section 2 wording does not carry on its own.
    spec_literal = "true" if (inside_all_lags and ncc_high) else "false"
    spec_literal_note = (
        "spec section 2 P1: 'true iff rec-gt autocorrelation inside WM falls INSIDE the "
        "white-noise surrogate band AND its structure is same-family as zf-gt'. Measured: "
        f"inside_band_all_lags={inside_all_lags} (lag-1 autocorrelation {profile[1]:.4f} against a "
        f"surrogate band of [{lo[1]:.4f}, {hi[1]:.4f}]), same-family={ncc_high} "
        f"(ncc_rec_zf={r_rec_zf:.4f}). "
        + ("The implemented rule returns 'insufficient' instead because the calibration stake "
           "failed: zf's OWN residual -- pure aliasing, no prior -- does not read 'inside band' "
           "under this same rule either, so the first clause is not diagnostic and per spec "
           "section 3 it is discarded rather than used. Read this literal reading as 'the "
           "autocorrelation is not delta-like', NOT as 'the speckle is not aliasing'."
           if not zf_inside_all_lags else
           "The implemented rule agrees with the literal reading here.")
    )

    return {
        "verdict": verdict,
        "reason": reason,
        "spec_literal_verdict": spec_literal,
        "spec_literal_note": spec_literal_note,
        "autocorr_profile": profile.tolist(),
        "surrogate_band_lo": lo.tolist(),
        "surrogate_band_hi": hi.tolist(),
        "inside_band_all_lags": bool(inside_all_lags),
        "ncc_rec_zf": r_rec_zf,
        "ncc_self_permutation_null_hi97_5": null_hi,
        "ncc_high_relative_to_null": ncc_high,
        "stake_zf_inside_band_all_lags": bool(zf_inside_all_lags),
        "stake_zf_autocorr_profile": zf_profile.tolist(),
        "limitation_all_lags_false_positive_rate_on_white_noise": fp_rate,
    }


# ---------------------------------------------------------------------------
# P2 — outer-rim / skull-scalp band = aliasing (mask dependence, control B)
# ---------------------------------------------------------------------------

P2_ANATOMY_SIGMA = 4.0
P2_RIM_EROSION = 6


def _rim_mask(brain: np.ndarray, iterations: int = P2_RIM_EROSION) -> np.ndarray:
    """Spec section 2 P2's own region: the OUTER-RIM / skull-scalp band, i.e. the head mask minus
    its interior. `iterations` is the shell thickness in voxels."""
    from scipy import ndimage

    return brain & ~ndimage.binary_erosion(brain, iterations=int(iterations))


def _p2(rec: np.ndarray, gt: np.ndarray, zf: np.ndarray, brain: np.ndarray, contrast: str,
       ctrlB_path: str | None) -> dict:
    """P2's rule is spec section 2's: the band's error pattern must MOVE when the mask instance
    changes; if it does not move, it is not aliasing. Three things the final whole-branch review
    changed, none of them the rule:

    C1 -- THE NULL. This function used to gate on `_permutation_null_ncc(e_shipped, e_shipped, ...)`
    without checking the exchangeability that null assumes, while `_p3` refuses to use the same
    primitive until `_residual_is_exchangeable` passes. On the real volumes it does NOT pass
    (|rec-gt| autocorrelation at lags 1-4 is 0.189/0.139/0.118/0.103 against a permutation band of
    ~0.0002-0.0009), so P2's gate understated the null width by about two orders of magnitude and
    "high vs low" degenerated into "ncc != 0". The exchangeability check now runs here too, and
    when it fails the gate switches to the circular-shift null, which is valid regardless. BOTH
    nulls and the exchangeability reading are recorded, exactly as `_p3` records them. The verdict
    is unchanged by this (ncc 0.398 against a shift-null 97.5th percentile of ~0.085) -- that is
    the point: the conclusion survived a valid null, the gate did not.

    I4 -- THE SCOPE. The reading is now taken on the rim band spec section 2 names, not on the
    whole head. The whole-head reading is kept alongside (it answers a slightly different, broader
    question) but the verdict binds to the registered region.

    I1 -- THE STAKES. The anatomy control and the zf reference used to be reported and never
    consulted. Both now gate, and either failing yields "insufficient" rather than a verdict:
      * anatomy control: both error maps share a brain-shaped envelope, so a raw ncc could be high
        purely from that. The high-pass (sigma=4 residual) component must ALSO clear its own null,
        or the raw correlation is explained by shared anatomy and says nothing about the mask.
      * zf reference: `zf-gt` is this mask's pure aliasing with no prior. For "the pattern is
        indifferent to which mask instance drew it" to mean anything, a DIFFERENT mask instance's
        error must resemble the shipped error at least as much as the shipped mask's own aliasing
        reference does. Both are comparisons between measured quantities; no constant is invented.
    """
    from scipy import ndimage

    if not ctrlB_path or not os.path.exists(ctrlB_path):
        return {"verdict": "insufficient", "reason": "control B volumes not available"}

    vB = load_volumes(ctrlB_path, contrast)
    e_shipped = np.abs(rec.astype(np.float64) - gt.astype(np.float64))
    e_ctrlB = np.abs(vB["rec"].astype(np.float64) - vB["gt"].astype(np.float64))
    e_zf = np.abs(zf.astype(np.float64) - gt.astype(np.float64))

    rim = _rim_mask(brain)
    r = ncc(e_ctrlB, e_shipped, rim)

    # --- the two nulls, and the exchangeability reading that decides which one gates (C1)
    perm_null = _permutation_null_ncc(e_shipped, e_shipped, rim)
    perm_hi = float(np.percentile(np.abs(perm_null), 97.5))
    exchangeable = _residual_is_exchangeable(e_shipped, rim)
    shift_null = _circular_shift_null_ncc(e_shipped, e_ctrlB, rim, observed=r)
    shift_hi = shift_null["hi97_5"]

    gating_null = "permutation" if exchangeable else "circular_shift"
    null_hi = perm_hi if exchangeable else shift_hi
    low = bool(abs(r) <= null_hi)

    # --- stake 1: anatomy control, with its OWN null (I1)
    lp_shipped = ndimage.gaussian_filter(e_shipped, P2_ANATOMY_SIGMA)
    lp_ctrlB = ndimage.gaussian_filter(e_ctrlB, P2_ANATOMY_SIGMA)
    hp_shipped, hp_ctrlB = e_shipped - lp_shipped, e_ctrlB - lp_ctrlB
    r_lowpass = ncc(lp_shipped, lp_ctrlB, rim)
    r_highpass = ncc(hp_shipped, hp_ctrlB, rim)
    hp_null = _circular_shift_null_ncc(hp_shipped, hp_ctrlB, rim, observed=r_highpass)
    hp_null_hi = hp_null["hi97_5"]
    highpass_clears_null = bool(abs(r_highpass) > hp_null_hi)

    # --- stake 2: the zf pure-aliasing reference (I1)
    r_zf_vs_shipped = ncc(e_zf, e_shipped, rim)
    exceeds_zf_reference = bool(abs(r) > abs(r_zf_vs_shipped))

    # the same three readings on the whole head, for continuity with the pre-I4 record
    whole_brain = {
        "ncc_ctrlB_vs_shipped": ncc(e_ctrlB, e_shipped, brain),
        "ncc_highpass_ctrlB_vs_shipped": ncc(hp_shipped, hp_ctrlB, brain),
        "ncc_zf_vs_shipped": ncc(e_zf, e_shipped, brain),
        "note": ("whole-head (brain_mask) scope, the pre-I4 reading -- reported for continuity; "
                 "the verdict binds to the rim scope spec section 2 registers"),
    }

    # The two stakes qualify a HIGH correlation ("the pattern is indifferent to the mask"), which
    # is the only reading they can invalidate: if the raw ncc already sits inside its own null
    # there is no correlation for a shared envelope to have manufactured, and nothing to compare
    # against the zf reference. So they gate the "false" branch only.
    stakes_passed = bool(highpass_clears_null and exceeds_zf_reference)
    if low:
        verdict = "true"
        reason = ("rim error pattern is indistinguishable from its own null -> it moved with the "
                  "mask instance")
    elif stakes_passed:
        verdict = "false"
        reason = (f"rim error pattern correlates with the shipped mask's beyond chance "
                  f"(ncc={r:.4f} vs the {gating_null} null's 97.5th percentile {null_hi:.4f}) -> "
                  f"indifferent to the mask instance")
    else:
        verdict = "insufficient"
        failed = ([] if highpass_clears_null else ["anatomy control: the correlation does not "
                                                  "survive removing the shared low-frequency "
                                                  "envelope"]) + \
                 ([] if exceeds_zf_reference else ["zf reference: a different mask instance's "
                                                   "error resembles the shipped error LESS than "
                                                   "the shipped mask's own aliasing does"])
        reason = "stake failed -- " + "; ".join(failed)

    return {
        "verdict": verdict,
        "reason": reason,
        "spec_literal_verdict": "true" if low else "false",
        "spec_literal_note": (
            "spec section 2 P2: 'true iff the band's error pattern MOVES when the mask instance "
            "changes'. The implemented rule is that rule, so the two agree whenever the stakes "
            "pass; the literal reading here ignores only the stake gate."),
        "scope": f"outer rim: brain_mask minus binary_erosion(brain_mask, {P2_RIM_EROSION})",
        "scope_n_voxels": int(rim.sum()),
        "ncc_ctrlB_vs_shipped": r,
        "permutation_null_hi97_5": perm_hi,
        "residual_is_exchangeable": bool(exchangeable),
        "shift_null_hi97_5": shift_hi,
        "shift_null_empirical_p": shift_null["empirical_p"],
        "shift_null_detail": shift_null,
        "gating_null": gating_null,
        "gating_null_hi97_5": null_hi,
        "null_choice_note": (
            "the permutation null assumes the residual is spatially exchangeable; that assumption "
            "is tested here (residual_is_exchangeable) and, when it fails, the circular-shift null "
            "-- valid regardless -- is what the verdict is gated on. Both are recorded."),
        "low": low,
        "ctrlB_npz": ctrlB_path,
        "anatomy_control": {
            "sigma": P2_ANATOMY_SIGMA,
            "ncc_lowpass_ctrlB_vs_shipped": r_lowpass,
            "ncc_highpass_ctrlB_vs_shipped": r_highpass,
            "highpass_shift_null_hi97_5": hp_null_hi,
            "stake_highpass_clears_null": highpass_clears_null,
            "note": ("low-pass ncc is inflated by both error maps sharing the same brain-shaped "
                    "envelope; the high-pass component is the discriminating reading, and it must "
                    "clear its own shift null for the verdict to stand"),
        },
        "ncc_zf_vs_shipped": r_zf_vs_shipped,
        "stake_exceeds_zf_reference": exceeds_zf_reference,
        "zf_vs_ctrlB_note": ("a DIFFERENT mask instance's error (ctrlB) resembles the shipped "
                             "error MORE than the shipped mask's own pure-aliasing zf error does "
                             "when ncc_ctrlB_vs_shipped > ncc_zf_vs_shipped -- a stronger version "
                             "of the same conclusion, and a stake the verdict is gated on"),
        "stakes_passed": stakes_passed,
        "whole_brain_scope": whole_brain,
    }


# ---------------------------------------------------------------------------
# P3 — ventricle / midline streaks = displaced structure
# ---------------------------------------------------------------------------

def _p3(err_rec: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    """Fix round 1 (C1): `_permutation_null_ncc`'s within-mask reshuffle assumes `err_rec` is
    spatially exchangeable. That assumption is a STAKE, and it is tested directly here with
    `_residual_is_exchangeable` — exactly P1's own all-lags check, applied to the same `err_rec`
    and the same `mask` P3 uses. When it fails (as it does on real data: P1 independently records
    `autocorr_profile[1] = 0.228` for this exact residual, far outside a surrogate band of
    ±0.003-0.006), the permutation null is not trusted, and per spec §3 ("a metric that fails its
    stake is discarded on the spot") the verdict is `insufficient` — mirroring P1's own stake
    discipline exactly, not a new rule invented for P3. The circular-shift null (valid regardless
    of exchangeability — see `_circular_shift_null_ncc`) is always computed and reported alongside
    for transparency, but is NOT used to construct a substitute "true"/"false" call: doing so would
    just be a second, un-pre-registered rule rescuing a verdict the first one lost, which is exactly
    what the calibration discipline forbids. The half-voxel-shift positive control still shows the
    metric WOULD see a real displacement (see `tests/diag_artefact/test_report.py`), which is what
    licenses trusting an "insufficient" here as an honest gap rather than a broken measurement.
    """
    perm_valid = _residual_is_exchangeable(err_rec, mask)

    per_axis = {}
    any_significant_perm = False
    for a in range(3):
        g = np.gradient(gt.astype(np.float64), axis=a)
        real_corr = ncc(err_rec, g, mask)

        null = _permutation_null_ncc(err_rec, g, mask)
        null_hi = float(np.percentile(np.abs(null), 97.5))
        significant_perm = bool(abs(real_corr) > null_hi)
        any_significant_perm = any_significant_perm or significant_perm

        shift_null = _circular_shift_null_ncc(err_rec, g, mask, observed=real_corr)
        shift_null_hi = shift_null["hi97_5"]
        shift_empirical_p = shift_null["empirical_p"]
        significant_shift = bool(abs(real_corr) > shift_null_hi)

        shifted = half_voxel_shift(gt, axis=a) - gt
        shift_corr = gradient_correlation(shifted, gt, mask)[f"axis{a}"]

        per_axis[f"axis{a}"] = {
            "real_corr": real_corr,
            "permutation_null_hi97_5": null_hi,
            "significant_under_permutation_null": significant_perm,
            "shift_null_hi97_5": shift_null_hi,
            "shift_null_empirical_p": shift_empirical_p,
            "shift_null_detail": shift_null,
            "significant_under_shift_null": significant_shift,
            "half_voxel_shift_calibration": shift_corr,
            "real_over_shift_ratio": float(abs(real_corr) / (abs(shift_corr) + 1e-30)),
        }

    if not perm_valid:
        verdict = "insufficient"
        reason = ("stake failed: err_rec is not spatially exchangeable (escapes its own "
                  "surrogate band -- the same all-lags check P1 applies), so the permutation "
                  "null's exchangeability assumption is invalid and understates the true null "
                  "spread; the circular-shift null is valid and is reported per-axis, but per "
                  "spec section 3 the reading is discarded rather than used to construct a "
                  "different verdict")
    elif any_significant_perm:
        verdict = "true"
        reason = "at least one axis's gradient correlation clearly exceeds its own permutation-null (independent-noise) level, and the permutation null's exchangeability stake passed"
    else:
        verdict = "false"
        reason = "every axis's gradient correlation sits inside its own permutation-null band, and the permutation null's exchangeability stake passed"

    # Final review (I2), the highest-value line in this file for the person who asked the question:
    # spec section 2 fixes P3's stake as "non-zero on at least one axis BEYOND WHAT THE HALF-PIXEL-
    # SHIFT POSITIVE CONTROL CALIBRATES AS DETECTABLE". The implementation substituted a permutation
    # null (not pre-registered), then correctly discarded it when it failed validity -- which cost
    # the answer. The registered rule does not depend on that null at all: it compares the real
    # correlation against the calibration reading itself, and the calibration reading IS the
    # referent, so no threshold is invented by applying it. Both readings agree here.
    ratios = {a: per_axis[a]["real_over_shift_ratio"] for a in per_axis}
    max_ratio = max(ratios.values())
    at_or_above_calibration = bool(max_ratio >= 1.0)
    any_significant_shift = any(per_axis[a]["significant_under_shift_null"] for a in per_axis)
    spec_literal = "true" if at_or_above_calibration else "false"

    return {
        "verdict": verdict,
        "reason": reason,
        "spec_literal_verdict": spec_literal,
        "spec_literal_note": (
            "spec section 2 P3: 'true iff corr(rec-gt, d(gt)/dx) is non-zero on at least one axis "
            "beyond what the half-pixel-shift positive control calibrates as detectable'. "
            f"Measured: the largest |real_corr| / |half_voxel_shift_calibration| over the three "
            f"axes is {max_ratio:.4f} -- any coherent displacement present is at most "
            f"{100 * max_ratio:.1f}% of a detectable half-voxel shift, and no axis clears the "
            f"circular-shift null either (any_significant_under_shift_null="
            f"{any_significant_shift}). The implemented rule returned '{verdict}'; the "
            f"pre-registered rule needs no null at all and answers '{spec_literal}'."),
        "spec_literal_basis": {
            "real_over_shift_ratio_per_axis": ratios,
            "max_real_over_shift_ratio": max_ratio,
            "any_axis_at_or_above_calibration": at_or_above_calibration,
            "any_axis_significant_under_shift_null": bool(any_significant_shift),
        },
        "permutation_null_valid": perm_valid,
        "axes": per_axis,
    }


# ---------------------------------------------------------------------------
# P4 — background contains recon-introduced ghosts
# ---------------------------------------------------------------------------

def _load_mask2d(mask_path: str) -> np.ndarray:
    d = sio.loadmat(mask_path)
    m3 = np.asarray(d["mask"])
    return (m3[:, :, 0] != 0).astype(np.float64)


PSF_RADIUS_SWEEP = (2, 3, 4, 5, 6)  # documentation only (I1) -- adjudicate()'s own mainlobe_radius default (2) is untouched


def _p4(rec: np.ndarray, gt: np.ndarray, zf: np.ndarray, brain: np.ndarray,
       mask_path: str) -> dict:
    curve = air_energy_curve(rec, gt, brain)
    curve_zf = air_energy_curve(zf, gt, brain)  # stake: zf must read "full scale" if the metric is sensitive

    mask2d = _load_mask2d(mask_path)
    psf = psf_spread(mask2d)  # mainlobe_radius=2 (default, unchanged) -- this is the reading adjudicate() reads

    uniform2d = np.zeros_like(mask2d)
    uniform2d[::4, :] = 1.0
    psf_uniform = psf_spread(uniform2d)  # stake: positive control, must show a discrete replica

    # Fix round 1 (I1): document, do not act on, the fact that the r>2 reading sits one voxel
    # inside a wide ACS mainlobe skirt (24-px calib region). `mainlobe_radius` is NOT changed --
    # retuning it until adjudicate()'s pre-registered `< 0.1` bar flips is exactly what this plan
    # exists to prevent. The sweep is reported so a reader can see the radius the verdict is
    # blocked on, alongside the diffuse-vs-discrete comparison that needs no absolute constant.
    psf_radius_sweep = {r: psf_spread(mask2d, mainlobe_radius=r)["peak_sidelobe"]
                        for r in PSF_RADIUS_SWEEP}

    ratios = [row["ratio_rec"] for row in curve if np.isfinite(row["ratio_rec"])]
    zf_ratios = [row["ratio_rec"] for row in curve_zf if np.isfinite(row["ratio_rec"])]

    # --- the two stakes, now GATING (final review I1). Both are comparisons between measured
    # quantities -- no constant is introduced, and neither can turn "insufficient" into a verdict:
    # `adjudicate` consults `stake_gate.passed` only to DISCARD, per spec section 3.
    air_metric_is_sensitive = bool(len(zf_ratios) == len(ratios) and len(ratios) > 0 and
                                   all(z > r for z, r in zip(zf_ratios, ratios)))
    psf_control_is_discrete = bool(psf_uniform["peak_sidelobe"] > psf["peak_sidelobe"])
    stake_gate = {
        "air_curve_zf_reads_above_rec_at_every_threshold": air_metric_is_sensitive,
        "psf_uniform_control_more_discrete_than_shipped": psf_control_is_discrete,
        "passed": bool(air_metric_is_sensitive and psf_control_is_discrete),
        "note": ("stake 1: zf is this mask's pure aliasing with no prior, so it must read higher "
                 "air energy than the reconstruction at every threshold -- if it did not, the air "
                 "ratio would be blind to the very thing P4 asks about. stake 2: the uniform-mask "
                 "positive control must show a more discrete PSF than the shipped blue-noise "
                 "mask, or 'diffuse' is not being measured. A failure of either sends P4 to "
                 "insufficient in adjudicate(); neither can create a true/false."),
    }

    # --- I3: name the clause that actually blocks the verdict. `adjudicate` reaches insufficient
    # from the AIR RATIOS alone; `diffuse` is only ever consulted inside the `false` branch, which
    # the ratios have already closed, so the PSF reading cannot be what blocks P4.
    worst_dev = max(abs(r - 1.0) for r in ratios) if ratios else float("nan")
    reason = (
        f"air-energy ratio_rec = {['%.3f' % r for r in ratios]} across thresholds "
        f"{[row['threshold'] for row in curve]} (n_voxels {[row['n_voxels'] for row in curve]}). "
        f"adjudicate() is blocked by these ratios alone: the `true` branch needs every ratio > 2.0 "
        f"and the maximum here is {max(ratios):.3f}; the `false` branch needs every "
        f"|ratio - 1| < 0.2 and the worst is {worst_dev:.3f}. The PSF term NEVER ENTERS the branch "
        f"taken -- `diffuse` is read only inside the `false` branch, which the ratios already "
        f"closed, so a perfectly diffuse PSF would change nothing. SEPARATE CONTEXT (not the "
        f"blocking clause): psf_peak_sidelobe={psf['peak_sidelobe']:.4f} at the pre-registered "
        f"mainlobe_radius=2, {psf_radius_sweep.get(3):.4f} at r=3, against the uniform-mask "
        f"positive control's {psf_uniform['peak_sidelobe']:.4f} (sidelobe_rms "
        f"{psf['sidelobe_rms']:.4f} vs {psf_uniform['sidelobe_rms']:.4f}) -- diffuse without "
        f"needing any absolute constant. DIRECTION of the air reading: the ratio rises as the air "
        f"threshold tightens for BOTH the reconstruction and the zero-fill reference (a selection "
        f"effect -- air is defined by thresholding the GT, which shrinks the denominator without "
        f"constraining the numerator), and the reconstruction sits below the zero-fill reference "
        f"at every threshold ({['%.3f' % z for z in zf_ratios]}), i.e. the opposite of leakage."
    )

    spec_literal = "true" if all(r > 2.0 for r in ratios) else "false"

    return {
        "air_curve": curve,
        "psf_peak_sidelobe": psf["peak_sidelobe"],
        "psf_sidelobe_rms": psf["sidelobe_rms"],
        "stake_air_curve_zf": curve_zf,
        "stake_psf_uniform_mask_peak_sidelobe": psf_uniform["peak_sidelobe"],
        "stake_psf_uniform_mask_sidelobe_rms": psf_uniform["sidelobe_rms"],
        "stake_gate": stake_gate,
        "psf_radius_sweep": psf_radius_sweep,
        "mask_path": mask_path,
        "reason": reason,
        "spec_literal_verdict": spec_literal,
        "spec_literal_note": (
            "spec section 2 P4: 'true iff E_air(rec)/E_air(gt) >> 1 AND air-region error is "
            "same-family as zf's', with '>> 1' operationalised by the pre-registered rule as "
            f"every point on the curve above 2.0. Measured: the curve sweeps from "
            f"{max(ratios):.3f} at the tightest air threshold to {min(ratios):.3f} over the whole "
            "air mask, so it is not above 2 across the curve and the first conjunct fails -- which "
            "settles the conjunction whatever the second reads. The rise at the tightest threshold "
            "is a selection effect shared with the pure-aliasing reference, which sits ABOVE the "
            f"reconstruction at every threshold ({['%.3f' % z for z in zf_ratios]}): there is no "
            "threshold at which the reconstruction puts more energy into air than zero-fill does. "
            "The implemented rule returns 'insufficient' rather than 'false' because the ratios "
            "are also not flat at 1. Spec section 1's own decision box reads this as 'the "
            "background grain is in the GT too -> section 8b stands'; nothing here overturns "
            "section 8b."),
    }


# ---------------------------------------------------------------------------
# patchiness attribution — control A, two lambda arms
# ---------------------------------------------------------------------------

PATCHINESS_CURVE_Q = (0.10, 0.50, 0.90, 0.999)

# spec section 3's calibration-stakes table quotes this number directly: "rc_lam000 (no TV) ->
# patchiness metrics must read 'no patchiness' on it (section 14b measured: selectivity 1.20,
# whole ratio curve above 1)". It is used below ONLY as a comparison anchor (which of two measured
# distances is smaller) for an additional, informational check -- it does not gate
# `stake_discriminates`/`agree`/`attribution`, all of which stay exactly as before.
M5_REFERENCE_SELECTIVITY = 1.20


def _m5_reference_check(rc_selectivity: float, shipped_selectivity: float) -> dict:
    """Spec section 3's REGISTERED M5 stake is 'rc_lam000 must read NO patchiness', anchored to
    section 14b's own measurement (selectivity ~1.20, curve above 1 at every quantile) -- a
    property of rc_lam000's OWN reading. `stake_rc_lam000_selectivity_discriminates` (rc_sel <
    shipped_sel) approximates that with a DIRECTION PROXY relative to the run under test, which is
    a different, weaker question (it can pass even when rc_lam000 itself reads nothing like
    section 14b's reference). This checks the literal one, as a comparison between two measured
    distances -- no new absolute threshold is introduced.
    """
    rc_dist = abs(rc_selectivity - M5_REFERENCE_SELECTIVITY)
    shipped_dist = abs(shipped_selectivity - M5_REFERENCE_SELECTIVITY)
    return {
        "reference": M5_REFERENCE_SELECTIVITY,
        "rc_lam000_distance_from_reference": rc_dist,
        "shipped_distance_from_reference": shipped_dist,
        "rc_lam000_closer_to_reference_than_shipped": bool(rc_dist < shipped_dist),
    }


def _gauge_normalization_provenance_note(contrast: str, raw_agree: bool, agree: bool,
                                         d010_raw: float, d060_raw: float,
                                         d010_norm: float, d060_norm: float) -> str:
    """Fix (DEFECT 1a, honesty-fix-report): the note this replaces claimed the gauge normalisation
    "is conservative (it removes an attribution rather than creating one)" -- true of the T1 run,
    false of the FLAIR run (there it creates one: raw arms disagree, normalised arms agree). The
    direction is DATA-DEPENDENT, not conservative by construction, so it is computed per-run from
    the actual raw_agree/agree readings rather than asserted as a blanket claim -- a record that
    stated a fixed direction could silently go stale on a rerun; this cannot.
    """
    if raw_agree and not agree:
        effect = "REMOVED an attribution the raw comparison had asserted"
    elif (not raw_agree) and agree:
        effect = "CREATED an attribution the raw comparison did not support"
    elif raw_agree and agree:
        effect = "left the attribution direction unchanged (raw and gauge-normalised both agree)"
    else:
        effect = "left both readings without an attribution (raw and gauge-normalised both disagree)"
    return (
        "POST-HOC: dividing each arm's selectivity by its OWN zf anchor was introduced at fix "
        "round 1 (2026-08-11), after the raw comparison had already been computed. It is "
        "defensible (memory.md section 3's metric-gauge trap: each arm is a different "
        "acquisition, so their anchors are not interchangeable), but its DIRECTION OF EFFECT IS "
        "DATA-DEPENDENT, NOT conservative by construction -- it can CREATE an attribution the raw "
        f"arms did not support just as readily as it can remove one. On this run (contrast="
        f"{contrast}) it {effect}: raw deltas {d010_raw:+.4f}/{d060_raw:+.4f} -> "
        f"raw_agree={raw_agree}; gauge-normalised deltas {d010_norm:+.4f}/{d060_norm:+.4f} -> "
        f"agree={agree}. It was NOT pre-registered and is recorded as such here rather than only "
        "in a session log."
    )


def _attribution_not_for_citation_reasons(raw_agree: bool, d010_raw: float, d060_raw: float,
                                          d010_norm: float, d060_norm: float,
                                          reference_check: dict | None,
                                          rc_sel: float | None, shipped_sel: float | None) -> list:
    """Fix (DEFECT 1b, honesty-fix-report): why an attribution the `elif agree:` branch asserts may
    not be safe to cite, named explicitly in the JSON rather than left implicit in `agree=True`.
    Two independent reasons, both checked; either alone is enough to flag it.
    """
    reasons = []
    if not raw_agree:
        reasons.append(
            "created by the post-hoc gauge normalisation: the raw (un-gauge-corrected) arm deltas "
            f"disagree in sign ({d010_raw:+.4f} / {d060_raw:+.4f} -> raw_agree=False) while the "
            f"gauge-normalised ones agree ({d010_norm:+.4f} / {d060_norm:+.4f} -> agree=True) -- "
            "the normalisation is what produced this attribution, not what removed one.")
    if reference_check is not None and not reference_check["rc_lam000_closer_to_reference_than_shipped"]:
        reasons.append(
            "the M5 stake it passes (`stake_rc_lam000_selectivity_discriminates`, i.e. "
            "rc_sel < shipped_sel) is only a DIRECTION PROXY, not spec section 3's registered "
            f"stake -- 'rc_lam000 must read NO patchiness', anchored to section 14b's reference "
            f"selectivity of {M5_REFERENCE_SELECTIVITY:.2f} with the curve above 1 at every "
            f"quantile. Here rc_lam000's own selectivity ({rc_sel:.4f}) sits FURTHER from "
            f"{M5_REFERENCE_SELECTIVITY:.2f} than the shipped run's ({shipped_sel:.4f}) does, so "
            "under the spec's own wording the stake arguably fails here. (Its curve LEVEL does "
            "match section 14b's 'above 1 everywhere' signature -- that is the honest part of the "
            "evidence.)")
    return reasons


def _patchiness(rec: np.ndarray, gt: np.ndarray, wm: np.ndarray, contrast: str,
               ctrlA_af2_lam010_path: str | None, ctrlA_af2_lam060_path: str | None,
               rc_lam000_path: str, zf: np.ndarray, ctrlB_path: str | None = None) -> dict:
    """Fix round 1 (C2) rewrote this from a bare scalar-selectivity comparison to one that
    enforces the two stakes it already computed:

    1. `rc_lam000`'s stake reads 1.1988 against the shipped run's 1.1830 -- essentially the SAME
       scalar for "no TV, no patchiness by construction" and "the run under test". Per spec section
       3 that is a stake FAILURE for `selectivity` as a discriminator here: it is reported but not
       used to gate anything, and the full `gradient_ratio_curve` (whose LEVEL and SHAPE is what
       section 14b actually reads) is reported for every arm instead of trusting the scalar alone.
    2. Each arm's `zf`-anchor moves between acquisitions (AF=2's own zf reads far from the shipped
       AF=12 zf, while a second AF=12 draw, ctrlB, reads close to it) -- project memory section 3's
       metric-gauge trap. Every selectivity is gauge-normalised by ITS OWN zf-anchor before the two
       lambda arms are compared, and the brief's existing agree/disagree rule is applied to THAT
       normalised reading, not the raw one (also reported, for transparency, clearly labelled).
    """
    from .metrics import gradient_ratio_curve

    shipped_sel = selectivity(rec, gt, wm)
    shipped_zf_sel = selectivity(zf, gt, wm)
    out = {
        "shipped_selectivity": shipped_sel,
        "stake_zf_selectivity": shipped_zf_sel,
        "shipped_gradient_ratio_curve": gradient_ratio_curve(rec, gt, wm, quantiles=PATCHINESS_CURVE_Q),
    }

    # The M5 stake, spec section 3: the no-TV control must read LESS patchiness than the run under
    # test on whatever index is used to attribute patchiness. That is a DIRECTION, not a threshold
    # -- no constant is introduced by checking it (final review I1). It fails here, which is what
    # disqualifies the scalar as a discriminator.
    stake_discriminates = None
    reference_check = None
    if rc_lam000_path and os.path.exists(rc_lam000_path):
        v0 = load_volumes(rc_lam000_path, contrast)
        wm0 = wm_mask(v0["gt"], brain_mask(v0["gt"]))
        rc_sel = selectivity(v0["rec"], v0["gt"], wm0)
        stake_discriminates = bool(rc_sel < shipped_sel)
        reference_check = _m5_reference_check(rc_sel, shipped_sel)
        out["stake_rc_lam000_selectivity"] = rc_sel
        out["stake_rc_lam000_m5_reference_check"] = reference_check
        out["stake_rc_lam000_gradient_ratio_curve"] = gradient_ratio_curve(
            v0["rec"], v0["gt"], wm0, quantiles=PATCHINESS_CURVE_Q)
        out["stake_rc_lam000_vs_shipped_selectivity_delta"] = rc_sel - shipped_sel
        out["stake_rc_lam000_selectivity_discriminates"] = stake_discriminates
        out["stake_rc_lam000_scalar_note"] = (
            "near-zero delta: the scalar selectivity does NOT discriminate shipped from the "
            "no-TV control here -- discarded as a discriminator; read the curve LEVEL/SHAPE "
            "instead (rc_lam000 sits above 1 at every quantile per section 14b, shipped below). "
            "Final review (I1): this is now GATED, not merely noted -- when the stake fails, no "
            "attribution is asserted from the scalar, whichever way the two lambda arms happen to "
            "point. The curve level/shape is reported for every arm so a reader can see what the "
            "scalar could not.")
    else:
        out["stake_rc_lam000_selectivity"] = None
        out["stake_rc_lam000_selectivity_discriminates"] = None
        out["stake_rc_lam000_m5_reference_check"] = None

    have_both = (ctrlA_af2_lam010_path and ctrlA_af2_lam060_path and
                os.path.exists(ctrlA_af2_lam010_path) and os.path.exists(ctrlA_af2_lam060_path))
    if not have_both:
        out["attribution"] = "control A undecided"
        out["reason"] = "control A (AF=2, both lambda arms) volumes not available"
        return out

    v010 = load_volumes(ctrlA_af2_lam010_path, contrast)
    v060 = load_volumes(ctrlA_af2_lam060_path, contrast)
    wm010 = wm_mask(v010["gt"], brain_mask(v010["gt"]))
    wm060 = wm_mask(v060["gt"], brain_mask(v060["gt"]))

    sel010 = selectivity(v010["rec"], v010["gt"], wm010)
    sel060 = selectivity(v060["rec"], v060["gt"], wm060)
    curve010 = gradient_ratio_curve(v010["rec"], v010["gt"], wm010, quantiles=PATCHINESS_CURVE_Q)
    curve060 = gradient_ratio_curve(v060["rec"], v060["gt"], wm060, quantiles=PATCHINESS_CURVE_Q)

    # each arm's OWN zf-anchor (fix round 1, C2.3) -- not the shipped run's.
    zf010_sel = selectivity(v010["zf"], v010["gt"], wm010)
    zf060_sel = selectivity(v060["zf"], v060["gt"], wm060)

    # corroborating evidence the AF=2 zf-shift is acquisition-specific, not noise: a second AF=12
    # draw's own zf-anchor should stay close to the shipped AF=12 zf-anchor.
    # Final review (I1): ctrlB is also a free RUN-TO-RUN NOISE FLOOR for this index -- it is the
    # same acquisition, same lambda, same iterations, only a different mask draw, so whatever it
    # differs from the shipped run by is what "no change in patchiness" looks like on this scale.
    # Only its zf-anchor was being read; its own reconstruction's selectivity was not.
    ctrlB_zf_sel = ctrlB_sel = ctrlB_norm = None
    if ctrlB_path and os.path.exists(ctrlB_path):
        vB = load_volumes(ctrlB_path, contrast)
        wmB = wm_mask(vB["gt"], brain_mask(vB["gt"]))
        ctrlB_zf_sel = selectivity(vB["zf"], vB["gt"], wmB)
        ctrlB_sel = selectivity(vB["rec"], vB["gt"], wmB)
        ctrlB_norm = ctrlB_sel / ctrlB_zf_sel

    norm_shipped = shipped_sel / shipped_zf_sel
    norm010 = sel010 / zf010_sel
    norm060 = sel060 / zf060_sel
    d010_norm = norm010 - norm_shipped
    d060_norm = norm060 - norm_shipped
    agree = bool(d010_norm != 0 and d060_norm != 0 and np.sign(d010_norm) == np.sign(d060_norm))

    # raw (un-gauge-corrected) comparison kept ONLY for transparency -- NOT what `agree` is based
    # on; the metric-gauge trap this fix closes is exactly that the raw comparison looks like
    # agreement (both arms read lower than shipped) while the gauge-corrected one does not.
    d010_raw = sel010 - shipped_sel
    d060_raw = sel060 - shipped_sel
    raw_agree = bool(d010_raw != 0 and d060_raw != 0 and np.sign(d010_raw) == np.sign(d060_raw))

    noise_floor = None if ctrlB_norm is None else abs(ctrlB_norm - norm_shipped)

    out.update({
        "ctrlA_af2_lam010_npz": ctrlA_af2_lam010_path,
        "ctrlA_af2_lam060_npz": ctrlA_af2_lam060_path,
        "selectivity_ctrlA_af2_lam010": sel010,
        "selectivity_ctrlA_af2_lam060": sel060,
        "gradient_ratio_curve_ctrlA_af2_lam010": curve010,
        "gradient_ratio_curve_ctrlA_af2_lam060": curve060,
        "zf_selectivity_ctrlA_af2_lam010": zf010_sel,
        "zf_selectivity_ctrlA_af2_lam060": zf060_sel,
        "zf_selectivity_ctrlB": ctrlB_zf_sel,
        "selectivity_ctrlB": ctrlB_sel,
        "gauge_normalized_selectivity_shipped": norm_shipped,
        "gauge_normalized_selectivity_ctrlA_af2_lam010": norm010,
        "gauge_normalized_selectivity_ctrlA_af2_lam060": norm060,
        "gauge_normalized_selectivity_ctrlB": ctrlB_norm,
        "gauge_normalized_delta_lam010_vs_shipped": d010_norm,
        "gauge_normalized_delta_lam060_vs_shipped": d060_norm,
        "run_to_run_noise_floor_ctrlB_vs_shipped": noise_floor,
        "run_to_run_noise_floor_note": (
            "|gauge-normalised selectivity(ctrlB) - gauge-normalised selectivity(shipped)|. ctrlB "
            "is the SAME acquisition, lambda and iteration count as the shipped run with only a "
            "different AF=12 mask draw, so this is what 'no change in patchiness' measures as on "
            "this index. Compare it against the two arms' deltas before reading either as a "
            "signal: a delta smaller than this floor is not distinguishable from a re-draw."),
        "agree": agree,
        "raw_selectivity_delta_lam010_vs_shipped": d010_raw,
        "raw_selectivity_delta_lam060_vs_shipped": d060_raw,
        "raw_agree_UNCORRECTED_do_not_use_for_attribution": raw_agree,
        "gauge_normalization_provenance": _gauge_normalization_provenance_note(
            contrast, raw_agree, agree, d010_raw, d060_raw, d010_norm, d060_norm),
    })

    # Final review (I1): the attribution must not be computed from an index whose own calibration
    # stake failed. `agree` is still recorded (the sign rule itself is pre-registered and
    # untouched), but it is no longer allowed to assert an attribution when the M5 stake is down.
    if stake_discriminates is False:
        out["attribution"] = "control A undecided"
        out["reason"] = (
            "M5's calibration stake failed: the no-TV control rc_lam000 reads selectivity "
            f"{out['stake_rc_lam000_selectivity']:.4f} against the shipped run's {shipped_sel:.4f} "
            "-- the run that has no TV at all, and therefore no TV-induced patchiness by "
            "construction, scores HIGHER than the run under test. Per spec section 3 the scalar is "
            "discarded on the spot and may not support a verdict, so no attribution is asserted "
            "from it in either direction. For the record, the arms would not have agreed anyway "
            f"(gauge-normalised deltas {d010_norm:+.4f} / {d060_norm:+.4f})"
            + ("" if noise_floor is None else
               f", and the larger of them sits at {max(abs(d010_norm), abs(d060_norm)) / noise_floor:.2f}x "
               f"the run-to-run noise floor {noise_floor:.4f} measured from a second AF=12 draw")
            + ". Read the per-arm gradient_ratio_curve levels instead: section 14b's own "
            "classification is that a curve above 1 at every quantile is under-regularised "
            "roughness and a curve below 1 that rises with the quantile is selective suppression.")
    elif agree:
        out["attribution"] = ("less patchiness at AF=2" if d010_norm < 0 else
                              "patchiness persists or worsens at AF=2")
        # Fix (DEFECT 1b, honesty-fix-report): an attribution asserted here is not automatically
        # safe to cite -- name, in the JSON itself, whichever of the two known failure modes apply.
        _not_for_citation_reasons = _attribution_not_for_citation_reasons(
            raw_agree, d010_raw, d060_raw, d010_norm, d060_norm, reference_check,
            rc_sel if reference_check is not None else None,
            shipped_sel)
        out["attribution_not_for_citation"] = bool(_not_for_citation_reasons)
        out["attribution_not_for_citation_reasons"] = _not_for_citation_reasons
    else:
        out["attribution"] = "control A undecided"
        out["reason"] = ("gauge-normalised selectivity (each arm divided by its own zf anchor) "
                         "disagrees in direction between the two lambda arms, even though the "
                         "raw (un-gauge-corrected) selectivity agreed -- the raw agreement was "
                         "the metric-gauge trap (memory.md section 3), not a real signal")
    return out


# ---------------------------------------------------------------------------
# run_all — the brief's promised interface
# ---------------------------------------------------------------------------

def run_all(npz_path: str, contrast: str, controls: dict) -> dict:
    """Wire M1-M5 and the controls into the `readings` dict `adjudicate` consumes.

    `controls` maps logical names to *_volumes.npz paths (or overrides), all optional:
      "ctrlB"            -- control B (different mask seed), for P2 and the patchiness zf-gauge check
      "ctrlA_af2_lam010" -- control A, AF=2, lambda_geo=0.10, for the patchiness attribution
      "ctrlA_af2_lam060" -- control A, AF=2, lambda_geo=0.60, for the patchiness attribution
      "rc_lam000"        -- no-TV run, the M5 "no patchiness" calibration stake
      "mask_path"        -- shipped mask .mat, for M1's PSF (default: the ALIGNED AF=12 mask)
    Fix round 1: keys renamed from "ctrlA_lam010"/"ctrlA_lam060" to "ctrlA_af2_lam010"/
    "ctrlA_af2_lam060" -- the old names collided, in spelling only, with the manifest's OWN
    `ctrlA_lam010` key, which is the unrelated all-ones completed_vacuous run that must never be
    cited (I2).
    A control missing from `controls` (or whose file does not exist) degrades its proposition to
    "insufficient" / "control A undecided" rather than raising.
    """
    v = load_volumes(npz_path, contrast)
    rec, gt, zf = v["rec"], v["gt"], v["zf"]
    rec64, gt64, zf64 = rec.astype(np.float64), gt.astype(np.float64), zf.astype(np.float64)

    brain = brain_mask(gt)
    wm = wm_mask(gt, brain)

    err_rec = rec64 - gt64
    err_zf = zf64 - gt64

    readings = {
        "npz_path": npz_path,
        "contrast": contrast,
        "P1": _p1(err_rec, err_zf, wm),
        "P2": _p2(rec, gt, zf, brain, contrast, controls.get("ctrlB")),
        "P3": _p3(err_rec, gt, wm),
        "P4": _p4(rec, gt, zf, brain, controls.get("mask_path", DEFAULT_MASK_PATH)),
        "patchiness": _patchiness(rec, gt, wm, contrast,
                                  controls.get("ctrlA_af2_lam010"), controls.get("ctrlA_af2_lam060"),
                                  controls.get("rc_lam000", DEFAULT_RC_LAM000), zf,
                                  ctrlB_path=controls.get("ctrlB")),
    }
    return readings


# ---------------------------------------------------------------------------
# adjudicate / write_report / adjudication_figure — task-9-brief.md Step 3, verbatim
# ---------------------------------------------------------------------------

def adjudicate(readings: dict) -> dict:
    v = {}
    p4 = readings.get("P4")
    if p4:
        # Final review (I1): P4's own calibration stakes must be able to reach its verdict, or
        # they are decoration. This guard is a spec section 3 DISCARD path and nothing else -- it
        # can only send P4 to "insufficient", never to true or false, and the pre-registered rule
        # below (2.0 / 0.2 / 0.1) is untouched and still the only thing that can produce a verdict.
        if p4.get("stake_gate", {}).get("passed") is False:
            v["P4"] = "insufficient"
        else:
            ratios = [r["ratio_rec"] for r in p4["air_curve"] if np.isfinite(r["ratio_rec"])]
            diffuse = p4["psf_peak_sidelobe"] < 0.1
            if ratios and all(r > 2.0 for r in ratios):
                v["P4"] = "true"
            elif ratios and all(abs(r - 1.0) < 0.2 for r in ratios) and diffuse:
                v["P4"] = "false"
            else:
                v["P4"] = "insufficient"
    for key in ("P1", "P2", "P3"):
        r = readings.get(key)
        v[key] = r.get("verdict", "insufficient") if isinstance(r, dict) else "insufficient"
    return v


def _p1_literal_support(r: dict) -> tuple:
    """DEFECT 2 (honesty-fix-report): is P1's `spec_literal_verdict` supported, i.e. NOT resting on
    a clause its own calibration stake invalidated? `spec_literal` is 'true' iff
    `inside_band_all_lags AND ncc_high` (spec section 2). Here it reads 'false' whenever
    `inside_band_all_lags` is False -- but `stake_zf_inside_band_all_lags` is the SAME rule applied
    to zf's own pure-aliasing residual (P1's own section-3 calibration stake), and when IT also
    reads False, the clause fails identically on a known-aliasing reference and per section 3 ("a
    metric that fails its stake is discarded on the spot") cannot be trusted as diagnostic either
    way. `ncc_high` is not itself gated by that stake and is named as the countervailing evidence
    when it points the other way.
    """
    inside = r.get("inside_band_all_lags")
    stake = r.get("stake_zf_inside_band_all_lags")
    if inside is False and stake is False:
        ncc = r.get("ncc_rec_zf")
        null_hi = r.get("ncc_self_permutation_null_hi97_5")
        countervailing = ""
        if r.get("ncc_high_relative_to_null") and ncc is not None and null_hi is not None:
            countervailing = (f" The other conjunct points the opposite way: "
                              f"ncc_rec_zf={ncc:.4f} against a null ceiling of {null_hi:.5f} says "
                              "the residual IS same-family with zf's aliasing -- evidence FOR P1.")
        return (False,
                "the literal 'false' is driven by inside_band_all_lags=False, but this "
                "proposition's own calibration stake (stake_zf_inside_band_all_lags) reads False "
                "identically on zf's pure-aliasing residual, so that clause cannot be trusted as "
                "diagnostic here (spec section 3)." + countervailing)
    return (True, f"inside_band_all_lags={inside}, stake_zf_inside_band_all_lags={stake} -- not "
                  "both False, so the literal reading is not resting on an invalidated clause.")


def _p2_literal_support(r: dict) -> tuple:
    """P2's `spec_literal_verdict` is 'true' iff `low` (the rim error pattern sits inside its own
    null). That null is chosen by validity -- the circular-shift null when `residual_is_exchangeable`
    is False, the permutation null otherwise (`_p2`'s `gating_null`) -- rather than asserted
    unconditionally, so the clause it rests on is not an invalidated one; the exchangeability CHECK
    is exactly what keeps it valid. Only flagged unsupported if the code somehow gated on the
    permutation null despite exchangeability having failed (it does not, on real data), which would
    mean the literal reading rests on the SAME invalidated null P2's own docstring (C1) describes.
    """
    exch = r.get("residual_is_exchangeable")
    gating = r.get("gating_null")
    if exch is False and gating not in (None, "circular_shift"):
        return (False,
                f"residual_is_exchangeable=False (the permutation null's assumption fails) but "
                f"gating_null={gating!r}, not the valid circular-shift fallback -- the literal "
                "reading would be resting on an invalidated null.")
    return (True,
            f"the literal reading rests on `low`, gated by the {gating!r} null; "
            f"residual_is_exchangeable={exch} correctly routes to a valid null (circular-shift "
            "when the permutation null's exchangeability assumption fails), so the clause is not "
            "invalidated.")


def _p3_literal_support(r: dict) -> tuple:
    """P3's `spec_literal_verdict` compares `real_corr` against the half-voxel-shift positive
    control directly (`real_over_shift_ratio`, in `spec_literal_basis`); `_p3`'s own
    `spec_literal_note` states this rule 'needs no null at all'. `permutation_null_valid` (P3's
    section-3 calibration stake) only demotes the IMPLEMENTED rule's verdict to 'insufficient' when
    it fails -- it plays no role in `spec_literal_basis`, so there is no invalidated clause for the
    literal reading to rest on.
    """
    valid = r.get("permutation_null_valid")
    has_basis = bool(r.get("spec_literal_basis"))
    return (True,
            "the literal reading (`real_over_shift_ratio` vs the half-voxel-shift positive "
            f"control) does not depend on the permutation null at all -- permutation_null_valid="
            f"{valid} only demotes the IMPLEMENTED verdict, and spec_literal_basis is computed "
            f"independently of it (present={has_basis}).")


def _p4_literal_support(r: dict) -> tuple:
    """P4's `spec_literal_verdict` is computed straight from the air-energy ratios
    (`all(ratio > 2.0 ...)`, spec_literal_basis is the air_curve itself); its calibration
    `stake_gate` (the zf-sensitivity stake AND the PSF positive-control stake) is what would
    invalidate those ratios if either failed. On the real volumes both pass.
    """
    passed = r.get("stake_gate", {}).get("passed")
    if passed is False:
        return (False,
                "stake_gate.passed=False -- the air-energy ratios the literal reading rests on "
                "failed their own calibration stake and cannot be trusted.")
    return (True,
            f"stake_gate.passed={passed} -- the air-energy ratios the literal reading rests on are "
            "validated (both the zf-sensitivity and PSF positive-control stakes).")


_LITERAL_SUPPORT_CHECKS = {"P1": _p1_literal_support, "P2": _p2_literal_support,
                          "P3": _p3_literal_support, "P4": _p4_literal_support}


def spec_literal_verdicts(readings: dict) -> dict:
    """What spec section 2's rules, applied literally as written, yield from the same numbers.

    Recorded ALONGSIDE `adjudicate`'s verdicts, never instead of them (final review I2). The two
    differ wherever the implemented rule has an "insufficient" branch the section 2 wording does
    not: section 2 states each proposition as "true iff <condition>", a biconditional that always
    answers, while the implementation adds spec section 3's discard discipline on top. P3 is the
    case that matters -- its implemented rule lost the answer to a null that was never
    pre-registered and then failed validity, while the registered rule (compare against the
    half-pixel-shift positive control) answers cleanly from the data.

    DEFECT 2 (honesty-fix-report): a bare `{"P1": "false", ...}` block is readable as four
    standalone verdicts, but P1's 'false' is derived from `inside_band_all_lags`, exactly the
    clause P1's own calibration stake invalidates (see `_p1_literal_support`). Each entry now
    carries `supported` (computed via `_LITERAL_SUPPORT_CHECKS`, not asserted) and a `reason`
    alongside `value`, so this block cannot be read as a standalone verdict without also seeing
    whether it is entitled to be. `value` and every underlying `spec_literal_verdict` are
    unchanged -- only this aggregate's SHAPE changes, adding information, never altering an
    existing reading.
    """
    out = {}
    for key in ("P1", "P2", "P3", "P4"):
        r = readings.get(key)
        if not isinstance(r, dict):
            continue
        value = r.get("spec_literal_verdict", "insufficient")
        supported, reason = _LITERAL_SUPPORT_CHECKS[key](r)
        out[key] = {"value": value, "supported": supported, "reason": reason}
    return out


def write_report(readings: dict, verdicts: dict, out_dir: str = "results_arc/diag_artefact",
                 filename: str = "verdicts.json") -> str:
    return write_json(os.path.join(out_dir, filename),
                      {"readings": readings, "verdicts": verdicts,
                       "spec_literal_verdicts": spec_literal_verdicts(readings)},
                      generated_by="diag_artefact/report.py (python -m diag_artefact.report)")


def _figure_verdict_lines(readings: dict, verdicts: dict) -> list:
    """One line per proposition for the figure stamp: the verdict, the spec-literal reading when
    it differs, and the single number the outcome turns on."""
    literal = spec_literal_verdicts(readings)
    titles = {
        "P1": "P1 WM speckle = residual incoherent aliasing",
        "P2": "P2 outer-rim band = aliasing (moves with the mask)",
        "P3": "P3 ventricle/midline streaks = displaced structure",
        "P4": "P4 background contains recon-introduced ghosts",
    }
    facts = {}
    p1 = readings.get("P1", {})
    if p1:
        facts["P1"] = (f"stake down: zf's own residual also escapes the band "
                       f"(autocorr lag1 {p1['stake_zf_autocorr_profile'][1]:.3f}); "
                       f"ncc(rec-gt, zf-gt) = {p1['ncc_rec_zf']:.3f}")
    p2 = readings.get("P2", {})
    if p2 and "ncc_ctrlB_vs_shipped" in p2:
        facts["P2"] = (f"rim ncc(ctrlB, shipped) = {p2['ncc_ctrlB_vs_shipped']:.3f} vs "
                       f"{p2['gating_null']} null 97.5% = {p2['gating_null_hi97_5']:.3f}")
    p3 = readings.get("P3", {})
    if p3 and "spec_literal_basis" in p3:
        facts["P3"] = (f"largest displacement = "
                       f"{100 * p3['spec_literal_basis']['max_real_over_shift_ratio']:.1f}% of a "
                       f"detectable half-voxel shift; permutation null invalid -> implemented rule "
                       f"abstains")
    p4 = readings.get("P4", {})
    if p4 and "air_curve" in p4:
        rr = [r["ratio_rec"] for r in p4["air_curve"] if np.isfinite(r["ratio_rec"])]
        facts["P4"] = (f"air-energy ratio sweeps {max(rr):.2f} -> {min(rr):.2f} with the air "
                       f"threshold (blocking clause); PSF term never enters")
    lines = []
    for key in ("P1", "P2", "P3", "P4"):
        if key not in verdicts:
            continue
        entry = literal.get(key)
        lit = entry.get("value") if entry else None
        tag = verdicts[key].upper()
        if lit and lit != verdicts[key]:
            tag += f"  (spec-literal rule: {lit.upper()}"
            # DEFECT 2 (honesty-fix-report): a reader of the figure alone must also see when the
            # spec-literal reading rests on a clause its own stake invalidated -- not just the JSON.
            if entry and entry.get("supported") is False:
                tag += ", NOT SUPPORTED -- see verdicts JSON"
            tag += ")"
        lines.append(f"{titles[key]}:  {tag}\n     {facts.get(key, '')}")
    return lines


def adjudication_figure(vols: dict, regions: dict, out_path: str, verdicts: dict | None = None,
                        readings: dict | None = None, contrast: str = "T1") -> str:
    """GT | zf error | rec error, error panels in jet at the fixed [0, 0.1] convention.

    Final review (I6): the four verdicts are STAMPED on the figure. The panel used to carry no
    outcome at all, so a reader had to parse a 200-line JSON to learn what the picture was
    evidence for -- and the figure was gitignored on top of that.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt, rec, zf = vols["gt"], vols["rec"], vols["zf"]
    z = 160
    peak = float(np.abs(gt).max()) + 1e-9
    panels = [("ground truth", np.abs(gt[:, :, z]), "gray", 0, peak),
              ("|zf - GT|", np.abs(zf[:, :, z] - gt[:, :, z]) / peak, "jet", 0, 0.1),
              ("|recon - GT|", np.abs(rec[:, :, z] - gt[:, :, z]) / peak, "jet", 0, 0.1)]

    lines = _figure_verdict_lines(readings or {}, verdicts or {})
    fig_h = 4.6 + (0.42 * len(lines) if lines else 0.0)
    fig, ax = plt.subplots(1, 3, figsize=(13, fig_h))
    for a, (title, img, cmap, vmin, vmax) in zip(ax, panels):
        a.imshow(np.rot90(img), cmap=cmap, vmin=vmin, vmax=vmax)
        a.set_title(title, fontsize=11)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{contrast} axial z={z} — adjudication panel (error maps jet [0, 0.1])",
                 fontsize=12)
    if lines:
        fig.tight_layout(rect=(0, len(lines) * 0.42 / fig_h, 1, 1))
        y = len(lines) * 0.42 / fig_h
        for i, line in enumerate(lines):
            fig.text(0.02, y - (i + 1) * 0.42 / fig_h + 0.10 / fig_h, line, fontsize=8.5,
                     family="monospace", va="bottom")
    else:
        fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


# ---------------------------------------------------------------------------
# main — Task 9 Step 6 ("run the full diagnosis and record the verdicts"), committed rather than
# run as an ad hoc heredoc. Resolves the checkpoint from m0_provenance.json's own standing rule,
# builds `controls` from run_manifest.json citing ONLY status=="completed_genuine" runs (the
# vacuous ctrlA_lam010 and the not_run ctrlA_lam120 are never referenced), and writes the JSON +
# figure this task delivers.
# ---------------------------------------------------------------------------

DEFAULT_M0_JSON = "results_arc/diag_artefact/m0_provenance.json"
DEFAULT_MANIFEST_JSON = "results_arc/diag_artefact/run_manifest.json"
FALLBACK_NPZ = "results_arc/tvs4k_e040_volumes.npz"  # spec's standing rule when M0 is non-unique


def _resolve_checkpoint(m0_json_path: str = DEFAULT_M0_JSON) -> str:
    """M0's own standing rule: if `resolved` is null (non-unique match), conclusions bind to
    tvs4k_e040 rather than guessing. Returns the resolved candidate if M0 DID resolve.
    """
    with open(m0_json_path) as fh:
        m0 = json.load(fh)
    return m0["resolved"] or FALLBACK_NPZ


def _controls_from_manifest(manifest_path: str = DEFAULT_MANIFEST_JSON) -> dict:
    """Build the `controls` dict `run_all` expects, citing only runs the manifest itself marks
    `completed_genuine`. `ctrlA_lam010` in the MANIFEST is the ALL-ONES, completed_vacuous run
    (task-9-brief.md point 3: must not be cited for anything) -- Control A here is instead the
    AF=2 pair `ctrlA_af2_lam010` / `ctrlA_af2_lam060`. `ctrlA_lam120` is not_run and so is simply
    absent. A run whose status is anything other than "completed_genuine" is skipped rather than
    cited, whatever its name suggests.

    Fix round 1 (I2): the RETURNED dict's own keys are also `ctrlA_af2_lam010`/`ctrlA_af2_lam060`
    (not `ctrlA_lam010`/`ctrlA_lam060`) so nothing downstream can be misread as citing the
    manifest's vacuous entry just because a key happened to share its name.
    """
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    runs = manifest["runs"]

    def _npz_if_genuine(name):
        r = runs.get(name)
        if r and r.get("status") == "completed_genuine" and r.get("exists"):
            return r["npz"]
        return None

    return {
        "ctrlB": _npz_if_genuine("ctrlB_seed1"),
        "ctrlA_af2_lam010": _npz_if_genuine("ctrlA_af2_lam010"),
        "ctrlA_af2_lam060": _npz_if_genuine("ctrlA_af2_lam060"),
        "rc_lam000": DEFAULT_RC_LAM000,
        "mask_path": DEFAULT_MASK_PATH,
    }


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contrast", default="T1")
    ap.add_argument("--npz", default=None, help="override the M0-resolved checkpoint")
    ap.add_argument("--m0-json", default=DEFAULT_M0_JSON)
    ap.add_argument("--manifest-json", default=DEFAULT_MANIFEST_JSON)
    ap.add_argument("--out-dir", default="results_arc/diag_artefact")
    ap.add_argument("--figure-out", default=None,
                    help="default: <out-dir>/adjudication.png for T1, adjudication_<contrast>.png "
                         "otherwise")
    ap.add_argument("--json-name", default=None,
                    help="default: verdicts.json for T1, verdicts_<contrast>.json otherwise")
    args = ap.parse_args(argv)

    suffix = "" if args.contrast == "T1" else f"_{args.contrast}"
    json_name = args.json_name or f"verdicts{suffix}.json"
    figure_out = args.figure_out or os.path.join(args.out_dir, f"adjudication{suffix}.png")

    npz_path = args.npz or _resolve_checkpoint(args.m0_json)
    controls = _controls_from_manifest(args.manifest_json)

    print(f"checkpoint: {npz_path}")
    print(f"controls: {controls}")

    readings = run_all(npz_path, args.contrast, controls)

    # Fix round 1 (I2): record WHY this checkpoint, in the durable JSON, not only in prose.
    with open(args.m0_json) as fh:
        m0 = json.load(fh)
    readings["provenance"] = {
        "m0_unique_match": m0.get("unique_match"),
        "m0_resolved": m0.get("resolved"),
        "checkpoint_used": npz_path,
        "checkpoint_source": ("M0 resolved uniquely" if m0.get("resolved") else
                              "spec section 7 FALLBACK: M0 did not resolve uniquely "
                              "(unique_match=false, resolved=null); binds to tvs4k_e040 per the "
                              "standing rule. The figure's own provenance is unrecoverable."),
    }

    verdicts = adjudicate(readings)
    literal = spec_literal_verdicts(readings)
    json_path = write_report(readings, verdicts, out_dir=args.out_dir, filename=json_name)

    v = load_volumes(npz_path, args.contrast)
    fig_path = adjudication_figure(v, {}, figure_out, verdicts=verdicts, readings=readings,
                                   contrast=args.contrast)

    print(f"wrote {json_path}")
    print(f"wrote {fig_path}")
    print("verdicts:", verdicts)
    print("spec-literal verdicts:", literal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
