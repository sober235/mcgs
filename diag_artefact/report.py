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
from .volumes import air_mask, brain_mask, wm_mask, load_volumes

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
    """
    rng = np.random.default_rng(seed)
    vals = field[mask].astype(np.float64)
    scratch = np.zeros_like(field, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        scratch[mask] = rng.permutation(vals)
        out[i] = ncc(scratch, ref, mask)
    return out


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

    return {
        "verdict": verdict,
        "reason": reason,
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

def _p2(rec: np.ndarray, gt: np.ndarray, brain: np.ndarray, contrast: str,
       ctrlB_path: str | None) -> dict:
    if not ctrlB_path or not os.path.exists(ctrlB_path):
        return {"verdict": "insufficient", "reason": "control B volumes not available"}

    vB = load_volumes(ctrlB_path, contrast)
    e_shipped = np.abs(rec.astype(np.float64) - gt.astype(np.float64))
    e_ctrlB = np.abs(vB["rec"].astype(np.float64) - vB["gt"].astype(np.float64))

    r = ncc(e_ctrlB, e_shipped, brain)
    null = _permutation_null_ncc(e_shipped, e_shipped, brain)
    null_hi = float(np.percentile(np.abs(null), 97.5))
    low = bool(abs(r) <= null_hi)

    verdict = "true" if low else "false"
    reason = ("error pattern indistinguishable from a permutation-null of itself -> moved with "
              "the mask instance" if low else
              "error pattern correlates with the shipped mask's beyond chance -> indifferent "
              "to the mask instance")
    return {
        "verdict": verdict,
        "reason": reason,
        "ncc_ctrlB_vs_shipped": r,
        "permutation_null_hi97_5": null_hi,
        "low": low,
        "ctrlB_npz": ctrlB_path,
    }


# ---------------------------------------------------------------------------
# P3 — ventricle / midline streaks = displaced structure
# ---------------------------------------------------------------------------

def _p3(err_rec: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    per_axis = {}
    any_significant = False
    for a in range(3):
        g = np.gradient(gt.astype(np.float64), axis=a)
        real_corr = ncc(err_rec, g, mask)
        null = _permutation_null_ncc(err_rec, g, mask)
        null_hi = float(np.percentile(np.abs(null), 97.5))

        shifted = half_voxel_shift(gt, axis=a) - gt
        shift_corr = gradient_correlation(shifted, gt, mask)[f"axis{a}"]

        significant = bool(abs(real_corr) > null_hi)
        any_significant = any_significant or significant
        per_axis[f"axis{a}"] = {
            "real_corr": real_corr,
            "permutation_null_hi97_5": null_hi,
            "significant": significant,
            "half_voxel_shift_calibration": shift_corr,
            "real_over_shift_ratio": float(abs(real_corr) / (abs(shift_corr) + 1e-30)),
        }

    verdict = "true" if any_significant else "false"
    reason = ("at least one axis's gradient correlation clearly exceeds its own "
              "permutation-null (independent-noise) level" if any_significant else
              "every axis's gradient correlation sits inside its own permutation-null band")
    return {"verdict": verdict, "reason": reason, "axes": per_axis}


# ---------------------------------------------------------------------------
# P4 — background contains recon-introduced ghosts
# ---------------------------------------------------------------------------

def _load_mask2d(mask_path: str) -> np.ndarray:
    d = sio.loadmat(mask_path)
    m3 = np.asarray(d["mask"])
    return (m3[:, :, 0] != 0).astype(np.float64)


def _p4(rec: np.ndarray, gt: np.ndarray, zf: np.ndarray, brain: np.ndarray,
       mask_path: str) -> dict:
    curve = air_energy_curve(rec, gt, brain)
    curve_zf = air_energy_curve(zf, gt, brain)  # stake: zf must read "full scale" if the metric is sensitive

    mask2d = _load_mask2d(mask_path)
    psf = psf_spread(mask2d)

    uniform2d = np.zeros_like(mask2d)
    uniform2d[::4, :] = 1.0
    psf_uniform = psf_spread(uniform2d)  # stake: positive control, must show a discrete replica

    return {
        "air_curve": curve,
        "psf_peak_sidelobe": psf["peak_sidelobe"],
        "psf_sidelobe_rms": psf["sidelobe_rms"],
        "stake_air_curve_zf": curve_zf,
        "stake_psf_uniform_mask_peak_sidelobe": psf_uniform["peak_sidelobe"],
        "mask_path": mask_path,
    }


# ---------------------------------------------------------------------------
# patchiness attribution — control A, two lambda arms
# ---------------------------------------------------------------------------

def _patchiness(rec: np.ndarray, gt: np.ndarray, wm: np.ndarray, contrast: str,
               ctrlA_lam010_path: str | None, ctrlA_lam060_path: str | None,
               rc_lam000_path: str, zf: np.ndarray) -> dict:
    shipped_sel = selectivity(rec, gt, wm)
    out = {"shipped_selectivity": shipped_sel}

    # calibration stakes (§0/§14b established readings): zf ~ 0.97 flat; rc_lam000 (no TV) ~ 1.20.
    out["stake_zf_selectivity"] = selectivity(zf, gt, wm)
    if rc_lam000_path and os.path.exists(rc_lam000_path):
        v0 = load_volumes(rc_lam000_path, contrast)
        wm0 = wm_mask(v0["gt"], brain_mask(v0["gt"]))
        out["stake_rc_lam000_selectivity"] = selectivity(v0["rec"], v0["gt"], wm0)
    else:
        out["stake_rc_lam000_selectivity"] = None

    have_both = (ctrlA_lam010_path and ctrlA_lam060_path and
                os.path.exists(ctrlA_lam010_path) and os.path.exists(ctrlA_lam060_path))
    if not have_both:
        out["attribution"] = "control A undecided"
        out["reason"] = "control A (AF=2, both lambda arms) volumes not available"
        return out

    v010 = load_volumes(ctrlA_lam010_path, contrast)
    v060 = load_volumes(ctrlA_lam060_path, contrast)
    wm010 = wm_mask(v010["gt"], brain_mask(v010["gt"]))
    wm060 = wm_mask(v060["gt"], brain_mask(v060["gt"]))
    sel010 = selectivity(v010["rec"], v010["gt"], wm010)
    sel060 = selectivity(v060["rec"], v060["gt"], wm060)
    d010 = sel010 - shipped_sel
    d060 = sel060 - shipped_sel
    agree = bool(d010 != 0 and d060 != 0 and np.sign(d010) == np.sign(d060))

    out.update({
        "ctrlA_lam010_npz": ctrlA_lam010_path, "ctrlA_lam060_npz": ctrlA_lam060_path,
        "selectivity_ctrlA_lam010": sel010, "selectivity_ctrlA_lam060": sel060,
        "delta_lam010_vs_shipped": d010, "delta_lam060_vs_shipped": d060, "agree": agree,
    })
    if agree:
        out["attribution"] = ("less patchiness at AF=2 (mask-dependent)" if d010 < 0 else
                              "patchiness persists or worsens at AF=2 (regulariser-intrinsic)")
    else:
        out["attribution"] = "control A undecided"
    return out


# ---------------------------------------------------------------------------
# run_all — the brief's promised interface
# ---------------------------------------------------------------------------

def run_all(npz_path: str, contrast: str, controls: dict) -> dict:
    """Wire M1-M5 and the controls into the `readings` dict `adjudicate` consumes.

    `controls` maps logical names to *_volumes.npz paths (or overrides), all optional:
      "ctrlB"            -- control B (different mask seed), for P2
      "ctrlA_lam010"     -- control A, AF=2, lambda_geo=0.10, for the patchiness attribution
      "ctrlA_lam060"     -- control A, AF=2, lambda_geo=0.60, for the patchiness attribution
      "rc_lam000"        -- no-TV run, the M5 "no patchiness" calibration stake
      "mask_path"        -- shipped mask .mat, for M1's PSF (default: the ALIGNED AF=12 mask)
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
        "P2": _p2(rec, gt, brain, contrast, controls.get("ctrlB")),
        "P3": _p3(err_rec, gt, wm),
        "P4": _p4(rec, gt, zf, brain, controls.get("mask_path", DEFAULT_MASK_PATH)),
        "patchiness": _patchiness(rec, gt, wm, contrast,
                                  controls.get("ctrlA_lam010"), controls.get("ctrlA_lam060"),
                                  controls.get("rc_lam000", DEFAULT_RC_LAM000), zf),
    }
    return readings


# ---------------------------------------------------------------------------
# adjudicate / write_report / adjudication_figure — task-9-brief.md Step 3, verbatim
# ---------------------------------------------------------------------------

def adjudicate(readings: dict) -> dict:
    v = {}
    p4 = readings.get("P4")
    if p4:
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


def write_report(readings: dict, verdicts: dict, out_dir: str = "results_arc/diag_artefact") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "verdicts.json")
    with open(path, "w") as fh:
        json.dump({"readings": readings, "verdicts": verdicts}, fh, indent=2, default=float)
    return os.path.abspath(path)


def adjudication_figure(vols: dict, regions: dict, out_path: str) -> str:
    """GT | zf error | rec error, error panels in jet at the fixed [0, 0.1] convention."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt, rec, zf = vols["gt"], vols["rec"], vols["zf"]
    z = 160
    peak = float(np.abs(gt).max()) + 1e-9
    panels = [("ground truth", np.abs(gt[:, :, z]), "gray", 0, peak),
              ("|zf - GT|", np.abs(zf[:, :, z] - gt[:, :, z]) / peak, "jet", 0, 0.1),
              ("|recon - GT|", np.abs(rec[:, :, z] - gt[:, :, z]) / peak, "jet", 0, 0.1)]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for a, (title, img, cmap, vmin, vmax) in zip(ax, panels):
        a.imshow(np.rot90(img), cmap=cmap, vmin=vmin, vmax=vmax)
        a.set_title(title, fontsize=11)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("T1 axial z=160 — adjudication panel (error maps jet [0, 0.1])", fontsize=12)
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
    `completed_genuine`. `ctrlA_lam010` in the manifest is the ALL-ONES, completed_vacuous run
    (task-9-brief.md point 3: must not be cited for anything) -- Control A here is instead the
    AF=2 pair `ctrlA_af2_lam010` / `ctrlA_af2_lam060`. `ctrlA_lam120` is not_run and so is simply
    absent. A run whose status is anything other than "completed_genuine" is skipped rather than
    cited, whatever its name suggests.
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
        "ctrlA_lam010": _npz_if_genuine("ctrlA_af2_lam010"),
        "ctrlA_lam060": _npz_if_genuine("ctrlA_af2_lam060"),
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
    ap.add_argument("--figure-out", default="results_arc/diag_artefact/adjudication.png")
    args = ap.parse_args(argv)

    npz_path = args.npz or _resolve_checkpoint(args.m0_json)
    controls = _controls_from_manifest(args.manifest_json)

    print(f"checkpoint: {npz_path}")
    print(f"controls: {controls}")

    readings = run_all(npz_path, args.contrast, controls)
    verdicts = adjudicate(readings)
    json_path = write_report(readings, verdicts, out_dir=args.out_dir)

    v = load_volumes(npz_path, args.contrast)
    fig_path = adjudication_figure(v, {}, args.figure_out)

    print(f"wrote {json_path}")
    print(f"wrote {fig_path}")
    print("verdicts:", verdicts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
