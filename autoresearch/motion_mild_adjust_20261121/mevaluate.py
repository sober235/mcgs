#!/usr/bin/env python3
"""Evaluation of the second version (runs on amax, CPU).

    mevaluate.py --feas          the feasibility check (design section 10 step 1): DTa under the interleaved order
    mevaluate.py --dev TAG       report of a development round

Scores. The frozen scorer inserts every sampled line (ledger head_psnr). An arm that weights the echo trains
(M2 adaptive or truth) is scored on the image with only the kept lines inserted (w >= 0.5, reference state):
head_psnr_wdc; that is its head_psnr_eval. Arms without weights keep the scorer's value. head_psnr_raw = raw render.
No clock value or duration is read, printed or written.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import weights
from mcommon import ARMS2, DATA2, LEDGER2, RES2, THRESH2

log = lambda *a: print(*a, flush=True)


def load_ledger(path=None):
    p = Path(path or LEDGER2)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def weighted(arm):
    return ARMS2[arm]["m2"] in ("adaptive", "truth")


def _file_scores(exp, key, single=False):
    """Head metrics of one array of <exp>_canon.npz; None when the file or the array is missing."""
    canon, vols = RES2 / f"{exp}_canon.npz", RES2 / f"{exp}_volumes.npz"
    if not (canon.exists() and vols.exists()):
        return None
    with np.load(canon) as z:
        if key not in z.files:
            return None
    import mrun
    return mrun.head_scores(exp, key, single=single)


def table(ledger, scores=None):
    """One row per experiment (the last ledger line of an experiment wins): exp, af, order, cond, arm, seed, tag,
    T1, T2. T2 (the only contrast of a single-contrast arm) carries head_psnr (scorer), head_psnr_raw,
    head_psnr_wdc (when the kept-lines image exists) and head_psnr_eval."""
    scores = scores or _file_scores
    rows = {}
    for r in ledger:
        single = ARMS2[r["arm"]]["trainer"] == "single"
        t2 = dict(r["plain"] if single else r["contrast"])
        raw, wdc = scores(r["exp"], "rec_raw", single), scores(r["exp"], "rec_w_dc", single)
        if raw:
            t2.update(head_psnr_raw=raw["head_psnr"], head_ssim_raw=raw["head_ssim"])
        if wdc:
            t2.update(head_psnr_wdc=wdc["head_psnr"], head_ssim_wdc=wdc["head_ssim"], head_hfen_wdc=wdc.get("head_hfen"))
        t2["head_psnr_eval"] = t2.get("head_psnr_wdc") if weighted(r["arm"]) else t2["head_psnr"]
        rows[r["exp"]] = dict(exp=r["exp"], af=r["af"], order=r["order"], cond=r["cond"], arm=r["arm"],
                              seed=r["seed"], tag=r.get("tag", ""), T1=None if single else dict(r["plain"]), T2=t2)
    return list(rows.values())


def val(tab, order, cond, arm, c="T2", key="head_psnr_eval", tag="", seed=42):
    """One number of the table; None when the run or the contrast is missing. T1 has no weighted variant, so its
    evaluation score is the scorer's head_psnr."""
    if c == "T1" and key == "head_psnr_eval":
        key = "head_psnr"
    for r in tab:
        if (r["order"], r["cond"], r["arm"], r["tag"], r["seed"]) == (order, cond, arm, tag, seed):
            return None if r[c] is None else r[c].get(key)
    return None


def recovered(tab, order, cond, arm, tag="", base="B0"):
    """Share of the T2 loss that an arm takes back: loss = base(C0) - base(cond); value = (arm - base(cond)) / loss.
    base = B0 (the unchanged joint model) or I-T2 for a single-contrast arm. The baselines are looked up with the
    arm's tag first, then untagged. value is None when the loss is below min_loss_db. None if a run is missing."""
    pick = lambda o, c: next((v for v in (val(tab, o, c, base, tag=tag), val(tab, o, c, base)) if v is not None), None)
    b0_c0, b0, a = pick("il", "C0"), pick(order, cond), val(tab, order, cond, arm, tag=tag)
    if b0_c0 is None or b0 is None or a is None:
        return None
    loss = b0_c0 - b0
    return dict(loss=loss, value=(a - b0) / loss if loss >= THRESH2["min_loss_db"] else None, b0_c0=b0_c0, b0=b0,
                arm_psnr=a, base=base)


def weight_detection(mm, truth):
    """recall = share of the contaminated trains with w < 0.5; false_rate = share of the clean trains with w < 0.5;
    frac_w1 = share of all trains with w = 1; auc of the last statistic z for the contaminated trains."""
    w = np.asarray(mm["m2"]["w"], dtype=np.float64)
    cont = np.asarray(truth["contaminated"], dtype=bool)
    z = mm["m2"].get("z")
    a = weights.auc(np.asarray(z, dtype=np.float64), cont) if z is not None else float("nan")
    return dict(recall=float((w[cont] < 0.5).mean()) if cont.any() else None,
                false_rate=float((w[~cont] < 0.5).mean()) if (~cont).any() else None,
                frac_w1=float((w == 1.0).mean()), auc=None if np.isnan(a) else a,
                n_contaminated=int(cont.sum()), n_down=int((w < 1.0).sum()))


def _load_mm(exp):
    p = RES2 / f"{exp}_mm.json"
    return json.loads(p.read_text()) if p.exists() else None


def _truth(order, cond):
    return json.loads((DATA2 / order / cond / "motion_truth.json").read_text())


def feasibility(tab, mm=None, truth=None):
    """Design section 10 step 1, on DTa under the interleaved order: (1) the statistic of the unchanged model (B0,
    monitor) separates the contaminated echo trains, AUC >= feas_auc at the last computation; (2) truth weights (Ow)
    take back at least feas_recover of the loss."""
    mm = mm or _load_mm("mm_af12_il_DTa_B0")
    truth = truth or _truth("il", "DTa")
    cont = np.asarray(truth["contaminated"], dtype=bool)
    hist = [dict(it=h["it"], auc=weights.auc(np.asarray(h["z"], dtype=np.float64), cont)) for h in mm["m2"]["history"]]
    auc = hist[-1]["auc"] if hist else None
    r = recovered(tab, "il", "DTa", "Ow") or dict(loss=None, value=None)
    out = dict(auc=auc, auc_history=hist, loss=r["loss"], recover=r["value"], b0_c0=r.get("b0_c0"), b0=r.get("b0"),
               ow=r.get("arm_psnr"), n_contaminated=int(cont.sum()), n_trains=int(cont.size),
               pass_auc=bool(auc is not None and auc >= THRESH2["feas_auc"]),
               pass_recover=bool(r["value"] is not None and r["value"] >= THRESH2["feas_recover"]))
    if r["loss"] is not None and r["loss"] < THRESH2["min_loss_db"]:
        out["decision"] = "stop: motion too small to measure"
    else:
        out["decision"] = "go" if out["pass_auc"] and out["pass_recover"] else "stop"
    return out


def dev_report(tab, tag):
    """Rows of one development round (and the untagged rows they are compared with)."""
    out = []
    for r in tab:
        if r["tag"] != tag:
            continue
        d = dict(exp=r["exp"], T1=None if r["T1"] is None else r["T1"]["head_psnr"], T2_all=r["T2"]["head_psnr"],
                 T2_raw=r["T2"].get("head_psnr_raw"), T2_eval=r["T2"]["head_psnr_eval"])
        base = "I-T2" if ARMS2[r["arm"]]["trainer"] == "single" else "B0"
        d["recovered"] = recovered(tab, r["order"], r["cond"], r["arm"], tag=tag, base=base)
        d["vs_C0_B0"] = None if val(tab, "il", "C0", "B0") is None or d["T2_eval"] is None \
            else d["T2_eval"] - val(tab, "il", "C0", "B0")
        mm = _load_mm(r["exp"])
        if mm is not None:
            d["weights"] = weight_detection(mm, _truth(r["order"], r["cond"]))
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feas", action="store_true")
    ap.add_argument("--dev", metavar="TAG", help="development round, e.g. d1; '' = the untagged rows")
    args = ap.parse_args()
    tab = table(load_ledger())
    if args.feas:
        f = feasibility(tab)
        for name in ("mm_af12_il_C0_B0", "mm_af12_il_DTa_B0", "mm_af12_il_DTa_Ow"):
            r = next((x for x in tab if x["exp"] == name), None)
            if r:
                log(f"{name}: T1 {r['T1']['head_psnr']:.3f}  T2 all lines {r['T2']['head_psnr']:.3f}  "
                    f"raw {r['T2'].get('head_psnr_raw')}  eval {r['T2']['head_psnr_eval']}")
        log(json.dumps(f, indent=1))
        mm = _load_mm("mm_af12_il_DTa_B0")
        log("FEAS z of the last computation, per echo train (C = contaminated): "
            + " ".join(f"{z:.1f}{'C' if c else ''}" for z, c in zip(mm["m2"]["z"], _truth("il", "DTa")["contaminated"])))
        log(f"FEASIBILITY decision: {f['decision']}")
    elif args.dev is not None:
        log(json.dumps(dev_report(tab, args.dev), indent=1))
    else:
        ap.error("nothing to do")


if __name__ == "__main__":
    main()
