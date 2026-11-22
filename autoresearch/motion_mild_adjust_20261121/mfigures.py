#!/usr/bin/env python3
"""Figures of the second version (runs on amax, CPU). No clock value appears on any figure.

    mfigures.py --feas     the feasibility check: z score of every echo train, AUC along the training, T2 PSNR
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from mcommon import DATA2, RES2, ROOT2, THRESH2

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT2 / "figures"
C_CLEAN, C_BAD, C_LINE = "#4c72b0", "#c44e52", "#555555"


def feas_figure(feas, mm, truth, out_png, dpi=200):
    """Three panels: (a) z score of every echo train of the unchanged model on DTa at the last computation, in
    acquisition order; (b) AUC of that score for the contaminated trains along the training; (c) T2 head PSNR of
    the three runs. feas = mevaluate.feasibility(...)."""
    z = np.asarray(mm["m2"]["z"], dtype=np.float64)
    cont = np.asarray(truth["contaminated"], dtype=bool)
    n = np.arange(z.size)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4), gridspec_kw=dict(width_ratios=[2.2, 1.0, 1.0]))
    a = ax[0]
    a.bar(n[~cont], z[~cont], color=C_CLEAN, width=0.9, label=f"clean echo trains ({int((~cont).sum())})")
    a.bar(n[cont], z[cont], color=C_BAD, width=0.9, label=f"trains acquired during the motion ({int(cont.sum())})")
    st = mm["m2"].get("settings") or {}
    for key, ls in (("z0", "--"), ("z1", ":")):
        if key in st:
            a.axhline(float(st[key]), color=C_LINE, ls=ls, lw=1.0)
            a.text(z.size - 1, float(st[key]), f" {key} = {st[key]:g}", va="bottom", ha="right", fontsize=9, color=C_LINE)
    a.set_xlabel("echo train (acquisition order)")
    a.set_ylabel("robust z score of the residual statistic")
    a.set_title("(a) Per-echo-train residual of MC-GS unchanged", fontsize=11, loc="left")
    a.legend(frameon=False, fontsize=9, loc="upper left")
    a = ax[1]
    hist = feas.get("auc_history") or []
    if hist:
        a.plot([h["it"] for h in hist], [h["auc"] for h in hist], "-o", ms=3, color=C_CLEAN)
    a.axhline(THRESH2["feas_auc"], color=C_LINE, ls="--", lw=1.0)
    a.text(0.98, THRESH2["feas_auc"], f"required {THRESH2['feas_auc']:.2f}", transform=a.get_yaxis_transform(),
           ha="right", va="bottom", fontsize=9, color=C_LINE)
    a.set_ylim(0.4, 1.02)
    a.set_xlabel("iteration")
    a.set_ylabel("AUC (contaminated vs clean trains)")
    auc = feas.get("auc")
    a.set_title("(b) Separation along the training" + ("" if auc is None else f"; final {auc:.3f}"), fontsize=11, loc="left")
    a = ax[2]
    bars = [("no motion\nMC-GS", feas.get("b0_c0"), C_CLEAN), ("motion\nMC-GS", feas.get("b0"), C_BAD),
            ("motion\ntrue weights", feas.get("ow"), "#55a868")]
    vals = [v for _, v, _ in bars if v is not None]
    for i, (name, v, col) in enumerate(bars):
        if v is None:
            continue
        a.bar(i, v, color=col, width=0.6)
        a.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    a.set_xticks(range(len(bars)))
    a.set_xticklabels([b[0] for b in bars], fontsize=9)
    if vals:
        a.set_ylim(min(vals) - 0.6, max(vals) + 0.4)
    a.set_ylabel("T2 head PSNR (dB)")
    rec, loss = feas.get("recover"), feas.get("loss")
    note = "" if loss is None else f"loss {loss:.3f} dB" + ("" if rec is None else f", recovered {100 * rec:.0f} %")
    a.set_title(f"(c) {note}" if note else "(c) T2 head PSNR", fontsize=11, loc="left")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Feasibility check (development condition DTa, interleaved order): {feas.get('decision')}",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feas", action="store_true")
    args = ap.parse_args()
    import mevaluate
    OUT.mkdir(parents=True, exist_ok=True)
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if args.feas:
        f = mevaluate.feasibility(mevaluate.table(mevaluate.load_ledger()))
        mm = json.loads((RES2 / "mm_af12_il_DTa_B0_mm.json").read_text())
        truth = json.loads((DATA2 / "il/DTa/motion_truth.json").read_text())
        p = feas_figure(f, mm, truth, OUT / "feas_il_DTa.png")
        (OUT / "feas_il_DTa.json").write_text(json.dumps(f, indent=1))
        print(f"wrote {p.name}", flush=True)
        if art:
            shutil.copy(p, Path(art) / p.name)
            shutil.copy(OUT / "feas_il_DTa.json", Path(art) / "feas_il_DTa.json")
    else:
        ap.error("nothing to do")


if __name__ == "__main__":
    main()
