#!/usr/bin/env python3
"""retro_more_arms.py -- add the four arms the first retrospective pass left out (2026-09-17).

The registered comparison set is nine methods besides ZF (CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU,
3D-DIP, 3D-INR, GS-single, MC-GS), plus MC-INR in the prospective topic. The first pass ran six of them; this adds
Joint SENSE, L1-SPIRiT, ZS-SSDU and MC-INR so the retrospective table matches the prospective one.

Every lambda is selected on this data, on 5% held-out phase-encode lines, never on truth -- carrying the
prospective lambda over is exactly what broke CS-SENSE in the first pass:

  Joint SENSE : vectorial TV over the pair, one lambda for both scans
  L1-SPIRiT   : kernel 3 and 5 x a lambda grid, the kernel settled on the same held-out number
  ZS-SSDU     : no lambda, zero-shot self-supervised, 60 epochs as in the prospective runs
  MC-INR      : no lambda; needs the GR_pt1.10 interpreter (tiny-cuda-nn)

  python scripts/retro_more_arms.py --case UCSF-PDGM-0004 --gpu 3 [--arms jsense,spirit,zsssdu,mcinr]
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

T = Path(__file__).resolve().parents[1]
CE = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/scripts")
PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
PY_TCNN = "/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python"
ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--arms", default="jsense,spirit,zsssdu,mcinr")
ap.add_argument("--lams", default="5e-6,1.25e-5,2.5e-5,5e-5,1e-4,2e-4")
ap.add_argument("--sp_lams", default="1e-4,2.5e-4,5e-4,1e-3,2e-3")
ap.add_argument("--sp_fixed", default="",
                help="'kernel:lambda' to skip the L1-SPIRiT grid and refit directly. The grid costs about 3 h per "
                     "case under contention, and the choice was identical across cases, so it is settled once on "
                     "UCSF-PDGM-0004 (kernel 5, 1e-4) and reused within this simulation family (2026-09-17)")
ap.add_argument("--outer", type=int, default=200)
a = ap.parse_args()
S = T / "sim" / a.case
R = T / "recon" / a.case
env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID",
           CE_WORK_PRE=str(S / "work_pre"), CE_WORK_POST=str(S / "work_post"),
           CE_SPLIT=str(S / "split_95_5_seed0.npz"))
run = lambda cmd, py=PY: subprocess.run([py, "-u"] + cmd, env=env, cwd=str(T), check=True)
log = lambda *m: print(*m, flush=True)
sel = {}
if (R / "more_arms_selection.json").exists():
    sel = json.load(open(R / "more_arms_selection.json"))

# ---- Joint SENSE: vectorial TV over the pair, one lambda ------------------------------------------------------
if "jsense" in a.arms:
    lams = [float(v) for v in a.lams.split(",")]
    for l in lams:
        if not (R / "tv" / f"jsense_o{a.outer}_lam{l:g}_fit_pre_meta.json").exists():
            run([str(CE / "ce_tv_sense.py"), "--mode", "joint", "--mask", "fit", "--lams", f"{l:g}",
                 "--outer", str(a.outer), "--gpu", str(a.gpu), "--out", str(R / "tv")])
    rows = {f"{l:g}": sum(json.load(open(R / "tv" / f"jsense_o{a.outer}_lam{l:g}_fit_{s}_meta.json"))["kres_heldout"]
                          for s in ("pre", "post")) / 2 for l in lams}
    best = min(rows, key=rows.get)
    sel["jsense"] = dict(lam=float(best), heldout=rows[best], grid=rows)
    log(f"{a.case} Joint SENSE: " + "  ".join(f"{k}:{v:.5f}" for k, v in rows.items()) + f"  -> lambda {best}")
    if not (R / "tv" / f"jsense_o{a.outer}_lam{float(best):g}_full_pre_meta.json").exists():
        run([str(CE / "ce_tv_sense.py"), "--mode", "joint", "--mask", "full", "--lams", best,
             "--outer", str(a.outer), "--gpu", str(a.gpu), "--out", str(R / "tv")])

# ---- L1-SPIRiT: kernel 3 and 5 x lambda, both settled on the held-out lines ------------------------------------
if "spirit" in a.arms and a.sp_fixed:
    k, l = int(a.sp_fixed.split(":")[0]), float(a.sp_fixed.split(":")[1])
    for scan in ("pre", "post"):
        sel[f"spirit_{scan}"] = dict(ksize=k, lam=l, heldout=None, grid="fixed from UCSF-PDGM-0004")
        if not (R / "spirit" / f"spirit_ks{k}_i150_lam{l:g}_full_{scan}_meta.json").exists():
            run([str(CE / "ce_spirit.py"), "--scan", scan, "--mask", "full", "--ksize", str(k),
                 "--lams", f"{l:g}", "--gpu", str(a.gpu), "--out", str(R / "spirit")])
    log(f"{a.case} L1-SPIRiT: kernel {k}, lambda {l:g} (fixed)")
elif "spirit" in a.arms:
    lams = [float(v) for v in a.sp_lams.split(",")]
    for scan in ("pre", "post"):
        rows = {}
        for k in (3, 5):
            todo = [l for l in lams if not (R / "spirit" / f"spirit_ks{k}_i150_lam{l:g}_fit_{scan}_meta.json").exists()]
            if todo:
                run([str(CE / "ce_spirit.py"), "--scan", scan, "--mask", "fit", "--ksize", str(k),
                     "--lams", ",".join(f"{l:g}" for l in todo), "--gpu", str(a.gpu), "--out", str(R / "spirit")])
            for l in lams:
                m = json.load(open(R / "spirit" / f"spirit_ks{k}_i150_lam{l:g}_fit_{scan}_meta.json"))
                rows[f"ks{k}_lam{l:g}"] = m["kres_heldout"]
        best = min(rows, key=rows.get)
        k, l = int(best.split("_")[0][2:]), float(best.split("lam")[1])
        sel[f"spirit_{scan}"] = dict(ksize=k, lam=l, heldout=rows[best], grid=rows)
        log(f"{a.case} L1-SPIRiT {scan}: best {best} ({rows[best]:.5f})")
        if not (R / "spirit" / f"spirit_ks{k}_i150_lam{l:g}_full_{scan}_meta.json").exists():
            run([str(CE / "ce_spirit.py"), "--scan", scan, "--mask", "full", "--ksize", str(k),
                 "--lams", f"{l:g}", "--gpu", str(a.gpu), "--out", str(R / "spirit")])

# ---- ZS-SSDU and MC-INR: nothing to select --------------------------------------------------------------------
if "zsssdu" in a.arms:
    for scan in ("pre", "post"):
        if not (R / f"zsssdu_{scan}_meta.json").exists():
            run([str(CE / "ce_classical.py"), "--method", "zsssdu", "--scan", scan, "--epochs", "60",
                 "--gpu", str(a.gpu), "--out", str(R)])
if "mcinr" in a.arms:
    if not all((R / "mcinr" / f"mcinr_full_{s}_meta.json").exists() for s in ("pre", "post")):
        run([str(CE / "ce_mcinr.py"), "--mask", "full", "--gpu", str(a.gpu), "--out", str(R / "mcinr")], py=PY_TCNN)

json.dump(sel, open(R / "more_arms_selection.json", "w"), indent=1)
log(f"wrote {R / 'more_arms_selection.json'}")
