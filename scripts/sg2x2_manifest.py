#!/usr/bin/env python3
"""Provenance + regulariser audit of one sg2x2 run (called by run_sg2x2.sh after arc/cycle.py; re-runnable).

Reads results_arc/EXP.log and the trainer's resolved config results_arc/EXP/EXP/config.yaml, checks them against
scripts/sg2x2_cells.resolved_switches(CELL) in three ways (the spec asks for the weighted losses, not only the config):
  resolved_ok   every switch in the resolved config equals the cell's value; in every cell also lambda_nl (and its
                plain / contrast variants) == 0, no guide path set, kspace_freq_adaptive True, kspace_high_weight 2.2
  startup_ok    a reg-on cell printed [edge_tv] / [wavelet] / [phase-tv]; a reg-off cell printed none of them, and no
                [dtv-proj] / [guide] / [nl] / [FLAIR-private] line appeared in any cell
  loss_ok       dc_p and R_dtv / R_edge / R_vtv / R_dtvp / R_nl are present and finite on the first and last logged
                iteration, and the R_* are 0 there in every cell; in a reg-off cell R_geo / R_wav (and R_ptv when
                logged) are present, finite and 0 too
resolved_ok also pins every knob the spec freezes (k-space weighting, no whitening, no private points, no densification,
max_radius, 3-NN scales, tv_eps, kappa source, known_zero by channel name; reg-on cells: db4 / 1 level / kappa_q / w_floor),
each read with the trainer's own default when the key is absent. With --seed / --npts / --init / --support / --iters
the launcher's arguments are compared with the resolved config as well (audit(..., args=dict) does the same).
and writes results_arc/EXP_manifest.json. Exit 0 when ok, 3 when not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sg2x2_cells import CELLS, resolved_switches  # noqa: E402

_ITER = re.compile(r"^\[iter\s+(\d+)/(\d+)\]\s+(.*)$")
_KV = re.compile(r"(\w+)=(-?nan|-?inf|[0-9.eE+-]+|True|False)")
STARTUP_PREFIXES = ("[edge_tv]", "[wavelet]", "[phase-tv]", "[init_cloud]", "[single-contrast]", "[dtv-proj]", "[guide]",
                    "[nl]", "[FLAIR-private]", "[tv_shape]")
CROSS_R = ("R_dtv", "R_edge", "R_vtv", "R_dtvp", "R_nl")
PRIOR_R = ("R_geo", "R_wav", "R_ptv")
# fixed in every cell, independent of resolved_switches (ruling R6)
FIXED_EXPECT = {"lambda_nl": 0.0, "lambda_nl_plain": 0.0, "lambda_nl_contrast": 0.0, "guide_set": False,
                "kspace_freq_adaptive": True, "kspace_high_weight": 2.2,
                "kspace_high_weight_plain": None, "kspace_freq_curriculum": False, "kspace_center_fraction": 0.05,
                "kspace_center_weight": 0.3, "kspace_mid_weight": 1.0, "prewhiten": False, "lambda_private_sp": 0.0,
                "num_private_points": 0, "adaptive_control_enable": False, "max_radius": 20, "init_scale_mode": "3nn",
                "tv_eps": 0.04, "edge_tv_kappa_source": "zf"}
REG_ON_EXPECT = {"wav_name": "db4", "wav_levels": 1, "edge_tv_kappa_q": 0.85, "edge_tv_wfloor": 0.1}


def md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_iter_line(line: str) -> dict:
    m = _ITER.match(line.strip())
    if not m:
        raise ValueError(f"not an iter line: {line!r}")
    d = {"it": int(m.group(1)), "max_iter": int(m.group(2))}
    for k, v in _KV.findall(m.group(3)):
        if v in ("True", "False"):
            d[k] = v == "True"
            continue
        try:
            d[k] = float(v)
        except ValueError:
            d[k] = float("nan")
    return d


def _finite(d: dict, k: str) -> bool:
    v = d.get(k)
    return isinstance(v, float) and math.isfinite(v)


def _resolved_from_cfg(cfg: dict) -> dict:
    loss = cfg.get("loss", {})
    geo = loss["lambda_geo"] if "lambda_geo" in loss else loss.get("tv_weight", 0.0)
    out = {"lambda_geo": float(geo), "edge_tv": bool(loss.get("edge_tv", False))}
    for k in ("lambda_wav", "lambda_wav_plain", "lambda_ptv", "lambda_edge", "lambda_dtv", "lambda_vtv",
              "lambda_dtv_proj", "edge_tv_guide_alpha", "ptv_guide_alpha"):
        out[k] = float(loss.get(k, 0.0) or 0.0)
    if "lambda_wav_plain" not in loss:
        out["lambda_wav_plain"] = out["lambda_wav"]
    out["lambda_nl"] = float(loss.get("lambda_nl", 0.0) or 0.0)
    out["lambda_nl_plain"] = float(loss.get("lambda_nl_plain", loss.get("lambda_nl", 0.0)) or 0.0)
    out["lambda_nl_contrast"] = float(loss.get("lambda_nl_contrast", loss.get("lambda_nl", 0.0)) or 0.0)
    out["kspace_high_weight"] = float(loss.get("kspace_high_weight", 0.0) or 0.0)
    out["kspace_freq_adaptive"] = bool(loss.get("kspace_freq_adaptive", False))
    out["guide_set"] = bool(loss.get("guide_path_plain") or loss.get("guide_path_contrast"))
    # knobs the spec freezes; absent -> the trainer's own default (joint_trainer.py / single_contrast_trainer.py)
    khp = loss.get("kspace_high_weight_plain")
    out["kspace_high_weight_plain"] = None if khp is None else float(khp)
    out["kspace_freq_curriculum"] = bool(loss.get("kspace_freq_curriculum", False))
    out["kspace_center_fraction"] = float(loss.get("kspace_center_fraction", 0.05))
    out["kspace_center_weight"] = float(loss.get("kspace_center_weight", 0.3))
    out["kspace_mid_weight"] = float(loss.get("kspace_mid_weight", 1.0))
    out["prewhiten"] = bool(loss.get("prewhiten", False))
    out["lambda_private_sp"] = float(loss.get("lambda_private_sp", 0.0) or 0.0)
    out["num_private_points"] = int(cfg.get("gaussian", {}).get("num_private_points", 0))
    out["adaptive_control_enable"] = bool(cfg.get("adaptive_control", {}).get("enable", False))
    out["max_radius"] = float(cfg.get("voxelizer", {}).get("max_radius", 20))
    out["init_scale_mode"] = str(cfg.get("gaussian", {}).get("init_scale_mode", "fixed"))
    out["tv_eps"] = float(loss.get("tv_eps", 0.0))
    out["edge_tv_kappa_source"] = str(loss.get("edge_tv_kappa_source", "render"))
    out["wav_name"] = str(loss.get("wav_name", "db2"))
    out["wav_levels"] = int(loss.get("wav_levels", 3))
    out["edge_tv_kappa_q"] = float(loss.get("edge_tv_kappa_q", 0.75))
    out["edge_tv_wfloor"] = float(loss.get("edge_tv_wfloor", 0.1))
    data = cfg.get("data", {})
    for ch, blk in data.items():
        out[f"prewhiten_{ch}"] = None if blk.get("prewhiten") is None else bool(blk["prewhiten"])
        out[f"known_zero_{ch}"] = bool(blk.get("known_zero", False))
    out["plain_name"] = str(data.get("plain", {}).get("name", ""))
    out["seed"] = int(cfg.get("training", {}).get("seed", 42))
    out["max_iterations"] = int(cfg.get("training", {}).get("max_iterations", 0))
    g = cfg.get("gaussian", {})
    out["initial_num_points"] = int(g.get("initial_num_points", 200000))
    out["init_mode"] = str(g.get("init_mode", "uniform"))
    out["init_support_path"] = str(g.get("init_support_path", ""))
    return out


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, str) or isinstance(b, str) or isinstance(a, bool) or isinstance(b, bool):
        return a == b
    return abs(float(a) - float(b)) < 1e-12


def audit(cell: str, resolved_cfg: dict, log_text: str, args: dict | None = None) -> dict:
    """args (optional): the launcher's seed / npts / init / support / iters, compared with the resolved config."""
    want = {**resolved_switches(cell), **FIXED_EXPECT}
    got = _resolved_from_cfg(resolved_cfg)
    if CELLS[cell].reg:
        want.update(REG_ON_EXPECT)
    for ch in resolved_cfg.get("data", {}):
        want[f"prewhiten_{ch}"] = None
    want["known_zero_plain"] = got.get("plain_name") == "FLAIR"
    if "contrast" in resolved_cfg.get("data", {}):
        want["known_zero_contrast"] = True
    if args is not None:
        want.update({"seed": int(args["seed"]), "initial_num_points": int(args["npts"]), "init_mode": str(args["init"]),
                     "max_iterations": int(args["iters"])})
        if args["init"] == "support_file":
            want["init_support_path"] = str(args["support"])
    resolved_ok = all(k in got and _same(got[k], want[k]) for k in want)
    lines = log_text.splitlines()
    startup = [l for l in lines if l.startswith(STARTUP_PREFIXES)]
    reg = CELLS[cell].reg
    has = {p: any(l.startswith(p) for l in startup) for p in STARTUP_PREFIXES}
    startup_ok = (not any(has[p] for p in ("[dtv-proj]", "[guide]", "[nl]", "[FLAIR-private]"))
                  and (all(has[p] for p in ("[edge_tv]", "[wavelet]", "[phase-tv]")) if reg
                       else not any(has[p] for p in ("[edge_tv]", "[wavelet]", "[phase-tv]"))))
    iters = [parse_iter_line(l) for l in lines if _ITER.match(l.strip())]
    loss_ok = bool(iters)
    for d in ([iters[0], iters[-1]] if iters else []):
        need = ("dc_p",) + CROSS_R + (("R_geo", "R_wav") if not reg else ())
        if not all(_finite(d, k) for k in need):
            loss_ok = False
        if any(d.get(k, 0.0) != 0.0 for k in CROSS_R):
            loss_ok = False
        if not reg and any(d.get(k, 0.0) != 0.0 for k in PRIOR_R):   # R_ptv only when the trainer logs it
            loss_ok = False
        if "R_ptv" in d and not _finite(d, "R_ptv"):
            loss_ok = False
    return {"resolved_ok": resolved_ok, "startup_ok": startup_ok, "loss_ok": loss_ok,
            "ok": resolved_ok and startup_ok and loss_ok, "want": want, "got": got,
            "startup_lines": startup, "iter_first": iters[0] if iters else None, "iter_last": iters[-1] if iters else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp"); ap.add_argument("cell", choices=sorted(CELLS))
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--npts", type=int, required=True)
    ap.add_argument("--init", required=True); ap.add_argument("--support", required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--arc", default="results_arc")
    a = ap.parse_args()
    arc = ROOT / a.arc
    log_text = (arc / f"{a.exp}.log").read_text()
    cfg = yaml.safe_load(open(arc / a.exp / a.exp / "config.yaml"))
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    git_dirty = [l[3:] for l in subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                               text=True).stdout.splitlines()]
    data_md5 = {}
    for ch, blk in cfg["data"].items():
        data_md5[ch] = {k: md5(blk[k]) for k in ("data_path", "csm_path", "mask_path") if blk.get(k)}
    res = audit(a.cell, cfg, log_text, {"seed": a.seed, "npts": a.npts, "init": a.init, "support": a.support,
                                        "iters": a.iters})
    cell = CELLS[a.cell]
    man = {"exp": a.exp, "cell": a.cell, "trainer": cell.trainer, "contrast": cell.contrast, "reg": cell.reg,
           "seed": a.seed, "npts": a.npts, "iters": a.iters, "init": a.init, "support": a.support,
           "support_md5": md5(a.support) if a.init == "support_file" else None,
           "git_head": git_head, "git_dirty": git_dirty, "config": cell.config, "config_md5": md5(ROOT / cell.config),
           "data_md5": data_md5, "resolved": res["got"], "expected": res["want"], "startup_lines": res["startup_lines"],
           "iter_first": res["iter_first"], "iter_last": res["iter_last"],
           "audit": {k: res[k] for k in ("resolved_ok", "startup_ok", "loss_ok", "ok")}}
    (arc / f"{a.exp}_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"[sg2x2-manifest] {a.exp} cell={a.cell} audit={man['audit']}")
    sys.exit(0 if man["audit"]["ok"] else 3)


if __name__ == "__main__":
    main()
