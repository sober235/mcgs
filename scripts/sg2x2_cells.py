# scripts/sg2x2_cells.py
#!/usr/bin/env python3
"""The four cells of the shared-geometry 2x2 ablation (spec docs/experiments/2026-10-09-mcgs-shared-geometry-2x2-ablation.md,
section 2-4) as arc/cycle.py arguments. This file is the only place the cells are defined: run_sg2x2.sh asks it for
the argument list, scripts/sg2x2_manifest.py asks it what the resolved config must say.

    A      MC-GS + Reg        joint   shared geometry, frozen per-channel recipe, no cross-contrast weight
    B_T1   GS-single + Reg    single  T1 alone, the same per-channel recipe, its own wavelet weight
    B_FLAIR                   single  FLAIR alone
    C      MC-GS - Reg        joint   data consistency only
    D_T1   GS-single - Reg    single  T1 alone, data consistency only
    D_FLAIR                   single  FLAIR alone, data consistency only

Everything the spec freezes across cells (k-space weighting, warm-up, max_radius, 3-NN scales, point count, seed, the
init support file) is in COMMON; the recipe block equals run_wd_mcgs.sh with lambda_edge 0.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

WAV = {"T1": "0.12", "FLAIR": "0.10"}          # the frozen AF16 phase-3 per-contrast wavelet weights


@dataclass(frozen=True)
class Cell:
    trainer: str          # "joint" | "single"
    contrast: str | None  # None for joint, "T1" | "FLAIR" for single
    reg: bool             # per-channel explicit priors on (A/B) or off (C/D)
    config: str           # base config under configs/sg2x2/


CELLS = {
    "A": Cell("joint", None, True, "configs/sg2x2/af16_jyz_joint.yaml"),
    "B_T1": Cell("single", "T1", True, "configs/sg2x2/af16_jyz_single_T1.yaml"),
    "B_FLAIR": Cell("single", "FLAIR", True, "configs/sg2x2/af16_jyz_single_FLAIR.yaml"),
    "C": Cell("joint", None, False, "configs/sg2x2/af16_jyz_joint.yaml"),
    "D_T1": Cell("single", "T1", False, "configs/sg2x2/af16_jyz_single_T1.yaml"),
    "D_FLAIR": Cell("single", "FLAIR", False, "configs/sg2x2/af16_jyz_single_FLAIR.yaml"),
}

CROSS_OFF = ["--lambda_edge", "0", "--lambda_dtv", "0", "--lambda_vtv", "0",
             "--set", "loss.lambda_dtv_proj=0", "--set", "loss.edge_tv_guide_alpha=0", "--set", "loss.ptv_guide_alpha=0",
             "--set", "loss.lambda_nl=0"]


def _wavelet(cell: Cell) -> tuple[str, str]:
    """(lambda_wav_plain, lambda_wav) for the cell: the joint model keeps T1 on the plain slot and FLAIR on the
    contrast slot; a single-contrast run puts its one contrast's weight on both keys."""
    if cell.contrast is None:
        return WAV["T1"], WAV["FLAIR"]
    return WAV[cell.contrast], WAV[cell.contrast]


def cell_args(cell: str, npts: int, seed: int, support: str, init: str = "support_file",
              iters: int = 4000, warmup: int = 500) -> list[str]:
    c = CELLS[cell]
    args = ["--iters", str(iters), "--warmup", str(warmup), "--max_radius", "20", "--kspace_high", "2.2",
            "--trainer", c.trainer, "--base_cfg", c.config]
    if c.reg:
        wp, wc = _wavelet(c)
        args += ["--lambda_geo", "0.10", "--edge_tv", "--edge_tv_kappa_q", "0.85", "--edge_tv_wfloor", "0.10",
                 "--tv_eps", "0.04", "--lambda_ptv", "0.03",
                 "--set", "loss.wav_name=db4", "--set", "loss.wav_levels=1",
                 "--set", f"loss.lambda_wav_plain={wp}", "--set", f"loss.lambda_wav={wc}"]
    else:
        args += ["--lambda_geo", "0", "--tv_eps", "0.04", "--lambda_ptv", "0",
                 "--set", "loss.edge_tv=false", "--set", "loss.lambda_wav=0", "--set", "loss.lambda_wav_plain=0"]
    args += CROSS_OFF
    args += ["--set", "gaussian.init_scale_mode=3nn", "--set", "loss.edge_tv_kappa_source=zf",
             "--set", f"gaussian.initial_num_points={int(npts)}", "--set", f"training.seed={int(seed)}",
             "--set", "gaussian.num_private_points=0", "--set", "adaptive_control.enable=false"]
    if init == "support_file":
        args += ["--set", "gaussian.init_mode=support_file", "--set", f"gaussian.init_support_path={support}"]
    elif init == "head":
        args += ["--set", "gaussian.init_mode=head"]
    else:
        raise ValueError(f"init must be 'support_file' or 'head', got {init!r}")
    return args


def resolved_switches(cell: str) -> dict:
    """What the trainer's resolved config (results_arc/<exp>/<exp>/config.yaml) must say for this cell."""
    c = CELLS[cell]
    out = {"lambda_edge": 0, "lambda_dtv": 0, "lambda_vtv": 0, "lambda_dtv_proj": 0,
           "edge_tv_guide_alpha": 0, "ptv_guide_alpha": 0}
    if c.reg:
        wp, wc = _wavelet(c)
        out.update({"lambda_geo": 0.10, "edge_tv": True, "lambda_wav_plain": float(wp), "lambda_wav": float(wc),
                    "lambda_ptv": 0.03})
    else:
        out.update({"lambda_geo": 0, "edge_tv": False, "lambda_wav_plain": 0, "lambda_wav": 0, "lambda_ptv": 0})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell", choices=sorted(CELLS))
    ap.add_argument("--npts", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--init", default="support_file", choices=["support_file", "head"])
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--warmup", type=int, default=500)
    a = ap.parse_args()
    print("\n".join(cell_args(a.cell, a.npts, a.seed, a.support, a.init, a.iters, a.warmup)))


if __name__ == "__main__":
    main()
