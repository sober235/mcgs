#!/usr/bin/env python3
"""build_ctrl_mask.py — generalised end-to-end driver for a vd-Poisson control mask at an
arbitrary target acceleration, DC-aligned exactly like the shipped AF=12 mask.

Sibling of `scripts/build_ctrlB_mask.py` (kept untouched — it is bound to control B's own
shipped artifact, `ctrlB_indep_af12_seed1_ALIGNED.mat`, which this script must not overwrite or
otherwise disturb). This script generalises the same draw/gate/save pipeline so the target
acceleration is a CLI parameter, for control A's repair: instead of an all-ones (fully sampled)
mask — which under hard-DC (`k_dc = m*y + (1-m)*k_model`, (1-m)=0 everywhere) discards the
model's output entirely and is vacuous — draw a MILD-acceleration plane (target AF=2 by default)
that still leaves a real null space for the model to fill.

Same generator, same fixed knobs as the shipped AF=12 recipe (`calib=24, ellipse=0, pp=1`); only
the per-axis accel factors change with the target AF, via the formula used by this repo's other
AF-parameterised generator (`.../3dgsVC/mask/gen_compl_indep.m`):
    ax = sqrt(AF * 3/4),  ay = sqrt(AF * 4/3)        (ax*ay = AF; ax=3,ay=4 at AF=12, matching
                                                       the shipped call exactly)

CRITICAL FINDING carried over from build_ctrlB_mask.py (re-verified for the AF=2 case, see the
docstring of `draw_plane` below): `rng()` has no observed effect on `vdPoisMex`'s output within a
single MATLAB process; the entropy is process-level. A fresh process is required for a new draw.

SELECTION DISCIPLINE — identical structure to build_ctrlB_mask.py, not an implementation detail:
the retry loop below decides accept/reject using `net_AF` ONLY, against the tolerance documented
at the CLI default below. `mask@DC==1` is evaluated exactly ONCE after the loop, as a hard
assertion that ABORTS the run — never a retry condition. There is no "differs from shipped" gate
here: control A's AF=2 mask differs from the shipped AF=12 mask by design (different generator
parameters entirely), so that check does not apply and is deliberately not ported.

TOLERANCE, why 0.15 (documented, not tuned to the accepted draw): three independent calibration
launches at AF=2 (ax=sqrt(1.5), ay=sqrt(8/3)) before this gate was fixed landed net_AF at
2.077, 2.082, 2.081 — a tight cluster (spread 0.005) sitting consistently ~4% ABOVE the nominal
target of 2.0 (the opposite direction from the AF=12 case, where net_AF landed ~1.4% BELOW its
nominal 12 because the forced-ACS region adds samples back; at AF=2 the ACS is a much smaller
fraction of the already-~48%-sampled plane, so the net effect of ax,ay not being exact integers
dominates instead). Unlike control B, there is no shipped instance to rate-match — "close to 2"
only needs to keep the mask unambiguously MILD (nowhere near AF=12's ~11.8, nowhere near the
vacuous all-ones case). `--af-tol 0.15` (7.5% relative) comfortably covers the observed cluster
plus margin for further launch-to-launch drift, while keeping the accepted net_AF pinned to
roughly [1.85, 2.15].

Usage:
    PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python \\
        scripts/build_ctrl_mask.py --af 2.0 [--af-tol 0.15] [--out PATH] [--json-out PATH] \\
        [--overwrite] [--max-attempts N] [--seed N]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diag_artefact.controls import align_like_shipped, assert_target_af_mask, derive_roll

MATLAB = "/usr/local/bin/matlab"
ESPIRIT_UTILS = "/data0/congcong/code/ESPIRiT/utils"
MDIR = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask"
UNALIGNED_MASK = f"{MDIR}/mask3d_vdpois_af12_transverse.mat"
ALIGNED_MASK = f"{MDIR}/mask3d_vdpois_af12_transverse_ALIGNED.mat"
DATA_PATH = ("/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/"
             "Train_3dBrain_huangyuwei_20230705-160927-1451_161015_T1_trainer.mat")
DEFAULT_OUT = f"{MDIR}/ctrlA_af2_indep_seed1_ALIGNED.mat"
DEFAULT_JSON = "results_arc/diag_artefact/ctrlA_af2_mask_report.json"
DEFAULT_AF = 2.0
DEFAULT_AF_TOL = 0.15      # see module docstring "TOLERANCE" for the calibration behind this
MAX_ATTEMPTS_DEFAULT = 20  # generous, matching build_ctrlB_mask.py's observed pass-rate margin


def accel_factors(target_af: float) -> tuple[float, float]:
    """ax,ay per `.../3dgsVC/mask/gen_compl_indep.m`'s AF-parameterised formula. ax*ay=target_af;
    reduces to the shipped (3,4) exactly at target_af=12."""
    return math.sqrt(target_af * 3 / 4), math.sqrt(target_af * 4 / 3)


def draw_plane(tmp_mat: Path, target_af: float, seed: int) -> tuple[np.ndarray, float]:
    """Launch ONE fresh MATLAB process and draw one vd-Poisson PE-plane at `target_af`.

    A fresh process is required for a new draw (see module docstring). `seed` is passed through
    to rng() for provenance only.
    """
    ax, ay = accel_factors(target_af)
    script = (
        f"addpath('{ESPIRIT_UTILS}');\n"
        f"rng({seed});\n"
        f"mask = single(vdPoisMex(220,256,220,256,{ax!r},{ay!r},24,0,1));\n"
        "net = numel(mask)/nnz(mask);\n"
        f"save('{tmp_mat}', 'mask', 'net', '-v7');\n"
        "fprintf('DRAW nnz=%d net_AF=%.6f\\n', nnz(mask), net);\n"
    )
    script_path = tmp_mat.with_suffix(".m")
    script_path.write_text(script)
    proc = subprocess.run(
        [MATLAB, "-batch", f"run('{script_path}')"],
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0 or not tmp_mat.exists():
        raise RuntimeError(
            f"MATLAB draw failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    mask2d = sio.loadmat(str(tmp_mat))["mask"].astype(np.float32)
    net_af = float(mask2d.size / mask2d.sum())
    return mask2d, net_af


def find_data_dc() -> tuple:
    """Locate the data's true k-space DC, mirroring dataset.py's _load_h5_kspace /
    _normalize_kspace_layout. Identical to build_ctrlB_mask.py's helper (duplicated rather than
    imported so this script stays a self-contained sibling and never touches that file)."""
    with h5py.File(DATA_PATH, "r") as f:
        raw = f["ksp"][:]
    if raw.dtype.names is not None and "real" in raw.dtype.names:
        ksp = raw["real"].astype(np.float64) + 1j * raw["imag"].astype(np.float64)
    else:
        ksp = raw.astype(np.complex128)
    energy = (np.abs(ksp) ** 2).sum(axis=-1)  # coil axis is last for this file
    return np.unravel_index(int(np.argmax(energy)), energy.shape)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--af", type=float, default=DEFAULT_AF, help="target net acceleration factor")
    ap.add_argument("--af-tol", type=float, default=DEFAULT_AF_TOL,
                     help="acceptance tolerance |net_AF - af| < af-tol (see module docstring)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="3-D mask .mat output path")
    ap.add_argument("--json-out", default=DEFAULT_JSON, help="gate/attempt-log report path")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing --out/--json-out")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    ap.add_argument("--seed", type=int, default=1,
                     help="passed to MATLAB rng(); has NO observed effect on the draw (see module docstring)")
    args = ap.parse_args()

    out_path = Path(args.out)
    json_path = Path(args.json_out)
    if not args.overwrite:
        existing = [p for p in (out_path, json_path) if p.exists()]
        if existing:
            print(f"REFUSING to overwrite existing output(s) without --overwrite: "
                  f"{[str(p) for p in existing]}", file=sys.stderr)
            sys.exit(1)

    unal = sio.loadmat(UNALIGNED_MASK)["mask"].astype(np.float32)
    alig = sio.loadmat(ALIGNED_MASK)["mask"].astype(np.float32)
    roll = derive_roll(unal, alig)
    print(f"derived roll: {roll} (same derivation as build_ctrlB_mask.py, from the shipped "
          f"unaligned/ALIGNED AF=12 pair -- reused as-is since the roll corrects a data-registration "
          f"offset, not a mask-generator property, so it applies to any PE-plane mask of this shape)")

    dc = find_data_dc()
    print(f"data k-space DC at {dc}  (matrix center "
          f"{tuple(s // 2 for s in (256, 220, 256))})")

    ax, ay = accel_factors(args.af)
    print(f"target AF={args.af} -> ax={ax:.6f} ay={ay:.6f} (ax*ay={ax*ay:.6f})")

    attempts = []
    accepted_2d = None
    with tempfile.TemporaryDirectory() as td:
        for i in range(1, args.max_attempts + 1):
            tmp_mat = Path(td) / f"draw_{i}.mat"
            mask2d, net_af = draw_plane(tmp_mat, target_af=args.af, seed=args.seed)
            abs_delta = abs(net_af - args.af)
            nnz = int(mask2d.sum())
            # Selection on net_AF ONLY -- see module docstring "SELECTION DISCIPLINE". Do not add
            # any other condition to this `accept` expression.
            accept = abs_delta < args.af_tol
            attempts.append({"attempt": i, "nnz": nnz, "net_af": net_af,
                              "abs_delta_af": abs_delta, "accepted": accept})
            print(f"attempt {i}: nnz={nnz} net_AF={net_af:.6f} abs_delta={abs_delta:.6f} "
                  f"{'ACCEPTED' if accept else 'rejected'}")
            if accept:
                accepted_2d = mask2d
                break

    if accepted_2d is None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({"attempts": attempts, "status": "FAILED_NO_DRAW_WITHIN_TOLERANCE"}, indent=2))
        raise RuntimeError(
            f"No draw landed within |net_AF-{args.af}|<{args.af_tol} in {args.max_attempts} "
            f"attempts. Full attempt log written to {json_path}."
        )

    m3 = align_like_shipped(accepted_2d, (256, 220, 256), roll)

    # HARD ASSERTION -- not a retry condition (see module docstring). Raising aborts the whole
    # run right here; there is no `except` around it and no way back into the draw loop above.
    gate_report = assert_target_af_mask(m3, dc_index=(dc[0], dc[1]), target_af=args.af, tol=args.af_tol)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(out_path), {"mask": m3, "net_af": np.float32(gate_report["net_af"])})

    report = {
        **gate_report,
        "roll": [int(x) for x in roll],
        "dc": [int(x) for x in dc],
        "ax": ax, "ay": ay,
        "attempts": attempts,
        "n_attempts": len(attempts),
        "out": str(out_path),
        "status": "OK",
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    print(f"wrote {json_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
