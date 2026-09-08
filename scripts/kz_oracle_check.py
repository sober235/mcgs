#!/usr/bin/env python3
"""ORACLE diagnostic (uses the full k-space; never part of the fit): grades the estimated Z per contrast for the
winner_v2_edgeTV data config — line precision/recall against the true numerically-zero lines, and the ENERGY the
false positives carry: total, and the excess above the readout-end noise floor (a false positive at or below the
noise floor is harmless: the model cannot predict noise, so forcing a zero there never raises the expected error)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paired_gs"))
from trainers.joint_trainer import _load_dataset                       # noqa: E402
from data.private_paths import resolve_subject_paths                   # noqa: E402
from data.known_zero import ZERO_REL, estimate_known_zero, estimate_noise_cov   # noqa: E402

cfg = yaml.safe_load(open(ROOT / "configs/winner_v2_edgeTV.yaml"))
out = {}
for sub in ("plain", "contrast"):
    d = dict(cfg["data"][sub])
    for k in ("data_path", "csm_path", "mask_path"):
        d[k] = resolve_subject_paths(d[k])
    ds = _load_dataset(d)
    y, m, ro = ds.kspace_undersampled_complex, ds.mask, int(ds.readout_axis)
    Z, info = estimate_known_zero(y, m, ro)
    _, winfo = estimate_noise_cov(y, m, Z, info)
    nl = winfo["noise_level"]
    Cn, Rn = y.shape[0], m.shape[ro]
    a_full = torch.movedim(ds.kspace_multicoil.abs().amax(0), ro, -1)
    true_zero = (a_full.amax(-1) < ZERO_REL * float(a_full.max())).numpy()                 # ORACLE
    e_line = torch.movedim((ds.kspace_multicoil.abs() ** 2).sum(0), ro, -1).sum(-1).numpy()  # ORACLE line energies
    z_pe = torch.movedim(Z, ro, -1)[:, :, 0].numpy()
    tp = int((z_pe & true_zero).sum()); fp_mask = z_pe & ~true_zero; fp = int(fp_mask.sum()); fn = int((~z_pe & true_zero).sum())
    E = float(e_line.sum()); fp_e = e_line[fp_mask]
    excess = float(np.maximum(fp_e - (nl * Cn * Rn if nl else 0.0), 0.0).sum()) if fp else 0.0
    out[sub] = {"name": d.get("name", sub), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                "false_positive_lines": fp, "fp_energy_fraction": float(fp_e.sum() / E) if fp else 0.0,
                "fp_excess_over_noise_fraction": excess / E,
                "max_fp_line_energy_over_noise": float(fp_e.max() / (nl * Cn * Rn)) if (fp and nl) else 0.0,
                "true_zero_fraction": float(true_zero.mean()), "frac_z": info.frac_z, "cuts": info.cuts,
                "ellipse": info.ellipse, "noise_level": nl, "n_outliers": info.n_outliers,
                "n_significant_outliers": info.n_significant_outliers, "n_nonzero_in_z": info.n_nonzero_in_z}
    print(sub, json.dumps(out[sub]), flush=True)
Path(ROOT / "results_arc" / "kz_oracle_check.json").write_text(json.dumps(out, indent=2))
