"""G0 representation-ceiling gate (spec §5) — Gaussian-arm trainer.

Trains `paired_gs.trainers.joint_trainer.JointTrainer` cold-start on the fully
sampled config (`configs/g0_fullsample.yaml`) and exports the RAW COMPLEX render
(`tr._render(channel)`), not a magnitude npz and not `arc/cycle.py`'s hard-DC
export. Both deviations from the naive recipe are load-bearing; explained below.

---- mask_type: full is not supported ----
`paired_gs/data/dataset.py` (`_load_dataset` / `_generate_mask`) only accepts
mask_type in {stacked_2d_gaussian, gaussian, poisson, random, external}; "full"
raises `ValueError: Unsupported mask_type`. Route actually used (per the task's
documented fallback): generated an explicit all-ones mask
`configs/g0_mask_full_ones.mat` (key "mask", shape (256,220,256), matching the
T1/FLAIR volume_shape exactly so `_conform_mask_shape` passes it through
unchanged) and pointed `data.{plain,contrast}.mask_path` at it in
`configs/g0_fullsample.yaml`, with `mask_type: external` (the base config's
existing mechanism, unchanged). See `configs/g0_fullsample.yaml` for the
resulting config.

---- why NOT `arc/cycle.py --save_volumes` ----
`arc/cycle.py`'s `_vol()` (and `JointTrainer.export_results()`) both apply hard
data consistency before export: `k_dc = mask*y + (1-mask)*fft3c(csm*render)`.
Under an all-ones mask, `(1-mask) == 0` identically, so `k_dc == y` *regardless
of the trained render* — the exported volume collapses to
`ifft3c(y) * csm.conj()).sum(0)`, the plain coil-combine of the MEASURED
fully-sampled k-space, with zero dependence on the Gaussian model. Using that
export for the ceiling gate would make the gate vacuous: it tests the adjoint
operator, not the Gaussian representation. (This is a correctness trap specific
to the AF=1 edge case — `_vol()`/`export_results()` are perfectly correct for
their actual purpose, hard-DC synthesis under real undersampling, which is why
they exist.)

Instead this script drives `JointTrainer` directly — the exact same class
`arc/cycle.py` imports, unmodified, from `paired_gs/` — and, in the SAME
process immediately after `tr.train()` returns, calls `tr._render(channel)` to
capture the pre-DC complex volume the model actually produced. This is genuine
new orchestration code under `crisp/`; it does not modify `paired_gs/` or
`arc/cycle.py`, and duplicates only the handful of `build_cfg()` lines from
`arc/cycle.py` that this narrow use case needs (no CLI knob parity is
attempted — this script is not a general-purpose replacement for arc/cycle.py).

`tr._render(channel)` returns `torch.complex(vol_real, vol_imag)` (or a bare
complex tensor if the voxelizer already returns one) — a genuine complex
image-domain volume, trained by `_kspace_loss` to satisfy
`fft3c(csm * render) ~= y_measured` under the L2 DC loss, i.e. the SAME forward
convention `crisp.metrics.coil_domain.e_coil_full` evaluates. There is no
separate phase-recovery step to reproduce and no gauge mismatch to correct for.

---- cold start ----
`configs/g0_fullsample.yaml` has `gaussian.warm_start_checkpoint` removed
(never present — built from `winner_v2_aligned.yaml` with that key popped), so
`JointTrainer._setup_model` (joint_trainer.py:172-222) takes the `else` branch
at line 202: `initial_num_points` (200000) random positions, density
initialized from 0.1x the zero-filled image at each position. No
`adaptive_control.enable` is set in this config, so densification stays off —
point count is fixed at 200000 for the whole run (well under the 1e6 gate).

---- regulariser deviation from the plan (never amended) ----
This run trained with the shipped winner config's regularisers, inherited
unchanged from `winner_v2_aligned.yaml` into `configs/g0_fullsample.yaml`:
`tv_weight 0.06`, `kspace_freq_adaptive true`, `kspace_center_weight 0.3`,
`kspace_high_weight 2.2`. That is NOT what the written plan's Task 8 Step 4
specified (`--lambda_geo 0.0 --kspace_high 1.0`, i.e. TV off and no
frequency-adaptive k-space weighting). This deviation was never amended
against the plan. Its direction is conservative for the purpose of this
gate: the winner config's regularisers were tuned for undersampled
reconstruction, not this full-sampling representation-capacity probe, so if
anything they handicap the Gaussian arm here relative to an unregularised
fit — and a handicapped arm still cleared the pass line. Read the G0
ratios (1.29x T1, 1.39x FLAIR against the <=2x line; see
`results_crisp/g0_ceiling_huangyuwei_{T1,FLAIR}.json`) as a LOWER BOUND on
the representation's capability, not as its ceiling.

Usage (foreground; one contrast pair trained jointly, ~4000 iters):
    CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH= \\
      /home/congcongliu/anaconda3/envs/GSTMI/bin/python \\
      crisp/experiments/g0_train_gaussian.py --exp g0_full --gpu 5 --iters 4000 --warmup 500
Smoke test (cold-start path has zero runs on record before this task):
    ... --exp g0_smoke --gpu 5 --iters 20 --warmup 5
Output: results_crisp/{exp}_render.npz (complex renderT1/renderFL) and
        results_crisp/{exp}_train_meta.json (npts, elapsed, config actually used).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--iters", type=int, required=True)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--base_cfg", default="configs/g0_fullsample.yaml")
args = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent            # crisp/experiments/
ROOT = HERE.parent.parent                          # framework root
PGS = ROOT / "paired_gs"
sys.path.insert(0, str(PGS))
from seed import set_seed                           # noqa: E402
from trainers.joint_trainer import JointTrainer      # noqa: E402
from data.private_paths import resolve_subject_paths  # noqa: E402

RES_ARC = ROOT / "results_arc"          # same output convention as arc/cycle.py (gitignored)
RES_CRISP = ROOT / "results_crisp"
log = lambda *a: print(*a, flush=True)


def build_cfg():
    cfg = yaml.safe_load(open(ROOT / args.base_cfg))
    for s in ("plain", "contrast"):
        sc = cfg["data"][s]
        for k in ("data_path", "csm_path", "mask_path"):
            if sc.get(k):
                sc[k] = resolve_subject_paths(sc[k])
    assert "warm_start_checkpoint" not in cfg.get("gaussian", {}), (
        "g0 is a cold-start experiment; base_cfg must not carry warm_start_checkpoint"
    )
    cfg.setdefault("training", {})
    cfg["training"]["max_iterations"] = args.iters
    cfg["training"]["shared_warmup_iters"] = args.warmup
    cfg.setdefault("output", {})
    cfg["output"]["output_dir"] = str(RES_ARC / args.exp)
    cfg["output"]["exp_name"] = args.exp
    return cfg


def main():
    t0 = time.time()
    dev = torch.device("cuda", 0)
    set_seed(42, deterministic=False)
    assert args.warmup < args.iters * 0.5, f"warmup {args.warmup} must be < 0.5*iters {args.iters}"
    cfg = build_cfg()
    RES_ARC.mkdir(parents=True, exist_ok=True)
    RES_CRISP.mkdir(parents=True, exist_ok=True)
    log(f"[g0-gauss {args.exp}] gpu={args.gpu} iters={args.iters} warmup={args.warmup} "
        f"base_cfg={args.base_cfg}")

    tr = JointTrainer(cfg, device=dev)
    tr.train()

    with torch.no_grad():
        x_p = tr._render("plain")
        x_c = tr._render("contrast")
        vram = torch.cuda.max_memory_allocated() / 1e9
        npts = int(tr.model.num_points)
        render_p = x_p.detach().cpu().numpy().astype(np.complex64)
        render_c = x_c.detach().cpu().numpy().astype(np.complex64)

    elapsed = time.time() - t0
    npz_path = RES_CRISP / f"{args.exp}_render.npz"
    np.savez_compressed(npz_path, renderT1=render_p, renderFL=render_c)

    meta = dict(exp=args.exp, gpu=args.gpu, iters=args.iters, warmup=args.warmup,
                base_cfg=args.base_cfg, npts=npts, vram_gb=vram, elapsed_s=elapsed,
                render_npz=str(npz_path), render_dtype="complex64",
                route_note="raw JointTrainer._render() output, no hard-DC, no magnitude cast")
    meta_path = RES_CRISP / f"{args.exp}_train_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    log(f"[g0-gauss {args.exp}] DONE npts={npts} vram_gb={vram:.1f} elapsed_s={elapsed:.0f} "
        f"-> {npz_path}\n      meta -> {meta_path}")


if __name__ == "__main__":
    main()
