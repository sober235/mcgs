#!/usr/bin/env python3
"""smoke_test.py — prove the GENERALIZED paired_gs core + winner_v2 config reload the
validated winner-v2 checkpoint and reproduce its ESPIRiT-CSM scores.

What it exercises end-to-end (all through the generalized, contrast-agnostic code):
  * import rewiring (paired_gs as root, `from seed import set_seed`)
  * config schema with data.plain.name / data.contrast.name
  * dataset loading (k-space + external ESPIRiT CSM + transverse vdPoisson mask)
  * JointGaussianModel3D warm-start from the winner-v2 best.pth (576902 Gaussians)
  * tile-CUDA voxelizer JIT compile + _render(plain) / _render(contrast)
  * hard-DC synthesis + SENSE coil-combine (the scored reconstruction path)

It does NOT retrain (max_iterations=2, warmup=0) — reloading the trained Gaussians and
re-scoring is a faithful, ~3-4 min reproduction of the winner numbers without the 43-min
retrain. Scores must land within tolerance of the reference:
    plain (T1)      PSNR 37.738  SSIM 0.8755
    contrast (FLAIR) PSNR 38.143  SSIM 0.894

Run on amax GSTMI (torch 2.7.1+cu118) pinned to an idle GPU:
    PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      /home/congcongliu/anaconda3/envs/GSTMI/bin/python tests/smoke_test.py --gpu <gpu> \
        --ckpt <winner_v2 best.pth>
"""
import os, sys, json, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=2)
ap.add_argument("--ckpt", required=True, help="winner-v2 best.pth")
ap.add_argument("--config", default="configs/winner_v2.yaml")
ap.add_argument("--psnr_tol", type=float, default=0.6, help="dB tolerance (nondeterministic run)")
ap.add_argument("--ssim_tol", type=float, default=0.02)
args = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent          # tests/
ROOT = HERE.parent                               # framework root
PGS = ROOT / "paired_gs"                         # import root
sys.path.insert(0, str(PGS))
from seed import set_seed
from trainers.joint_trainer import JointTrainer
from data.private_paths import resolve_subject_paths
from data.transforms import fft3c, ifft3c
from metrics import evaluate_reconstruction

REFERENCE = {"plain": dict(psnr=37.738, ssim=0.8755),
             "contrast": dict(psnr=38.143, ssim=0.894)}


def build_trainer(dev):
    set_seed(42, deterministic=False)
    cfg = yaml.safe_load(open(ROOT / args.config))
    for s in ("plain", "contrast"):
        sc = cfg["data"][s]
        for k in ("data_path", "csm_path", "mask_path"):
            if sc.get(k):
                sc[k] = resolve_subject_paths(sc[k])
    # No retrain: build a fresh 200k-point model, then load the winner's FULL trained
    # model_state directly (both plain AND contrast densities). We do NOT use the
    # warm_start_checkpoint path: that path reads only plain params from a
    # `gaussian_state`/top-level layout and re-clones contrast=plain, which would
    # discard the trained contrast densities. A training-time best.pth is
    # {"iter","model_state","optim_state"} where model_state has all 7 params.
    cfg.setdefault("gaussian", {}).pop("warm_start_checkpoint", None)
    ckpt = torch.load(str(Path(args.ckpt).resolve()), map_location="cpu", weights_only=False)
    ms = ckpt["model_state"]
    n = ms["positions"].shape[0]
    # size the fresh model to match the checkpoint so load_state_dict aligns
    cfg["gaussian"]["initial_num_points"] = int(n)
    cfg.setdefault("training", {})["max_iterations"] = 2
    cfg["training"]["shared_warmup_iters"] = 0
    cfg.setdefault("output", {})["output_dir"] = str(ROOT / "results_smoke")
    cfg["output"]["exp_name"] = "smoke"
    tr = JointTrainer(cfg, device=dev)
    missing, unexpected = tr.model.load_state_dict(ms, strict=False)
    print(f"[smoke] loaded model_state: {n} Gaussians, "
          f"missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
    return cfg, tr


def score(tr, which, dev):
    """render + hard-DC + SENSE-combine vs ESPIRiT-CSM GT, then ALPHA-FIT PSNR/SSIM.

    This mirrors the authoritative reference scorer (score_winner_espirit_final.py):
        rec = |x_dc|;  alpha = sum(rec*gt)/sum(rec*rec);  r = rec*alpha
        psnr = 10*log10(gt.max()^2 / mean((r-gt)^2));  ssim(gt, r, data_range=gt.max())
    The alpha-fit removes the global intensity-scale ambiguity so the number is
    comparable to the shipped winner-v2 espirit_scores.json.
    """
    from skimage.metrics import structural_similarity as sk_ssim
    if which == "plain":
        csm, y, mask, ds = tr.csm_p, tr.kspace_p, tr.mask_p, tr.ds_p
    else:
        csm, y, mask, ds = tr.csm_c, tr.kspace_c, tr.mask_c, tr.ds_c
    gt_t = ds.get_data()["ground_truth"].to(dev)
    m = (mask if mask.ndim == 3 else mask[0]).to(dev).float()
    with torch.no_grad():
        x = tr._render(which)                                   # NO train() — exact loaded state
        k_model = fft3c(csm * x.unsqueeze(0))
        k_dc = m.unsqueeze(0) * y + (1.0 - m).unsqueeze(0) * k_model
        x_dc = (ifft3c(k_dc) * csm.conj()).sum(0)
    rec = x_dc.abs().cpu().numpy().astype("float64")
    gt = gt_t.abs().cpu().numpy().astype("float64")
    alpha = float((rec * gt).sum() / (rec * rec).sum())
    r = rec * alpha
    mx = float(gt.max())
    psnr = 10.0 * np.log10(mx ** 2 / np.mean((r - gt) ** 2))
    ss = sk_ssim(gt, r, data_range=mx)
    return float(psnr), float(ss)


def main():
    dev = torch.device("cuda", 0)
    print(f"[smoke] gpu={args.gpu} ckpt={args.ckpt}")
    cfg, tr = build_trainer(dev)
    print(f"[smoke] names: plain={tr.name_plain!r} contrast={tr.name_contrast!r}  "
          f"N_gaussians={tr.model.num_points}")
    # No training: the winner model_state is already loaded. Rendering the exact
    # loaded Gaussians is the faithful reproduction (train() would perturb them).

    results, ok = {}, True
    for which in ("plain", "contrast"):
        psnr, ssim = score(tr, which, dev)
        ref = REFERENCE[which]
        dp, dsim = psnr - ref["psnr"], ssim - ref["ssim"]
        passed = abs(dp) <= args.psnr_tol and abs(dsim) <= args.ssim_tol
        ok &= passed
        name = tr.name_plain if which == "plain" else tr.name_contrast
        # Store enough precision that the residual (nondeterministic run) is visible;
        # do NOT round the deltas to 0. The run is not bit-reproducible.
        results[which] = dict(name=name, psnr=round(psnr, 5), ssim=round(ssim, 5),
                              ref_psnr=ref["psnr"], ref_ssim=ref["ssim"],
                              d_psnr=round(dp, 5), d_ssim=round(dsim, 5), pass_=passed)
        print(f"[smoke] {name:5s} PSNR={psnr:.5f} (ref {ref['psnr']}, d={dp:+.5f})  "
              f"SSIM={ssim:.5f} (ref {ref['ssim']}, d={dsim:+.5f})  {'PASS' if passed else 'FAIL'}")

    verdict = "SMOKE_PASS" if ok else "SMOKE_FAIL"
    print(f"[smoke] {verdict}")
    out = ROOT / "results_smoke" / "smoke_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(verdict=verdict, results=results,
                                    psnr_tol=args.psnr_tol, ssim_tol=args.ssim_tol), indent=2))
    print(f"[smoke] wrote {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
