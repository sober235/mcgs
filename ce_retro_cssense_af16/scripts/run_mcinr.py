#!/usr/bin/env python3
"""run_mcinr.py -- MC-INR (Niessen 2025 reproduction, /data0/congcong/code/GS/mcinr_niessen2025) on the synthetic pair.

Runs scripts/run_volume.py of that repository under its own environment (GR_pt1.10, tinycudann) with this topic's
files: pre in the T1 slot, post in the FLAIR slot, one AF16 mask for both, each scan's own maps. Then converts the
network images (paper units: data / M * 100) to this topic's gauge (x * M/100 / scale, no fit to the truth) and scores
them with t1t2_common.Pair as method 'mcinr'.
  python scripts/run_mcinr.py --gpu 0                 # all 256 readout slices (~15-25 min) -> recon/mcinr_<scan>_complex.npy
  python scripts/run_mcinr.py --gpu 0 --slices 120:122  # smoke: writes recon/mcinr_run_smoke/, scores nothing
  python scripts/run_mcinr.py --gpu 0 --score_only    # convert + score an existing full run
  python scripts/run_mcinr.py --gpu 0 --n_epochs 800 --lr 3e-3 --tag _e800_lr3e-3            # sensitivity sweep (2026-10-09)
  python scripts/run_mcinr.py --gpu 0 --n_epochs 800 --slices 100:120 --tag _e800 --score_subset   # head PSNR on 20 slices
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1t2_common import Pair, R, SCANS   # noqa: E402

MCINR = Path("/data0/congcong/code/GS/mcinr_niessen2025")
PY_TCNN = "/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python"
SLOT = {"pre": "T1", "post": "FLAIR"}
NX = 256


def to_gauge(x_norm, M, scale):
    return x_norm * (M / 100.0) / scale


def assemble(slices, xs, nx=NX):
    v = np.zeros((nx,) + tuple(slices.shape[1:]), np.complex64)
    v[np.asarray(xs)] = slices
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--slices", default="all")
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--n_epochs", type=int, default=None, help="steps per slice (default: the paper's 200)")
    ap.add_argument("--lr", type=float, default=None, help="learning rate (default: the paper's 1e-2)")
    ap.add_argument("--tag", default="", help="suffix of the run directory, the method key and the output files; empty = the registered arm")
    ap.add_argument("--score_subset", action="store_true", help="with --slices a:b: head-only PSNR / NMSE on those slices (gauge head mask and peak), written to logs/mcinr<tag>_subset_<scan>.json")
    a = ap.parse_args()
    smoke = a.slices != "all"
    out = R / (f"recon/mcinr_run{a.tag}_smoke" if smoke else f"recon/mcinr_run{a.tag}")
    if not a.score_only:
        cmd = [PY_TCNN, "-u", str(MCINR / "scripts/run_volume.py"), "--af", "16", "--gpu", str(a.gpu), "--slices", a.slices, "--resume",
               *(["--n_epochs", str(a.n_epochs)] if a.n_epochs is not None else []), *(["--lr", str(a.lr)] if a.lr is not None else []),
               "--ksp_T1", str(R / "sim/gs/pre_kspace.h5"), "--ksp_FLAIR", str(R / "sim/gs/post_kspace.h5"),
               "--csm_T1", str(R / "sim/gs/pre_csm.h5"), "--csm_FLAIR", str(R / "sim/gs/post_csm.h5"),
               "--mask_T1", str(R / "sim/mask/mask_af16_3d.h5"), "--mask_FLAIR", str(R / "sim/mask/mask_af16_3d.h5"),
               "--out", str(out)]
        env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=env)
    if smoke and not a.score_subset:
        print("smoke run done, nothing scored"); return
    if smoke:
        score_subset(a, out); return
    meta = json.load(open(out / "meta.json"))
    assert meta["slices"] == "all", "score needs a full run"
    xs = list(range(NX))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                       # scoring on the CPU in double precision
    for scan in SCANS:
        P = Pair(scan, torch.device("cpu"))
        P.gt_c = P.gt_c.to(torch.complex128); P.gt = P.gt.to(torch.float64)
        x_norm = torch.from_numpy(np.load(out / f"mcinr_norm_{SLOT[scan]}_complex.npy"))
        x = to_gauge(x_norm.to(torch.complex128), meta["M_acquired"], P.scale)
        assert tuple(x.shape) == P.G, (x.shape, P.G)
        row = P.score(x, "mcinr" + a.tag, seconds=meta["seconds_loop"], n_epochs=meta["hparams"]["n_epochs"], lr=meta["hparams"]["lr"])
        np.save(R / f"recon/mcinr{a.tag}_{scan}_complex.npy", x.numpy().astype(np.complex64))
        json.dump({"contrast": scan, "rows": [row], "run_meta": str(out / "meta.json"), "M_acquired": meta["M_acquired"],
                   "gauge_scale": P.scale}, open(R / f"logs/mcinr{a.tag}_{scan}.json", "w"), indent=2)
        print(f"[{scan}] mcinr{a.tag} head_psnr={row['head_psnr']:.3f} head_ssim={row['head_ssim']:.4f} head_nmse={row['head_nmse']:.5f}", flush=True)


def score_subset(a, out):
    """Head-only PSNR / NMSE restricted to the slices of a subset run: the gauge's head mask (|GT| > 0.05 P99.9 of the whole
    volume) and the whole-volume peak, so numbers of different subsets of the same slices are comparable with each other --
    NOT with the whole-volume table."""
    from diag_artefact.headonly import head_mask
    meta = json.load(open(out / "meta.json"))
    xs = meta["slices"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    res = {}
    for scan in SCANS:
        P = Pair(scan, torch.device("cpu"))
        gt = P.gt.to(torch.float64).numpy()
        head = head_mask(gt); gmax = float(gt.max())
        x_norm = torch.from_numpy(np.load(out / f"mcinr_norm_{SLOT[scan]}_complex.npy"))
        rec = to_gauge(x_norm.to(torch.complex128), meta["M_acquired"], P.scale).abs().numpy()
        g, h = gt[xs], head[xs]
        d2 = float(((rec - g)[h] ** 2).mean()); e2 = float((g[h] ** 2).sum())
        res[scan] = dict(head_psnr=float(20 * np.log10(gmax) - 10 * np.log10(d2)), head_nmse=float(((rec - g)[h] ** 2).sum() / e2),
                         slices=xs, n_epochs=meta["hparams"]["n_epochs"], lr=meta["hparams"]["lr"], seconds_loop=meta["seconds_loop"])
        print(f"[{scan}] mcinr{a.tag} subset {xs[0]}..{xs[-1]} ({len(xs)} slices): head_psnr={res[scan]['head_psnr']:.3f} "
              f"head_nmse={res[scan]['head_nmse']:.5f} n_epochs={res[scan]['n_epochs']} lr={res[scan]['lr']}", flush=True)
        json.dump(res[scan], open(R / f"logs/mcinr{a.tag}_subset_{scan}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
