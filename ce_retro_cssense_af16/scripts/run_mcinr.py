#!/usr/bin/env python3
"""run_mcinr.py -- MC-INR (Niessen 2025 reproduction, /data0/congcong/code/GS/mcinr_niessen2025) on the synthetic pair.

Runs scripts/run_volume.py of that repository under its own environment (GR_pt1.10, tinycudann) with this topic's
files: pre in the T1 slot, post in the FLAIR slot, one AF16 mask for both, each scan's own maps. Then converts the
network images (paper units: data / M * 100) to this topic's gauge (x * M/100 / scale, no fit to the truth) and scores
them with t1t2_common.Pair as method 'mcinr'.
  python scripts/run_mcinr.py --gpu 0                 # all 256 readout slices (~15-25 min) -> recon/mcinr_<scan>_complex.npy
  python scripts/run_mcinr.py --gpu 0 --slices 120:122  # smoke: writes recon/mcinr_run_smoke/, scores nothing
  python scripts/run_mcinr.py --gpu 0 --score_only    # convert + score an existing full run
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
    a = ap.parse_args()
    smoke = a.slices != "all"
    out = R / ("recon/mcinr_run_smoke" if smoke else "recon/mcinr_run")
    if not a.score_only:
        cmd = [PY_TCNN, "-u", str(MCINR / "scripts/run_volume.py"), "--af", "16", "--gpu", str(a.gpu), "--slices", a.slices, "--resume",
               "--ksp_T1", str(R / "sim/gs/pre_kspace.h5"), "--ksp_FLAIR", str(R / "sim/gs/post_kspace.h5"),
               "--csm_T1", str(R / "sim/gs/pre_csm.h5"), "--csm_FLAIR", str(R / "sim/gs/post_csm.h5"),
               "--mask_T1", str(R / "sim/mask/mask_af16_3d.h5"), "--mask_FLAIR", str(R / "sim/mask/mask_af16_3d.h5"),
               "--out", str(out)]
        env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=env)
    if smoke:
        print("smoke run done, nothing scored"); return
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
        row = P.score(x, "mcinr", seconds=meta["seconds_loop"])
        np.save(R / f"recon/mcinr_{scan}_complex.npy", x.numpy().astype(np.complex64))
        json.dump({"contrast": scan, "rows": [row], "run_meta": str(out / "meta.json"), "M_acquired": meta["M_acquired"],
                   "gauge_scale": P.scale}, open(R / f"logs/mcinr_{scan}.json", "w"), indent=2)
        print(f"[{scan}] mcinr head_psnr={row['head_psnr']:.3f} head_ssim={row['head_ssim']:.4f} head_nmse={row['head_nmse']:.5f}", flush=True)


if __name__ == "__main__":
    main()
