#!/usr/bin/env python3
"""extract_gs_volumes.py -- pull an MC-GS run's volumes out of cycle.py's npz, put them in this topic's gauge, score.

cycle.py normalises each channel's data by P99.9 of ITS OWN reference (a SENSE adjoint of the noisy full k-space), so
its volumes are in trainer units; this topic's gauge divides by P99.9 of the CS-SENSE truth. The conversion is
x_gauge = x_trainer * scale_trainer / scale_gauge, with scale_trainer parsed from the run log (one line per channel,
plain first) and cross-checked against the zero-filled volumes: the npz carries the trainer's ZF of the same data,
Pair.sense_adj(Pair.y) is ours, and their norm ratio must equal the factor (<1e-3) or the run is not on our data.
npz keys are fixed by channel: recT1/recT1_complex/zfT1_complex = plain, recFL/recFL_complex/zfFL_complex = contrast.
  extract_gs_volumes.py --exp ce_retro_af16_joint --key gs_joint --map pre:plain,post:contrast
  extract_gs_volumes.py --exp ce_retro_af16_single_pre --key gs_single --map pre:plain
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1t2_common import Pair, R, TRAINER   # noqa: E402

NPZ_C = {"plain": "recT1_complex", "contrast": "recFL_complex"}
NPZ_ZF = {"plain": "zfT1_complex", "contrast": "zfFL_complex"}
CH_IDX = {"plain": 0, "contrast": 1}
PAT = re.compile(r"Normalized MRI intensity scale by P99\.9 magnitude = ([0-9.eE+-]+)")


def parse_scales(text):
    return [float(v) for v in PAT.findall(text)]


def convert(rec_complex, factor):
    return (np.asarray(rec_complex) * np.float32(factor)).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--map", default="pre:plain,post:contrast"); ap.add_argument("--gpu", type=int, default=-1)
    a = ap.parse_args()
    arc = TRAINER / "results_arc"
    m = dict(kv.split(":") for kv in a.map.split(","))
    score_json = json.load(open(arc / f"{a.exp}_score.json"))
    for scan, ch in m.items():
        assert score_json.get(f"{ch}_name") == scan, (ch, score_json.get(f"{ch}_name"), scan)
    scales = parse_scales(open(arc / f"{a.exp}.log").read())
    assert len(scales) >= 2, f"expected two normalisation lines in {a.exp}.log, found {scales}"
    dev = torch.device("cpu") if a.gpu < 0 else torch.device(f"cuda:{a.gpu}")
    with np.load(arc / f"{a.exp}_volumes.npz") as z:
        for scan, ch in m.items():
            P = Pair(scan, dev)
            factor = scales[CH_IDX[ch]] / P.scale
            zf_tr = torch.from_numpy(np.asarray(z[NPZ_ZF[ch]])).to(torch.complex64)
            zf_us = P.sense_adj(P.y).cpu()
            ratio = float(zf_us.abs().norm() / zf_tr.abs().norm())
            if abs(ratio / factor - 1) > 1e-3:
                raise SystemExit(f"[{scan}] ZF norm ratio {ratio:.6f} != scale factor {factor:.6f}: the run is not on this topic's data or the FFT convention differs")
            x = convert(z[NPZ_C[ch]], factor)
            np.save(R / f"recon/{a.key}_{scan}_complex.npy", x)
            np.save(R / f"recon/{a.key}_{scan}_mag.npy", np.abs(x).astype(np.float32))
            row = P.score(torch.from_numpy(x).to(dev), a.key, seconds=float(score_json.get("elapsed_s", 0)), exp=a.exp, factor_scale=factor)
            json.dump({"contrast": scan, "rows": [row], "exp": a.exp, "npz": str(arc / f"{a.exp}_volumes.npz"),
                       "scale_trainer": scales[CH_IDX[ch]], "scale_gauge": P.scale, "zf_ratio_check": ratio,
                       "vram_gb": score_json.get("vram_gb"), "npts": score_json.get("npts")},
                      open(R / f"logs/{a.key}_{scan}.json", "w"), indent=2)
            print(f"[{scan}] {a.key} <- {a.exp}:{ch}  factor {factor:.5f} (zf ratio {ratio:.5f})  head_psnr={row['head_psnr']:.3f} head_ssim={row['head_ssim']:.4f}", flush=True)


if __name__ == "__main__":
    main()
