#!/usr/bin/env python3
"""score_mcinr_arm.py -- register the MC-INR arm (Niessen et al. 2025 reproduction) in this topic.

The reproduction ran on this topic's own mask, k-space and ESPIRiT maps (its data/ is symlinks back
to here) and scored itself on CPU in double precision. This re-scores its volumes through
t1t2_common.Pair.score -- the identical code path every other arm of the table used -- writes
logs/mcinr_<c>.json in the topic's row format, and prints the disagreement with the numbers the
reproduction reported. Anything above ~1e-6 relative means the two paths do not agree and the row
would not be comparable with the rest of the table.

  $PY scripts/score_mcinr_arm.py --gpu 0     # then scripts/summarize.py

Noise study (2026-09-16): the MC-INR run fitted this topic's noisy measurements (run_volume.py --af 16 --tag _noise1x --ksp_T1/--ksp_FLAIR data/meas_<c>.h5) with this topic's mask, and mcinr_niessen2025's score.py --af 16 --tag _noise1x put its volumes in gauge units against the clean AF16 topic. This topic's ground truth and scale are the same clean ones, so the two gauges agree by construction.
Also links recon/mcinr_<c>_complex.npy to the reproduction's volume, as in the AF20 topic, for fig_three_planes_compact.py --with_mcinr.
"""
import argparse, json, os, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--af", type=int, default=16, choices=[16])
ap.add_argument("--tag", default="_noise1x", help="the MC-INR run directory suffix: recon/af<AF><tag>/")
args = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np, torch

R = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise1x_v3_jiangyizhou")
M = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou")
sys.path.insert(0, str(R / "scripts"))
from t1t2_common import Pair  # noqa: E402

theirs = json.load(open(M / f"logs/score_af{args.af}{args.tag}.json"))
meta = json.load(open(M / f"recon/af{args.af}{args.tag}/meta.json"))
dev = torch.device("cuda:0")
KEYS = ("psnr", "ssim", "nmse", "hfen", "head_psnr", "head_ssim", "head_hfen", "sharp", "grain")
worst = 0.0

for c in ("T1", "FLAIR"):
    v = np.load(M / f"recon/af{args.af}{args.tag}/mcinr_{c}_complex.npy")
    print(f"[{c}] volume {v.shape} {v.dtype}")
    p = Pair(c, dev)
    assert v.shape == tuple(p.gt.shape), (v.shape, p.gt.shape)
    row = p.score(torch.from_numpy(v).to(torch.complex64).to(dev), "mcinr", seconds=meta["seconds_loop"])
    ref = theirs["contrasts"][c]["row"]
    print(f"  {'metric':11s} {'topic rescore':>16s} {'mcinr repo':>16s} {'|diff|':>10s}")
    for k in KEYS:
        d = abs(row[k] - ref[k])
        worst = max(worst, d / max(1.0, abs(ref[k])))
        print(f"  {k:11s} {row[k]:16.8f} {ref[k]:16.8f} {d:10.2e}")
    json.dump({"contrast": c, "rows": [row], "scale_p999": float(p.scale), "coils": int(p.NC),
               "source": str(M / f"recon/af{args.af}{args.tag}/mcinr_{c}_complex.npy"),
               "note": "multi-contrast INR of Niessen et al. 2025, per-slice 2-D, paper recipe; one network "
                       "outputs both contrasts, so `seconds` is the whole two-contrast run"},
              open(R / f"logs/mcinr_{c}.json", "w"), indent=1)
    print(f"  wrote {R / f'logs/mcinr_{c}.json'}")
    link = R / f"recon/mcinr_{c}_complex.npy"
    target = M / f"recon/af{args.af}{args.tag}/mcinr_{c}_complex.npy"
    if not link.exists():
        link.symlink_to(target)
    assert link.resolve() == target.resolve(), (link, link.resolve())
    print(f"  linked {link} -> {target}")
    del p
    torch.cuda.empty_cache()

print(f"\nworst relative disagreement across both contrasts: {worst:.2e}")
