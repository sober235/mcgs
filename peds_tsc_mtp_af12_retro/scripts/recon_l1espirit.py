#!/usr/bin/env python3
"""recon_l1espirit.py -- L1-wavelet ESPIRiT (BART pics) on this topic's data and gauge.

The k-space and the ESPIRiT maps come from the same h5 files every other arm uses, normalised
by the same P99.9(|GT|) scale, so BART's SENSE image lands in the gauge units directly:
    bart pics -l1 -r LAM -i ITERS -S ksp maps out      (CPU: this BART build has no GPU support)
(-S rescales the output back to the input units; without it pics returns its internally scaled
image). Sensitivities have SoS == 1 everywhere (ecalib -c 0), so the SENSE image is directly
comparable with the conjugate-SENSE ground truth.

usage: recon_l1espirit.py --topic <dir> --contrast T1 --gpu N --lam 0.005 [--iters 100 --suffix _lam5e-3]
Writes recon/l1espirit<suffix>_<c>_complex.npy and logs/l1espirit_<c><suffix>.json.
"""
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--topic", required=True)
ap.add_argument("--contrast", required=True)
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--lam", type=float, default=0.005)
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--suffix", default="")
ap.add_argument("--bart", default="/home/congcongliu/src/bart/bart")
ap.add_argument("--threads", type=int, default=32, help="OpenMP threads for BART (this build has no GPU support)")
args = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
import torch

sys.path.insert(0, str(Path(args.topic) / "scripts"))
sys.path.insert(0, "/home/congcongliu/src/bart/python")
import t1t2_common as C                                            # noqa: E402
import cfl                                                         # noqa: E402  (BART's writer/reader)

R = Path(args.topic)
dev = torch.device("cuda:0")
log = lambda *a: print(*a, flush=True)

t0 = time.time()
P = C.Pair(args.contrast, dev)
log(f"[{args.contrast}] coils={P.NC} vol={P.G} scale={P.scale:.4e}")
# BART dims: (x, y, z, coil); ours: (coil, x, y, z)
ksp = P.y.cpu().numpy().transpose(1, 2, 3, 0).astype(np.complex64)
maps = P.S.cpu().numpy().transpose(1, 2, 3, 0).astype(np.complex64)
del P.y_full
torch.cuda.empty_cache()
work = Path(tempfile.mkdtemp(prefix=f"l1espirit_{args.contrast}_", dir=str(R / "recon")))
cfl.writecfl(str(work / "ksp"), ksp)
cfl.writecfl(str(work / "maps"), maps)
# Stage-2 spec section 5: without -p, BART infers the sampling pattern from the NONZERO k-space
# entries and would read the measured zeros of Z as "not sampled". The pattern is Omega | Z.
pat = str(R / "data/pattern_union")
assert Path(pat + ".cfl").exists(), f"missing BART pattern {pat}.cfl (run scripts/prep_mask.py)"
cmd = [args.bart, "pics", "-p", pat, "-l1", "-r", str(args.lam), "-i", str(args.iters), "-S",
       str(work / "ksp"), str(work / "maps"), str(work / "out")]
os.environ["OMP_NUM_THREADS"] = str(args.threads)
log(" ".join(cmd))
t1 = time.time()
res = subprocess.run(cmd, capture_output=True, text=True)
log(res.stdout[-2000:]); log(res.stderr[-2000:])
if res.returncode != 0:
    raise SystemExit(f"bart pics failed rc={res.returncode}")
x = np.squeeze(cfl.readcfl(str(work / "out"))).astype(np.complex64)
assert x.shape == tuple(P.G), (x.shape, P.G)
x_t = torch.from_numpy(x).to(dev)
row = P.score(x_t, "l1espirit" + args.suffix, seconds=time.time() - t1, lam=args.lam, iters=args.iters)
np.save(R / f"recon/l1espirit{args.suffix}_{args.contrast}_complex.npy", x)
for f in work.iterdir():
    f.unlink()
work.rmdir()
log(f"[{args.contrast}] l1espirit{args.suffix} psnr={row['psnr']:.3f} ssim={row['ssim']:.4f} head_psnr={row['head_psnr']:.3f} "
    f"head_ssim={row['head_ssim']:.4f} head_hfen={row['head_hfen']:.4f} sharp={row['sharp']:.3f} ({row['seconds']:.0f} s)")
p = R / f"logs/l1espirit_{args.contrast}{args.suffix}.json"
json.dump({"contrast": args.contrast, "lam": args.lam, "iters": args.iters, "pattern": pat,
           "mask_frac": P.mask_frac, "mask_frac_omega": P.mask_frac_omega,
           "rows": [row], "total_seconds": time.time() - t0}, open(p, "w"), indent=2)
log(f"WROTE {p}")
