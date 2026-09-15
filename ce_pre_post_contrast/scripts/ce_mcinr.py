#!/usr/bin/env python3
"""ce_mcinr.py -- MC-INR (Niessen et al., MICCAI 2025 RIME, arXiv 2509.04888) on the UIH prospective CE pair.

The method code is the faithful reproduction in /data0/congcong/code/GS/mcinr_niessen2025 (mcinr/, vendored
paper config: tcnn hash grid + ReLU MLP, one fresh network per readout slice jointly emitting BOTH scans,
weighted k-space MSE, Adam 1e-2, 200 steps, last step taken, no regulariser), used unchanged; only the data
wiring is new: pre and post each keep their own coil maps, both use the acquired plane (identical for the two
scans), and hybrid space is taken along this data's readout axis (axis 0, 256 positions, fully sampled).
Same protocol as the jyz T1+FLAIR reproduction (that topic's README, 2026-09-13).

  --mask full : every acquired line (the column shown in the figure)
  --mask fit  : the 95% lines of heldout_split/split_95_5_seed0.npz, to report a held-out residual

Must run in the GR_pt1.10 env (tinycudann). Writes <out>/mcinr_<mask>_<scan>_{complex.npy,meta.json},
alpha-scaled onto the measured data like every column of this topic, plus losses_<mask>.npy.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mask", required=True, choices=["fit", "full"])
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--slices", default="all", help="all | <start>:<stop> (smoke test; nothing is scored)")
args = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/data0/congcong/code/GS/mcinr_niessen2025")
import ce_common as C  # noqa: E402
from mcinr.data import acquired_max, mask_stack, slice_inputs, to_hybrid  # noqa: E402
from mcinr.model import paper_hparams  # noqa: E402
from mcinr.recon import reconstruct_slice  # noqa: E402

t_start = time.time()
dev = torch.device("cuda:0")
torch.zeros(1, device=dev)            # hold a CUDA context from the start (other schedulers read it as "card busy")
torch.set_num_threads(16)
log = lambda *a: print(*a, flush=True)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
hp = paper_hparams()

fit_plane, ho_plane = C.load_split()
hybrids, csms, m2s = [], [], []
for s in C.SCANS:
    y, S, m2 = C.load_scan(s)
    hybrids.append(to_hybrid(y)); csms.append(S); m2s.append(m2)
    del y
assert bool((m2s[0] == m2s[1]).all()), "pre and post acquired planes differ"
plane = fit_plane if args.mask == "fit" else m2s[0]
planes = [plane, plane]
M = max(acquired_max(h, plane) for h in hybrids)
X = hybrids[0].shape[1]
xs = list(range(X)) if args.slices == "all" else list(range(*(int(v) for v in args.slices.split(":"))))
log(f"MC-INR on CE, mask {args.mask}: {int(plane.sum())} lines, M {M:.6f}, slices {len(xs)}/{X}, "
    f"load {time.time() - t_start:.0f}s, gpu {torch.cuda.get_device_name(0)}, hparams {hp}")

vol = {s: np.zeros((X,) + tuple(plane.shape), np.complex64) for s in C.SCANS}
losses = np.zeros((X, hp["n_epochs"]))
secs = np.zeros(X)
mask_dev = mask_stack(planes).to(dev)
t_loop = time.time()
for x in xs:
    t0 = time.time()
    kdata, csm = slice_inputs(hybrids, csms, planes, M, x)
    img, losses[x] = reconstruct_slice(kdata.to(dev), csm.to(dev), mask_dev, seed=x, device=dev)
    if not bool(torch.isfinite(torch.view_as_real(img)).all()):
        sys.exit(f"non-finite output at slice {x}")
    for n, s in enumerate(C.SCANS):
        vol[s][x] = img[..., n].numpy()
    secs[x] = time.time() - t0
    if x % 16 == 0 or x == xs[-1]:
        log(f"slice {x:3d}  loss {losses[x, 0]:.4e} -> {losses[x, -1]:.4e}  {secs[x]:.2f}s  elapsed {time.time() - t_loop:.0f}s")
if len(xs) != X:
    log(f"smoke run over {len(xs)} slices done, median {np.median(secs[xs]):.2f}s/slice; not scored"); sys.exit(0)
np.save(OUT / f"losses_{args.mask}.npy", losses)
del hybrids

for n, s in enumerate(C.SCANS):
    y, S, m2 = C.load_scan(s, dev)
    x = torch.from_numpy(vol[s]).to(dev)
    r = C.kres_metrics(x, y, S, {"full": m2, "fit": fit_plane.to(dev), "ho": ho_plane.to(dev)})
    alpha = r["full"]["alpha"]
    tag = f"mcinr_{args.mask}_{s}"
    np.save(OUT / f"{tag}_complex.npy", (alpha * x).cpu().numpy().astype(np.complex64))
    C.dump_json(dict(method="mcinr", scan=s, mask=args.mask, M=M, hparams=hp, seed="slice index",
                     alpha_re=float(np.real(alpha)), alpha_im=float(np.imag(alpha)),
                     kres=r["full"]["kres"], kres_fit=r["fit"]["kres"], kres_heldout=r["ho"]["kres"],
                     heldout_is_unseen=args.mask == "fit", shape=list(x.shape),
                     seconds_loop=float(secs.sum()), seconds_per_slice_median=float(np.median(secs)),
                     torch=torch.__version__, gpu=torch.cuda.get_device_name(0)), OUT / f"{tag}_meta.json")
    log(f"[{tag}] kres full {r['full']['kres']:.5f} fit {r['fit']['kres']:.5f} held-out {r['ho']['kres']:.5f}  |alpha| {abs(alpha):.4f}")
    del y, S, x
    torch.cuda.empty_cache()
log(f"DONE {time.time() - t_start:.0f}s")
