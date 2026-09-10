#!/usr/bin/env python3
"""ploraks_import.py -- assemble P-LORAKS slice reconstructions into a volume and score it.

Full run (all 256 slices): FFT the reconstructed hybrid data back along the readout, combine with
the ESPIRiT maps exactly as the ground truth is (conj(S) . ifft3c), save recon/ploraks<suffix>_<c>_complex.npy
and a json row on the shared gauge. Subset run (--slices given at export): score each slice's
2-D SENSE combine against the GT slice inside the head (rank sweep), no volume written.

usage: ploraks_import.py --topic <dir> --contrast T1 --infile export.h5 --recfile recon.h5 [--suffix _r120]
"""
import argparse, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--topic", required=True); ap.add_argument("--contrast", required=True)
ap.add_argument("--infile", required=True); ap.add_argument("--recfile", required=True)
ap.add_argument("--suffix", default=""); ap.add_argument("--gpu", type=int, default=None)
args = ap.parse_args()
import os
if args.gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
sys.path.insert(0, str(Path(args.topic) / "scripts"))
import t1t2_common as C
R = Path(args.topic)
dev = torch.device("cuda:0" if args.gpu is not None else "cpu")
P = C.Pair(args.contrast, dev)
with h5py.File(args.infile) as f:
    idx = [int(i) for i in f["slices"][...]]
with h5py.File(args.recfile) as f:
    image_domain = "xre" in f
    rank = int(np.array(f["rank"][...]).ravel()[0]); Rn = int(np.array(f["R"][...]).ravel()[0])
    lam = float(np.array(f["lambda"][...]).ravel()[0]) if "lambda" in f else None
    if "slices_sel" in f:                                                                # probe on a sub-selection of the exported slices
        idx = [idx[int(k) - 1] for k in np.array(f["slices_sel"][...]).ravel()]
    if image_domain:
        ximg = torch.from_numpy(f["xre"][...] + 1j * f["xim"][...]).to(torch.complex64)          # (nx, ky, kz): SENSE-LORAKS returns images
        ximg = ximg * float(np.sqrt(ximg.shape[1] * ximg.shape[2]))     # toolbox uses MATLAB ifft2 (1/N); the gauge is orthonormal (1/sqrt N)
    else:
        rec = torch.from_numpy(f["rre"][...] + 1j * f["rim"][...]).to(torch.complex64)   # (nx, ky, kz, coil)
# MATLAB h5write of (coil, kz, ky, nx) arrays is read back by h5py as (nx, ky, kz, coil)
rec = None if image_domain else rec.permute(3, 0, 1, 2).to(dev)                     # -> (coil, nx, ky, kz)
full = len(idx) == P.G[0] and idx == list(range(P.G[0]))
if full:
    if image_domain:
        x = ximg.to(dev)                                                   # the stack of per-readout images IS the volume
    else:
        k = C._c(torch.fft.fftn, rec, (1,))
        x = P.sense_adj(k)
    row = P.score(x, "ploraks" + args.suffix, seconds=0.0, rank=rank, R=Rn, lam=lam)
    np.save(R / f"recon/ploraks{args.suffix}_{args.contrast}_complex.npy", x.cpu().numpy().astype(np.complex64))
    p = R / f"logs/ploraks_{args.contrast}{args.suffix}.json"
    json.dump({"contrast": args.contrast, "rank": rank, "R": Rn, "rows": [row]}, open(p, "w"), indent=2)
    print(f"[{args.contrast}] ploraks{args.suffix} rank={rank} psnr={row['psnr']:.3f} ssim={row['ssim']:.4f} head_psnr={row['head_psnr']:.3f} "
          f"head_ssim={row['head_ssim']:.4f} head_hfen={row['head_hfen']:.4f} sharp={row['sharp']:.3f}")
    print("WROTE", p)
else:
    # per-slice 2-D check: hybrid image slice = conj(S_x) . ifft2c(rec_x), against the GT slice
    head = P.gt > 0.05 * torch.quantile(P.gt.reshape(-1)[:: max(1, P.gt.numel() // 2_000_000)], 0.999)
    gmax = float(P.gt.max()); out = {}
    for j, i in enumerate(idx):
        xs = ximg[j].to(dev) if image_domain else (P.S[:, i].conj() * C.ifft2c(rec[:, j])).sum(0)          # (ky, kz) image at readout i
        hp = head[i]; g = P.gt[i][hp].abs(); e = (xs.abs()[hp] - g)
        out[i] = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
    print(json.dumps({"rank": rank, "R": Rn, "lambda": lam, "mean_head_psnr": float(np.mean(list(out.values()))), "per_slice": out}))
