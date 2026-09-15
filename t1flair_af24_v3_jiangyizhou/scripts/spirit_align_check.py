#!/usr/bin/env python3
"""spirit_align_check.py -- does the GPU port in recon_spirit.py reproduce Lustig's MATLAB toolbox?

Exports one readout slice's undersampled 2-D k-space and its ACS block, runs scripts/spirit_align.m
(calibSPIRiT + SPIRiT('fft') + pocsSPIRiT, the originals), and compares three things:
  1. the calibration kernel               -- pure linear algebra, must agree to ~1e-6
  2. (G - I) applied to the measured data -- the operator, same tolerance
  3. the POCS output after nIter          -- the whole pipeline, wavelet included

On (3) the two wavelets are not the same code: Wavelab's FWT2_PO here, a periodised orthogonal DWT
with the same Daubechies 4-tap filter there. They differ by a signed permutation of the coefficients,
and the joint soft threshold commutes with that (it is pointwise in the coefficient index, uses only
the coil-wise magnitude, and is equivariant under unimodular scaling), so the reconstructions must
still agree -- that is exactly what this checks.

usage: spirit_align_check.py --contrast T1 --gpu N [--slice 128 --ksize 3 --iters 10 --lam 0.002]
Writes logs/spirit_align_<contrast>_ks<k>.json.
"""
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", default="T1", choices=["T1", "FLAIR"])
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--slice", type=int, default=128)
ap.add_argument("--ksize", type=int, default=3)
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--lam", type=float, default=0.002)
ap.add_argument("--tyk", type=float, default=0.01)
ap.add_argument("--matlab", default="/usr/local/bin/matlab")
ap.add_argument("--double", action="store_true",
                help="run the GPU path in complex128, to separate single-precision noise from a real difference")
a = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np, torch, h5py, scipy.io as sio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = ["recon_spirit", "--contrast", a.contrast, "--gpu", str(a.gpu)]
import recon_spirit as S                                             # noqa: E402
from t1t2_common import Pair, R, ifft1c                              # noqa: E402

dev = torch.device("cuda:0")
log = lambda *x: print(*x, flush=True)

P = Pair(a.contrast, dev)
NC, (NX, NY, NZ) = P.NC, P.G
ch, cw, cy0, cz0 = S.acs_box(P.m2)
yh = torch.empty_like(P.y)
for i in range(0, NC, P.chunk):
    yh[i:i + P.chunk] = ifft1c(P.y[i:i + P.chunk], -3)
CT = torch.complex128 if a.double else torch.complex64
data = yh[:, a.slice].unsqueeze(0).contiguous().to(CT)               # (1, NC, NY, NZ)
acs = data[..., cy0:cy0 + ch, cz0:cz0 + cw]
log(f"[{a.contrast}] slice {a.slice}  ACS {ch}x{cw}  ksize {a.ksize}  iters {a.iters}  lam {a.lam}")

lo, hi = S.DEC_LO.to(dev).to(CT), S.DEC_HI.to(dev).to(CT)
ker = S.calib_kernels(S.build_AtA(acs, a.ksize), NC, a.ksize, a.tyk)
K = S.image_kernels(ker, NY, NZ)
gx = S.gop(data, K)
t0 = time.time()
res = S.pocs(data, K, (~P.m2).view(1, 1, NY, NZ), a.iters, a.lam, 4, lo, hi,
             1 << max(NY - 1, NZ - 1).bit_length())
log(f"GPU POCS {a.iters} iters: {time.time() - t0:.1f}s")

work = Path(tempfile.mkdtemp(prefix="spirit_align_", dir=str(R / "recon")))
sio.savemat(str(work / "in.mat"),
            {"DATA": data[0].permute(1, 2, 0).cpu().numpy(),          # (sx, sy, nc)
             "kCalib": acs[0].permute(1, 2, 0).cpu().numpy(),
             "kSize": np.array([a.ksize, a.ksize], float), "CalibTyk": float(a.tyk),
             "nIter": float(a.iters), "wavWeight": float(a.lam)})
cmd = [a.matlab, "-batch", f"infile='{work}/in.mat'; outfile='{work}/out.mat'; "
                           f"run('{HERE}/spirit_align.m')"]
log(" ".join(cmd))
t0 = time.time()
r = subprocess.run(cmd, capture_output=True, text=True)
log(r.stdout[-3000:]); log(r.stderr[-3000:])
if r.returncode != 0:
    raise SystemExit(f"matlab failed rc={r.returncode}")
log(f"MATLAB: {time.time() - t0:.1f}s")


def rd(f, name):
    d = f[name][:]
    x = (d["real"] + 1j * d["imag"]) if d.dtype.names else d.astype(np.complex128)
    return x.T                                                        # h5py gives MATLAB order reversed


with h5py.File(work / "out.mat", "r") as f:
    m_ker, m_gx, m_res = rd(f, "kernel"), rd(f, "gx"), rd(f, "res")

# MATLAB kernel (k1, k2, nc_in, nc_out) -> ours (nc_in, nc_out, k1, k2)
g_ker = ker[0].permute(2, 3, 0, 1).cpu().numpy()
out = {}
for name, g, m in (("kernel", g_ker, m_ker),
                   ("gop", gx[0].permute(1, 2, 0).cpu().numpy(), m_gx),
                   ("pocs", res[0].permute(1, 2, 0).cpu().numpy(), m_res)):
    assert g.shape == m.shape, (name, g.shape, m.shape)
    rel = float(np.linalg.norm(g - m) / np.linalg.norm(m))
    out[name] = {"rel_l2": rel, "max_abs": float(np.abs(g - m).max()),
                 "ref_max_abs": float(np.abs(m).max())}
    log(f"  {name:7s} rel L2 = {rel:.3e}   max|diff| = {np.abs(g - m).max():.3e} "
        f"(ref peak {np.abs(m).max():.3e})")

for f in work.iterdir():
    f.unlink()
work.rmdir()
tag = "_f64" if a.double else ""
p = R / f"logs/spirit_align_{a.contrast}_ks{a.ksize}{tag}.json"
json.dump({"contrast": a.contrast, "slice": a.slice, "ksize": a.ksize, "iters": a.iters, "dtype": str(CT),
           "lam": a.lam, "tyk": a.tyk, "acs": f"{ch}x{cw}", **out}, open(p, "w"), indent=2)
log(f"wrote {p}")
