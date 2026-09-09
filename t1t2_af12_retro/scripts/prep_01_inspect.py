#!/usr/bin/env python3
"""prep_01_inspect.py -- verify the readout axis, the shutter geometry and the coil
spectrum of the MC_Jia SPIRiT-diffusion CFL volumes (T1 + T2, one subject).

Writes a JSON report; makes no decisions on its own.
"""
import json, os, sys
import numpy as np

SRC = "/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL/vol1_huangyuwei"
OUT = "/data0/congcong/code/GS/Results_20260908/T1T2_AF12_retro"
FILES = {"T1": "t1_fsp_train_huangyuwei_cc", "T2": "t2_train_huangyuwei_cc",
         "FLAIR": "flair_train_huangyuwei_cc"}


def cfl(base):
    with open(base + ".hdr") as f:
        dims = [int(x) for x in f.read().split("\n")[1].split()]
    n = int(np.prod(dims))
    a = np.memmap(base + ".cfl", dtype=np.complex64, mode="r", shape=(n,))
    return a.reshape(dims, order="F")[..., 0]           # drop trailing singleton -> (X,Y,Z,C)


def report(tag, base):
    k = cfl(base)
    nx, ny, nz, nc = k.shape
    r = {"tag": tag, "dims": [nx, ny, nz, nc]}

    # --- line-wise zero structure along each candidate readout axis -------------------
    # A Cartesian 3D acquisition samples the readout axis fully: every acquired PE line
    # is nonzero over its WHOLE readout extent, and a rejected line is zero over all of it.
    for ax, name in ((0, "x"), (1, "y"), (2, "z")):
        amax = np.abs(np.moveaxis(k, ax, -1)).max(axis=(0, 1, 2) if False else 2)  # placeholder
        break
    # do it explicitly and cheaply: per-axis, |k| max over coils, then per-line stats
    absk = None
    stats = {}
    for ax, name in ((0, "x"), (1, "y"), (2, "z")):
        # move candidate readout to last, collapse coils
        blocks = []
        step = 16
        mv = np.moveaxis(k, ax, 2)                        # (A,B,RO,C)
        A, B, RO, C = mv.shape
        line_max = np.zeros((A, B), dtype=np.float32)
        line_min = np.zeros((A, B), dtype=np.float32)
        for a0 in range(0, A, step):
            blk = np.abs(np.asarray(mv[a0:a0 + step]))     # (b,B,RO,C)
            m = blk.max(axis=-1)                           # over coils -> (b,B,RO)
            line_max[a0:a0 + step] = m.max(axis=-1)
            line_min[a0:a0 + step] = m.max(axis=-1)        # reuse; refined below
            # fraction of readout samples that are (numerically) zero on this line
            thr = 0.0
            if a0 == 0:
                thr_global = float(m.max())
            zfrac = (m <= 1e-10 * max(thr_global, 1e-30)).mean(axis=-1)
            if a0 == 0:
                zf = np.zeros((A, B), dtype=np.float32)
            zf[a0:a0 + step] = zfrac
        gmax = float(line_max.max())
        is_zero_line = line_max <= 1e-10 * gmax
        # a "clean" readout axis: every line is either all-zero or has ~no zero samples
        partial = float(((~is_zero_line) & (zf > 0.02)).mean())
        stats[name] = {"axis": ax, "plane_shape": [int(A), int(B)],
                       "zero_line_frac": float(is_zero_line.mean()),
                       "partial_lines_frac": partial}
    r["readout_candidates"] = stats

    # --- coil energy spectrum (SVD over coils, from a central block) ------------------
    cy, cz = ny // 2, nz // 2
    blk = np.asarray(k[nx // 2 - 16:nx // 2 + 16, cy - 32:cy + 32, cz - 24:cz + 24, :])
    M = blk.reshape(-1, nc)
    s = np.linalg.svd(M, compute_uv=False)
    cum = np.cumsum(s ** 2) / (s ** 2).sum()
    r["coil_svd_cum_energy"] = [float(x) for x in cum]
    r["coils_for_99.0pct"] = int(np.searchsorted(cum, 0.990) + 1)
    r["coils_for_99.9pct"] = int(np.searchsorted(cum, 0.999) + 1)
    r["coils_for_99.99pct"] = int(np.searchsorted(cum, 0.9999) + 1)

    # --- DC location -----------------------------------------------------------------
    e = np.zeros((ny, nz))
    for x in range(0, nx, 8):
        e += (np.abs(np.asarray(k[x])) ** 2).sum(axis=-1)
    r["dc_yz"] = [int(v) for v in np.unravel_index(np.argmax(e), e.shape)]
    r["centre_yz"] = [ny // 2, nz // 2]
    ex = np.zeros(nx)
    for x in range(nx):
        ex[x] = (np.abs(np.asarray(k[x, cy - 2:cy + 2, cz - 2:cz + 2, :])) ** 2).sum()
    r["dc_x"] = int(np.argmax(ex)); r["centre_x"] = nx // 2
    r["support_frac_yz"] = float((e > 0).mean())
    return r


if __name__ == "__main__":
    out = {"source": SRC, "contrasts": []}
    for tag, f in FILES.items():
        print("probing", tag, flush=True)
        out["contrasts"].append(report(tag, os.path.join(SRC, f)))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "logs", "prep_01_inspect.json")
    json.dump(out, open(p, "w"), indent=2)
    print(json.dumps(out, indent=2))
