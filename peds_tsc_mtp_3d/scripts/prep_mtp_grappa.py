#!/usr/bin/env python
"""Stage 1 of the pediatric MTP line: one acquired contrast from the UIH raw file -> readout oversampling
removed -> 3D GRAPPA (PE acceleration 2, 24-line ACS block) -> completed multi-coil k-space, masks,
root-sum-of-squares images and self-check numbers.

Usage (through ./py, the launcher sets CUDA_VISIBLE_DEVICES):
    ./py prep_mtp_grappa.py --fa 16 --echo 0 --device cuda:0
    ./py prep_mtp_grappa.py --fa 4  --echo 4 --device cpu

Outputs in data/FA{fa}_TE{te}/: kspace_grappa.h5 (dataset "kspace", (coil, RO, PE, SPE) complex64,
DC-centred), masks.npz, rss_grappa.npy, rss_zf.npy, meta.json. The raw file is never modified.
Design: mcgs_peds_wt/docs/superpowers/specs/2026-09-12-peds-tsc-mtp-grappa-design.md
"""
import argparse, json, os, sys, time
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch

WT = "/data0/congcong/code/GS/mcgs_peds_wt"
sys.path.insert(0, WT)
from uih_mtp.uih_raw import LineIndex, read_lines, read_protocol, scan_index   # noqa: E402
from uih_mtp.grappa import acs_holdout_check, grappa3d                        # noqa: E402

RAW = ("/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/rawdata/"
       "UID_126819142341469_mtp_cartesian_.7x.7x1.raw")
TOPIC = "/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_3D"
INDEX = os.path.join(TOPIC, "data", "line_index.npz")
U1_OF_FA = {4: 0, 16: 1}                  # user index 1 selects the flip-angle pass (verified 2026-09-12)
SET_MAIN = 1                              # set 0 is a repeat of echo 0 and is not used


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save_index(idx: LineIndex, path: str):
    np.savez(path, offsets=idx.offsets, ref=idx.ref, rxgain=idx.rxgain, noise_offsets=idx.noise_offsets,
             scalars=np.array([idx.samples, idx.channels, idx.dhl_version, idx.ro_center, idx.pe_center,
                               idx.spe_center, idx.pe_len, idx.spe_len, idx.n_phasecor, idx.n_feedback]),
             **{"c_" + k: v for k, v in idx.counters.items()})


def load_index(path: str) -> LineIndex:
    z = np.load(path)
    s = [int(v) for v in z["scalars"]]
    return LineIndex(path=RAW, samples=s[0], channels=s[1], dhl_version=s[2], ro_center=s[3], pe_center=s[4],
                     spe_center=s[5], pe_len=s[6], spe_len=s[7], offsets=z["offsets"],
                     counters={k[2:]: z[k] for k in z.files if k.startswith("c_")}, ref=z["ref"],
                     rxgain=z["rxgain"], noise_offsets=z["noise_offsets"], n_phasecor=s[8], n_feedback=s[9])


def protocol_te_ms(xml: bytes):
    vals = [float(v.text) / 1000.0 for v in ET.fromstring(xml).findall("Root/Seq/Basic/TE/Value")]
    if len(vals) != 5:
        raise ValueError(f"expected 5 echo times in the protocol, found {vals}")
    return vals


def rss(k: torch.Tensor) -> np.ndarray:
    """Root-sum-of-squares image of a centred (C, X, Y, Z) k-space, coil by coil to bound memory."""
    acc = None
    for c in range(k.shape[0]):
        img = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(k[c]), norm="ortho"))
        p = img.real ** 2 + img.imag ** 2
        acc = p if acc is None else acc + p
    return torch.sqrt(acc).cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa", type=int, choices=(4, 16), required=True)
    ap.add_argument("--echo", type=int, choices=range(5), required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lam", type=float, default=1e-3, help="Tikhonov, relative to the mean Gram diagonal")
    ap.add_argument("--n-pe-sources", type=int, default=4)
    ap.add_argument("--dx", type=int, default=2)
    ap.add_argument("--dz", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    magic, prot_len, xml = read_protocol(RAW)
    te_ms = protocol_te_ms(xml)
    name = f"FA{args.fa}_TE{te_ms[args.echo]:.2f}"
    out = os.path.join(TOPIC, "data", name)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "meta.json")) and not args.overwrite:
        log(f"{name}: meta.json exists, use --overwrite to redo"); return
    device = torch.device(args.device)
    log(f"{name}: u1={U1_OF_FA[args.fa]} echo={args.echo} device={device}")

    t0 = time.time()
    if os.path.exists(INDEX):
        idx = load_index(INDEX); log(f"index loaded from cache ({idx.n_imaging} imaging lines)")
    else:
        idx = scan_index(RAW); save_index(idx, INDEX); log(f"index built and cached ({idx.n_imaging} imaging lines)")
    t_index = time.time() - t0

    t0 = time.time()
    k_np = read_lines(RAW, idx, set_=SET_MAIN, con=args.echo, u1=U1_OF_FA[args.fa], ro_crop=idx.ro_center * 2)
    m_acq = idx.mask(set_=SET_MAIN, con=args.echo, u1=U1_OF_FA[args.fa])
    t_read = time.time() - t0
    log(f"read {int(m_acq.sum())} lines -> k {k_np.shape}, {k_np.nbytes / 1e9:.2f} GB, {t_read:.0f} s")
    sel = idx.select(SET_MAIN, args.echo, U1_OF_FA[args.fa])
    ref_pe = np.unique(idx.counters["pe"][sel & idx.ref])
    acs = (int(ref_pe.min()), int(ref_pe.max()) + 1)
    log(f"ACS block from the PPA reference flags: PE {acs[0]}..{acs[1] - 1} ({len(ref_pe)} lines)")

    k = torch.from_numpy(k_np)
    t0 = time.time()
    res = grappa3d(k, m_acq, acs_lines=acs, R=2, n_pe_sources=args.n_pe_sources, dx=args.dx, dz=args.dz,
                   lam_rel=args.lam, device=device)
    t_grappa = time.time() - t0
    log(f"GRAPPA: {res.info['n_targets']} positions filled ({res.info['n_equations']} calibration equations, "
        f"{res.info['n_unknowns']} unknowns), {t_grappa:.0f} s")
    t0 = time.time()
    holdout = acs_holdout_check(k, m_acq, acs_lines=acs, R=2, n_pe_sources=args.n_pe_sources, dx=args.dx,
                                dz=args.dz, lam_rel=args.lam, device=device)
    log(f"ACS hold-out check: relative k-space error on the hidden odd lines {holdout:.4f} ({time.time() - t0:.0f} s)")

    kg = res.kspace
    e_acq = float(torch.linalg.vector_norm(kg[:, :, torch.from_numpy(m_acq)]) ** 2)
    e_grp = float(torch.linalg.vector_norm(kg[:, :, torch.from_numpy(res.m_grappa)]) ** 2)
    log(f"energy: acquired {e_acq:.4g}, interpolated {e_grp:.4g} ({100 * e_grp / (e_acq + e_grp):.2f} % of the total)")

    t0 = time.time()
    rss_g = rss(kg.to(device)); rss_z = rss(k.to(device))
    log(f"RSS images {rss_g.shape}, {time.time() - t0:.0f} s")

    t0 = time.time()
    with h5py.File(os.path.join(out, "kspace_grappa.h5"), "w") as f:
        d = f.create_dataset("kspace", data=kg.numpy(), chunks=(1,) + tuple(kg.shape[1:]))
        d.attrs["axes"] = "coil, RO, PE, SPE (DC-centred; RO oversampling removed)"
        d.attrs["centres"] = [idx.ro_center, idx.pe_center, idx.spe_center]
        d.attrs["voxel_mm"] = [0.7, 0.7, 1.35]
        d.attrs["source"] = f"{os.path.basename(RAW)} set={SET_MAIN} con={args.echo} u1={U1_OF_FA[args.fa]}"
    np.savez(os.path.join(out, "masks.npz"), m_acq=m_acq, m_grappa=res.m_grappa, m_zero=res.m_zero,
             envelope=res.envelope, acs=np.array(acs))
    np.save(os.path.join(out, "rss_grappa.npy"), rss_g)
    np.save(os.path.join(out, "rss_zf.npy"), rss_z)
    t_save = time.time() - t0
    meta = dict(contrast=name, flip_angle_deg=args.fa, echo=args.echo, te_ms=te_ms[args.echo], te_all_ms=te_ms,
                raw=RAW, set=SET_MAIN, u1=U1_OF_FA[args.fa], shape=list(kg.shape), centres=[idx.ro_center, idx.pe_center, idx.spe_center],
                voxel_mm=[0.7, 0.7, 1.35], acs_lines=list(acs), n_acquired=int(m_acq.sum()), n_grappa=int(res.m_grappa.sum()),
                n_zero_by_design=int(res.m_zero.sum()), af_vs_band=float(227 * 104 / m_acq.sum()),
                af_vs_full_grid=float(idx.pe_len * idx.spe_len / m_acq.sum()), grappa=res.info,
                acs_holdout_rel_error=holdout, energy_acquired=e_acq, energy_interpolated=e_grp,
                timings_s=dict(index=t_index, read=t_read, grappa=t_grappa, save=t_save), device=str(device),
                gain_codes=[int(v) for v in np.unique(idx.rxgain[sel])])
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    log(f"saved to {out}")


if __name__ == "__main__":
    main()
