#!/usr/bin/env python3
"""compare_masks.py -- the BART-mask line against the vdPoisMex-mask line, arm by arm (audit 2026-09-29).

Reads both topics' logs/summary.json for the head-only metrics (summarize.py must have run in each), loads the
registered volume of every arm present in both (ZF, CG-SENSE, CS-SENSE at each topic's own best lambda, GS-single,
MC-GS), and adds the one number the audit is about: the NMSE per k-space band (normalised radius on the PE plane,
error and truth energies summed over the readout axis). Writes $NEW/logs/mask_swap_compare.json, prints the table,
and draws two figures into $NEW/figures/ and /home/congcongliu/figs/ce_retro_af16_audit/:
  mask_swap_axial_z201.png   GT | BART recon | BART error | vdPois recon | vdPois error, one row per arm, per scan
  mask_swap_band_nmse.png    band NMSE curves, solid = BART mask, dashed = vdPoisMex mask
  python scripts/compare_masks.py [--old <topic>] [--new <topic>] [--z 201]
"""
import argparse, json, os, shutil
from pathlib import Path

import numpy as np

OLD = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16")
NEW = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois")
ARMS = [("zf", "ZF"), ("cgsense", "CG-SENSE"), ("cssense_o200", "CS-SENSE"), ("gs_single", "GS-single"), ("gs_joint", "MC-GS")]
BANDS = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.5)]
METRICS = ("head_psnr", "head_ssim", "head_nmse", "head_hfen")


def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1), norm="ortho"), axes=(-2, -1))


def volume_path(topic, key, scan, row):
    if key == "cssense_o200":
        return topic / f"recon/cssense_o200_lam{row['lam']:g}_{scan}_complex.npy"
    return topic / f"recon/{key}_{scan}_complex.npy"


def band_nmse(vol, gt, rad):
    E = np.mean(np.abs(fft2c(vol - gt)) ** 2, axis=0); G = np.mean(np.abs(fft2c(gt)) ** 2, axis=0)
    return {f"{a}-{b}": float(E[(rad >= a) & (rad < b)].sum() / max(G[(rad >= a) & (rad < b)].sum(), 1e-30)) for a, b in BANDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=str(OLD)); ap.add_argument("--new", default=str(NEW)); ap.add_argument("--z", type=int, default=201)
    a = ap.parse_args(); old, new = Path(a.old), Path(a.new)
    summ = {t: {(r["contrast"], r["method"]): r for r in json.load(open(t / "logs/summary.json"))} for t in (old, new)}
    yy, zz = np.meshgrid((np.arange(232) - 116) / 116, (np.arange(176) - 88) / 88, indexing="ij"); rad = np.sqrt(yy ** 2 + zz ** 2)
    out, panels = {}, {}
    for scan in ("pre", "post"):
        gt = np.load(old / f"recon/gt_{scan}_complex.npy"); p999 = float(np.quantile(np.abs(gt), 0.999))
        out[scan] = {}
        for key, label in ARMS:
            rows = {t: summ[t].get((scan, key)) for t in (old, new)}
            if any(r is None for r in rows.values()):
                print(f"[skip] {scan} {label}: missing in", [str(t) for t, r in rows.items() if r is None]); continue
            rec = {}
            for t in (old, new):
                v = np.load(volume_path(t, key, scan, rows[t]))
                rec[t] = {"metrics": {m: float(rows[t][m]) for m in METRICS}, "lam": rows[t].get("lam"), "band_nmse": band_nmse(v, gt, rad)}
                panels[(scan, label, t)] = (np.abs(v[a.z]) / p999, np.abs(np.abs(v[a.z]) - np.abs(gt[a.z])) / p999)
            panels[(scan, "GT")] = np.abs(gt[a.z]) / p999
            out[scan][label] = {"bart": rec[old], "vdpois": rec[new],
                                "delta": {m: rec[new]["metrics"][m] - rec[old]["metrics"][m] for m in METRICS}}
    new.joinpath("logs").mkdir(exist_ok=True); json.dump(out, open(new / "logs/mask_swap_compare.json", "w"), indent=1)
    # table
    print(f"{'scan':5} {'arm':10} {'PSNR bart':>10} {'vdpois':>8} {'delta':>7} | {'SSIM bart':>9} {'vdpois':>7} | {'NMSE bart':>9} {'vdpois':>7} | {'HFEN bart':>9} {'vdpois':>7} | band 0.2-0.3 bart/vd | 0.3-0.4 bart/vd")
    for scan in out:
        for label, r in out[scan].items():
            b, v = r["bart"], r["vdpois"]
            print(f"{scan:5} {label:10} {b['metrics']['head_psnr']:10.3f} {v['metrics']['head_psnr']:8.3f} {r['delta']['head_psnr']:+7.3f} | "
                  f"{b['metrics']['head_ssim']:9.4f} {v['metrics']['head_ssim']:7.4f} | {b['metrics']['head_nmse']:9.5f} {v['metrics']['head_nmse']:7.5f} | "
                  f"{b['metrics']['head_hfen']:9.4f} {v['metrics']['head_hfen']:7.4f} | {b['band_nmse']['0.2-0.3']:.3f}/{v['band_nmse']['0.2-0.3']:.3f} | "
                  f"{b['band_nmse']['0.3-0.4']:.3f}/{v['band_nmse']['0.3-0.4']:.3f}")
    # figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    figdir = new / "figures"; figdir.mkdir(exist_ok=True); pub = Path("/home/congcongliu/figs/ce_retro_af16_audit"); pub.mkdir(exist_ok=True)
    for scan in out:
        labels = list(out[scan]); n = len(labels)
        if n == 0:
            continue
        fig, ax = plt.subplots(n, 5, figsize=(14, 2.9 * n))
        ax = np.atleast_2d(ax)
        y0, y1, x0, x1 = 40, 200, 20, 160
        for i, label in enumerate(labels):
            ax[i, 0].imshow(panels[(scan, "GT")][y0:y1, x0:x1].T, cmap="gray", vmin=0, vmax=1, origin="lower"); ax[i, 0].set_title(f"GT {scan} z{a.z}" if i == 0 else "GT", fontsize=9)
            for j, (t, name) in enumerate([(old, "BART poisson -v"), (new, "vdPoisMex")]):
                im, err = panels[(scan, label, t)]
                m = out[scan][label]["bart" if t == old else "vdpois"]["metrics"]
                ax[i, 1 + 2 * j].imshow(im[y0:y1, x0:x1].T, cmap="gray", vmin=0, vmax=1, origin="lower")
                ax[i, 1 + 2 * j].set_title(f"{label} | {name}\nhead PSNR {m['head_psnr']:.2f}  SSIM {m['head_ssim']:.3f}", fontsize=8)
                ax[i, 2 + 2 * j].imshow(err[y0:y1, x0:x1].T, cmap="jet", vmin=0, vmax=0.1, origin="lower"); ax[i, 2 + 2 * j].set_title("|err|/max(GT), jet 0-0.1", fontsize=8)
        for x in ax.ravel(): x.axis("off")
        plt.tight_layout(); fn = figdir / f"mask_swap_axial_z{a.z}_{scan}.png"; plt.savefig(fn, dpi=110); plt.close(); shutil.copy(fn, pub / fn.name)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    centres = [(b[0] + b[1]) / 2 for b in BANDS]; colors = dict(zip([l for _, l in ARMS], ["k", "tab:blue", "tab:orange", "tab:purple", "tab:red"]))
    for k, scan in enumerate(out):
        for label, r in out[scan].items():
            ax[k].semilogy(centres, [r["bart"]["band_nmse"][f"{b[0]}-{b[1]}"] for b in BANDS], "-o", color=colors[label], label=f"{label} BART", ms=4)
            ax[k].semilogy(centres, [r["vdpois"]["band_nmse"][f"{b[0]}-{b[1]}"] for b in BANDS], "--s", color=colors[label], label=f"{label} vdPois", ms=4)
        ax[k].set_title(f"{scan}: NMSE per k-space band (solid BART, dashed vdPoisMex)"); ax[k].set_xlabel("normalised radius"); ax[k].grid(alpha=.3)
        if k == 0: ax[k].legend(fontsize=7, ncol=2)
    plt.tight_layout(); fn = figdir / "mask_swap_band_nmse.png"; plt.savefig(fn, dpi=120); shutil.copy(fn, pub / fn.name)
    print("wrote", new / "logs/mask_swap_compare.json", "and figures in", figdir)


if __name__ == "__main__":
    main()
