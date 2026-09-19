#!/usr/bin/env python3
"""lesion_preservation.py -- does an arm keep the lesion? Measured inside fixed regions of interest, with no
segmentation model anywhere in the chain.

WHY NOT A SEGMENTER. Both segmentation routes tried on this project failed for reasons that have nothing to do with
the arms: on this topic the truth is itself a CS-SENSE reconstruction, so a segmenter reads the CS-SENSE family as
most truth-like (SEGMENTATION.md); on the PDGM topic MedSAM under-segments the tumour core badly enough that
zero-filling outscores every real reconstruction (../ce_retro_pdgm_af16/FINDINGS.md). A metric that passes through a
black box only works when the box's own error is small next to the differences being measured. Fixed ROIs have no
such box: the region is the same for every arm, and every number below is monotone in image quality by construction.

THE REGIONS come from lesion_candidates/enhance_rois_cssense.npz, drawn on the truth:
  core     6039 voxels, the lesion. NOT a radiologist's outline -- a person pointed at the lesion on an axial
           montage, and the region is the largest 26-connected blob of smoothed pre-contrast signal below 0.32
           white-matter units inside a hand-set box. It is fixed, and identical for all eleven arms, which is all an
           ROI has to be; it is not a gold standard and must not be quoted as one.
  shell    dilate(core, 2) AND NOT erode(core, 2): the boundary band, ~2 mm on this 1 mm grid.
  rim      the 0-3 mm collar outside the core (roi1).
  wm       486769 voxels of normal-appearing white matter (roi2), the intensity unit and the noise reference.

WHAT IS MEASURED, per arm and scan, over the whole 3-D lesion (not one slice):
  core PSNR / NRMSE     plain fidelity inside the lesion, in the topic's gauge (peak = max|GT| of the volume)
  shell RMSE            intensity agreement in the boundary band -- no assumption about where the edge "is"
  edge RMSE             RMSE of the gradient-magnitude field over the shell: boundary geometry, not brightness
  HFEN (lesion box)     high-frequency error inside the lesion's bounding box, the topic's own hfen definition
  contrast              (mean_wm - mean_core) / mean_wm, and its error against the truth's contrast. This is the
                        clinical question "is the lesion still as conspicuous as it should be"
  CNR                   (mean_wm - mean_core) / std_wm, and its error against the truth's. An arm that smooths the
                        lesion away loses contrast; an arm that adds noise loses CNR without losing contrast
  edge width            20-80% rise distance of the intensity profile against the signed distance to the core
                        boundary, and its error against the truth's. Blur widens it, over-sharpening narrows it
  enhancement in core   mean(post - pre) inside the core in white-matter units, against the truth's. This lesion
                        does not enhance, so the truth's value is the floor: an arm above it is inventing uptake
  per-slice             core PSNR on every axial slice holding >= MIN_SLICE_CORE lesion voxels, and how often each
                        arm ranks first of eleven -- one lesion is one sample, the slices are what there is

  env PYTHONNOUSERSITE=1 PYTHONPATH= <GSTMI python> scripts/lesion_preservation.py

Writes logs/lesion_preservation.json.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

# `arms` is imported inside main(), not here: it binds RETRO_TOPIC at import time, and the test suite's tiny_topic
# fixture only sets that variable, so a module-level import would pin every later test to the real topic. The same
# reason medsam_seg.py imports it inside main().

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
PROS = Path(os.environ.get("CE_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast"))
ROI_NPZ = PROS / "lesion_candidates/enhance_rois_cssense.npz"

SCANS = ("pre", "post")
K_SHELL = 2
BOX_PAD = 8
MIN_SLICE_CORE = 50
EDGE_MM = 4


def gmag(v):
    """forward-difference gradient magnitude; the same formula t1t2_common._gmag uses, in numpy"""
    gx = np.zeros_like(v); gy = np.zeros_like(v); gz = np.zeros_like(v)
    gx[:-1] = v[1:] - v[:-1]
    gy[:, :-1] = v[:, 1:] - v[:, :-1]
    gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
    return np.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)


def hfen_box(rec, gt):
    """the topic's HFEN (Laplacian of the normalised volumes), evaluated on a crop"""
    def lap(v):
        return (6.0 * v - np.roll(v, 1, 0) - np.roll(v, -1, 0) - np.roll(v, 1, 1)
                - np.roll(v, -1, 1) - np.roll(v, 1, 2) - np.roll(v, -1, 2))
    g = gt / (gt.max() + 1e-9)
    r = rec / (gt.max() + 1e-9)
    lg = lap(g)
    return float(np.linalg.norm(lap(r) - lg) / (np.linalg.norm(lg) + 1e-9))


def psnr_in(rec, gt, roi, peak):
    mse = float(((rec[roi] - gt[roi]) ** 2).mean())
    return float("inf") if mse == 0 else float(20 * np.log10(peak) - 10 * np.log10(mse))


def nrmse_in(rec, gt, roi):
    d = rec[roi] - gt[roi]
    return float(np.sqrt((d * d).sum() / ((gt[roi] ** 2).sum() + 1e-12)))


def contrast(vol_wm_units, core, wm):
    """(mean_wm - mean_core) / mean_wm -- positive because this lesion is darker than white matter"""
    mw = float(vol_wm_units[wm].mean())
    return float((mw - float(vol_wm_units[core].mean())) / (mw + 1e-12))


def cnr(vol_wm_units, core, wm):
    """(mean_wm - mean_core) / std_wm: conspicuity against the background's own spread"""
    return float((float(vol_wm_units[wm].mean()) - float(vol_wm_units[core].mean()))
                 / (float(vol_wm_units[wm].std()) + 1e-12))


def signed_distance(core):
    """+ outside the core, - inside, in voxels (1 mm here)"""
    return ndi.distance_transform_edt(~core) - ndi.distance_transform_edt(core)


def edge_width(vol_wm_units, sd, mm=EDGE_MM):
    """20-80% rise distance of the mean intensity profile against the signed distance to the lesion boundary.
    Returns NaN if the profile is not a clean rise (no crossing inside the window)."""
    band = np.abs(sd) <= mm
    d = np.rint(sd[band]).astype(int)
    v = vol_wm_units[band]
    # only bins that actually hold voxels: the exact boundary (d = 0) is never occupied, because the signed
    # distance is <= -1 strictly inside the core and >= +1 strictly outside it
    xs = np.array([x for x in range(-mm, mm + 1) if (d == x).any()])
    if len(xs) < 4:
        return float("nan")
    prof = np.array([v[d == x].mean() for x in xs])
    lo, hi = prof[0], prof[-1]
    if hi - lo <= 0:
        return float("nan")

    def cross(frac):
        target = lo + frac * (hi - lo)
        for i in range(len(xs) - 1):
            a, b = prof[i], prof[i + 1]
            if (a - target) * (b - target) <= 0 and b != a:
                return xs[i] + (target - a) / (b - a)
        return float("nan")

    return float(cross(0.8) - cross(0.2))


def main():
    from arms import ORDER, LABEL, arm_files

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="logs/lesion_preservation.json")
    a = ap.parse_args()

    z = np.load(ROI_NPZ, allow_pickle=True)
    core = z["core"]
    rim, wm = z["roi1"], z["roi2"]
    shell = ndi.binary_dilation(core, iterations=K_SHELL) & ~ndi.binary_erosion(core, iterations=K_SHELL)
    sd = signed_distance(core)
    idx = np.argwhere(core)
    box = tuple(slice(max(int(idx[:, i].min()) - BOX_PAD, 0),
                      min(int(idx[:, i].max()) + BOX_PAD + 1, core.shape[i])) for i in range(3))
    slices_ax = [k for k in range(core.shape[0]) if core[k].sum() >= MIN_SLICE_CORE]
    print(f"core {int(core.sum())} vox, shell {int(shell.sum())}, rim {int(rim.sum())}, wm {int(wm.sum())}; "
          f"lesion box {[ (s.start, s.stop) for s in box ]}; {len(slices_ax)} axial slices scored", flush=True)

    summary = json.load(open(R / "logs/summary.json"))
    rows, per_slice = [], {}
    vols_wm = {}
    for scan in SCANS:
        gt = np.abs(np.load(R / f"recon/gt_{scan}_complex.npy")).astype(np.float64)
        peak = float(gt.max())
        gt_wm = gt / float(np.median(gt[wm]))
        gt_g = gmag(gt)
        vols_wm[("truth", scan)] = gt_wm
        ref = dict(contrast=contrast(gt_wm, core, wm), cnr=cnr(gt_wm, core, wm),
                   edge_width=edge_width(gt_wm, sd))
        rows.append(dict(scan=scan, arm="truth", label="truth", core_psnr=float("inf"), core_nrmse=0.0,
                         shell_rmse=0.0, edge_rmse=0.0, hfen_box=0.0, **ref,
                         contrast_err=0.0, cnr_err=0.0, edge_width_err=0.0))
        print(f"\n[{scan}] truth: contrast {ref['contrast']:.4f}, CNR {ref['cnr']:.3f}, "
              f"edge width {ref['edge_width']:.2f} mm", flush=True)
        print(f"{'arm':>12} {'corePSNR':>9} {'coreNRMSE':>10} {'shellRMSE':>10} {'edgeRMSE':>9} {'HFENbox':>8} "
              f"{'contrast':>9} {'dContr%':>8} {'CNR':>7} {'dCNR%':>7} {'edgeW':>6} {'dEdge':>6}", flush=True)
        ps_slice = {}
        for key, lbl, path in arm_files(scan, summary):
            v = np.abs(np.load(path)).astype(np.float64)
            v_wm = v / float(np.median(v[wm]))
            vols_wm[(key, scan)] = v_wm
            r = dict(scan=scan, arm=key, label=lbl,
                     core_psnr=psnr_in(v, gt, core, peak), core_nrmse=nrmse_in(v, gt, core),
                     shell_rmse=float(np.sqrt((((v - gt)[shell]) ** 2).mean())),
                     edge_rmse=float(np.sqrt((((gmag(v) - gt_g)[shell]) ** 2).mean())),
                     hfen_box=hfen_box(v[box], gt[box]),
                     contrast=contrast(v_wm, core, wm), cnr=cnr(v_wm, core, wm),
                     edge_width=edge_width(v_wm, sd))
            r["contrast_err"] = (r["contrast"] - ref["contrast"]) / ref["contrast"]
            r["cnr_err"] = (r["cnr"] - ref["cnr"]) / ref["cnr"]
            r["edge_width_err"] = r["edge_width"] - ref["edge_width"]
            rows.append(r)
            ps_slice[key] = [psnr_in(v[k], gt[k], core[k], peak) for k in slices_ax]
            print(f"{lbl:>12} {r['core_psnr']:9.3f} {r['core_nrmse']:10.4f} {r['shell_rmse']:10.5f} "
                  f"{r['edge_rmse']:9.5f} {r['hfen_box']:8.4f} {r['contrast']:9.4f} "
                  f"{100 * r['contrast_err']:+8.2f} {r['cnr']:7.3f} {100 * r['cnr_err']:+7.2f} "
                  f"{r['edge_width']:6.2f} {r['edge_width_err']:+6.2f}", flush=True)
        per_slice[scan] = dict(slices=slices_ax,
                               first={LABEL[k]: int(sum(1 for i in range(len(slices_ax))
                                                        if max(ps_slice, key=lambda q: ps_slice[q][i]) == k))
                                      for k in ORDER})

    # enhancement inside the core: this lesion does not enhance, so the truth's value is the floor
    print("\nenhancement inside the lesion core (post - pre, white-matter units; the truth is the floor)")
    enh_truth = float((vols_wm[("truth", "post")][core] - vols_wm[("truth", "pre")][core]).mean())
    print(f"{'truth':>12} {enh_truth:+.4f}")
    enh = {}
    for key in ORDER:
        e = float((vols_wm[(key, "post")][core] - vols_wm[(key, "pre")][core]).mean())
        enh[key] = dict(enh=e, excess=e - enh_truth)
        print(f"{LABEL[key]:>12} {e:+.4f}   excess {e - enh_truth:+.4f}")

    print("\nper-slice core PSNR, times first of eleven:")
    for scan in SCANS:
        f = per_slice[scan]["first"]
        print(f"  [{scan}] " + ", ".join(f"{k} {v}" for k, v in sorted(f.items(), key=lambda t: -t[1]) if v))

    out = R / a.out
    json.dump(dict(rois=dict(core=int(core.sum()), shell=int(shell.sum()), rim=int(rim.sum()), wm=int(wm.sum())),
                   note="ROIs are drawn on the truth and identical for every arm; the core is a hand-guided "
                        "threshold region, not a radiologist's outline",
                   rows=rows, per_slice=per_slice,
                   enhancement=dict(truth=enh_truth, arms={LABEL[k]: v for k, v in enh.items()})),
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
