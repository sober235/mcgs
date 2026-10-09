#!/usr/bin/env python3
"""fig_lesion_planes_zoom.py -- the lesion figure on three planes, pre on top, post below (user layout 2026-10-09).

Rows, per scan: transverse image, transverse zoom, coronal image, coronal zoom, sagittal image, sagittal zoom.
  image row  the whole slice, grey P99.9 window divided by --brightness (display only), an orange box at the zoom location,
             in-panel head-masked slice PSNR / SSIM / NMSE (bold = best of the eleven arms on that slice)
  zoom row   left half = the boxed region of the image above, magnified; right half = |rec - GT| / max|GT| of the same
             region, jet [0, err_vmax], head-masked. The truth column's right half is the sampling mask cut through the
             k-space centre in that plane.
Columns: truth plus the eleven arms (arms.py order). No lesion outline is drawn; the box says where the lesion is.
The zoom box is a square around the lesion core's cross-section in that slice, grown by --pad voxels.
  env RETRO_TOPIC=<topic> PYTHONNOUSERSITE=1 PYTHONPATH= <GSTMI python> scripts/fig_lesion_planes_zoom.py --tra 204 --cor 130 --sag 44
"""
import argparse, json, os
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

from arms import arm_files

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
ROI = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz")
SERVED = Path("/home/congcongliu/figs/ce_retro_af16")


def plane(v, ax, i):
    """volume slice -> display array, the topic's convention (fig_three_planes_compact / fig_lesion_axial): transverse
    np.rot90(x.T, 1), coronal and sagittal as cut; every panel is drawn with origin='lower'."""
    s = np.take(v, i, axis=ax)
    return np.rot90(s.T, 1) if ax == 0 else s


def mask_plane(m2, ax, shape3):
    """2-D PE mask (A-P, L-R) cut through the k-space centre in the plane of axis ax, displayed like the images."""
    nx = shape3[0]
    if ax == 0:
        return np.rot90(m2.T, 1)
    if ax == 1:                                  # (S-I, L-R): readout fully sampled, L-R lines where ky = centre is sampled
        return np.broadcast_to(m2[m2.shape[0] // 2][None, :], (nx, m2.shape[1]))
    return np.broadcast_to(m2[:, m2.shape[1] // 2][None, :], (nx, m2.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tra", type=int, default=204); ap.add_argument("--cor", type=int, default=130); ap.add_argument("--sag", type=int, default=44)
    ap.add_argument("--pad", type=int, default=10); ap.add_argument("--brightness", type=float, default=1.8)
    ap.add_argument("--err_vmax", type=float, default=0.1); ap.add_argument("--cw", type=float, default=1.45, help="column width, inches")
    ap.add_argument("--dpi", type=int, default=150); ap.add_argument("--pdf", action="store_true"); ap.add_argument("--tag", default="")
    ap.add_argument("--order", default="plane", choices=["plane", "scan"], help="plane: each plane shows pre then post; scan: pre's three planes then post's")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    with np.load(ROI) as r:
        names = [str(n) for n in r["names"]]; core3 = r["roi%d" % names.index("病灶核心")] > 0
    m2 = np.load(R / "sim/mask/mask_af16_pe.npy") > 0
    planes = [(0, a.tra, "transverse"), (1, a.cor, "coronal"), (2, a.sag, "sagittal")]
    scans = ("pre", "post")
    labels = ["GT"] + [lab for _, lab, _ in arm_files("pre")]

    # load once
    data = {}
    for sc in scans:
        gt = np.abs(np.load(R / f"recon/gt_{sc}_complex.npy"))
        files = {lab: p for _, lab, p in arm_files(sc)}
        vols = {lab: np.abs(np.load(files[lab], mmap_mode="r")) for lab in labels[1:]}
        p999 = float(np.quantile(gt, 0.999)); gmax = float(gt.max()); head = gt > 0.05 * p999
        data[sc] = dict(gt=gt, vols=vols, p999=p999, gmax=gmax, head=head)

    # geometry: one box per plane (same for both scans: same lesion, same slice)
    boxes, disp_shape = {}, {}
    for ax, i, _ in planes:
        c = plane(core3, ax, i); ys, xs = np.where(c)
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        half = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + a.pad
        y0, x0 = max(cy - half, 0), max(cx - half, 0)
        boxes[ax] = (y0, min(y0 + 2 * half, c.shape[0]), x0, min(x0 + 2 * half, c.shape[1]))
        disp_shape[ax] = c.shape

    rows = []
    if a.order == "plane":          # per plane: pre image, pre zoom, post image, post zoom (user 2026-10-09)
        for ax, i, pname in planes:
            for sc in scans:
                rows += [("img", sc, ax, i, pname), ("zoom", sc, ax, i, pname)]
    else:                           # per scan: pre's three planes, then post's
        for sc in scans:
            for ax, i, pname in planes:
                rows += [("img", sc, ax, i, pname), ("zoom", sc, ax, i, pname)]
    hr = [disp_shape[r[2]][0] / disp_shape[r[2]][1] if r[0] == "img" else 0.5 for r in rows]
    W = a.cw * len(labels); H = a.cw * sum(hr)
    fig = plt.figure(figsize=(W + 0.9, H + 2.1))
    gs = fig.add_gridspec(len(rows), len(labels), height_ratios=hr, left=0.9 / (W + 0.9), right=1 - 0.02, top=1 - 1.2 / (H + 2.1),
                          bottom=0.9 / (H + 2.1), wspace=0.03, hspace=0.04)
    scores = {}
    for ri, (kind, sc, ax, i, pname) in enumerate(rows):
        d = data[sc]; gt = plane(d["gt"], ax, i); head = plane(d["head"], ax, i); vmax = d["p999"] / a.brightness
        y0, y1, x0, x1 = boxes[ax]
        if kind == "img":
            vals = {}
            lg = ndi.gaussian_laplace(gt.astype(np.float64), 1.5)
            for lab in labels[1:]:
                v = plane(d["vols"][lab], ax, i).astype(np.float64)
                mse = np.mean((v[head] - gt[head]) ** 2)
                _, sm = structural_similarity(v, gt.astype(np.float64), data_range=d["gmax"], full=True)
                vals[lab] = dict(psnr=20 * np.log10(d["gmax"]) - 10 * np.log10(mse), ssim=float(sm[head].mean()),
                                 nmse=float(np.sum((v[head] - gt[head]) ** 2) / np.sum(gt[head] ** 2)),
                                 hfen=float(np.linalg.norm((ndi.gaussian_laplace(v, 1.5) - lg)[head]) / np.linalg.norm(lg[head])))
            best = {"psnr": max(vals, key=lambda k: vals[k]["psnr"]), "ssim": max(vals, key=lambda k: vals[k]["ssim"]),
                    "nmse": min(vals, key=lambda k: vals[k]["nmse"]), "hfen": min(vals, key=lambda k: vals[k]["hfen"])}
            scores[f"{sc}_{pname}_{i}"] = vals
        for ci, lab in enumerate(labels):
            axh = fig.add_subplot(gs[ri, ci]); axh.set_xticks([]); axh.set_yticks([])
            v = gt if lab == "GT" else plane(d["vols"][lab], ax, i)
            if kind == "img":
                axh.imshow(v, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest", origin="lower"); axh.set_aspect("equal")
                axh.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec="orange", lw=1.0))
                if ri == 0:
                    axh.set_title(lab, fontsize=12)
                if lab == "GT":
                    axh.text(0.03, 0.97, f"GT ({sc})", color="#ffd34d", fontsize=9, fontweight="bold", va="top", transform=axh.transAxes)
                else:
                    s = vals[lab]
                    for k, (m, txt) in enumerate((("psnr", f"{s['psnr']:.2f} dB"), ("ssim", f"SSIM {s['ssim']:.4f}"), ("nmse", f"NMSE {s['nmse']:.4f}"), ("hfen", f"HFEN {s['hfen']:.4f}"))):
                        axh.text(0.03, 0.97 - 0.085 * k * (1.0 / max(hr[ri], 0.6)), txt, color="#ffd34d", fontsize=7.5, va="top",
                                 transform=axh.transAxes, fontweight="bold" if best[m] == lab else "normal")
            else:
                for s_ in axh.spines.values(): s_.set_visible(False)
                L = axh.inset_axes([0.0, 0.0, 0.495, 1.0]); Rr = axh.inset_axes([0.505, 0.0, 0.495, 1.0])
                for q in (L, Rr):
                    q.set_xticks([]); q.set_yticks([])
                    for s_ in q.spines.values(): s_.set_color("orange"); s_.set_linewidth(1.0)
                L.imshow(v[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=vmax, interpolation="nearest", aspect="auto", origin="lower")
                if lab == "GT":
                    mk = mask_plane(m2, ax, d["gt"].shape)
                    Rr.imshow(mk, cmap="gray", interpolation="nearest", aspect="auto", origin="lower")
                else:
                    err = np.where(head, np.abs(v - gt) / d["gmax"], 0.0)
                    Rr.imshow(err[y0:y1, x0:x1], cmap="jet", vmin=0, vmax=a.err_vmax, interpolation="nearest", aspect="auto", origin="lower")
            if ci == 0:
                if kind == "img":
                    axh.set_ylabel(f"{sc}  {pname} {i}", fontsize=10)
                else:
                    axh.text(-0.04, 0.5, "zoom | err", rotation=90, ha="right", va="center", fontsize=8, transform=axh.transAxes)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, a.err_vmax))
    cax = fig.add_axes([1 - 0.02 - 2 * a.cw / (W + 0.9), 0.52 / (H + 2.1), 2 * a.cw / (W + 0.9), 0.13 / (H + 2.1)])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal"); cb.set_label("|err| / max|GT|", fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.suptitle(f"CE retro, nominal AF 16, actual 16.00x (vdPoisMex). Truth = CS-SENSE reconstruction of the AF 5.33 acquisition. "
                 f"{'Per plane: pre, then post' if a.order == 'plane' else 'Pre on top, post below'}; each block: whole slice (orange box = zoom location), then zoom (left) | error zoom (right).\n"
                 f"Image panels: head-masked slice PSNR / SSIM / NMSE / HFEN (2-D LoG, sigma 1.5), bold = best of the 11 arms. Error |rec - GT| / max|GT|, jet [0, {a.err_vmax:g}], head-masked.\n"
                 f"Grey window P99.9 / {a.brightness:g} (display only). GT column zoom row, right half: sampling mask in that plane.", fontsize=10)
    name = f"ce_retro_af16_lesion_planes_zoom_t{a.tra}_c{a.cor}_s{a.sag}{'_byplane' if a.order == 'plane' else ''}{a.tag}"
    out = R / "figures" / f"{name}.png"; out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=a.dpi)
    if a.pdf:
        fig.savefig(out.with_suffix(".pdf"), dpi=600)
    SERVED.mkdir(parents=True, exist_ok=True); fig.savefig(SERVED / f"{name}.png", dpi=a.dpi)
    json.dump(scores, open(R / "logs" / f"{name}_scores.json", "w"), indent=1)
    print(out)
    for k, vals in scores.items():
        print(k, "MC-GS %.3f GS-single %.3f margin %+.3f | best %s" % (vals["MC-GS"]["psnr"], vals["GS-single"]["psnr"],
              vals["MC-GS"]["psnr"] - vals["GS-single"]["psnr"], max(vals, key=lambda q: vals[q]["psnr"])))


if __name__ == "__main__":
    main()
