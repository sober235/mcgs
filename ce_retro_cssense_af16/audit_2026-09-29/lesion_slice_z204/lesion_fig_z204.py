import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.metrics import structural_similarity
A = "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois_ar"
os.environ["RETRO_TOPIC"] = A; sys.path.insert(0, A + "/scripts")
import arms
Z = 204
roi = np.load("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz")
names = [str(n) for n in roi["names"]]; core = roi["roi%d" % names.index("病灶核心")][Z] > 0
rot = lambda a: np.rot90(a, 1)
ys, xs = np.where(core); pad = 14
by0, by1, bx0, bx1 = ys.min() - pad, ys.max() + pad, xs.min() - pad, xs.max() + pad      # lesion box in (axis1, axis2)
cols = [("GT", None)] + [(lab, p) for k, lab, p in arms.arm_files("pre")]
res = {}
fig, ax = plt.subplots(4, 12, figsize=(30, 9.3), gridspec_kw=dict(wspace=0.02, hspace=0.08))
for si, sc in enumerate(("pre", "post")):
    files = dict((lab, p) for k, lab, p in arms.arm_files(sc))
    gt3 = np.abs(np.load(f"{A}/recon/gt_{sc}_complex.npy")); gmax = float(gt3.max()); p999 = float(np.quantile(gt3, 0.999))
    head3 = gt3 > 0.05 * p999; gt = gt3[Z]; head = head3[Z]
    _, smap = structural_similarity(gt, gt, data_range=gmax, full=True)
    vals = {}
    for j, (lab, _) in enumerate(cols):
        v = gt if lab == "GT" else np.abs(np.load(files[lab], mmap_mode="r")[Z])
        if lab != "GT":
            mse = np.mean((v[head] - gt[head]) ** 2); psnr = 20 * np.log10(gmax) - 10 * np.log10(mse)
            _, sm = structural_similarity(v.astype(np.float64), gt.astype(np.float64), data_range=gmax, full=True)
            box = np.zeros_like(head); box[by0:by1, bx0:bx1] = True; box &= head
            bpsnr = 20 * np.log10(gmax) - 10 * np.log10(np.mean((v[box] - gt[box]) ** 2))
            vals[lab] = dict(psnr=psnr, ssim=float(sm[head].mean()), box_psnr=bpsnr)
        a = ax[2 * si, j]; a.imshow(rot(v / p999), cmap="gray", vmin=0, vmax=0.667)
        a.contour(rot(core), levels=[0.5], colors="magenta", linewidths=0.8)
        H, Wd = rot(v).shape
        a.add_patch(Rectangle((by0, H - bx1), by1 - by0, bx1 - bx0, fill=False, ec="orange", lw=1.2))
        ins = a.inset_axes([0.58, 0.58, 0.42, 0.42]); ins.imshow(rot(v[by0:by1, bx0:bx1] / p999), cmap="gray", vmin=0, vmax=0.667)
        ins.contour(rot(core[by0:by1, bx0:bx1]), levels=[0.5], colors="magenta", linewidths=0.8); ins.set_xticks([]); ins.set_yticks([])
        for s in ins.spines.values(): s.set_color("orange"); s.set_linewidth(1.2)
        e = ax[2 * si + 1, j]
        if lab == "GT":
            e.imshow(rot(core), cmap="gray"); e.text(0.03, 0.95, "lesion core", color="w", fontsize=9, va="top", transform=e.transAxes)
        else:
            err = np.where(head, np.abs(v - gt) / p999, 0); e.imshow(rot(err), cmap="jet", vmin=0, vmax=0.1)
            ie = e.inset_axes([0.58, 0.58, 0.42, 0.42]); ie.imshow(rot(err[by0:by1, bx0:bx1]), cmap="jet", vmin=0, vmax=0.1); ie.set_xticks([]); ie.set_yticks([])
            for s in ie.spines.values(): s.set_color("orange"); s.set_linewidth(1.2)
        if si == 0: a.set_title(lab, fontsize=12)
    best = {m: max(vals, key=lambda k: vals[k][m]) for m in ("psnr", "ssim", "box_psnr")}
    for j, (lab, _) in enumerate(cols):
        if lab == "GT":
            ax[2 * si, 0].text(0.03, 0.97, f"GT ({sc})", color="yellow", fontsize=11, fontweight="bold", va="top", transform=ax[2 * si, 0].transAxes); continue
        v = vals[lab]; a = ax[2 * si, j]
        for k, (m, fmt) in enumerate((("psnr", "{:.2f} dB"), ("ssim", "SSIM {:.4f}"), ("box_psnr", "box {:.2f}"))):
            a.text(0.03, 0.97 - 0.075 * k, fmt.format(v[m]), color="yellow", fontsize=8.5, va="top", transform=a.transAxes,
                   fontweight="bold" if best[m] == lab else "normal")
    res[sc] = vals
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
mp = res["pre"]["MC-GS"]["psnr"] - res["pre"]["GS-single"]["psnr"]; mq = res["post"]["MC-GS"]["psnr"] - res["post"]["GS-single"]["psnr"]
bp = res["pre"]["MC-GS"]["box_psnr"] - res["pre"]["GS-single"]["box_psnr"]; bq = res["post"]["MC-GS"]["box_psnr"] - res["post"]["GS-single"]["box_psnr"]
fig.suptitle(f"Transverse slice {Z} through the lesion (magenta = lesion core). nominal AF 16, actual 16.00x, vdPoisMex, confirmed MC-GS recipe vs rule-A fair GS-single.\n"
             f"MC-GS - GS-single on this slice: head PSNR pre {mp:+.3f} / post {mq:+.3f} dB, lesion box pre {bp:+.3f} / post {bq:+.3f} dB "
             f"(whole volume +0.062 / +0.088 dB). Slice chosen among the 41 lesion slices: MC-GS ahead of GS-single both on the slice and in the orange lesion box, on both scans (14 qualify). Error |rec-GT|/P99.9, jet [0, 0.1], head only.", fontsize=12)
out = "/home/congcongliu/figs/ce_retro_af16/ce_retro_af16_lesion_mcgs_vs_single_z204.png"
plt.savefig(out, dpi=130, bbox_inches="tight"); plt.savefig(out.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
json.dump(res, open(A + "/logs/lesion_slice_z204_scores.json", "w"), indent=1)
print(out); print(json.dumps({s: {k: round(v["psnr"], 3) for k, v in res[s].items()} for s in res}))
print("box", {s: {k: round(v["box_psnr"], 3) for k, v in res[s].items()} for s in res})
