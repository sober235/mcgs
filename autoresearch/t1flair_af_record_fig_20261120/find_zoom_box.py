#!/usr/bin/env python3
"""find_zoom_box.py -- where, on the slices of the T1 + FLAIR record figure, do the error maps differ most?
(user, 2026-11-21: "identify the zoom-in section with the greatest difference in the error map shown in the figure")

For each of the six rows (transverse / coronal / sagittal x T1 / FLAIR, slices 180 / 124 / 76) a box of --box voxels
is slid over the slice. In each position the mean of the figure's error map (|recon - GT| / max GT) is taken per
method. Score = (lowest box mean among the ten other methods) - (MC-GS box mean): the box where MC-GS's error map is
furthest below the best competitor's. Only boxes with >= --head_frac of their pixels inside the head count.
Also printed: the same numbers for the box the figure uses now (where the three slices meet), and a joint choice
(one box position per plane shared by T1 and FLAIR, score = sum of the two contrasts).
Volumes, arms and head mask as fig_t1flair_af_record_format.py. Read-only, CPU. An analysis of finished
reconstructions against GT; nothing here enters a reconstruction.

  gpu.py run --cpu -- @python autoresearch/t1flair_af_record_fig_20261120/find_zoom_box.py --af 12 --single indep
"""
import argparse, json, os, re
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--af", type=int, default=12, choices=[12, 16, 20, 24])
ap.add_argument("--single", choices=["registered", "indep"], default="registered")
ap.add_argument("--slices", type=int, nargs=3, default=[180, 124, 76])
ap.add_argument("--box", type=int, default=48)
ap.add_argument("--head_frac", type=float, default=0.95)
args = ap.parse_args()
R = Path(f"/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF{args.af}_v3_jiangyizhou")
K = "p14" if args.af == 12 else "p3"
ARC_IS = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_t1flair_af_indep_single_20261121/trainer/results_arc")
ARC_FW = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_framework_ablation_20261110/trainer/results_arc")
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "zoom_out")); art.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)
MC = "MC-GS"


def vols(c):
    rows = {r["method"]: r for r in json.load(open(R / "logs/summary.json")) if r["contrast"] == c}

    def lam(key, prefix):
        hit = [p for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy")
               if abs(float(re.search(r"_lam([0-9.]+)_", p.name).group(1)) - float(rows[key]["lam"])) < 1e-12]
        assert len(hit) == 1, (key, hit)
        return hit[0]
    sp = rows["spirit"]
    P = {"ZF": R / f"recon/zf_{c}_complex.npy", "CG-SENSE": R / f"recon/cgsense_{c}_complex.npy",
         "CS-SENSE": lam("cssense_o200", "cssense_o200"), "Joint SENSE": lam("jcssense", "jcssense_o200"),
         "L1-SPIRiT": R / f"recon/spirit_ks{sp['ksize']}_lam{sp['lam']:g}_{c}_complex.npy",
         "ZS-SSDU": R / f"recon/zsssdu_{c}_complex.npy", "3D-DIP": R / f"recon/dip_{c}_complex.npy",
         "3D-INR": R / f"recon/inr_{c}_complex.npy", "MC-INR": R / f"recon/mcinr_{c}_complex.npy",
         "GS-single": R / f"recon/gs_single_{K}_{c}_mag.npy", MC: R / f"recon/gs_joint_{K}_{c}_mag.npy"}
    out = {k: np.abs(np.load(p)).astype(np.float32) for k, p in P.items()}
    if args.single == "indep":
        st = ARC_FW / f"fw_t1flair_nojoint_{c}" if args.af == 16 else ARC_IS / f"is_af{args.af}_indep_{c}"
        with np.load(f"{st}_volumes.npz") as z:
            out["GS-single"] = np.abs(np.asarray(z["rec"])).astype(np.float32)
    return out


D = lambda x, axis: np.rot90(x.T, 1) if axis == 0 else x          # the figure's display transform
B, hw = args.box, args.box // 2
res, panels = dict(af=args.af, single=args.single, box=B, slices=args.slices, rows=[]), []
data = {}
for c in ("T1", "FLAIR"):
    gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
    data[c] = dict(gt=gt, gmax=float(gt.max()), head=gt > 0.05 * float(np.quantile(gt, 0.999)), v=vols(c))
LABS = list(data["T1"]["v"])
OTH = [l for l in LABS if l != MC]
names = ("Transverse", "Coronal", "Sagittal")
for axis, idx in enumerate(args.slices):
    sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
    rest = [a for a in range(3) if a != axis]
    cur = tuple(args.slices[a] for a in rest)                      # centre of the figure's present box
    sc, bm, ok = {}, {}, None
    for c in ("T1", "FLAIR"):
        d = data[c]
        m = d["head"][sl]
        bm[c] = {l: uniform_filter(np.abs(d["v"][l][sl] - d["gt"][sl]) / d["gmax"] * m, size=B, mode="constant") for l in LABS}
        frac = uniform_filter(m.astype(np.float32), size=B, mode="constant")
        valid = frac >= args.head_frac
        valid[:hw] = valid[-hw:] = False; valid[:, :hw] = valid[:, -hw:] = False
        ok = valid if ok is None else ok & valid
        sc[c] = np.minimum.reduce([bm[c][l] for l in OTH]) - bm[c][MC]

    def report(c, pos, tag):
        means = {l: float(bm[c][l][pos]) for l in LABS}
        best = min(OTH, key=means.get)
        r = dict(plane=names[axis], index=idx, contrast=c, which=tag, centre=dict(zip((f"axis{rest[0]}", f"axis{rest[1]}"), map(int, pos))),
                 box_mean_error=means, best_other=best, mcgs_minus_best_other=means[MC] - means[best],
                 mcgs_over_best_other=means[MC] / means[best], mcgs_minus_gs_single=means[MC] - means["GS-single"])
        res["rows"].append(r)
        log(f"  {names[axis]:<10} {idx:>3} {c:<5} {tag:<8} centre {tuple(map(int, pos))}  MC-GS {means[MC]:.5f}  best other {best} "
            f"{means[best]:.5f}  ({100 * (1 - means[MC] / means[best]):+.1f}% lower)  GS-single {means['GS-single']:.5f}  "
            f"ZS-SSDU {means['ZS-SSDU']:.5f}  CS-SENSE {means['CS-SENSE']:.5f}")
        return r

    for c in ("T1", "FLAIR"):
        report(c, cur, "current")
        s = np.where(ok, sc[c], -np.inf)
        pos = np.unravel_index(np.argmax(s), s.shape)
        report(c, pos, "best")
    s = np.where(ok, sc["T1"] + sc["FLAIR"], -np.inf)
    pos = np.unravel_index(np.argmax(s), s.shape)
    for c in ("T1", "FLAIR"):
        r = report(c, pos, "joint")
        d = data[c]
        box = np.zeros(d["gt"][sl].shape, bool); box[pos[0] - hw:pos[0] + hw, pos[1] - hw:pos[1] + hw] = True
        rr, cc = np.nonzero(D(box, axis))
        crop = lambda x: D(x, axis)[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        panels.append((f"{names[axis]} {idx} {c}\ncentre {tuple(map(int, pos))}", D(d["gt"][sl], axis), (cc.min(), rr.min()),
                       [(l, crop(d["v"][l][sl]), crop(np.abs(d["v"][l][sl] - d["gt"][sl]) / d["gmax"] * d["head"][sl]), r["box_mean_error"][l]) for l in LABS],
                       float(np.quantile(d["gt"][sl], 0.999))))

(art / f"zoom_box_af{args.af}_{args.single}.json").write_text(json.dumps(res, indent=1))
# preview: per row the slice with the joint box, then the error crop of every method in that box
n = len(LABS)
fig, axs = plt.subplots(len(panels), n + 1, figsize=(1.9 * (n + 1), 1.95 * len(panels)))
for i, (title, g2, (x0, y0), items, vmax) in enumerate(panels):
    a = axs[i, 0]; a.imshow(g2, cmap="gray", vmin=0, vmax=vmax, origin="lower")
    a.add_patch(plt.Rectangle((x0 - .5, y0 - .5), B, B, fill=False, ec="#FFC23D", lw=1.5)); a.set_ylabel(title, fontsize=7)
    lo = min(it[3] for it in items)
    for j, (l, _, e, mean) in enumerate(items):
        a = axs[i, j + 1]; a.imshow(e, cmap="jet", vmin=0, vmax=0.1, origin="lower")
        a.set_title(f"{l}\n{mean:.4f}", fontsize=8, fontweight="bold" if mean == lo else "normal")
for a in axs.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"AF{args.af}: box ({B} voxels) where MC-GS's error map is furthest below the best other method, one position per plane "
             f"(T1 + FLAIR); number = mean |error| / max GT in the box; GS-single = {args.single}", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(art / f"zoom_box_af{args.af}_{args.single}.png", dpi=130)
log("DONE")
