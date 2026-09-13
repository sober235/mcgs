#!/usr/bin/env python3
"""best_gap_slice.py -- which single slice shows MC-GS ahead of the comparison methods by the most?

Same gauge as the delivered compact figures (fig_three_planes_compact.py): head mask |GT| > 0.05 * P99.9 of the
volume, PSNR at the volume's max|GT|, SSIM = 7x7-uniform SSIM map averaged over the head pixels (data range =
max|GT|), NMSE summed over the head pixels. Competitors are the non-GS arms only (our own single-contrast ablation
is not a competitor, the project's convention); the classical arms are taken at their best lambda per family, as
the figure does.

Ranks every slice of every axis with at least --min_head head voxels by the PSNR margin over the best competitor,
then reports SSIM and NMSE margins for the top --top of each axis, and the best slice common to both contrasts
(largest smaller-of-the-two PSNR margin).

usage: best_gap_slice.py [--mcgs_T1 gs_joint_p2] [--mcgs_FLAIR gs_joint_p2_fl11] [--min_head 20000] [--top 5]
Writes logs/best_gap_slice.json.
"""
import argparse, json, re
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

R = Path(__file__).resolve().parents[1]
SSIM_KW = dict(gaussian_weights=False, use_sample_covariance=False, win_size=7)

ap = argparse.ArgumentParser()
ap.add_argument("--mcgs_T1", default="gs_joint_p2", help="MC-GS arm for T1 (the all-AF12 figure)")
ap.add_argument("--mcgs_FLAIR", default="gs_joint_p2_fl11", help="MC-GS arm for FLAIR (the design-B figure)")
ap.add_argument("--min_head", type=int, default=20000)
ap.add_argument("--top", type=int, default=5)
ap.add_argument("--out", default="best_gap_slice", help="logs/<out>.json")
ap.add_argument("--single_T1", default=None, help="GS-single arm to COUNT AS A COMPETITOR for T1 "
                "(default: not counted, the project convention); pass gs_single_p2 to require MC-GS to lead every arm")
ap.add_argument("--single_FLAIR", default=None, help="same for FLAIR (gs_single_p2 at AF12, gs_single_p2_fl11 in design B)")
a = ap.parse_args()

rows_all = json.load(open(R / "logs/summary.json"))
ANAT = {0: "transverse", 1: "coronal", 2: "sagittal"}


def lam_file(prefix, row_key, c):
    """the classical arm's volume at the lambda summary.json keeps for it (the figure's rule)"""
    row = next((r for r in rows_all if r["contrast"] == c and r["method"] == row_key), None)
    lam = (row or {}).get("lam")
    if lam is None:
        p = R / f"recon/{prefix}_{c}_complex.npy"
        return p if p.exists() else None
    for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
        m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
        if m and abs(float(m.group(1)) - float(lam)) < 1e-12:
            return p
    canon = {"cssense_o200": R / f"logs/classical_{c}_o200.json", "jcssense": R / "logs/jcssense_o200.json"}.get(prefix)
    if canon is not None and canon.exists():
        for r in json.load(open(canon)).get("rows", []):
            if r.get("contrast", c) == c and r.get("lam") is not None and abs(float(r["lam"]) - float(lam)) < 1e-12:
                q = R / f"recon/{prefix}_{c}_complex.npy"
                return q if q.exists() else None
    return None


out = {}
for c, mcgs in (("T1", a.mcgs_T1), ("FLAIR", a.mcgs_FLAIR)):
    comp = {"cgsense": R / f"recon/cgsense_{c}_complex.npy",
            "cssense_o200": lam_file("cssense_o200", "cssense_o200", c),
            "jcssense": lam_file("jcssense", "jcssense", c),
            "l1espirit": lam_file("l1espirit", "l1espirit", c),
            "zsssdu": R / f"recon/zsssdu_{c}_complex.npy",
            "dip": R / f"recon/dip_{c}_complex.npy",
            "inr": R / f"recon/inr_{c}_complex.npy",
            "ploraks": R / f"recon/ploraks_{c}_complex.npy"}
    sgl = {"T1": a.single_T1, "FLAIR": a.single_FLAIR}[c]
    if sgl:
        comp[sgl] = R / f"recon/{sgl}_{c}_mag.npy"
    comp = {k: v for k, v in comp.items() if v is not None and Path(v).exists()}
    gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
    gmax = float(gt.max())
    head = gt > 0.05 * float(np.quantile(gt, 0.999))
    ours = np.abs(np.load(R / f"recon/{mcgs}_{c}_mag.npy")).astype(np.float32)
    vols = {k: np.abs(np.load(v)).astype(np.float32) for k, v in comp.items()}
    print(f"[{c}] ours = {mcgs}; competitors = {sorted(vols)}")

    cand = []
    for axis in (0, 1, 2):
        for idx in range(gt.shape[axis]):
            sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
            m2 = head[sl]
            n = int(m2.sum())
            if n < a.min_head:
                continue
            g = gt[sl].astype(np.float64)[m2]
            def psnr(v):
                e = v[sl].astype(np.float64)[m2] - g
                return 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
            po = psnr(ours)
            pc = {k: psnr(v) for k, v in vols.items()}
            best = max(pc, key=pc.get)
            cand.append(dict(axis=axis, index=idx, n_head=n, ours_psnr=po, best_other=best,
                             best_other_psnr=pc[best], margin_psnr=po - pc[best]))
    cand.sort(key=lambda r: -r["margin_psnr"])

    # the three metrics of the figure, for the top rows of each axis and for the overall top
    detail = {}
    todo = []
    for axis in (0, 1, 2):
        todo += [r for r in cand if r["axis"] == axis][: a.top]
    for r in todo:
        axis, idx = r["axis"], r["index"]
        sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
        g64 = gt[sl].astype(np.float64); m2 = head[sl]; g = g64[m2]
        def three(v):
            x = v[sl].astype(np.float64)
            e = x[m2] - g
            _, smap = structural_similarity(g64, x, data_range=gmax, full=True, **SSIM_KW)
            return (20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean())),
                    float(smap[m2].mean()), float((e * e).sum() / (g * g).sum()))
        po, so, no = three(ours)
        oc = {k: three(v) for k, v in vols.items()}
        r.update(ours_ssim=so, ours_nmse=no,
                 margin_ssim=so - max(v[1] for v in oc.values()),
                 margin_nmse=min(v[2] for v in oc.values()) - no,
                 best_other_ssim_arm=max(oc, key=lambda k: oc[k][1]),
                 best_other_nmse_arm=min(oc, key=lambda k: oc[k][2]),
                 best_other_ssim=max(v[1] for v in oc.values()),
                 best_other_nmse=min(v[2] for v in oc.values()))
        detail[f"{axis}:{idx}"] = r
    out[c] = dict(mcgs=mcgs, ranked=cand, detailed=[detail[k] for k in detail])

    print(f"\n[{c}] top {a.top} by PSNR margin, per axis (gauge = the figure's):")
    for axis in (0, 1, 2):
        print(f"  --- {ANAT[axis]} ---")
        for r in [x for x in cand if x["axis"] == axis][: a.top]:
            print(f"   idx {r['index']:3d}  head {r['n_head']:6d}  MC-GS {r['ours_psnr']:7.3f} dB  "
                  f"margin PSNR {r['margin_psnr']:+.3f} (vs {r['best_other']})  "
                  f"SSIM {r['margin_ssim']:+.4f} (vs {r['best_other_ssim_arm']})  "
                  f"NMSE {r['margin_nmse']:+.5f} (vs {r['best_other_nmse_arm']})")

# one slice that serves both contrasts: same axis and index, largest smaller-of-the-two PSNR margin
common = {}
for c in out:
    for r in out[c]["ranked"]:
        common.setdefault((r["axis"], r["index"]), {})[c] = r
both = [(min(v["T1"]["margin_psnr"], v["FLAIR"]["margin_psnr"]), k, v) for k, v in common.items() if len(v) == 2]
both.sort(key=lambda t: -t[0])
print("\nbest slice common to BOTH contrasts (ranked by the smaller of the two PSNR margins):")
for m, (axis, idx), v in both[:5]:
    print(f"  {ANAT[axis]} {idx:3d}: min margin {m:+.3f}  "
          f"T1 {v['T1']['margin_psnr']:+.3f} (MC-GS {v['T1']['ours_psnr']:.3f} vs {v['T1']['best_other']}), "
          f"FLAIR {v['FLAIR']['margin_psnr']:+.3f} (MC-GS {v['FLAIR']['ours_psnr']:.3f} vs {v['FLAIR']['best_other']})")
out["common_both_contrasts"] = [dict(axis=k[0], index=k[1], min_margin_psnr=m,
                                     T1=v["T1"], FLAIR=v["FLAIR"]) for m, k, v in both[:20]]
json.dump(out, open(R / f"logs/{a.out}.json", "w"), indent=1)
print("\nwrote", R / f"logs/{a.out}.json")
