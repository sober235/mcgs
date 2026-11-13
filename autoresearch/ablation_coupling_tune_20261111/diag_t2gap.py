#!/usr/bin/env python3
"""diag_t2gap.py -- read-only: where does the joint model lose to the independent models on tangbing T1+T2?

Compares the saved magnitude volumes of fw_t1t2tbv32_full (joint, V32) with fw_t1t2tb_nojoint_T1 / _T2
(independent, V32) against the same ground truth. For each contrast, inside a head mask taken from the ground
truth (> 10 % of its 99.9th percentile; a diagnostic region, never a training input):
  * squared-error energy of joint and independent, split into a low band (Gaussian sigma 2 voxels) and the
    high-band remainder;
  * the same split by region: edges of both contrasts, edges of this contrast only, edges of the other contrast
    only, flat (edge = top 15 % of the ground-truth gradient magnitude inside the head);
  * bias: mean of (recon - GT) and the fraction of high-band error that is correlated with the GT detail
    (blur) versus uncorrelated (noise / artefact).
Prints JSON; writes it to MCGS_ARTIFACT_DIR when run under gpu.py --cpu.
"""
import json
import os

import numpy as np
from scipy import ndimage as ndi

R = "/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_framework_ablation_20261110/trainer/results_arc"
J = np.load(f"{R}/fw_t1t2tbv32_full_volumes.npz")
I = {"T1": np.load(f"{R}/fw_t1t2tb_nojoint_T1_volumes.npz"), "T2": np.load(f"{R}/fw_t1t2tb_nojoint_T2_volumes.npz")}
KEY = {"T1": "T1", "T2": "FL"}            # the joint npz stores the second contrast under "FL"


def grad_mag(v):
    return np.sqrt(sum(g ** 2 for g in np.gradient(v)))


def bands(e):
    lo = ndi.gaussian_filter(e, 2.0)
    return lo, e - lo


out = {}
gt = {c: J[f"gt{KEY[c]}"].astype(np.float64) for c in ("T1", "T2")}
head = (gt["T1"] > 0.1 * np.quantile(gt["T1"], 0.999)) | (gt["T2"] > 0.1 * np.quantile(gt["T2"], 0.999))
head = ndi.binary_fill_holes(ndi.binary_closing(head, iterations=2))
gm = {c: grad_mag(gt[c]) for c in gt}
edge = {c: gm[c] > np.quantile(gm[c][head], 0.85) for c in gt}
for c, o in (("T1", "T2"), ("T2", "T1")):
    g = gt[c]
    # independent single-contrast npz: keys rec / gt
    ind = I[c]["rec"].astype(np.float64)
    assert np.allclose(I[c]["gt"], g, rtol=1e-5, atol=1e-6 * g.max()), "ground truths differ"
    jo = J[f"rec{KEY[c]}"].astype(np.float64)
    regions = {"both_edges": head & edge[c] & edge[o], "own_edges_only": head & edge[c] & ~edge[o],
               "other_edges_only": head & ~edge[c] & edge[o], "flat": head & ~edge[c] & ~edge[o]}
    detail = g - ndi.gaussian_filter(g, 2.0)
    r = {"head_voxels": int(head.sum())}
    for name, rec in (("joint", jo), ("indep", ind)):
        e = rec - g
        lo, hi = bands(e)
        hm = head
        tot = float((e[hm] ** 2).sum())
        # high-band error regressed on GT detail: negative slope = blur, the remainder = uncorrelated
        d = detail[hm]
        beta = float((hi[hm] * d).sum() / (d * d).sum())
        resid = hi[hm] - beta * d
        r[name] = {"sse": tot, "lo_frac": float((lo[hm] ** 2).sum() / tot), "hi_frac": float((hi[hm] ** 2).sum() / tot),
                   "bias_mean_over_gtmean": float(e[hm].mean() / g[hm].mean()),
                   "detail_slope": beta, "hi_uncorrelated_sse": float((resid ** 2).sum()),
                   "region_sse": {k: float((e[m] ** 2).sum()) for k, m in regions.items()}}
    j, i = r["joint"], r["indep"]
    r["joint_minus_indep_sse_frac_of_indep"] = {
        "total": (j["sse"] - i["sse"]) / i["sse"],
        "low_band": (j["lo_frac"] * j["sse"] - i["lo_frac"] * i["sse"]) / i["sse"],
        "high_band": (j["hi_frac"] * j["sse"] - i["hi_frac"] * i["sse"]) / i["sse"],
        **{f"region_{k}": (j["region_sse"][k] - i["region_sse"][k]) / i["sse"] for k in regions}}
    r["region_voxel_frac"] = {k: float(m.sum() / head.sum()) for k, m in regions.items()}
    out[c] = r
    print(c, json.dumps(r["joint_minus_indep_sse_frac_of_indep"]), flush=True)
    print(c, "detail slope joint/indep", round(j["detail_slope"], 4), round(i["detail_slope"], 4),
          "| uncorrelated hi sse joint/indep", round(j["hi_uncorrelated_sse"] / i["hi_uncorrelated_sse"], 4),
          "| bias joint/indep", round(j["bias_mean_over_gtmean"], 5), round(i["bias_mean_over_gtmean"], 5), flush=True)
art = os.environ.get("MCGS_ARTIFACT_DIR")
if art:
    json.dump(out, open(os.path.join(art, "diag_t2gap.json"), "w"), indent=2)
print("DIAG DONE " + json.dumps(out))
