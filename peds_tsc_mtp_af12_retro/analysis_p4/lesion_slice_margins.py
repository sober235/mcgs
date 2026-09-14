"""Stage 4, revised target (user 2026-09-14 21:xx): ONE slice that contains a lesion, on which MC-GS beats every other
arm by at least 1 dB. Per-slice head PSNR exactly as the topic's slice figures: head = |GT| > 0.05 * P99.9 of the volume,
restricted to SPE 12..91; PSNR at the volume max |GT|; magnitude error.

Lesion slices (from the 2026-09-14 lesion check, analysis_2026-09-14_mcgs_best_slice_lesions):
  calcified-looking focus at (RO 166, PE 196, SPE 68..70)  -> transverse 68..70, coronal RO 163..169, sagittal PE 193..199
  subependymal nodules on vendor slices 38 / 46             -> transverse SPE 52..54 and 44..46
Competitors = the ten-arm table's arms at their table lambda, plus EVERY GS-single volume available (stage 2, tuned);
candidates = every joint MC-GS volume available. Output: per candidate, the best lesion slice and its margin."""
import json, re, sys
from pathlib import Path
import numpy as np

R = Path("/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro")
ARC = Path("/data0/congcong/code/GS/mcgs_peds_wt/results_arc")
rows = json.load(open(R / "logs/summary.json"))

def lam_file(c, key, prefix):
    lam = [r for r in rows if r["method"] == key and r["contrast"] == c][0]["lam"]
    for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
        m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
        if m and abs(float(m.group(1)) - float(lam)) < 1e-12:
            return p
    raise FileNotFoundError((c, key))

SLICES = {"transverse": (2, [44, 45, 46, 52, 53, 54, 68, 69, 70]),
          "coronal": (0, list(range(163, 170))),
          "sagittal": (1, list(range(193, 200)))}
out = {}
for c, ch in (("T2s", "recFL_complex"), ("T1w", "recT1_complex")):
    gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float64)
    head = gt > 0.05 * np.quantile(gt, 0.999); head[:, :, :12] = False; head[:, :, 92:] = False
    gmax = float(gt.max())
    comp = {"CG-SENSE": R / f"recon/cgsense_{c}_complex.npy", "CS-SENSE": lam_file(c, "cssense_o200", "cssense_o200"),
            "Joint SENSE": lam_file(c, "jcssense", "jcssense_o200"), "L1-ESPIRiT": lam_file(c, "l1espirit", "l1espirit"),
            "ZS-SSDU": R / f"recon/zsssdu_{c}_complex.npy", "3D-DIP": R / f"recon/dip_{c}_complex.npy",
            "3D-INR": R / f"recon/inr_{c}_complex.npy", "MC-INR": R / f"recon/mcinr_{c}_complex.npy",
            "GS-single (stage 2)": R / f"recon/gs_single_ar_{c}_complex.npy"}
    if c == "T1w":
        comp["GS-single (tuned p3)"] = R / "recon/gs_single_p3_T1w_complex.npy"
        comp["GS-single p3_single_T1w_c1"] = (ARC / "p3_single_T1w_c1_volumes.npz", "recT1_complex")
    cand = {"gs_joint_ar": R / f"recon/gs_joint_ar_{c}_complex.npy", "gs_joint_p3": R / f"recon/gs_joint_p3_{c}_complex.npy"}
    for e in ("p3_c2_wav006_gp0015", "p3_edge0010", "p3_geo_003", "p3_geo_006", "p3_i6000", "p3_n60", "p3_wavT1_003", "p3_wavT1_006"):
        cand[e] = (ARC / f"{e}_volumes.npz", ch)

    def load(src):
        if isinstance(src, tuple):
            with np.load(src[0]) as z:
                return np.abs(z[src[1]]).astype(np.float64)
        return np.abs(np.load(src)).astype(np.float64)

    def slice_psnr(v):
        res = {}
        for plane, (ax, idxs) in SLICES.items():
            for i in idxs:
                sl = [slice(None)] * 3; sl[ax] = i; sl = tuple(sl)
                m = head[sl]
                if m.sum() < 50:
                    continue
                e = v[sl][m] - gt[sl][m]
                res[f"{plane} {i}"] = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
        return res

    comp_p = {k: slice_psnr(load(p)) for k, p in comp.items()}
    cand_p = {k: slice_psnr(load(p)) for k, p in cand.items()}
    keys = list(next(iter(comp_p.values())).keys())
    bar = {s: max(comp_p[k][s] for k in comp_p) for s in keys}
    who = {s: max(comp_p, key=lambda k: comp_p[k][s]) for s in keys}
    out[c] = dict(bar=bar, bar_arm=who, candidates=cand_p, competitors=comp_p)
    print(f"\n== {c}: margin of each MC-GS candidate over the best other arm, per lesion slice")
    print(f"{'slice':16s} {'bar':>7s} {'bar arm':22s} " + " ".join(f"{k[:11]:>11s}" for k in cand_p))
    for s in keys:
        print(f"{s:16s} {bar[s]:7.2f} {who[s]:22s} " + " ".join(f"{cand_p[k][s] - bar[s]:+11.3f}" for k in cand_p))
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
