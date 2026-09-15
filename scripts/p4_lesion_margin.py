#!/usr/bin/env python3
"""Stage-4 target (user 2026-09-14): one slice containing a lesion on which MC-GS beats every other arm by >= 1 dB.

Per-slice PSNR exactly as the topic's row figure (fig_row_final.py): head = |GT| > 0.05 * P99.9 of the whole volume,
magnitude error over the head pixels of the slice, peak = max |GT| of the volume. Lesion slices (T2*w, where the lesions are visible), see SLICES below:
  subependymal nodules, vendor slices 38 / 46 = SPE 53 / 45, centres read off the GT zoom -> transverse +-1 partition,
    coronal / sagittal slices +-2 voxels through each centre
  calcified-looking focus at RO 166, PE 196, SPE 68..70 -> transverse 68..70, coronal 163..169, sagittal 193..199
Competitors: the table's other arms at their table lambda and EVERY GS-single volume available (stage 2, tuned, and
any results_arc single run of the same contrast), so the bar on a slice is the best any of them reaches there.

  $PY scripts/p4_lesion_margin.py --exp p4_edge0020 [--exp ...] [--contrast T2s]
Writes results_arc/<exp>_lesion.json and prints the best lesion slice of each run.
"""
import argparse, json, re
from pathlib import Path
import numpy as np

R = Path("/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro")
ARC = Path(__file__).resolve().parents[1] / "results_arc"
# nodule centres read off the GT T2s zoom (2026-09-14 21:05): SPE 53 at (RO, PE) (138, 140), (147, 139), (181, 128);
# SPE 45 at (148, 172), (166, 172). Calcified focus at (166, 196), SPE 68..70. In-plane slices are taken +-2 voxels
# around each centre, transverse ones +-1 partition.
SLICES = {"transverse": (2, [44, 45, 46, 52, 53, 54, 68, 69, 70]),
          "coronal": (0, sorted(set(range(136, 141)) | set(range(145, 151)) | set(range(163, 170)) | set(range(179, 184)))),
          "sagittal": (1, sorted(set(range(126, 131)) | set(range(137, 143)) | set(range(170, 175)) | set(range(193, 200))))}
ap = argparse.ArgumentParser()
ap.add_argument("--exp", action="append", default=[])
ap.add_argument("--contrast", default="T2s", choices=["T2s", "T1w"])
a = ap.parse_args()
c = a.contrast
rows = json.load(open(R / "logs/summary.json"))


def lam_file(key, prefix):
    lam = [r for r in rows if r["method"] == key and r["contrast"] == c][0]["lam"]
    for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
        m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
        if m and abs(float(m.group(1)) - float(lam)) < 1e-12:
            return p
    raise FileNotFoundError(key)


gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float64)
head = gt > 0.05 * float(np.quantile(gt, 0.999))
gmax = float(gt.max())


def load(src):
    if isinstance(src, tuple):
        with np.load(src[0]) as z:
            return np.abs(z[src[1]]).astype(np.float64)
    return np.abs(np.load(src)).astype(np.float64)


def per_slice(v):
    out = {}
    for plane, (ax, idxs) in SLICES.items():
        for i in idxs:
            sl = [slice(None)] * 3; sl[ax] = i; sl = tuple(sl)
            m = head[sl]
            e = v[sl][m] - gt[sl][m]
            out[f"{plane} {i}"] = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
    return out


comp = {"CG-SENSE": R / f"recon/cgsense_{c}_complex.npy", "CS-SENSE": lam_file("cssense_o200", "cssense_o200"),
        "Joint SENSE": lam_file("jcssense", "jcssense_o200"), "L1-ESPIRiT": lam_file("l1espirit", "l1espirit"),
        "ZS-SSDU": R / f"recon/zsssdu_{c}_complex.npy", "3D-DIP": R / f"recon/dip_{c}_complex.npy",
        "3D-INR": R / f"recon/inr_{c}_complex.npy", "MC-INR": R / f"recon/mcinr_{c}_complex.npy",
        "GS-single": R / f"recon/gs_single_ar_{c}_complex.npy"}
for sp in sorted(R.glob(f"recon/gs_single_*_{c}_complex.npy")):          # every registered GS-single arm of this contrast
    tag = sp.name[len("gs_single_"):-len(f"_{c}_complex.npy")]
    if tag != "ar":
        comp[f"GS-single ({tag})"] = sp
for sj in ARC.glob(f"*single_{c}*_volumes.npz"):
    comp[f"GS-single run {sj.name[:-len('_volumes.npz')]}"] = (sj, "recT1_complex")
cache = ARC / f"lesion_competitors_{c}.json"
key = sorted(str(v) for v in comp.values()) + [json.dumps(SLICES)]
if cache.exists() and json.load(open(cache)).get("key") == key:
    comp_p = json.load(open(cache))["psnr"]
else:
    comp_p = {k: per_slice(load(p)) for k, p in comp.items()}
    json.dump({"key": key, "psnr": comp_p}, open(cache, "w"), indent=1)
slices = list(next(iter(comp_p.values())).keys())
bar = {s: max(comp_p[k][s] for k in comp_p) for s in slices}
who = {s: max(comp_p, key=lambda k: comp_p[k][s]) for s in slices}
ch = "recFL_complex" if c == "T2s" else "recT1_complex"
for exp in a.exp:
    p = per_slice(load((ARC / f"{exp}_volumes.npz", ch)))
    marg = {s: p[s] - bar[s] for s in slices}
    best = max(marg, key=marg.get)
    non_gs = {s: p[s] - max(comp_p[k][s] for k in comp_p if not k.startswith("GS-single")) for s in slices}
    json.dump(dict(exp=exp, contrast=c, psnr=p, bar=bar, bar_arm=who, margin=marg, margin_over_non_gs=non_gs,
                   best_slice=best, best_margin=marg[best]), open(ARC / f"{exp}_lesion.json", "w"), indent=1)
    top = sorted(marg.items(), key=lambda kv: -kv[1])[:3]
    print(f"{exp} [{c}]: best lesion slice {best}: {p[best]:.3f} dB vs {who[best]} {bar[best]:.3f} -> margin {marg[best]:+.3f} "
          f"(over non-GS arms {non_gs[best]:+.3f}); next: " + ", ".join(f"{s} {m:+.3f}" for s, m in top[1:]))
