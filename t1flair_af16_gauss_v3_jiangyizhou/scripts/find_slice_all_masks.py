#!/usr/bin/env python3
"""find_slice_all_masks.py -- transverse slices on which MC-GS is first on ALL THREE sampling patterns of the diff-mask study.

Gauge = the stacked figure's (fig_masks_transverse_stack.py): head mask |GT| > 0.05 * P99.9 of the volume; per slice and arm,
head-masked PSNR (peak = max|GT| of the volume), SSIM (2-D map, 7x7 uniform window, data range max|GT|, averaged over the
head pixels) and NMSE. Arms = the twelve columns of that figure (ZF, nine methods at their kept lambdas, GS-single, MC-GS);
"first" means ahead of every other column, GS-single included (user 2026-09-13). Eligible slices: head area >= --min_area
of the largest slice in both contrasts (best_slice_mcgs.py's rule).
Reports, per contrast and for both together: the slices where MC-GS is first on PSNR + SSIM + NMSE on all three patterns
(and, separately, on PSNR alone), ranked by the smallest PSNR margin over the best other arm across the patterns (and
contrasts); candidates listed at least --gap slices apart. Writes logs/find_slice_all_masks.json.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity

ap = argparse.ArgumentParser()
ap.add_argument("--min_area", type=float, default=0.5)
ap.add_argument("--gap", type=int, default=6)
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--mcgs_key", default="gs_joint_p3")
ap.add_argument("--single_key", default="gs_single_p3")
args = ap.parse_args()

RES = Path("/data0/congcong/code/GS/Results_20260908")
TOPICS = [("vdpois", RES / "T1FLAIR_AF16_v3_jiangyizhou"), ("gauss", RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou"),
          ("radial", RES / "T1FLAIR_AF16_radial_v3_jiangyizhou")]
OUT = RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou" / "logs" / "find_slice_all_masks.json"


def arms_for(R, c):
    rows = {r["method"]: r for r in json.load(open(R / "logs/summary.json")) if r["contrast"] == c}

    def lam_file(row_key, prefix):
        lam = rows.get(row_key, {}).get("lam")
        for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
            m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
            if m and lam is not None and abs(float(m.group(1)) - float(lam)) < 1e-12:
                return p
        raise FileNotFoundError(f"{R.name} {row_key} {c}")

    r = rows["spirit"]
    return [("zf", R / f"recon/zf_{c}_complex.npy"), ("cgsense", R / f"recon/cgsense_{c}_complex.npy"),
            ("cssense", lam_file("cssense_o200", "cssense_o200")), ("jcssense", lam_file("jcssense", "jcssense_o200")),
            ("spirit", R / f"recon/spirit_ks{r['ksize']}_lam{r['lam']:g}_{c}_complex.npy"),
            ("zsssdu", R / f"recon/zsssdu_{c}_complex.npy"), ("dip", R / f"recon/dip_{c}_complex.npy"),
            ("inr", R / f"recon/inr_{c}_complex.npy"), ("mcinr", R / f"recon/mcinr_{c}_complex.npy"),
            ("gs_single", R / f"recon/{args.single_key}_{c}_mag.npy"), ("mcgs", R / f"recon/{args.mcgs_key}_{c}_mag.npy")]


# per contrast: eligible slices from the GT (same GT in all topics); per topic: metrics[metric][arm] -> array over slices
res = {}
elig = {}
for c in ("T1", "FLAIR"):
    gt = np.abs(np.load(TOPICS[0][1] / f"recon/gt_{c}_complex.npy")).astype(np.float32)
    gmax = float(gt.max())
    head = gt > 0.05 * float(np.quantile(gt, 0.999))
    area = head.reshape(head.shape[0], -1).sum(1)
    elig[c] = area >= args.min_area * area.max()
    res[c] = {}
    for name, R in TOPICS:
        g_t = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
        assert abs(float(g_t.max()) - gmax) <= 1e-6 * gmax, f"{R.name}: GT differs"
        arms = arms_for(R, c)
        met = {m: {} for m in ("psnr", "ssim", "nmse")}
        for key, path in arms:
            v = np.abs(np.load(path)).astype(np.float32)
            ps = np.full(gt.shape[0], np.nan); ss = np.full(gt.shape[0], np.nan); nm = np.full(gt.shape[0], np.nan)
            for i in np.flatnonzero(elig[c]):
                m2 = head[i]; g2 = gt[i].astype(np.float64); r2 = v[i].astype(np.float64)
                e = r2[m2] - g2[m2]
                ps[i] = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
                _, smap = structural_similarity(g2, r2, data_range=gmax, full=True, win_size=7)
                ss[i] = float(smap[m2].mean())
                nm[i] = float((e * e).sum() / (g2[m2] ** 2).sum())
            met["psnr"][key], met["ssim"][key], met["nmse"][key] = ps, ss, nm
        res[c][name] = met
        print(f"{c} {name}: scored {int(elig[c].sum())} eligible slices x {len(arms)} arms", flush=True)

n = len(elig["T1"])
both_elig = elig["T1"] & elig["FLAIR"]


def first_and_margin(met, i):
    """(first on psnr, ssim, nmse) flags and the PSNR margin of MC-GS over the best other arm on slice i"""
    others = [k for k in met["psnr"] if k != "mcgs"]
    p_m = met["psnr"]["mcgs"][i]; p_o = max(met["psnr"][k][i] for k in others)
    s_m = met["ssim"]["mcgs"][i]; s_o = max(met["ssim"][k][i] for k in others)
    n_m = met["nmse"]["mcgs"][i]; n_o = min(met["nmse"][k][i] for k in others)
    return (p_m > p_o, s_m > s_o, n_m < n_o), p_m - p_o


table = {}
for i in range(n):
    if not both_elig[i]:
        continue
    row = {}
    for c in ("T1", "FLAIR"):
        for name, _ in TOPICS:
            flags, mg = first_and_margin(res[c][name], i)
            row[f"{c}_{name}"] = dict(psnr=bool(flags[0]), ssim=bool(flags[1]), nmse=bool(flags[2]), all3=bool(all(flags)), margin=float(mg))
    table[i] = row


def report(label, cond, key="all3"):
    hits = [i for i, row in table.items() if all(row[k][key] for k in cond)]
    if not hits:
        print(f"{label}: NONE")
        return []
    score = {i: min(table[i][k]["margin"] for k in cond) for i in hits}
    order = sorted(hits, key=lambda i: -score[i])
    keep = []
    for i in order:
        if all(abs(i - k) >= args.gap for k in keep):
            keep.append(i)
        if len(keep) >= args.top:
            break
    runs = []
    start = prev = hits[0]
    for i in hits[1:]:
        if i != prev + 1:
            runs.append((start, prev)); start = i
        prev = i
    runs.append((start, prev))
    print(f"{label}: {len(hits)} slices  ranges {runs}")
    for i in keep:
        det = "  ".join(f"{k}:{table[i][k]['margin']:+.3f}" for k in cond)
        print(f"    slice {i:3d}  min margin {score[i]:+.3f} dB   {det}")
    return hits


COND = {"T1": [f"T1_{t}" for t, _ in TOPICS], "FLAIR": [f"FLAIR_{t}" for t, _ in TOPICS]}
COND["both"] = COND["T1"] + COND["FLAIR"]
out = {}
for key, lab in (("all3", "first on PSNR + SSIM + NMSE"), ("psnr", "first on PSNR only")):
    for cname, cond in COND.items():
        print(f"\n== {lab}, all three patterns, {cname} ==")
        out[f"{key}_{cname}"] = report(f"{cname}", cond, key)
print(f"\neligible slices (head area >= {args.min_area:g} of max, both contrasts): {int(both_elig.sum())} of {n}, "
      f"{np.flatnonzero(both_elig).min()}..{np.flatnonzero(both_elig).max()}")
OUT.parent.mkdir(exist_ok=True)
json.dump(dict(min_area=args.min_area, eligible=[int(i) for i in np.flatnonzero(both_elig)], hits=out,
               table={str(i): r for i, r in table.items()}), open(OUT, "w"), indent=1)
print("wrote", OUT)
