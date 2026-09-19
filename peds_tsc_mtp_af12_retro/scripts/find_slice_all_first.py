#!/usr/bin/env python3
"""find_slice_all_first.py -- every slice on which MC-GS leads EVERY comparison arm on the figure's three panel
metrics (user 2026-09-16: "find one slice, MC-GS 全部超过对比方法").

Per-slice gauge = the compact / inset figures': head = |GT| > 0.05 * P99.9 of the volume, PSNR at the volume's
max |GT| over the head pixels of the slice, SSIM = 7x7 uniform-window map averaged over the head pixels, NMSE over
the head pixels (NMSE ranks exactly as PSNR within one slice, so the search is PSNR-then-SSIM). Comparison arms =
the counted table rows at their table point (L1 = L1-SPIRiT since 2026-09-15; --l1 espirit for the old column),
EVERY registered GS-single volume of the contrast and EVERY results_arc single run of the contrast (the stage-4
rule: the bar on a slice is the best any single reaches there). Earlier MC-GS versions are the same method and
are not competitors.

  ./scripts/py scripts/find_slice_all_first.py --contrast T2s [--key gs_joint_p3f] [--min_head 0.15]

Writes logs/find_slice_all_first_<c>[_espirit].json (every considered slice, PSNR of every arm; SSIM where MC-GS
leads PSNR). Selecting a slice on the reported metric is illustration, not evidence: the counts are printed too.
"""
import argparse, json, re
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

R = Path(__file__).resolve().parents[1]
ARC = Path("/data0/congcong/code/GS/mcgs_peds_wt/results_arc")
PLANES = [(2, "transverse"), (0, "coronal"), (1, "sagittal")]

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1w", "T2s"])
ap.add_argument("--key", default="gs_joint_p3f", help="the MC-GS volume under test (recon/<key>_<c>_mag.npy)")
ap.add_argument("--l1", choices=["spirit", "espirit"], default="spirit")
ap.add_argument("--min_head", type=float, default=0.15, help="skip slices whose head fraction of the slice is below this")
ap.add_argument("--no_arc_singles", action="store_true", help="do not add the results_arc single runs as competitors")
args = ap.parse_args()
c = args.contrast

summary = {r["method"]: r for r in json.load(open(R / "logs/summary.json")) if r["contrast"] == c}


def lam_file(row_key, prefix):
    lam = summary.get(row_key, {}).get("lam")
    for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
        m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
        if lam is not None and m and abs(float(m.group(1)) - float(lam)) < 1e-12:
            return p
    raise FileNotFoundError(f"{row_key} {c} lam={lam}")


comp = {"cgsense": R / f"recon/cgsense_{c}_complex.npy",
        "cssense_o200": lam_file("cssense_o200", "cssense_o200"),
        "jcssense": lam_file("jcssense", "jcssense_o200")}
if args.l1 == "spirit":
    comp["spirit"] = R / f"recon/{summary['spirit']['src']}_{c}_complex.npy"
else:
    comp["l1espirit"] = lam_file("l1espirit", "l1espirit")
comp.update({"zsssdu": R / f"recon/zsssdu_{c}_complex.npy", "dip": R / f"recon/dip_{c}_complex.npy",
             "inr": R / f"recon/inr_{c}_complex.npy", "mcinr": R / f"recon/mcinr_{c}_complex.npy"})
for p in sorted(R.glob(f"recon/gs_single_*_{c}_mag.npy")):          # every registered GS-single arm of this contrast
    comp[p.name[:-len(f"_{c}_mag.npy")]] = p
if not args.no_arc_singles:
    for p in sorted(ARC.glob(f"*single_{c}*_volumes.npz")):          # every single run of this contrast in the worktree
        comp[f"run:{p.name[:-len('_volumes.npz')]}"] = (p, "recT1")   # a single run's only contrast is its plain channel
test = R / f"recon/{args.key}_{c}_mag.npy"
for k, p in comp.items():
    assert (p[0] if isinstance(p, tuple) else p).exists(), f"missing {k}: {p}"
assert test.exists(), test


def load(src):
    if isinstance(src, tuple):
        with np.load(src[0]) as z:
            return np.abs(z[src[1]]).astype(np.float32)
    return np.abs(np.load(src)).astype(np.float32)


gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
gmax = float(gt.max())
head = gt > 0.05 * float(np.quantile(gt, 0.999))
vols = {k: load(p) for k, p in comp.items()}
vols[args.key] = load(test)
print(f"{c}: MC-GS = {args.key}; {len(comp)} comparison arms: {', '.join(comp)}", flush=True)

out, n_cons, n_ps, n_all = [], 0, 0, 0
for axis, pname in PLANES:
    areas = head.sum(axis=tuple(a for a in range(3) if a != axis))
    amax = float(areas.max())
    for idx in range(gt.shape[axis]):
        sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
        m2 = head[sl]
        if m2.mean() < args.min_head:
            continue
        n_cons += 1
        g64 = gt[sl].astype(np.float64); g = g64[m2]
        mse = {k: float(((v[sl].astype(np.float64)[m2] - g) ** 2).mean()) for k, v in vols.items()}
        ps = {k: 20 * np.log10(gmax) - 10 * np.log10(m) for k, m in mse.items()}
        rec = dict(axis=axis, plane=pname, idx=idx, head_frac=float(m2.mean()), area_rel=float(m2.sum() / amax),
                   psnr=ps, ssim=None, wins_psnr=False, wins_all=False)
        others = {k: v for k, v in ps.items() if k != args.key}
        runner = max(others, key=others.get)
        rec["d_psnr"], rec["runner_psnr"] = ps[args.key] - others[runner], runner
        if rec["d_psnr"] > 0:
            rec["wins_psnr"] = True; n_ps += 1
            ss = {}
            for k, v in vols.items():
                _, smap = structural_similarity(g64, v[sl].astype(np.float64), data_range=gmax, full=True, win_size=7)
                ss[k] = float(smap[m2].mean())
            rec["ssim"] = ss
            o = {k: v for k, v in ss.items() if k != args.key}
            rs = max(o, key=o.get)
            rec["d_ssim"], rec["runner_ssim"] = ss[args.key] - o[rs], rs
            if rec["d_ssim"] > 0:
                rec["wins_all"] = True; n_all += 1
        out.append(rec)

tag = "" if args.l1 == "spirit" else "_espirit"
p = R / f"logs/find_slice_all_first_{c}{tag}.json"
json.dump(dict(contrast=c, key=args.key, l1=args.l1, competitors=list(comp), min_head=args.min_head,
               n_considered=n_cons, n_wins_psnr=n_ps, n_wins_all=n_all, slices=out), open(p, "w"), indent=1)
print(f"{n_cons} slices considered (head fraction >= {args.min_head:.0%}); MC-GS first on PSNR/NMSE on {n_ps}, "
      f"first on PSNR, SSIM and NMSE on {n_all}")
for axis, pname in PLANES:
    w = [r for r in out if r["axis"] == axis and r["wins_all"]]
    n = sum(1 for r in out if r["axis"] == axis)
    print(f"  {pname:11s}: {len(w):3d} of {n:3d} slices; large sections (area >= 90 % of the plane's largest): "
          f"{sum(1 for r in w if r['area_rel'] >= 0.9)}")
wins = sorted([r for r in out if r["wins_all"]], key=lambda r: -r["d_psnr"])
print("\ntop 12 by PSNR margin over the best other arm:")
for r in wins[:12]:
    print(f"  {r['plane']:11s} {r['idx']:3d}  area {r['area_rel']:.2f}  PSNR {r['psnr'][args.key]:7.3f} (+{r['d_psnr']:.3f} over "
          f"{r['runner_psnr']})  SSIM {r['ssim'][args.key]:.4f} (+{r['d_ssim']:.4f} over {r['runner_ssim']})")
print("wrote", p)
