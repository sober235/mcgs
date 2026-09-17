#!/usr/bin/env python3
"""eval_zoom_box.py -- how MC-GS stands inside given zoom boxes on one transverse slice, both contrasts, three patterns.
  eval_zoom_box.py --tra 205 --box 40 160 131 --box 32 148 146 ...      (W J K per box; J, K = box centre, volume axes 1 and 2)
Gauge = the stacked figure's, restricted to the box (head pixels; PSNR at the volume's max|GT|; SSIM map 7x7 averaged there);
columns = ZF, nine methods, GS-single, MC-GS. Prints, per contrast and pattern: MC-GS box PSNR, lead over the best other column,
over the best non-GS column and over GS-single, the SSIM lead, and whether MC-GS is first on both.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity

ap = argparse.ArgumentParser()
ap.add_argument("--tra", type=int, default=205)
ap.add_argument("--box", type=int, nargs=3, action="append", required=True, metavar=("W", "J", "K"))
ap.add_argument("--mcgs_key", default="gs_joint_p3")
ap.add_argument("--single_key", default="gs_single_p3")
args = ap.parse_args()
RES = Path("/data0/congcong/code/GS/Results_20260908")
TOPICS = [("vd-Poisson", RES / "T1FLAIR_AF16_v3_jiangyizhou"), ("Gaussian", RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou"),
          ("Pseudo-radial", RES / "T1FLAIR_AF16_radial_v3_jiangyizhou")]
NONGS = {"zf", "cgsense", "cssense_o200", "jcssense", "spirit", "zsssdu", "dip", "inr", "mcinr"}


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
            ("cssense_o200", lam_file("cssense_o200", "cssense_o200")), ("jcssense", lam_file("jcssense", "jcssense_o200")),
            ("spirit", R / f"recon/spirit_ks{r['ksize']}_lam{r['lam']:g}_{c}_complex.npy"),
            ("zsssdu", R / f"recon/zsssdu_{c}_complex.npy"), ("dip", R / f"recon/dip_{c}_complex.npy"),
            ("inr", R / f"recon/inr_{c}_complex.npy"), ("mcinr", R / f"recon/mcinr_{c}_complex.npy"),
            (args.single_key, R / f"recon/{args.single_key}_{c}_mag.npy"), (args.mcgs_key, R / f"recon/{args.mcgs_key}_{c}_mag.npy")]


t = args.tra
data = {}
for c in ("T1", "FLAIR"):
    gt = np.abs(np.load(TOPICS[0][1] / f"recon/gt_{c}_complex.npy")).astype(np.float32)
    gmax = float(gt.max()); head = gt > 0.05 * float(np.quantile(gt, 0.999))
    g2 = gt[t].astype(np.float64); m2 = head[t]
    for name, R in TOPICS:
        vs, sm = {}, {}
        for key, path in arms_for(R, c):
            v = np.abs(np.asarray(np.load(path, mmap_mode="r")[t])).astype(np.float64)
            vs[key] = v
            sm[key] = structural_similarity(g2, v, data_range=gmax, full=True, win_size=7)[1]
        data[(c, name)] = (g2, m2, gmax, vs, sm)
for W, J, K in args.box:
    h = W // 2
    print(f"== transverse {t}, box {W} centre ({J},{K})")
    for c in ("T1", "FLAIR"):
        worst_all = worst_n = 9e9; allfirst = True
        for name, _ in TOPICS:
            g2, m2, gmax, vs, sm = data[(c, name)]
            cut = (slice(J - h, J + h), slice(K - h, K + h)); m = m2[cut]; g = g2[cut][m]
            ps = {k: 20 * np.log10(gmax) - 10 * np.log10(((v[cut][m] - g) ** 2).mean()) for k, v in vs.items()}
            ss = {k: s[cut][m].mean() for k, s in sm.items()}
            mc = ps[args.mcgs_key]; oth = {k: v for k, v in ps.items() if k != args.mcgs_key}
            bo = max(oth, key=oth.get); bn = max((k for k in oth if k in NONGS), key=oth.get)
            so = max(v for k, v in ss.items() if k != args.mcgs_key)
            first = mc > oth[bo] and ss[args.mcgs_key] > so
            allfirst &= first; worst_all = min(worst_all, mc - oth[bo]); worst_n = min(worst_n, mc - oth[bn])
            print(f"   {c:5s} {name:13s} MC-GS {mc:6.2f}  vs best {bo:13s} {mc - oth[bo]:+.3f}  vs non-GS {bn:13s} {mc - oth[bn]:+.3f}  "
                  f"vs GS-single {mc - oth[args.single_key]:+.3f}  SSIM lead {ss[args.mcgs_key] - so:+.4f}  first {first}")
        print(f"   {c}: all first {allfirst}, weakest lead over all {worst_all:+.3f}, over non-GS {worst_n:+.3f}")
