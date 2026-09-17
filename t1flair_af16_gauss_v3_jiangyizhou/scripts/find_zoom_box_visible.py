#!/usr/bin/env python3
"""find_zoom_box_visible.py -- zoom boxes for the stacked three-pattern figure chosen for a VISIBLE MC-GS advantage, one contrast at
a time (the T1 and FLAIR figures are separate, so each gets its own box; within a figure one box serves all three pattern rows).

For every transverse slice in --slices, box size W in --sizes and every top-left corner (step 1), with the stacked figure's gauge
restricted to the box (head pixels inside the box; PSNR at the volume's max|GT|; SSIM = the slice's 2-D SSIM map, 7x7 uniform,
averaged over those pixels), over the twelve columns (ZF, nine methods, GS-single, MC-GS):
  eligible  = head fraction >= --min_head, box clear of the panel's top-left text, GT edge content (mean 2-D gradient magnitude of
              the GT inside the box) >= the --edge_q quantile of the eligible boxes of that slice / size (structure to look at),
              MC-GS first on box PSNR AND box SSIM in all three pattern rows (GS-single counts as a competitor);
  ranked by the smallest box-PSNR lead over the best other column across the three rows.
Also reported per pick: the lead over the best non-GS column and over GS-single, and RMSE(MC-GS) / RMSE(best other) per row
(a ratio of 0.8 = 20 % less error, which is where the difference starts to show at the fixed jet [0, 0.1] scale).
Writes logs/find_zoom_box_visible_<contrast>.json. Selection is done on these data; quote whole-slice / whole-volume numbers.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1", "FLAIR"])
ap.add_argument("--slices", type=int, nargs="+", default=[203, 204, 205, 206])
ap.add_argument("--sizes", type=int, nargs="+", default=[32, 40, 48])
ap.add_argument("--min_head", type=float, default=0.95)
ap.add_argument("--edge_q", type=float, default=0.5)
ap.add_argument("--clear", type=int, default=3)
ap.add_argument("--top", type=int, default=6)
ap.add_argument("--rank", choices=["all", "nongs"], default="all",
                help="rank eligible boxes by the smallest lead over every other column (all, default) or over the best non-GS "
                     "column (nongs: the visible gap); MC-GS must be first over every column in both cases")
ap.add_argument("--mcgs_key", default="gs_joint_p3")
ap.add_argument("--single_key", default="gs_single_p3")
args = ap.parse_args()
c = args.contrast
RES = Path("/data0/congcong/code/GS/Results_20260908")
TOPICS = [("vd-Poisson", RES / "T1FLAIR_AF16_v3_jiangyizhou"), ("Gaussian", RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou"),
          ("Pseudo-radial", RES / "T1FLAIR_AF16_radial_v3_jiangyizhou")]
OUT = RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou" / "logs" / f"find_zoom_box_visible_{c}{'' if args.rank == 'all' else '_' + args.rank}.json"
NONGS = {"zf", "cgsense", "cssense_o200", "jcssense", "spirit", "zsssdu", "dip", "inr", "mcinr"}


def arms_for(R):
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


def integral(a):
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    s[1:, 1:] = a.cumsum(0).cumsum(1)
    return s


def boxsum(S, W):                               # all W x W boxes, indexed by top-left corner (J0, K0)
    return S[W:, W:] - S[:-W, W:] - S[W:, :-W] + S[:-W, :-W]


gt_full = np.abs(np.load(TOPICS[0][1] / f"recon/gt_{c}_complex.npy")).astype(np.float32)
gmax = float(gt_full.max())
head_full = gt_full > 0.05 * float(np.quantile(gt_full, 0.999))
armsets = [(name, arms_for(R)) for name, R in TOPICS]
picks_all = []
for t in args.slices:
    g2 = gt_full[t].astype(np.float64); m2 = head_full[t]
    edge = np.hypot(sobel(g2, 0), sobel(g2, 1)) / gmax
    I_head, I_edge = integral(m2.astype(float)), integral(edge * m2)
    rows = []
    for name, arms in armsets:
        se, ss = {}, {}
        for key, path in arms:
            v = np.abs(np.asarray(np.load(path, mmap_mode="r")[t])).astype(np.float64)
            _, smap = structural_similarity(g2, v, data_range=gmax, full=True, win_size=7)
            se[key] = integral((v - g2) ** 2 * m2)
            ss[key] = integral(smap * m2)
        rows.append((name, se, ss))
    H, Wd = g2.shape
    for W in args.sizes:
        n = boxsum(I_head, W)
        frac = n / (W * W)
        J0 = np.arange(H - W + 1)[:, None]; K0 = np.arange(Wd - W + 1)[None, :]
        # display: rows = 255 - J (origin lower, eyes up), cols = K; metric text top-left (cols < 84, rows >= 208), label rows >= 232
        rmax = (H - 1) - J0
        clear = ((K0 >= 84 + args.clear) | (rmax < 208 - args.clear)) & (rmax < 232 - args.clear)
        ok = (frac >= args.min_head) & clear
        ecnt = np.where(n > 0, boxsum(I_edge, W) / np.maximum(n, 1), 0)
        if ok.sum() == 0:
            continue
        ok &= ecnt >= np.quantile(ecnt[ok], args.edge_q)
        nn = np.maximum(n, 1)
        lead_min = np.full(ok.shape, np.inf); lead_s_min = np.full(ok.shape, np.inf); lead_n_min = np.full(ok.shape, np.inf)
        per_row = []
        for name, se, ss in rows:
            ps = {k: 20 * np.log10(gmax) - 10 * np.log10(np.maximum(boxsum(S, W) / nn, 1e-30)) for k, S in se.items()}
            sm = {k: boxsum(S, W) / nn for k, S in ss.items()}
            others = [k for k in ps if k != args.mcgs_key]
            best_o = np.max(np.stack([ps[k] for k in others]), 0)
            best_s = np.max(np.stack([sm[k] for k in others]), 0)
            best_n = np.max(np.stack([ps[k] for k in others if k in NONGS]), 0)
            lead = ps[args.mcgs_key] - best_o
            lead_min = np.minimum(lead_min, lead)
            lead_n_min = np.minimum(lead_n_min, ps[args.mcgs_key] - best_n)
            lead_s_min = np.minimum(lead_s_min, sm[args.mcgs_key] - best_s)
            per_row.append(dict(name=name, ps=ps, lead=lead, lead_n=ps[args.mcgs_key] - best_n,
                                lead_g=ps[args.mcgs_key] - ps[args.single_key]))
        good = ok & (lead_min > 0) & (lead_s_min > 0)
        score = lead_min if args.rank == "all" else lead_n_min
        order = np.argsort(-np.where(good, score, -np.inf), axis=None)
        taken = []
        for idx in order[: 20000]:
            j0, k0 = np.unravel_index(idx, good.shape)
            if not good[j0, k0]:
                break
            if any(abs(j0 - a) < W and abs(k0 - b) < W for a, b in taken):
                continue
            taken.append((j0, k0))
            pick = dict(tra=int(t), W=int(W), J=int(j0 + W // 2), K=int(k0 + W // 2), head_frac=float(frac[j0, k0]),
                        edge=float(ecnt[j0, k0]), min_lead=float(lead_min[j0, k0]), min_lead_ssim=float(lead_s_min[j0, k0]),
                        min_lead_nongs=float(lead_n_min[j0, k0]), score=float(score[j0, k0]),
                        rows=[dict(pattern=r["name"], mcgs=float(r["ps"][args.mcgs_key][j0, k0]),
                                   lead_all=float(r["lead"][j0, k0]), lead_nongs=float(r["lead_n"][j0, k0]),
                                   lead_gs_single=float(r["lead_g"][j0, k0]),
                                   rmse_ratio=float(10 ** (-r["lead"][j0, k0] / 20))) for r in per_row])
            picks_all.append(pick)
            if len(taken) >= args.top:
                break
        print(f"{c} tra {t} W {W}: eligible {int(ok.sum())}, all-first {int(good.sum())}, best min lead "
              f"{(lead_min[good].max() if good.any() else float('nan')):+.3f}, best score ({args.rank}) "
              f"{(score[good].max() if good.any() else float('nan')):+.3f}")
picks_all.sort(key=lambda p: -p["score"])
print(f"\n{c}: best boxes over slices {args.slices} and sizes {args.sizes} (edge content >= q{args.edge_q:g})")
for p in picks_all[:12]:
    print(f"  tra {p['tra']} W {p['W']} centre ({p['J']},{p['K']}) min lead {p['min_lead']:+.3f} dB  min nonGS {p['min_lead_nongs']:+.3f}  SSIM {p['min_lead_ssim']:+.4f}  "
          f"edge {p['edge']:.3f}  | " + "  ".join(f"{r['pattern'][:5]} all {r['lead_all']:+.2f} nonGS {r['lead_nongs']:+.2f} "
                                                   f"GS {r['lead_gs_single']:+.2f} ratio {r['rmse_ratio']:.2f}" for r in p["rows"]))
json.dump(dict(contrast=c, rank=args.rank, slices=args.slices, sizes=args.sizes, min_head=args.min_head, edge_q=args.edge_q,
               picks=picks_all), open(OUT, "w"), indent=1)
print("wrote", OUT)
