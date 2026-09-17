#!/usr/bin/env python3
"""find_zoom_box_all_masks.py -- a W x W zoom box on one transverse slice for the stacked three-pattern figure
(fig_masks_transverse_stack.py --inset W --inset_center J K), the same box on all three sampling patterns and both contrasts.

Gauge = the stacked figure's, restricted to the box: head mask |GT| > 0.05 * P99.9 of the volume; box PSNR over the head pixels
inside the box (peak = max|GT| of the volume); box SSIM = the slice's 2-D SSIM map (7x7 uniform window, data range max|GT|)
averaged over the head pixels inside the box. Arms = the figure's twelve columns (ZF, nine methods at their kept lambdas,
GS-single, MC-GS); "first" = ahead of every other column, GS-single included.
Candidate boxes: W x W, centre (J, K) on a --step grid, head fraction inside the box >= --min_head, and clear (by --clear px)
of the top-right zoom inset and the top-left text of the rendered panel (display = transpose + one quarter turn, eyes up).
Reports (1) the reference box --ref J K (default 124 76 = the in-plane centre of the figure-of-record 48^3 box), and
(2) the boxes where MC-GS is first on box PSNR AND box SSIM in all six rows (3 patterns x 2 contrasts), ranked by the
smallest box-PSNR lead over the best other column. Selection is done on these data; quote whole-slice / whole-volume numbers.
Writes logs/find_zoom_box_all_masks_tra<I>.json.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity

ap = argparse.ArgumentParser()
ap.add_argument("--tra", type=int, default=205)
ap.add_argument("--w", type=int, default=48)
ap.add_argument("--step", type=int, default=2)
ap.add_argument("--min_head", type=float, default=0.95)
ap.add_argument("--clear", type=int, default=3)
ap.add_argument("--ref", type=int, nargs=2, default=[124, 76])
ap.add_argument("--top", type=int, default=10)
ap.add_argument("--mcgs_key", default="gs_joint_p3")
ap.add_argument("--single_key", default="gs_single_p3")
args = ap.parse_args()

RES = Path("/data0/congcong/code/GS/Results_20260908")
TOPICS = [("vd-Poisson", RES / "T1FLAIR_AF16_v3_jiangyizhou"), ("Gaussian", RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou"),
          ("Pseudo-radial", RES / "T1FLAIR_AF16_radial_v3_jiangyizhou")]
OUT = RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou" / "logs" / f"find_zoom_box_all_masks_tra{args.tra}.json"


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


def D(x):                                   # the figure's display transform (transpose + one quarter turn = eyes up)
    return np.rot90(x.T, 1)


def integral(a):
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    s[1:, 1:] = a.cumsum(0).cumsum(1)
    return s


W, hw = args.w, args.w // 2
rowsets = []                                # one per (pattern, contrast)
for c in ("T1", "FLAIR"):
    gt_full = np.abs(np.load(TOPICS[0][1] / f"recon/gt_{c}_complex.npy")).astype(np.float32)
    gmax = float(gt_full.max())
    head_full = gt_full > 0.05 * float(np.quantile(gt_full, 0.999))
    g2 = gt_full[args.tra].astype(np.float64)
    m2 = head_full[args.tra]
    for name, R in TOPICS:
        gchk = np.abs(np.load(R / f"recon/gt_{c}_complex.npy", mmap_mode="r")[args.tra]).astype(np.float64)
        assert np.allclose(gchk, g2, rtol=0, atol=1e-6 * gmax), f"{R.name}: GT slice differs"
        se, ss_ = {}, {}
        for key, path in arms_for(R, c):
            v = np.abs(np.asarray(np.load(path, mmap_mode="r")[args.tra])).astype(np.float64)
            e2 = (v - g2) ** 2 * m2
            _, smap = structural_similarity(g2, v, data_range=gmax, full=True, win_size=7)
            se[key] = integral(e2)
            ss_[key] = integral(smap * m2)
        rowsets.append(dict(pattern=name, contrast=c, gmax=gmax, cnt=integral(m2.astype(float)), se=se, ss=ss_))
    H, Wd = g2.shape                         # volume in-plane (J, K)

# display geometry of one transverse panel (as fig_masks_transverse_stack.py: pw = 2.6 in, ph = pw * 256 / 176)
disp_shape = D(np.zeros((H, Wd))).shape      # (rows, cols) of the displayed array, origin lower (high row = top)
DH, DW = disp_shape
ph_over_pw = DH / DW
inset_col0 = (1 - 0.012 - 0.38) * DW                      # top-right inset: cols >= this
inset_row0 = DH * (1 - (0.012 + 0.38) / ph_over_pw)       # ... and rows >= this
text_col1, text_row0 = 84, 208                             # top-left metric text: cols < 84 and rows >= 208
label_row0 = 232                                           # the pattern label across the top of the GT panel


def disp_box(J, K):
    bm = np.zeros((H, Wd), bool); bm[J - hw:J + hw, K - hw:K + hw] = True
    r, cc = np.nonzero(D(bm))
    return r.min(), r.max() + 1, cc.min(), cc.max() + 1


def clear_of_overlays(J, K):
    r0, r1, c0, c1 = disp_box(J, K)
    cl = args.clear
    in_inset = (c1 > inset_col0 - cl) and (r1 > inset_row0 - cl)
    in_text = (c0 < text_col1 + cl) and (r1 > text_row0 - cl)
    in_label = r1 > label_row0 - cl
    return not (in_inset or in_text or in_label)


def box_stats(J, K):
    a0, a1, b0, b1 = J - hw, J + hw, K - hw, K + hw
    out = []
    for rs in rowsets:
        n = rs["cnt"][a1, b1] - rs["cnt"][a0, b1] - rs["cnt"][a1, b0] + rs["cnt"][a0, b0]
        ps, ss = {}, {}
        for key in rs["se"]:
            I = rs["se"][key]; S = rs["ss"][key]
            e = (I[a1, b1] - I[a0, b1] - I[a1, b0] + I[a0, b0]) / n
            ps[key] = 20 * np.log10(rs["gmax"]) - 10 * np.log10(e)
            ss[key] = (S[a1, b1] - S[a0, b1] - S[a1, b0] + S[a0, b0]) / n
        others_p = {k: v for k, v in ps.items() if k != args.mcgs_key}
        others_s = {k: v for k, v in ss.items() if k != args.mcgs_key}
        bp = max(others_p, key=others_p.get); bs = max(others_s, key=others_s.get)
        out.append(dict(pattern=rs["pattern"], contrast=rs["contrast"], head_frac=n / W / W,
                        mcgs_psnr=ps[args.mcgs_key], best_other_psnr=others_p[bp], best_other_psnr_arm=bp,
                        lead_psnr=ps[args.mcgs_key] - others_p[bp],
                        mcgs_ssim=ss[args.mcgs_key], best_other_ssim=others_s[bs], best_other_ssim_arm=bs,
                        lead_ssim=ss[args.mcgs_key] - others_s[bs]))
    return out


def summarize(J, K):
    st = box_stats(J, K)
    return dict(J=J, K=K, disp_box_rows_cols=[int(x) for x in disp_box(J, K)], clear=clear_of_overlays(J, K),
                min_head_frac=min(s["head_frac"] for s in st),
                all_first_psnr=all(s["lead_psnr"] > 0 for s in st), all_first_ssim=all(s["lead_ssim"] > 0 for s in st),
                min_lead_psnr=min(s["lead_psnr"] for s in st), min_lead_ssim=min(s["lead_ssim"] for s in st), rows=st)


ref = summarize(*args.ref)
cands = []
for J in range(hw, H - hw + 1, args.step):
    for K in range(hw, Wd - hw + 1, args.step):
        cnt0 = rowsets[0]["cnt"]
        n = cnt0[J + hw, K + hw] - cnt0[J - hw, K + hw] - cnt0[J + hw, K - hw] + cnt0[J - hw, K - hw]
        if n / W / W < args.min_head or not clear_of_overlays(J, K):
            continue
        s = summarize(J, K)
        if s["all_first_psnr"] and s["all_first_ssim"]:
            cands.append(s)
cands.sort(key=lambda s: -s["min_lead_psnr"])
picked = []
for s in cands:                               # list well-separated candidates only
    if all(abs(s["J"] - p["J"]) + abs(s["K"] - p["K"]) >= W // 2 for p in picked):
        picked.append(s)
    if len(picked) >= args.top:
        break


def show(tag, s):
    print(f"{tag} centre (J {s['J']}, K {s['K']})  display rows {s['disp_box_rows_cols'][:2]} cols {s['disp_box_rows_cols'][2:]}  "
          f"clear={s['clear']}  head {s['min_head_frac']:.2f}  all-first PSNR {s['all_first_psnr']} SSIM {s['all_first_ssim']}  "
          f"min lead {s['min_lead_psnr']:+.3f} dB / {s['min_lead_ssim']:+.4f}")
    for r in s["rows"]:
        print(f"    {r['contrast']:5s} {r['pattern']:13s} MC-GS {r['mcgs_psnr']:.2f} dB vs {r['best_other_psnr_arm']:13s} "
              f"{r['best_other_psnr']:.2f} ({r['lead_psnr']:+.3f})   SSIM {r['mcgs_ssim']:.4f} vs {r['best_other_ssim_arm']:13s} "
              f"{r['best_other_ssim']:.4f} ({r['lead_ssim']:+.4f})")


print(f"transverse {args.tra}, {W}x{W} box, display {DH}x{DW}; inset zone cols >= {inset_col0:.0f} & rows >= {inset_row0:.0f}")
show("REFERENCE", ref)
print(f"\n{len(cands)} candidate centres with MC-GS first on box PSNR and SSIM in all six rows; best separated ones:")
for i, s in enumerate(picked):
    show(f"#{i + 1}", s)
json.dump(dict(tra=args.tra, w=W, step=args.step, min_head=args.min_head, reference=ref, n_candidates=len(cands),
               picked=picked), open(OUT, "w"), indent=1, default=float)
print("wrote", OUT)
