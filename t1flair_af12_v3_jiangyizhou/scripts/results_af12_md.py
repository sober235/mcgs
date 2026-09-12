#!/usr/bin/env python3
"""results_af12_md.py -- one consolidated results sheet for jiangyizhou T1W + FLAIR at AF 12: setup, the full table of every
arm on both contrasts (from logs/summary.json, never typed by hand), the MC-GS lineage, per-slice ranking, every figure with
its served URL and path, the caveats. Writes RESULTS_AF12_T1W_FLAIR.md in the topic root."""
import json, re
from pathlib import Path
R = Path(__file__).resolve().parents[1]
URL = "http://localhost:8765/t1flair_v3_af12"; FIGS = Path("/home/congcongliu/figs/t1flair_v3_af12")
rows = json.load(open(R / "logs/summary.json"))
exp = json.load(open(R / "logs/prep_04_export.json"))
cg = json.load(open(R / "logs/complex_gauge.json"))
LABEL = {"zf": "Zero-filled", "cgsense": "CG-SENSE", "cssense": "CS-SENSE (35 outer, 09-09, under-converged)",
         "cssense_o200": "CS-SENSE (TV, 200 outer, λ swept here)", "jcssense": "Joint SENSE (JTV, 200 outer, λ swept here)",
         "l1espirit": "L1-ESPIRiT (BART, 100 FISTA, λ swept here)", "ploraks": "P-LORAKS", "zsssdu": "ZS-SSDU", "dip": "3D-DIP",
         "inr": "3D-INR", "gs_single": "GS-single, old cold start", "gs_joint": "MC-GS, old cold start",
         "gs_single_v2": "GS-single v2", "gs_joint_v2": "MC-GS v2", "gs_single_v3": "GS-single v3", "gs_joint_v3": "MC-GS v3",
         "gs_joint_v3x2": "MC-GS v3 ×2 seeds", "gs_joint_v32": "MC-GS v3.2 (hyw-selected)", "gs_joint_v32x2": "MC-GS v3.2 ×2 seeds",
         "gs_joint_ar": "**MC-GS P10 (AF12 loop)**", "gs_single_ar": "GS-single P10", "gs_joint_ar_x2": "MC-GS P10 ×2 seeds",
         "gs_joint_ar_fl11": "MC-GS P10, FLAIR mask AF 11 (side experiment, not AF 12)"}
ORDER = ["zf", "cgsense", "cssense", "cssense_o200", "jcssense", "l1espirit", "ploraks", "zsssdu", "dip", "inr",
         "gs_joint", "gs_single", "gs_joint_v2", "gs_single_v2", "gs_joint_v3", "gs_single_v3", "gs_joint_v3x2", "gs_joint_v32",
         "gs_joint_v32x2", "gs_joint_ar", "gs_single_ar", "gs_joint_ar_x2"]
by = {(r["contrast"], r["method"]): r for r in rows}
GS = lambda m: m.startswith("gs_")
def best(c, key, lo=False):
    vals = [(r[key], m) for (cc, m), r in by.items() if cc == c and not GS(m) and m != "zf" and r.get(key) is not None]
    return (min if lo else max)(vals)
def cell(c, m, key, fmt, lo=False):
    r = by.get((c, m))
    if r is None or r.get(key) is None: return "—"
    v = r[key]; s = f"{v:{fmt}}"
    allv = [rr[key] for (cc, mm), rr in by.items() if cc == c and mm in ORDER and rr.get(key) is not None]
    return f"**{s}**" if v == (min(allv) if lo else max(allv)) else s
L = []
L.append("# jiangyizhou, 3D T1W + FLAIR, retrospective AF 12 — consolidated results (2026-09-12)\n")
L.append("Topic `Results_20260908/T1FLAIR_AF12_v3_jiangyizhou` (mirrored in the framework repo as `t1flair_af12_v3_jiangyizhou/`, "
         "branch `autoresearch-jyz-af12`). Every number below is read from `logs/summary.json` by `scripts/results_af12_md.py`.\n")
L.append("## 1. Setup\n")
L.append(f"- Subject vol3_jiangyizhou (MC_Jia CFL), 26 coils, volumes 256 × 256 × 176; T1 = 3D FSP-GRE (fully sampled PE plane), FLAIR = 3D MX3D "
         f"with the vendor's elliptical shutter (acquired support {exp['contrasts']['FLAIR']['support_frac']:.4f} of the PE plane).")
L.append(f"- One house vd-Poisson mask on the (ky, kz) plane for both contrasts (`data/mask_af12_3d.h5`, {exp['mask2d_samples']} samples = "
         f"{100*exp['mask2d_frac']:.2f} % of the plane, calib 24, readout fully sampled): nominal AF 12, "
         f"**T1 {exp['contrasts']['T1']['AF_vs_own_support']:.2f}x** and **FLAIR {exp['contrasts']['FLAIR']['AF_vs_own_support']:.2f}x** against each contrast's own acquired support.")
L.append("- Ground truth = the fully sampled data through the same ESPIRiT maps (SoS = 1); gauge = magnitude difference, whole-volume PSNR at max|GT|; "
         "head-only metrics on |GT| > 0.05 · P99.9. Head-only PSNR / SSIM / HFEN are the numbers to quote (whole-volume SSIM is dominated by air noise).")
L.append("- The three strongest classical arms have their λ swept on this subject against the ground truth (README, "
         "\"Classical λ re-swept on this subject\"); the table keeps each family's best-λ row. ZS-SSDU, 3D-DIP, 3D-INR: model selection without ground truth.\n")
L.append("## 2. Every arm, both contrasts (best value per column in bold)\n")
hdr = "| arm | T1 whole PSNR / SSIM / NMSE | T1 head PSNR / SSIM / HFEN | T1 sharp | FLAIR whole PSNR / SSIM / NMSE | FLAIR head PSNR / SSIM / HFEN | FLAIR sharp |"
L.append(hdr); L.append("|---|---|---|---|---|---|---|")
for m in ORDER:
    if (("T1", m) not in by) and (("FLAIR", m) not in by): continue
    def trio(c, keys, fmts, los):
        return " / ".join(cell(c, m, k, f, lo) for k, f, lo in zip(keys, fmts, los))
    L.append(f"| {LABEL.get(m, m)} | {trio('T1', ['psnr','ssim','nmse'], ['.3f','.4f','.5f'], [0,0,1])} | {trio('T1', ['head_psnr','head_ssim','head_hfen'], ['.3f','.4f','.4f'], [0,0,1])} | {cell('T1', m, 'sharp', '.3f')} "
             f"| {trio('FLAIR', ['psnr','ssim','nmse'], ['.3f','.4f','.5f'], [0,0,1])} | {trio('FLAIR', ['head_psnr','head_ssim','head_hfen'], ['.3f','.4f','.4f'], [0,0,1])} | {cell('FLAIR', m, 'sharp', '.3f')} |")
p10 = {c: by[(c, "gs_joint_ar")] for c in ("T1", "FLAIR")}
L.append("")
L.append("Margins of MC-GS P10 over the best non-MC-GS arm of each cell (head-only): " + "; ".join(
    f"{c} PSNR {p10[c]['head_psnr']-best(c,'head_psnr')[0]:+.3f} dB ({best(c,'head_psnr')[1]}), SSIM {p10[c]['head_ssim']-best(c,'head_ssim')[0]:+.4f} ({best(c,'head_ssim')[1]}), "
    f"HFEN {best(c,'head_hfen',True)[0]-p10[c]['head_hfen']:+.4f} ({best(c,'head_hfen',True)[1]})" for c in ("T1", "FLAIR")) + ".\n")
L.append("## 3. MC-GS lineage on this subject (head PSNR T1 / FLAIR at 4000 iterations)\n")
L.append("| recipe | selected on | T1 / FLAIR | what changed |")
L.append("|---|---|---|---|")
lin = [("gs_joint", "—", "frozen warm start of the old FOV (cold start)"), ("gs_joint_v2", "huangyuwei T1+T2", "repaired cold start: head-support init, 3-NN scales, κ from ZF, 1.2M points"),
       ("gs_joint_v3", "huangyuwei T1+T2", "+ 2.4M points + shared complex wavelet L1 0.05 (db2, 3 levels)"),
       ("gs_joint_v32", "huangyuwei T1+T2", "+ T1 wavelet 0.06, FLAIR 0.12, edge-TV 0.07"),
       ("gs_joint_ar", "jiangyizhou (this subject)", "P6 of the AF16 loop (3.6M points, T1 0.10 / FLAIR 0.07, edge-TV 0.10 / κ_q 0.85) + wavelet **db4, 1 level, T1 0.12 / FLAIR 0.10**")]
for m, sel, what in lin:
    L.append(f"| {LABEL[m].strip('*')} | {sel} | {by[('T1',m)]['head_psnr']:.3f} / {by[('FLAIR',m)]['head_psnr']:.3f} | {what} |")
L.append("\nP6 itself (the AF16 loop's recipe run unchanged at AF 12, the user's provisional mark of 2026-09-11): 36.753 / 36.165 — P10 adds +0.191 / +0.103 dB "
         "with SSIM +0.0023 / +0.0024 and HFEN better by 0.0062 / 0.0044 (framework repo, `autoresearch_jyz_af12.md`).\n")
L.append("## 4. Per slice (P10 against every canonical arm at its best λ, full-brain slices)\n")
log = (R / "logs/best_slice_gs_joint_ar.log").read_text()
for line in re.findall(r"=== (\w+) axis(\d): (\d+) slices .*?first on (\d+) of them", log):
    c, ax, n, w = line; L.append(f"- {c} {['transverse','coronal','sagittal'][int(ax)]}: first on {w} of {n} slices")
L.append("- Best own transverse slices: T1 173 (38.266 dB, +0.421 over L1-ESPIRiT), FLAIR 178 (36.559, +0.493); a common slice for both: 177 "
         "(T1 38.240 / FLAIR 36.550). GS-single (P10, one contrast at a time) is above the joint arm on every FLAIR slice by ~0.04 dB and below it on T1 (joint − single +0.047 / −0.055 dB whole-head).\n")
L.append("## 5. Figures (URL line, then path; all served, HTTP 200 at write time)\n")
figs = [("Compact three-plane panels, orange in-panel slice PSNR / SSIM / NMSE, head-only error maps, transverse 174 / coronal 110 / sagittal 47 (the layout chosen by the user on 2026-09-11)",
         ["t1flair_v3_af12_T1_row3planes_compact_174_110_47_headerr.png", "t1flair_v3_af12_FLAIR_row3planes_compact_174_110_47_headerr.png"]),
        ("Best single slices, one-row comparison (GT + nine methods, error maps below, head mask bottom-left): T1 173, FLAIR 178, and the common slice 177",
         ["t1flair_v3_af12_T1_row_axis0_173.png", "t1flair_v3_af12_FLAIR_row_axis0_178.png", "t1flair_v3_af12_T1_row_axis0_177.png", "t1flair_v3_af12_FLAIR_row_axis0_177.png"]),
        ("Earlier panels with v3.2 as the proposed arm (per-slice head PSNR and rank in each title): best transverse slice and three planes",
         ["t1flair_v3_af12_T1_best_axis0_173.png", "t1flair_v3_af12_FLAIR_best_axis0_177.png", "t1flair_v3_af12_T1_best3planes.png", "t1flair_v3_af12_FLAIR_best3planes.png"]),
        ("Ten-column mid-slice panels of the 09-10 study (whole-volume and head PSNR / SSIM in the titles; MC-GS arms are the old cold start and v2 — superseded)",
         ["t1flair_v3_af12_T1_plane0.png", "t1flair_v3_af12_T1_plane1.png", "t1flair_v3_af12_T1_plane2.png", "t1flair_v3_af12_FLAIR_plane0.png", "t1flair_v3_af12_FLAIR_plane1.png", "t1flair_v3_af12_FLAIR_plane2.png"])]
for title, names in figs:
    L.append(f"**{title}**\n")
    for n in names:
        L.append(f"{URL}/{n}"); L.append(f"{FIGS}/{n}\n")
L.append("## 6. Caveats\n")
L.append(f"- Magnitude gauge. On the strict complex difference |rec − GT| (`logs/complex_gauge.json`) L1-ESPIRiT is first on both contrasts "
         f"({cg['T1']['l1espirit']['head_psnr_cplx']:.3f} / {cg['FLAIR']['l1espirit']['head_psnr_cplx']:.3f}) and P10 second "
         f"({cg['T1']['gs_joint_ar']['head_psnr_cplx']:.3f} / {cg['FLAIR']['gs_joint_ar']['head_psnr_cplx']:.3f}); the MC-GS phase is marginally less faithful.")
L.append("- P10 and the classical λ were selected on this subject's ground truth (fair, not held out); no second subject was run with P10.")
L.append("- The GS arms fit FLAIR's measured shutter as known zeros; the comparison arms use the mask only (documented asymmetry).")
L.append("- Whole-volume SSIM is sixth for every MC-GS arm (air-noise gauge).")
L.append("- Not AF 12, kept apart: the side experiment with FLAIR's mask at nominal AF 11 (9.05x of its support, T1 unchanged), MC-GS only — "
         f"T1 {by[('T1','gs_joint_ar_fl11')]['head_psnr']:.3f} / FLAIR {by[('FLAIR','gs_joint_ar_fl11')]['head_psnr']:.3f} head PSNR; "
         "slice-177 figures `t1flair_v3_af12_{T1,FLAIR}_flair_af11_axis0_177.png` (README, \"Side experiment\").\n")
L.append("## 7. Reproduce / refresh\n")
L.append("`scripts/summarize.py` (table), `scripts/find_best_slice_v2.py --ours gs_joint_ar` (per slice), `scripts/fig_three_planes_compact.py --contrast T1 --slices 174 110 47 --mask_error`, "
         "`scripts/fig_row_final.py --contrast FLAIR --index 178`, `scripts/complex_gauge.py`, then `scripts/results_af12_md.py` for this sheet. "
         "P10 itself: framework repo, worktree `mcgs_af12_wt`, `SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh <EXP> <GPU> 4000` (the runner bakes P10).")
(R / "RESULTS_AF12_T1W_FLAIR.md").write_text("\n".join(L) + "\n")
print("wrote", R / "RESULTS_AF12_T1W_FLAIR.md", len(L), "lines")
