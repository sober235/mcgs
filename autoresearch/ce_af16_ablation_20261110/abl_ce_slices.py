#!/usr/bin/env python3
"""abl_ce_slices.py -- slice-level check of the September CE AF16 result against the genuinely independent
single-contrast GS of the framework ablation (CPU, read-only on the topic).

  gpu.py run --cpu -- @python autoresearch/ce_af16_ablation_20261110/abl_ce_slices.py

Question. The September topic (Results_20260908/CE_retro_cssense_af16) reports that MC-GS is first on 602/604 (pre)
and 604/604 (post) scored slices and wins all six panels of planes 201/131/48, with GS-single as the runner-up.
Its registered GS-single feeds the same scan into both channels of the joint model. The framework ablation trained a
genuinely independent single-contrast model per scan (fw_ce_nojoint_{pre,post}); on whole-head PSNR it is not below
the joint model. Do the slice-level claims survive when that model takes the GS-single place?

What it does.
  1. Puts the ablation volumes into the topic's gauge exactly as the topic's extract_gs_volumes.py does:
     x_gauge = x_trainer * scale_trainer / scale_gauge  (scale_trainer from the run log, scale_gauge from the topic's
     logs/gs_joint_<scan>.json). Checks: the same arithmetic applied to the September joint npz reproduces the
     registered recon/gs_joint_<scan>_complex.npy; the ablation runs' zero-filled volumes equal the September run's.
  2. Whole-head PSNR / NMSE in the topic gauge (head = |GT| > 0.05 P99.9, peak = max|GT|).
  3. Per-slice ranking with the topic's find_best_slice.py definition (three axes, slices with >= 2000 head voxels,
     head PSNR of the slice, eleven arms): first with the registered arms (must reproduce 602/604 and 604/604), then
     with the independent single model in the GS-single place.
  4. Lesion planes (>= 150 px of the lesion core, as the vdPois evaluator's plane list): PSNR / SSIM / NMSE per plane
     with the plane_metrics arithmetic of fig_lesion_plane_prepost.py; strict wins of MC-GS over every other arm, with
     either single arm. Planes 201 / 131 / 48 are reported in full.

Writes converted volumes, ce_slices.json and ce_slices.md into OUT (new directory under the ablation root); the json
and md also go to MCGS_ARTIFACT_DIR. Nothing is written into the topic. GT and ROI are used for scoring only.
"""
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

G = Path("/data0/congcong/code/GS")
TOPIC = G / "Results_20260908/CE_retro_cssense_af16"
A = G / "3DGS/outputs/codex_artifacts"
ABL = A / "mcgs_framework_ablation_20261110/trainer/results_arc"
SEPT_ARC = TOPIC / "trainer/_trainer_e778bf2/results_arc"
SUPFIX = A / "mcgs_ce_sept_cmp_20261109/recon"
ROI = G / "Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz"
OUT = A / "mcgs_framework_ablation_20261110/ce_slices"
SCANS = ("pre", "post")
THREE = {0: 201, 1: 131, 2: 48}
AXN = {0: "transverse", 1: "coronal", 2: "sagittal"}
MIN_HEAD, MIN_CORE = 2000, 150
PAT = re.compile(r"Normalized MRI intensity scale by P99\.9 magnitude = ([0-9.eE+-]+)")
ORDER = ["zf", "cgsense", "cssense_o200", "jcssense", "spirit", "zsssdu", "dip", "inr", "mcinr", "gs_single", "gs_joint"]
LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense_o200": "CS-SENSE", "jcssense": "Joint SENSE", "spirit": "L1-SPIRiT",
         "zsssdu": "ZS-SSDU", "dip": "3D-DIP", "inr": "3D-INR", "mcinr": "MC-INR", "gs_single": "GS-single",
         "gs_joint": "MC-GS", "gs_indep": "GS-single (independent)", "gs_rerun": "MC-GS (ablation rerun)",
         "dip_sup": "3D-DIP", "inr_sup": "3D-INR"}
log = lambda *a: print(*a, flush=True)


def arm_files(scan):
    """the topic's scripts/arms.py resolution"""
    sel = {r["method"]: r for r in json.load(open(TOPIC / "logs/summary.json")) if r["contrast"] == scan}
    out = {}
    for key in ORDER:
        if key == "cssense_o200":
            p = TOPIC / f"recon/cssense_o200_lam{sel[key]['lam']:g}_{scan}_complex.npy"
        elif key == "jcssense":
            p = TOPIC / f"recon/jcssense_o200_lam{sel[key]['lam']:g}_{scan}_complex.npy"
        elif key == "spirit":
            p = TOPIC / f"recon/spirit_ks{sel[key]['ksize']}_lam{sel[key]['lam']:g}_{scan}_complex.npy"
        else:
            p = TOPIC / f"recon/{key}_{scan}_complex.npy"
        out[key] = p
    return out


def scales(logfile):
    return [float(v) for v in PAT.findall(open(logfile).read())]


def plane_metrics(rec, truth, mask, gm):
    """fig_lesion_plane_prepost.plane_metrics (= the evaluator's arithmetic)"""
    t, v = truth.astype(np.float64), rec.astype(np.float64)
    _, smap = structural_similarity(v, t, data_range=gm, full=True)
    return {"psnr": float(20 * np.log10(gm) - 10 * np.log10(np.mean((v[mask] - t[mask]) ** 2))),
            "ssim": float(smap[mask].mean()),
            "nmse": float(np.sum((v[mask] - t[mask]) ** 2) / np.sum(t[mask] ** 2))}


def better(m, a, b):
    return a < b if m == "nmse" else a > b


OUT.mkdir(parents=True, exist_ok=True)
res = {"topic": str(TOPIC), "ablation": str(ABL), "checks": {}, "whole_head": {}, "per_slice": {}, "lesion_planes": {}}

# ---------------------------------------------------------------- 1. conversion into the topic gauge
gauge = {s: json.load(open(TOPIC / f"logs/gs_joint_{s}.json")) for s in SCANS}
sc_sept = scales(SEPT_ARC / "ce_retro_af16_joint.log")
sc_full = scales(ABL / "fw_ce_full.log")
assert len(sc_sept) >= 2 and sc_sept[:2] == sc_full[:2], (sc_sept, sc_full)
conv = {}
with np.load(SEPT_ARC / "ce_retro_af16_joint_volumes.npz") as zs, np.load(ABL / "fw_ce_full_volumes.npz") as zf:
    for i, (s, ch) in enumerate((("pre", "T1"), ("post", "FL"))):
        assert abs(gauge[s]["scale_trainer"] - sc_sept[i]) < 1e-9
        factor = sc_full[i] / gauge[s]["scale_gauge"]
        reg = np.load(TOPIC / f"recon/gs_joint_{s}_complex.npy")
        mine = (np.asarray(zs[f"rec{ch}_complex"]) * np.float32(factor)).astype(np.complex64)
        d_self = float(np.abs(mine - reg).max())
        zf_rel = float(np.abs(zf[f"zf{ch}_complex"] - zs[f"zf{ch}_complex"]).max() / np.abs(zs[f"zf{ch}_complex"]).max())
        assert d_self < 1e-6 and zf_rel < 1e-5, (s, d_self, zf_rel)
        conv[("gs_rerun", s)] = (np.asarray(zf[f"rec{ch}_complex"]) * np.float32(factor)).astype(np.complex64)
        zf_sept = np.asarray(zs[f"zf{ch}_complex"])
        with np.load(ABL / f"fw_ce_nojoint_{s}_volumes.npz") as zn:
            assert str(zn["contrast_name"]) == s, (str(zn["contrast_name"]), s)
            sc_n = scales(ABL / f"fw_ce_nojoint_{s}.log")
            assert sc_n and abs(sc_n[0] - sc_full[i]) < 1e-9, (sc_n, sc_full[i])
            zf_rel_n = float(np.abs(zn["zf_complex"] - zf_sept).max() / np.abs(zf_sept).max())
            assert zf_rel_n < 1e-5, (s, zf_rel_n)
            conv[("gs_indep", s)] = (np.asarray(zn["rec_complex"]) * np.float32(factor)).astype(np.complex64)
        res["checks"][s] = {"factor": factor, "september_npz_reproduces_registered_max_abs_diff": d_self,
                            "zf_full_vs_september_max_rel_diff": zf_rel, "zf_nojoint_vs_september_max_rel_diff": zf_rel_n}
        log(f"[gauge {s}] factor {factor:.6f}; Sept npz -> registered max|diff| {d_self:.2e}; "
            f"ZF full {zf_rel:.1e}, nojoint {zf_rel_n:.1e}")
for (key, s), x in conv.items():
    np.save(OUT / f"{key}_{s}_complex.npy", x)

# ---------------------------------------------------------------- volumes
gt, gmax, head, vol = {}, {}, {}, {}
for s in SCANS:
    gt[s] = np.abs(np.load(TOPIC / f"recon/gt_{s}_complex.npy")).astype(np.float32)
    gmax[s] = float(gt[s].max())
    head[s] = gt[s] > 0.05 * float(np.quantile(gt[s], 0.999))
    vol[s] = {k: np.abs(np.load(p)).astype(np.float32) for k, p in arm_files(s).items()}
    vol[s]["gs_indep"] = np.abs(conv[("gs_indep", s)]).astype(np.float32)
    vol[s]["gs_rerun"] = np.abs(conv[("gs_rerun", s)]).astype(np.float32)
    vol[s]["dip_sup"] = np.abs(np.load(SUPFIX / f"dip_sup_b96_tv0.03_{s}_complex.npy")).astype(np.float32)
    vol[s]["inr_sup"] = np.abs(np.load(SUPFIX / f"inr_i10_n512_sup6_tv0.03_{s}_complex.npy")).astype(np.float32)
with np.load(ROI, allow_pickle=False) as z:
    names = [str(n) for n in z["names"]]
    core = z[f"roi{names.index('病灶核心')}"] > 0
assert core.shape == gt["pre"].shape

# ---------------------------------------------------------------- 2. whole head
for s in SCANS:
    h, g = head[s], gt[s].astype(np.float64)
    res["whole_head"][s] = {}
    for k in ("gs_joint", "gs_rerun", "gs_single", "gs_indep", "zsssdu", "spirit"):
        e = vol[s][k].astype(np.float64)[h] - g[h]
        res["whole_head"][s][k] = {"psnr": float(20 * np.log10(gmax[s]) - 10 * np.log10(np.mean(e * e))),
                                   "nmse": float(np.sum(e * e) / np.sum(g[h] ** 2))}
    w = res["whole_head"][s]
    log(f"[whole head {s}] " + "  ".join(f"{LABEL[k]} {w[k]['psnr']:.3f}" for k in w))

# ---------------------------------------------------------------- 3. per-slice ranking (find_best_slice.py definition)
for s in SCANS:
    rows = []
    for ax in range(3):
        for i in range(gt[s].shape[ax]):
            sl = [slice(None)] * 3; sl[ax] = i; sl = tuple(sl)
            h = head[s][sl]
            if int(h.sum()) < MIN_HEAD:
                continue
            g = gt[s][sl][h].astype(np.float64)
            ps = {}
            for k in ORDER + ["gs_indep"]:
                e = vol[s][k][sl][h].astype(np.float64) - g
                ps[k] = float(20 * np.log10(gmax[s]) - 10 * np.log10((e * e).mean()))
            rows.append({"axis": ax, "index": i, "psnr": ps})
    out = {"scored_slices": len(rows)}
    for tag, others in (("registered", [k for k in ORDER if k != "gs_joint"]),
                        ("independent_single", [k for k in ORDER if k not in ("gs_joint", "gs_single")] + ["gs_indep"])):
        first = [r for r in rows if r["psnr"]["gs_joint"] > max(r["psnr"][k] for k in others)]
        lost = [r for r in rows if r not in first]
        winners = {}
        for r in lost:
            b = max(others, key=lambda k: r["psnr"][k])
            winners[b] = winners.get(b, 0) + 1
        out[tag] = {"mcgs_first": len(first),
                    "mcgs_first_by_axis": {AXN[a]: [sum(r["axis"] == a for r in first), sum(r["axis"] == a for r in rows)]
                                           for a in range(3)},
                    "winner_where_mcgs_is_not_first": winners}
    for tag, k in (("joint_minus_registered_single", "gs_single"), ("joint_minus_independent_single", "gs_indep")):
        d = np.array([r["psnr"]["gs_joint"] - r["psnr"][k] for r in rows])
        out[tag] = {"median": float(np.median(d)), "min": float(d.min()), "max": float(d.max()),
                    "positive": int((d > 0).sum()),
                    "by_axis_median": {AXN[a]: float(np.median([x for x, r in zip(d, rows) if r["axis"] == a]))
                                       for a in range(3)},
                    "by_axis_positive": {AXN[a]: [int(sum(x > 0 for x, r in zip(d, rows) if r["axis"] == a)),
                                                  sum(r["axis"] == a for r in rows)] for a in range(3)}}
    res["per_slice"][s] = out
    log(f"[per slice {s}] {len(rows)} slices; MC-GS first: registered {out['registered']['mcgs_first']}, "
        f"independent single {out['independent_single']['mcgs_first']}")

# ---------------------------------------------------------------- 4. lesion planes
FIELD = ["zf", "cgsense", "cssense_o200", "jcssense", "spirit", "zsssdu", "dip_sup", "inr_sup", "mcinr"]   # as the figure of record
planes = [(ax, i) for ax in range(3) for i in range(core.shape[ax]) if int(core.take(i, axis=ax).sum()) >= MIN_CORE]
LP = []
for ax, i in planes:
    sl = [slice(None)] * 3; sl[ax] = i; sl = tuple(sl)
    num = {s: {k: plane_metrics(vol[s][k][sl], gt[s][sl], head[s][sl], gmax[s])
               for k in FIELD + ["gs_single", "gs_indep", "gs_joint"]} for s in SCANS}
    rec = {"axis": ax, "index": i, "core": int(core[sl].sum()), "numbers": num}
    for tag, single in (("registered", "gs_single"), ("independent_single", "gs_indep")):
        wins, detail = 0, {}
        for s in SCANS:
            for m in ("psnr", "ssim", "nmse"):
                opp = {k: num[s][k][m] for k in FIELD + [single]}
                bk = (min if m == "nmse" else max)(opp, key=opp.get)
                win = better(m, num[s]["gs_joint"][m], opp[bk])
                wins += win
                detail[f"{s}:{m}"] = {"mcgs": num[s]["gs_joint"][m], "best_other": bk, "best_other_value": opp[bk], "win": bool(win)}
        rec[tag] = {"wins_of_6": int(wins), "detail": detail}
    LP.append(rec)
res["lesion_planes"] = {
    "n_planes": {AXN[a]: sum(p["axis"] == a for p in LP) for a in range(3)},
    "all_six_won": {tag: {AXN[a]: [p["index"] for p in LP if p["axis"] == a and p[tag]["wins_of_6"] == 6] for a in range(3)}
                    for tag in ("registered", "independent_single")},
    "three_planes": [p for p in LP if THREE[p["axis"]] == p["index"]],
    "planes": [{k: v for k, v in p.items() if k != "numbers"} for p in LP]}
for tag in ("registered", "independent_single"):
    a6 = res["lesion_planes"]["all_six_won"][tag]
    log(f"[lesion planes, {tag}] all six won: " + ", ".join(f"{k} {len(v)}/{res['lesion_planes']['n_planes'][k]}" for k, v in a6.items()))

# ---------------------------------------------------------------- report
L = ["# CE AF16 (September data): slice-level check with the independent single-contrast model", ""]
L += ["## Gauge checks", "", "| scan | factor | Sept npz -> registered, max abs diff | ZF full vs Sept | ZF nojoint vs Sept |", "|---|---|---|---|---|"]
for s in SCANS:
    c = res["checks"][s]
    L.append(f"| {s} | {c['factor']:.6f} | {c['september_npz_reproduces_registered_max_abs_diff']:.2e} | "
             f"{c['zf_full_vs_september_max_rel_diff']:.1e} | {c['zf_nojoint_vs_september_max_rel_diff']:.1e} |")
L += ["", "## Whole head, topic gauge (PSNR dB / NMSE)", "", "| arm | pre | post |", "|---|---|---|"]
for k in ("gs_joint", "gs_rerun", "gs_single", "gs_indep", "zsssdu", "spirit"):
    L.append(f"| {LABEL[k]} | " + " | ".join(f"{res['whole_head'][s][k]['psnr']:.3f} / {res['whole_head'][s][k]['nmse']:.5f}" for s in SCANS) + " |")
L += ["", "## Per-slice head PSNR: slices on which MC-GS is first among eleven arms", "",
      "| scan | scored | GS-single = registered | GS-single = independent | by axis, independent (tra / cor / sag) | winner elsewhere |", "|---|---|---|---|---|---|"]
for s in SCANS:
    o = res["per_slice"][s]
    ba = o["independent_single"]["mcgs_first_by_axis"]
    L.append(f"| {s} | {o['scored_slices']} | {o['registered']['mcgs_first']} | {o['independent_single']['mcgs_first']} | "
             + " / ".join(f"{ba[AXN[a]][0]}/{ba[AXN[a]][1]}" for a in range(3)) + " | "
             + ", ".join(f"{LABEL[k]} {v}" for k, v in o["independent_single"]["winner_where_mcgs_is_not_first"].items()) + " |")
L += ["", "| scan | MC-GS minus | median dB | min | max | slices with MC-GS higher | by axis (tra / cor / sag) |", "|---|---|---|---|---|---|---|"]
for s in SCANS:
    for tag, nm in (("joint_minus_registered_single", "registered single"), ("joint_minus_independent_single", "independent single")):
        d = res["per_slice"][s][tag]
        L.append(f"| {s} | {nm} | {d['median']:+.3f} | {d['min']:+.3f} | {d['max']:+.3f} | {d['positive']}/{res['per_slice'][s]['scored_slices']} | "
                 + " / ".join(f"{d['by_axis_positive'][AXN[a]][0]}/{d['by_axis_positive'][AXN[a]][1]}" for a in range(3)) + " |")
L += ["", "## Lesion planes: planes on which MC-GS wins all six (pre/post x PSNR/SSIM/NMSE)", "",
      "| GS-single arm | transverse | coronal | sagittal |", "|---|---|---|---|"]
for tag in ("registered", "independent_single"):
    a6, n = res["lesion_planes"]["all_six_won"][tag], res["lesion_planes"]["n_planes"]
    L.append(f"| {tag} | " + " | ".join(f"{len(a6[AXN[a]])}/{n[AXN[a]]}" for a in range(3)) + " |")
L += ["", "## Planes 201 / 131 / 48 with the independent single model", "",
      "| plane | scan | metric | MC-GS | independent single | registered single | best other arm | MC-GS first |", "|---|---|---|---|---|---|---|---|"]
for p in res["lesion_planes"]["three_planes"]:
    for s in SCANS:
        for m in ("psnr", "ssim", "nmse"):
            n, d = p["numbers"][s], p["independent_single"]["detail"][f"{s}:{m}"]
            others = {k: n[k][m] for k in FIELD}
            bo = (min if m == "nmse" else max)(others, key=others.get)
            f = "{:.3f}" if m == "psnr" else "{:.5f}"
            L.append(f"| {AXN[p['axis']]} {p['index']} | {s} | {m} | {f.format(n['gs_joint'][m])} | {f.format(n['gs_indep'][m])} | "
                     f"{f.format(n['gs_single'][m])} | {LABEL.get(bo, bo)} {f.format(others[bo])} | {'yes' if d['win'] else 'NO'} |")
md = "\n".join(L) + "\n"
(OUT / "ce_slices.md").write_text(md, encoding="utf-8")
json.dump(res, open(OUT / "ce_slices.json", "w"), indent=1, ensure_ascii=False)
art = os.environ.get("MCGS_ARTIFACT_DIR")
if art:
    Path(art).mkdir(parents=True, exist_ok=True)
    for n in ("ce_slices.md", "ce_slices.json"):
        shutil.copyfile(OUT / n, Path(art) / n)
print(md)
