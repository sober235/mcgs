#!/usr/bin/env python3
"""Print the numbers a topic's v3 README section needs: joint and single v3 against every other arm, per metric."""
import json, re, sys
from pathlib import Path
R = Path("/data0/congcong/code/GS/Results_20260908")
ARC = Path("/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_arc")
topic = sys.argv[1]; T = R / topic
OURS = sys.argv[2] if len(sys.argv) > 2 else "gs_joint_v32"
JOINT = ({"T1T2_AF12_retro": "ar19_n24_T1p06_geo007", "T1FLAIR_AF12_retro": "ar19_tf_hyw_full_C012", "T1FLAIR_AF12_v3_jiangyizhou": "ar19_tf_jyz_full_C012"}
         if OURS == "gs_joint_v32" else
         {"T1T2_AF12_retro": "ar12_n24_wav005_full4k", "T1FLAIR_AF12_retro": "ar13_tf_hyw_t1flair_v3b", "T1FLAIR_AF12_v3_jiangyizhou": "ar13_tf_jyz_t1flair_v3b"})[topic]
COMP = ["cgsense", "cssense_o200", "jcssense", "l1espirit", "ploraks", "zsssdu", "dip", "inr"]
LAB = {"cgsense": "CG-SENSE", "cssense_o200": "CS-SENSE (TV)", "jcssense": "joint CS-SENSE", "l1espirit": "L1-ESPIRiT",
       "ploraks": "P-LORAKS", "zsssdu": "ZS-SSDU", "dip": "3D-DIP", "inr": "3D-INR", "gs_joint_v3": "joint v3", "gs_joint_v32": "joint v3.2", "gs_single_v3": "single v3"}
rows = {(r["method"], r["contrast"]): r for r in json.load(open(T / "logs/summary.json"))}
cs = sorted({c for _, c in rows}, key=lambda c: (c != "T1", c))
METRICS = [("head_psnr", True, 3), ("head_ssim", True, 4), ("head_hfen", False, 4), ("psnr", True, 3), ("nmse", False, 5)]
for c in cs:
    print(f"== {c}   competitors present: {[m for m in COMP if (m, c) in rows]}")
    for ours in (OURS, "gs_single_v3"):
        if (ours, c) not in rows:
            print(f"   {ours}: MISSING"); continue
        out = []
        for k, hi, nd in METRICS:
            v = rows[(ours, c)][k]
            comp = {m: rows[(m, c)][k] for m in COMP if (m, c) in rows}
            best = (max if hi else min)(comp, key=comp.get)
            rank = 1 + sum((x > v) if hi else (x < v) for x in comp.values())
            out.append(f"{k}={v:.{nd}f} r{rank} vs {LAB[best]} {comp[best]:.{nd}f} ({v - comp[best]:+.{nd}f})")
        print(f"   {LAB[ours]:9s} " + " | ".join(out))
    j, s, v2, l1 = (rows.get((m, c)) for m in (OURS, "gs_single_v3", "gs_joint_v2", "l1espirit"))
    if j and s: print(f"   joint - single: head PSNR {j['head_psnr'] - s['head_psnr']:+.3f}, head SSIM {j['head_ssim'] - s['head_ssim']:+.4f}, HFEN {j['head_hfen'] - s['head_hfen']:+.4f}")
    if j and v2: print(f"   joint v3 - joint v2: head PSNR {j['head_psnr'] - v2['head_psnr']:+.3f}; sharp v3 {j['sharp']:.3f} / v2 {v2['sharp']:.3f} / L1 {l1['sharp']:.3f}; grain ratio v3 {j['grain_ratio']:.3f} / v2 {v2['grain_ratio']:.3f}")
log = (T / "logs/best_slice_v2.log").read_text()
for m in re.finditer(r"=== (\S+) axis(\d): (\d+) slices .*?; (\S+) first on (\d+) of them", log):
    print(f"   sweep {m.group(1)} axis{m.group(2)}: {m.group(4)} first on {m.group(5)}/{m.group(3)}")
for line in log.splitlines():
    if "margin taken against" in line: print("  ", line.strip())
b = json.load(open(T / "logs/best_slice_v2.json"))
for c in cs:
    if c in b:
        print(f"   best slices {c} (axis 0,1,2): " + ", ".join(str(max((r for r in b[c] if r['axis'] == ax), key=lambda r: r['ours'])['index']) for ax in range(3)))
sj = json.load(open(ARC / f"{JOINT}_score.json")); a = sj.get("argv") or []
print("   base_cfg:", a[a.index("--base_cfg") + 1] if "--base_cfg" in a else "?", "| elapsed min:", round(sj.get("elapsed_s", 0) / 60))
rp = ARC / f"{JOINT}_s43_score.json"
if rp.exists():
    r = json.load(open(rp)); print(f"   seed-43 repeat: {r['plain']['head_psnr']:.3f} / {r['contrast']['head_psnr']:.3f}")
