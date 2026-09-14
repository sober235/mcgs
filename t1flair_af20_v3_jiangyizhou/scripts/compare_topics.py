#!/usr/bin/env python3
"""compare_topics.py -- jiangyizhou at AF16 against AF20, every arm, one gauge.

The two topics share the subject, the k-space, the ESPIRiT maps, the ground truth and the scoring
module, and the MC-GS recipe is identical; only the mask differs (2966 -> 2427 PE points, T1 15.19x
-> 18.56x, FLAIR 12.49x -> 15.13x against its own shutter). So every AF20 - AF16 difference below
is the price of the extra acceleration for that arm.

usage: compare_topics.py            (both topics' logs/summary.json must exist)
"""
import json
from pathlib import Path

ROOT = Path("/data0/congcong/code/GS/Results_20260908")
A16, A20 = "T1FLAIR_AF16_v3_jiangyizhou", "T1FLAIR_AF20_v3_jiangyizhou"
ORDER = [("zf", "ZF"), ("cgsense", "CG-SENSE"), ("cssense", "CS-SENSE"),
         ("cssense_o200", "CS-SENSE (TV, converged)"), ("jcssense", "joint CS-SENSE (JTV)"),
         ("l1espirit", "L1-ESPIRiT"), ("zsssdu", "ZS-SSDU"), ("dip", "3D-DIP"), ("inr", "3D-INR"), ("mcinr", "MC-INR (Niessen 2025)"),
         ("gs_single_p3", "GS-single"), ("gs_joint_p3", "MC-GS")]
COMPETING = {"zf", "cgsense", "cssense", "cssense_o200", "jcssense", "l1espirit", "zsssdu", "dip", "inr", "mcinr"}


def load(topic):
    p = ROOT / topic / "logs/summary.json"
    return json.load(open(p)) if p.exists() else []


def row(rows, c, key, label):
    return next((r for r in rows if r["contrast"] == c and r["method"] in (key, label)), None)


def main():
    r16, r20 = load(A16), load(A20)
    for c in ("T1", "FLAIR"):
        print(f"\n=== jiangyizhou {c}: head-only PSNR and SSIM   (AF16 | AF20 | AF20-AF16)")
        print(f"{'method':26s} {'hPSNR16':>8s} {'hPSNR20':>8s} {'dPSNR':>7s}   {'hSSIM16':>8s} {'hSSIM20':>8s} {'dSSIM':>8s}   {'sharp20':>7s}")
        for key, label in ORDER:
            a, b = row(r16, c, key, label), row(r20, c, key, label)
            if not b:
                print(f"{label:26s} {'--':>8s}"); continue
            if a:
                print(f"{label:26s} {a['head_psnr']:8.3f} {b['head_psnr']:8.3f} {b['head_psnr']-a['head_psnr']:+7.3f}   "
                      f"{a['head_ssim']:8.4f} {b['head_ssim']:8.4f} {b['head_ssim']-a['head_ssim']:+8.4f}   {b['sharp']:7.3f}")
            else:
                print(f"{label:26s} {'--':>8s} {b['head_psnr']:8.3f} {'':7s}   {'--':>8s} {b['head_ssim']:8.4f} {'':8s}   {b['sharp']:7.3f}")

    print("\n=== MC-GS margin over the strongest COMPETING arm (ours excluded), head-only PSNR")
    for tag, rows in (("AF16", r16), ("AF20", r20)):
        for c in ("T1", "FLAIR"):
            mine = row(rows, c, "gs_joint_p3", "MC-GS")
            comp = [(r["head_psnr"], r["method"]) for r in rows if r["contrast"] == c and r["method"] in COMPETING]
            if mine and comp:
                h, m = max(comp)
                print(f"  {tag} {c:5s}  MC-GS {mine['head_psnr']:.3f}  best other {h:.3f} ({m})  margin {mine['head_psnr']-h:+.3f}")

    print("\n=== MC-GS minus GS-single (the coupling), head-only PSNR / SSIM")
    for tag, rows in (("AF16", r16), ("AF20", r20)):
        for c in ("T1", "FLAIR"):
            j, s = row(rows, c, "gs_joint_p3", "MC-GS"), row(rows, c, "gs_single_p3", "GS-single")
            if j and s:
                print(f"  {tag} {c:5s}  {j['head_psnr']-s['head_psnr']:+.3f} dB  {1000*(j['head_ssim']-s['head_ssim']):+.1f} mSSIM")

    print("\n=== rank of MC-GS among all arms, head-only PSNR (1 = best)")
    for tag, rows in (("AF16", r16), ("AF20", r20)):
        for c in ("T1", "FLAIR"):
            vals = sorted(((r["head_psnr"], r["method"]) for r in rows if r["contrast"] == c), reverse=True)
            rank = next((i + 1 for i, (_, m) in enumerate(vals) if m in ("gs_joint_p3", "MC-GS")), None)
            print(f"  {tag} {c:5s}  rank {rank}/{len(vals)}   (top: {vals[0][1]} {vals[0][0]:.3f})" if rank else f"  {tag} {c:5s}  --")


if __name__ == "__main__":
    main()
