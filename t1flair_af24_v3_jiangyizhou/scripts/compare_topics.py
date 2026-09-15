#!/usr/bin/env python3
"""compare_topics.py -- jiangyizhou at AF20 against AF24, every arm, one gauge.

The two topics share the subject, the k-space, the ESPIRiT maps, the ground truth and the scoring
module, and the MC-GS recipe is identical; only the mask differs (2427 -> 1877 PE points; AF = PE-plane
points / sampled points that carry data, user decision 2026-09-14: T1 18.56x -> 24.00x, FLAIR 19.96x ->
25.51x). So every AF24 - AF20 difference below
is the price of the extra acceleration for that arm.

usage: compare_topics.py            (both topics' logs/summary.json must exist)
"""
import json
from pathlib import Path

ROOT = Path("/data0/congcong/code/GS/Results_20260908")
A20, A24 = "T1FLAIR_AF20_v3_jiangyizhou", "T1FLAIR_AF24_v3_jiangyizhou"
ORDER = [("zf", "ZF"), ("cgsense", "CG-SENSE"), ("cssense", "CS-SENSE"),
         ("cssense_o200", "CS-SENSE (TV, converged)"), ("jcssense", "joint CS-SENSE (JTV)"),
         ("l1espirit", "L1-ESPIRiT"), ("spirit", "L1-SPIRiT"), ("zsssdu", "ZS-SSDU"), ("dip", "3D-DIP"), ("inr", "3D-INR"), ("mcinr", "MC-INR (Niessen 2025)"),
         ("gs_single_p3", "GS-single"), ("gs_joint_p3", "MC-GS")]
COMPETING = {"zf", "cgsense", "cssense", "cssense_o200", "jcssense", "l1espirit", "spirit", "zsssdu", "dip", "inr", "mcinr"}
# Comparison conventions (user): GS-single counts as a comparison arm (2026-09-13); L1-ESPIRiT is dropped from the
# comparison (2026-09-15; its rows stay in logs/summary.json). The quoted margin is the section that applies both.
DROP = {"l1espirit"}


def load(topic):
    p = ROOT / topic / "logs/summary.json"
    return json.load(open(p)) if p.exists() else []


def row(rows, c, key, label):
    return next((r for r in rows if r["contrast"] == c and r["method"] in (key, label)), None)


def main():
    r20, r24 = load(A20), load(A24)
    for c in ("T1", "FLAIR"):
        print(f"\n=== jiangyizhou {c}: head-only PSNR and SSIM   (AF20 | AF24 | AF24-AF20)")
        print(f"{'method':26s} {'hPSNR20':>8s} {'hPSNR24':>8s} {'dPSNR':>7s}   {'hSSIM20':>8s} {'hSSIM24':>8s} {'dSSIM':>8s}   {'sharp24':>7s}")
        for key, label in ORDER:
            a, b = row(r20, c, key, label), row(r24, c, key, label)
            if not b:
                print(f"{label:26s} {'--':>8s}"); continue
            if a:
                print(f"{label:26s} {a['head_psnr']:8.3f} {b['head_psnr']:8.3f} {b['head_psnr']-a['head_psnr']:+7.3f}   "
                      f"{a['head_ssim']:8.4f} {b['head_ssim']:8.4f} {b['head_ssim']-a['head_ssim']:+8.4f}   {b['sharp']:7.3f}")
            else:
                print(f"{label:26s} {'--':>8s} {b['head_psnr']:8.3f} {'':7s}   {'--':>8s} {b['head_ssim']:8.4f} {'':8s}   {b['sharp']:7.3f}")

    print("\n=== MC-GS margin over the strongest comparison arm: GS-single counts, L1-ESPIRiT dropped (the margin to quote), head-only PSNR")
    for tag, rows in (("AF20", r20), ("AF24", r24)):
        for c in ("T1", "FLAIR"):
            mine = row(rows, c, "gs_joint_p3", "MC-GS")
            comp = [(r["head_psnr"], r["method"]) for r in rows
                    if r["contrast"] == c and r["method"] not in DROP and r["method"] not in ("gs_joint_p3", "MC-GS")]
            nongs = [x for x in comp if x[1] not in ("gs_single_p3", "GS-single")]
            if mine and comp:
                h, m = max(comp); h2, m2 = max(nongs)
                print(f"  {tag} {c:5s}  MC-GS {mine['head_psnr']:.3f}  strongest {h:.3f} ({m})  margin {mine['head_psnr']-h:+.3f}"
                      f"   | strongest non-GS {h2:.3f} ({m2})  {mine['head_psnr']-h2:+.3f}")

    print("\n=== older framing, history only: MC-GS margin over the strongest arm with GS-single excluded and L1-ESPIRiT listed")
    for tag, rows in (("AF20", r20), ("AF24", r24)):
        for c in ("T1", "FLAIR"):
            mine = row(rows, c, "gs_joint_p3", "MC-GS")
            comp = [(r["head_psnr"], r["method"]) for r in rows if r["contrast"] == c and r["method"] in COMPETING]
            if mine and comp:
                h, m = max(comp)
                print(f"  {tag} {c:5s}  MC-GS {mine['head_psnr']:.3f}  best other {h:.3f} ({m})  margin {mine['head_psnr']-h:+.3f}")

    print("\n=== MC-GS minus GS-single (the coupling), head-only PSNR / SSIM")
    for tag, rows in (("AF20", r20), ("AF24", r24)):
        for c in ("T1", "FLAIR"):
            j, s = row(rows, c, "gs_joint_p3", "MC-GS"), row(rows, c, "gs_single_p3", "GS-single")
            if j and s:
                print(f"  {tag} {c:5s}  {j['head_psnr']-s['head_psnr']:+.3f} dB  {1000*(j['head_ssim']-s['head_ssim']):+.1f} mSSIM")

    print("\n=== rank of MC-GS among all arms, head-only PSNR (1 = best)")
    for tag, rows in (("AF20", r20), ("AF24", r24)):
        for c in ("T1", "FLAIR"):
            vals = sorted(((r["head_psnr"], r["method"]) for r in rows if r["contrast"] == c), reverse=True)
            rank = next((i + 1 for i, (_, m) in enumerate(vals) if m in ("gs_joint_p3", "MC-GS")), None)
            print(f"  {tag} {c:5s}  rank {rank}/{len(vals)}   (top: {vals[0][1]} {vals[0][0]:.3f})" if rank else f"  {tag} {c:5s}  --")


if __name__ == "__main__":
    main()
