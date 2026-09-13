#!/usr/bin/env python3
"""compare_topics.py -- AF20 against AF12, both subjects, every method, one gauge.

Reads each topic's logs/summary.json (written by that topic's summarize.py; the AF12 and AF20
topics share data, maps, GT and the scoring module, so the rows are directly comparable) and prints
(a) head-only PSNR of every method at AF12 and AF20 side by side with the AF20 - AF12 drop,
(b) the joint-minus-single table at AF20 for every channel-pairing, next to its AF12 value.
"""
import json
from pathlib import Path

ROOT = Path("/data0/congcong/code/GS/Results_20260908")
PAIRS = [("huangyuwei",  "T1FLAIR_AF12_retro",          "T1FLAIR_AF20_retro"),
         ("jiangyizhou", "T1FLAIR_AF12_v3_jiangyizhou", "T1FLAIR_AF20_v3_jiangyizhou")]
ORDER = [("zf", "ZF"), ("cgsense", "CG-SENSE"), ("cssense", "CS-SENSE"), ("zsssdu", "ZS-SSDU"),
         ("dip", "3D-DIP"), ("inr", "3D-INR"), ("gs_single", "MC-GS single"), ("gs_joint", "MC-GS joint")]


def load(topic):
    p = ROOT / topic / "logs/summary.json"
    return json.load(open(p)) if p.exists() else []


def row(rows, c, key, label):
    return next((r for r in rows if r["contrast"] == c and r["method"] in (key, label)), None)


def main():
    for subj, t12, t16 in PAIRS:
        r12, r16 = load(t12), load(t16)
        for c in ("T1", "FLAIR"):
            print(f"\n=== {subj} {c}: head-only PSNR / head-only SSIM   (AF12 | AF20 | AF20-AF12)")
            print(f"{'method':14s} {'hPSNR12':>8s} {'hPSNR16':>8s} {'dPSNR':>7s}   {'hSSIM12':>8s} {'hSSIM16':>8s} {'dSSIM':>7s}   {'PSNR16':>7s} {'SSIM16':>7s}")
            for key, label in ORDER:
                a, b = row(r12, c, key, label), row(r16, c, key, label)
                if not b:
                    print(f"{label:14s} {'--':>8s}"); continue
                if a:
                    print(f"{label:14s} {a['head_psnr']:8.3f} {b['head_psnr']:8.3f} {b['head_psnr']-a['head_psnr']:+7.3f}   "
                          f"{a['head_ssim']:8.4f} {b['head_ssim']:8.4f} {b['head_ssim']-a['head_ssim']:+7.4f}   "
                          f"{b['psnr']:7.3f} {b['ssim']:7.4f}")
                else:
                    print(f"{label:14s} {'--':>8s} {b['head_psnr']:8.3f} {'':7s}   {'--':>8s} {b['head_ssim']:8.4f}")
    print("\n=== MC-GS joint minus single, dB (whole | head-only | head-only SSIM x1000): AF12 -> AF20")
    print(f"{'subject':12s} {'channel':7s} {'AF12 whole':>11s} {'AF12 head':>10s} {'AF12 hSSIM':>11s} {'AF20 whole':>11s} {'AF20 head':>10s} {'AF20 hSSIM':>11s}")
    for subj, t12, t16 in PAIRS:
        for c in ("T1", "FLAIR"):
            out = [subj, c]
            for rows in (load(t12), load(t16)):
                j, s = row(rows, c, "gs_joint", "MC-GS joint"), row(rows, c, "gs_single", "MC-GS single")
                if j and s:
                    out += [f"{j['psnr']-s['psnr']:+11.3f}", f"{j['head_psnr']-s['head_psnr']:+10.3f}", f"{1000*(j['head_ssim']-s['head_ssim']):+11.1f}"]
                else:
                    out += [f"{'--':>11s}", f"{'--':>10s}", f"{'--':>11s}"]
            print(f"{out[0]:12s} {out[1]:7s} " + " ".join(out[2:]))
    print("\n=== rank of MC-GS joint among the eight arms, head-only PSNR (1 = best)")
    for subj, t12, t16 in PAIRS:
        for c in ("T1", "FLAIR"):
            line = f"{subj:12s} {c:7s}"
            for tag, rows in (("AF12", load(t12)), ("AF20", load(t16))):
                canon = {k for k, _ in ORDER} | {m for _, m in ORDER}   # the eight arms only: AF12 summaries may carry extra autoresearch rows
                vals = [(r["head_psnr"], r["method"]) for r in rows if r["contrast"] == c and r["method"] in canon]
                if not vals:
                    line += f"   {tag}: --"; continue
                vals.sort(reverse=True)
                rank = next((i + 1 for i, (_, m) in enumerate(vals) if m in ("gs_joint", "MC-GS joint")), None)
                gap = vals[0][0] - next(v for v, m in vals if m in ("gs_joint", "MC-GS joint")) if rank else float("nan")
                line += f"   {tag}: rank {rank}/{len(vals)}, {gap:.2f} dB behind {vals[0][1]}"
            print(line)


if __name__ == "__main__":
    main()
