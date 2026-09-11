#!/usr/bin/env python3
"""compare_topics.py -- one table across the three AF12 studies.

Reads each topic's logs/summary.json (written by that topic's summarize.py, all on the same
gauge) and prints (a) the joint-minus-single table for every channel-pairing, (b) head-only PSNR
of every method side by side, so a claim about the joint model rests on all subjects and both
pairings, not on whichever one looks best.
"""
import json
from pathlib import Path

ROOT = Path("/data0/congcong/code/GS/Results_20260908")
TOPICS = [("T1T2_AF12_retro",            "huangyuwei",  "T1+T2",    ("T1", "T2")),
          ("T1FLAIR_AF12_retro",         "huangyuwei",  "T1+FLAIR", ("T1", "FLAIR")),
          ("T1FLAIR_AF12_v3_jiangyizhou", "jiangyizhou", "T1+FLAIR", ("T1", "FLAIR"))]
ORDER = [("zf", "ZF"), ("cgsense", "CG-SENSE"), ("cssense", "CS-SENSE"), ("zsssdu", "ZS-SSDU"),
         ("dip", "3D-DIP"), ("inr", "3D-INR"), ("gs_single", "MC-GS single"), ("gs_joint", "MC-GS joint")]


def load(topic):
    p = ROOT / topic / "logs/summary.json"
    return json.load(open(p)) if p.exists() else []


def main():
    print("=== joint minus single, dB   (whole-volume PSNR | head-only PSNR | head-only SSIM x1000)")
    print(f"{'subject':12s} {'pairing':9s} {'channel':7s} {'d_whole':>8s} {'d_head':>8s} {'d_hSSIM':>8s}")
    deltas = []
    for topic, subj, pairing, contrasts in TOPICS:
        rows = load(topic)
        for c in contrasts:
            j = next((r for r in rows if r["contrast"] == c and r["method"] in ("gs_joint", "MC-GS joint")), None)
            s = next((r for r in rows if r["contrast"] == c and r["method"] in ("gs_single", "MC-GS single")), None)
            if not (j and s):
                continue
            dw, dh, ds = j["psnr"] - s["psnr"], j["head_psnr"] - s["head_psnr"], 1000 * (j["head_ssim"] - s["head_ssim"])
            deltas.append(dh)
            print(f"{subj:12s} {pairing:9s} {c:7s} {dw:+8.3f} {dh:+8.3f} {ds:+8.1f}")
    if deltas:
        import statistics as st
        print(f"{'':12s} {'mean':9s} {'':7s} {'':>8s} {st.mean(deltas):+8.3f}   over {len(deltas)} channel-pairings; "
              f"head-only gains > 0.1 dB: {sum(d > 0.1 for d in deltas)}, losses < -0.1 dB: {sum(d < -0.1 for d in deltas)}")

    print("\n=== head-only PSNR, every method, every study")
    cols = [(t, s, p, c) for t, s, p, cs in TOPICS for c in cs]
    print(f"{'method':14s}" + "".join(f"{s[:4]}/{p.replace('+', '')[:5]}/{c:5s}"[:14].rjust(15) for _, s, p, c in cols))
    for key, m in ORDER:
        line = f"{m:14s}"
        for topic, subj, pairing, c in cols:
            r = next((r for r in load(topic) if r["contrast"] == c and r["method"] in (key, m)), None)
            line += f"{r['head_psnr']:15.3f}" if r else f"{'--':>15s}"
        print(line)


if __name__ == "__main__":
    main()
