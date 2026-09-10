#!/usr/bin/env python3
"""find_best_slice_v2.py -- which slice shows MC-GS joint v2 at its best?

Per-slice head-masked PSNR (frozen gauge: head = |GT| > 0.05 * P99.9(|GT|), psnr_i over head & slice)
for every arm in recon/, every axis, every slice. Two rankings, kept apart:
  (a) v2's own head PSNR              -- the slice where v2 is reconstructed best, absolute
  (b) v2 minus the best OTHER arm     -- the slice where v2 leads the field by the most
Restricted to slices whose head mask holds >= --min_head voxels (default 20000 = a full brain
slice; a few thousand voxels at the vertex or skull base give large but meaningless numbers).

usage: find_best_slice_v2.py --topic <topic dir> --contrasts T1,T2 [--ours gs_joint_v2] [--min_head 20000]
Writes <topic>/logs/best_slice_v2.json and prints the top-5 of each ranking per axis.
"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--topic", required=True)
ap.add_argument("--contrasts", required=True)
ap.add_argument("--ours", default="gs_joint_v32")
ap.add_argument("--min_head", type=int, default=20000)
ap.add_argument("--others", default="zf,cgsense,cssense_o200,jcssense,l1espirit,ploraks,zsssdu,dip,inr",
                help="arms the margin is taken against: canonical, converged versions only (sweep variants in recon/ are ignored); missing ones are skipped")
args = ap.parse_args()
R = Path(args.topic)


def load(method, c):
    for suf in ("_complex.npy", "_mag.npy"):
        p = R / f"recon/{method}_{c}{suf}"
        if p.exists():
            return np.abs(np.load(p)).astype(np.float32)
    return None


def main():
    out = {}
    for c in args.contrasts.split(","):
        gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
        gmax = float(gt.max())
        head = gt > 0.05 * float(np.quantile(gt, 0.999))
        cand = list(dict.fromkeys(args.others.split(",") + [args.ours, "gs_joint_v3", "gs_single_v3"]))
        vols = {m: load(m, c) for m in cand}
        vols = {m: v for m, v in vols.items() if v is not None}
        if args.ours not in vols and args.ours == "gs_joint_v32" and "gs_joint_v3" in vols:
            print(f"[{c}] gs_joint_v32 volume missing; falling back to gs_joint_v3 as ours")
            args.ours = "gs_joint_v3"
        assert args.ours in vols, (args.ours, list(vols))
        other_set = [m for m in args.others.split(",") if m in vols and m != args.ours]
        print(f"[{c}] ours = {args.ours}; margin taken against {other_set}")
        rows = []
        for ax in range(3):
            n = gt.shape[ax]
            for i in range(n):
                sl = [slice(None)] * 3; sl[ax] = i; sl = tuple(sl)
                hp = head[sl]
                nh = int(hp.sum())
                if nh < args.min_head:
                    continue
                g = gt[sl][hp].astype(np.float64)
                ps = {}
                for m, v in vols.items():
                    e = v[sl][hp].astype(np.float64) - g
                    ps[m] = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
                others = {m: ps[m] for m in other_set}
                best_other = max(others, key=others.get)
                rows.append(dict(axis=ax, index=i, n_head=nh, ours=ps[args.ours],
                                 margin=ps[args.ours] - others[best_other], best_other=best_other,
                                 best_other_psnr=others[best_other], psnr=ps))
        out[c] = rows
        for ax in range(3):
            sub = [r for r in rows if r["axis"] == ax]
            if not sub:
                continue
            print(f"\n=== {c} axis{ax}: {len(sub)} slices with >= {args.min_head} head voxels; "
                  f"{args.ours} first on {sum(r['margin'] > 0 for r in sub)} of them")
            for key, label in (("ours", "own head PSNR"), ("margin", "margin over the best other arm")):
                top = sorted(sub, key=lambda r: -r[key])[:5]
                print(f"   top-5 by {label}:")
                for r in top:
                    print(f"      idx {r['index']:3d}  head {r['n_head']:6d}  v2 {r['ours']:.3f} dB  "
                          f"margin {r['margin']:+.3f} vs {r['best_other']} {r['best_other_psnr']:.3f}")
    (R / "logs/best_slice_v2.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {R / 'logs/best_slice_v2.json'}")


if __name__ == "__main__":
    main()
