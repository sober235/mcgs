#!/usr/bin/env python3
"""slice_summary.py -- summarise logs/slice_trend_<C>.json (AF 12 / 16 / 20 / 24): the figure-slice exceptions, the share of slices
(head >= 5% of the slice) on which each metric is worse at every step, and where the non-monotonic slices of the main arms sit.
Extension of the AF 12 / 16 / 20 review's slice_summary.py (2026-09-14); an arm without AF 24 (MC-INR) is judged on its steps."""
import json, numpy as np
T = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou"
AFS = ("12", "16", "20", "24")
ORDER = ["MC-GS", "GS-single", "L1-ESPIRiT", "CS-SENSE (converged TV)", "Joint SENSE", "CS-SENSE (35 it)",
         "ZS-SSDU", "CG-SENSE", "MC-INR", "3D-DIP", "3D-INR", "ZF"]
PL = {0: "transverse", 1: "coronal", 2: "sagittal"}
M = [("PSNR", 0, +1), ("SSIM", 1, +1), ("NMSE", 2, -1)]
# numbers printed in the figures of record (180/124/76), for a sanity check of the figure-slice path
FIGREF = {("20", "T1", "MC-GS", 0): (36.25, 0.9474, 0.00881), ("20", "T1", "MC-GS", 1): (35.38, 0.9103, 0.01986),
          ("20", "T1", "MC-GS", 2): (36.00, 0.9206, 0.01500), ("20", "T1", "L1-ESPIRiT", 0): (35.81, 0.9393, 0.00975),
          ("20", "FLAIR", "MC-GS", 0): (34.30, 0.9221, 0.01067), ("20", "FLAIR", "MC-GS", 1): (35.52, 0.9075, 0.01536),
          ("20", "FLAIR", "MC-GS", 2): (35.03, 0.9032, 0.01685),
          ("24", "T1", "MC-GS", 0): (35.20, 0.9366, 0.01121), ("24", "T1", "MC-GS", 1): (34.31, 0.8908, 0.02541),
          ("24", "T1", "MC-GS", 2): (34.99, 0.9059, 0.01892), ("24", "T1", "L1-ESPIRiT", 0): (34.95, 0.9318, 0.01190),
          ("24", "FLAIR", "MC-GS", 0): (33.25, 0.9084, 0.01359), ("24", "FLAIR", "MC-GS", 1): (34.52, 0.8884, 0.01931),
          ("24", "FLAIR", "MC-GS", 2): (34.17, 0.8888, 0.02054)}


def worse_every_step(vals, sgn):        # vals: {af: value}; consecutive available AFs; True if worse at each
    v = [vals[af] for af in AFS if af in vals]
    return len(v) >= 2 and all(sgn * (b - a) < 0 for a, b in zip(v[:-1], v[1:]))


for c in ("T1", "FLAIR"):
    d = json.load(open(f"{T}/logs/slice_trend_{c}.json"))
    print(f"\n==================== {c}")
    worst = {k: max(v[k] for v in d["validation"]) for k in ("dpsnr", "dssim", "dnmse")}
    print("vectorised vs skimage on the figure slices (AF20, AF24), worst |diff|:", {k: f"{x:.1e}" for k, x in worst.items()})
    bad = []
    for (af, cc, lab, ax), ref in FIGREF.items():
        if cc != c: continue
        got = d["figure"][f"{lab}|180/124/76|{ax}"][af]
        if not (abs(got[0] - ref[0]) < 0.006 and abs(got[1] - ref[1]) < 6e-5 and abs(got[2] - ref[2]) < 6e-6): bad.append((af, lab, ax, got, ref))
    print("AF20 and AF24 figure panel numbers reproduced:", "yes" if not bad else f"NO {bad}")
    for sname in ("180/124/76", "174/110/47"):
        exc, n_total = [], 0
        for lab in ORDER:
            for ax in (0, 1, 2):
                vals = d["figure"][f"{lab}|{sname}|{ax}"]
                for mname, mi, sgn in M:
                    n_total += 1
                    if not worse_every_step({af: vals[af][mi] for af in vals}, sgn):
                        fmt = "{:.3f}" if mname == "PSNR" else ("{:.4f}" if mname == "SSIM" else "{:.5f}")
                        exc.append(f"{lab} {PL[ax]} {mname}: " + " -> ".join(fmt.format(vals[af][mi]) if af in vals else "--" for af in AFS))
        print(f"\n[figure slices {sname}] worse at every step in {n_total-len(exc)}/{n_total} (method x plane x metric) cells; exceptions:")
        for e in exc: print("   ", e)
    gt = np.abs(np.load(f"/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou/recon/gt_{c}_complex.npy")).astype(np.float32)
    head = gt > 0.05 * float(np.quantile(gt, 0.999))
    OTHER = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    keep = {ax: head.sum(axis=OTHER[ax]) >= 0.05 * np.prod([gt.shape[a] for a in OTHER[ax]]) for ax in (0, 1, 2)}
    assert all(int(keep[ax].sum()) == d["keep"][str(ax)] for ax in (0, 1, 2)), "keep-mask rule differs from slice_trend.py"
    print(f"\n[all slices with head >= 5% of the slice: transverse {int(keep[0].sum())}, coronal {int(keep[1].sum())}, sagittal {int(keep[2].sum())}]")
    print("   share (%) of those slices where the metric is worse at EVERY step:  plane: PSNR SSIM NMSE all-three")
    for lab in ORDER:
        line = f"   {lab:24s}"
        for ax in (0, 1, 2):
            s_ = d["allslices"][f"{lab}|{ax}"]; afs = [af for af in AFS if af in s_]
            ok = {}
            for mname, _, sgn in M:
                a = np.array([s_[af][mname.lower()] for af in afs])[:, keep[ax]]
                ok[mname] = np.all(sgn * (a[1:] - a[:-1]) < 0, axis=0)
            allthree = ok["PSNR"] & ok["SSIM"] & ok["NMSE"]
            line += f" | {PL[ax][:3]} " + " ".join(f"{100*ok[m].mean():5.1f}" for m in ("PSNR", "SSIM", "NMSE")) + f" {100*allthree.mean():5.1f}"
        print(line + ("   (AF12-20 only)" if len(afs) < len(AFS) else ""))

    print("   non-monotonic head-PSNR slices of the arms that are not flat (index, head share, per-AF values):")
    for ax in (0, 1, 2):
        frac = head.sum(axis=OTHER[ax]) / np.prod([gt.shape[a] for a in OTHER[ax]]); kept = np.nonzero(keep[ax])[0]
        for lab in ("MC-GS", "GS-single", "L1-ESPIRiT", "CS-SENSE (converged TV)", "Joint SENSE", "ZS-SSDU", "CG-SENSE", "MC-INR", "3D-INR"):
            s_ = d["allslices"][f"{lab}|{ax}"]; afs = [af for af in AFS if af in s_]
            p = np.array([s_[af]["psnr"] for af in afs])
            bad = [i for i in kept if not np.all(p[1:, i] < p[:-1, i])]
            if not bad:
                continue
            edge = sum(1 for i in bad if min(i - kept.min(), kept.max() - i) < 10)
            eg = ", ".join(f"{i} ({100*frac[i]:.0f}%: " + "/".join(f"{p[k, i]:.3f}" for k in range(len(afs))) + ")" for i in bad[:3])
            print(f"     {PL[ax]:10s} {lab:24s} {len(bad):3d} slices, {edge} within 10 slices of the coverage edge; e.g. {eg}")
        s_ = d["allslices"][f"3D-DIP|{ax}"]
        p = np.array([s_[af]["psnr"] for af in AFS])[:, kept]
        print(f"     {PL[ax]:10s} 3D-DIP median head-PSNR change per step over kept slices: "
              + ", ".join(f"AF{a}->AF{b} {np.median(p[k+1]-p[k]):+.3f}" for k, (a, b) in enumerate(zip(AFS[:-1], AFS[1:]))) + " dB")
