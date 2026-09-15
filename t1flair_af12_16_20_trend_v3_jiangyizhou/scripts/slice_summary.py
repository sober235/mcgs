#!/usr/bin/env python3
"""slice_summary.py -- summarise logs/slice_trend_<C>.json: the figure-slice exceptions, the share of slices (head >= 5% of
the slice) on which each metric is worse at both steps, and where the non-monotonic slices of the main arms sit."""
import json, numpy as np
T = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_trend_v3_jiangyizhou"
ORDER = ["MC-GS", "GS-single", "CS-SENSE (converged TV)", "Joint SENSE", "CS-SENSE (35 it)",       # L1-ESPIRiT dropped
         "ZS-SSDU", "L1-SPIRiT", "CG-SENSE", "MC-INR", "3D-DIP", "3D-INR", "ZF"]                    # (user 2026-09-15)
PL = {0: "transverse", 1: "coronal", 2: "sagittal"}
M = [("PSNR", 0, +1), ("SSIM", 1, +1), ("NMSE", 2, -1)]
# numbers printed in the AF20 figures of record (180/124/76), for a sanity check of the figure-slice path
# (the L1-ESPIRiT reference panel was removed with the arm; the six MC-GS panels carry the check)
FIGREF = {("T1", "MC-GS", 0): (36.25, 0.9474, 0.00881), ("T1", "MC-GS", 1): (35.38, 0.9103, 0.01986), ("T1", "MC-GS", 2): (36.00, 0.9206, 0.01500),
          ("FLAIR", "MC-GS", 0): (34.30, 0.9221, 0.01067),
          ("FLAIR", "MC-GS", 1): (35.52, 0.9075, 0.01536), ("FLAIR", "MC-GS", 2): (35.03, 0.9032, 0.01685)}
def mono(v, sgn):                       # v = [AF12, AF16, AF20]; True if worse at both steps
    return sgn * (v[1] - v[0]) < 0 and sgn * (v[2] - v[1]) < 0
for c in ("T1", "FLAIR"):
    d = json.load(open(f"{T}/logs/slice_trend_{c}.json"))
    print(f"\n==================== {c}")
    worst = {k: max(v[k] for v in d["validation"]) for k in ("dpsnr", "dssim", "dnmse")}
    print("vectorised vs skimage on the figure slices, worst |diff|:", {k: f"{x:.1e}" for k, x in worst.items()})
    bad = []
    for (cc, lab, ax), ref in FIGREF.items():
        if cc != c: continue
        got = d["figure"][f"{lab}|180/124/76|{ax}"]["20"]
        if not (abs(got[0] - ref[0]) < 0.006 and abs(got[1] - ref[1]) < 6e-5 and abs(got[2] - ref[2]) < 6e-6): bad.append((lab, ax, got, ref))
    print("AF20 figure panel numbers reproduced:", "yes" if not bad else f"NO {bad}")
    for sname in ("180/124/76", "174/110/47"):
        exc = []
        for lab in ORDER:
            for ax in (0, 1, 2):
                vals = d["figure"][f"{lab}|{sname}|{ax}"]
                for mname, mi, sgn in M:
                    v = [vals[af][mi] for af in ("12", "16", "20")]
                    if not mono(v, sgn):
                        fmt = "{:.3f}" if mname == "PSNR" else ("{:.4f}" if mname == "SSIM" else "{:.5f}")
                        exc.append(f"{lab} {PL[ax]} {mname}: " + " -> ".join(fmt.format(x) for x in v))
        n_total = len(ORDER) * 3 * 3
        print(f"\n[figure slices {sname}] worse at both steps in {n_total-len(exc)}/{n_total} (method x plane x metric) cells; exceptions:")
        for e in exc: print("   ", e)
    gt = np.abs(np.load(f"/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou/recon/gt_{c}_complex.npy")).astype(np.float32)
    head = gt > 0.05 * float(np.quantile(gt, 0.999))
    OTHER = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    keep = {ax: head.sum(axis=OTHER[ax]) >= 0.05 * np.prod([gt.shape[a] for a in OTHER[ax]]) for ax in (0, 1, 2)}
    assert all(int(keep[ax].sum()) == d["keep"][str(ax)] for ax in (0, 1, 2)), "keep-mask rule differs from slice_trend.py"
    print(f"\n[all slices with head >= 5% of the slice: transverse {int(keep[0].sum())}, coronal {int(keep[1].sum())}, sagittal {int(keep[2].sum())}]")
    print("   share (%) of those slices where the metric is worse at BOTH steps:  plane: PSNR SSIM NMSE all-three")
    summary = {}
    for lab in ORDER:
        line = f"   {lab:24s}"
        for ax in (0, 1, 2):
            s_ = d["allslices"][f"{lab}|{ax}"]
            ok = {}
            for mname, _, sgn in M:
                a = np.array([s_[af][mname.lower()] for af in ("12", "16", "20")])[:, keep[ax]]
                ok[mname] = (sgn * (a[1] - a[0]) < 0) & (sgn * (a[2] - a[1]) < 0)
            allthree = ok["PSNR"] & ok["SSIM"] & ok["NMSE"]
            summary[(lab, ax)] = {m: float(ok[m].mean()) for m in ok} | {"all": float(allthree.mean())}
            line += f" | {PL[ax][:3]} " + " ".join(f"{100*ok[m].mean():5.1f}" for m in ("PSNR", "SSIM", "NMSE")) + f" {100*allthree.mean():5.1f}"
        print(line)

    print("   non-monotonic head-PSNR slices of the arms that are not flat (index, head share, AF12/16/20):")
    for ax in (0, 1, 2):
        frac = head.sum(axis=OTHER[ax]) / np.prod([gt.shape[a] for a in OTHER[ax]]); kept = np.nonzero(keep[ax])[0]
        for lab in ("MC-GS", "GS-single", "CS-SENSE (converged TV)", "Joint SENSE", "ZS-SSDU", "L1-SPIRiT", "CG-SENSE", "MC-INR", "3D-INR"):
            p = np.array([d["allslices"][f"{lab}|{ax}"][af]["psnr"] for af in ("12", "16", "20")])
            bad = [i for i in kept if not (p[1, i] < p[0, i] and p[2, i] < p[1, i])]
            if not bad:
                continue
            edge = sum(1 for i in bad if min(i - kept.min(), kept.max() - i) < 10)
            eg = ", ".join(f"{i} ({100*frac[i]:.0f}%: {p[0,i]:.3f}/{p[1,i]:.3f}/{p[2,i]:.3f})" for i in bad[:3])
            print(f"     {PL[ax]:10s} {lab:24s} {len(bad):3d} slices, {edge} within 10 slices of the coverage edge; e.g. {eg}")
        p = np.array([d["allslices"][f"3D-DIP|{ax}"][af]["psnr"] for af in ("12", "16", "20")])[:, kept]
        print(f"     {PL[ax]:10s} 3D-DIP median head-PSNR change AF16->AF20 over kept slices: {np.median(p[2]-p[1]):+.3f} dB")
