"""Physics check of the contrast labels: flip-angle ratio (T1 weighting) and the echo-train decay (T2*)."""
import numpy as np, os
T = "/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_3D/data"
def mag(c, kind):
    p = os.path.join(T, c, kind); return np.abs(np.load(p)) if os.path.exists(p) else None
TE = {0: 3.25, 1: 9.29, 2: 15.33, 3: 21.37, 4: 27.41}
fa16 = {e: mag(f"FA16_TE{TE[e]:.2f}", "sense_grappa.npy") for e in range(5)}
fa4_e0 = mag("FA4_TE3.25", "rss_grappa.npy")            # RSS == |SENSE| for unit-SoS maps (up to noise)
fa4_e4 = mag("FA4_TE27.41", "sense_grappa.npy")
t1w = fa16[0]
# crude tissue ROIs from the T1w image inside the brain (central partitions, central in-plane box to avoid scalp)
box = (slice(90, 230), slice(80, 224), slice(30, 74))          # (RO, PE, SPE) interior
sub = t1w[box]; ref = fa4_e4[box]
inside = sub > 0.15 * np.percentile(sub, 99.9)
wm = inside & (sub > np.percentile(sub[inside], 75))                          # brightest quarter of T1w = white matter
csf = inside & (sub < np.percentile(sub[inside], 15)) & (ref > np.percentile(ref[inside], 85))   # dark in T1w, bright at long TE = CSF
print(f"ROI voxels: WM {int(wm.sum())}, CSF {int(csf.sum())}")
r_wm = np.median(fa16[0][box][wm] / fa4_e0[box][wm]); r_csf = np.median(fa16[0][box][csf] / fa4_e0[box][csf])
print(f"FA16/FA4 signal ratio at TE 3.25: WM {r_wm:.2f}, CSF {r_csf:.2f}   (spoiled-GRE theory, TR 40.9 ms: WM ~2.3, CSF ~1.0)")
# T2* from the FA16 echo train, median over the ROI, log-linear fit
tes = np.array([TE[e] for e in range(5)])
for name, roi in (("WM", wm), ("CSF", csf)):
    s = np.array([np.median(fa16[e][box][roi]) for e in range(5)])
    slope, icpt = np.polyfit(tes, np.log(s), 1)
    print(f"FA16 echo train, {name}: signal vs TE {np.round(s / s[0], 3)} -> T2* = {-1 / slope:.0f} ms   (3T literature: WM 45-55, GM 50-65, CSF long)")
ratio_wm = np.median(fa4_e4[box][wm] / fa4_e0[box][wm]); ratio_csf = np.median(fa4_e4[box][csf] / fa4_e0[box][csf])
print(f"FA4 pass, TE 27.41 / TE 3.25: WM {ratio_wm:.3f} -> T2* {-24.16 / np.log(ratio_wm):.0f} ms; CSF {ratio_csf:.3f}")
print(f"CSF/WM contrast: T1w (FA16/TE3.25) {np.median(t1w[box][csf]) / np.median(t1w[box][wm]):.2f};  FA4/TE27.41 {np.median(fa4_e4[box][csf]) / np.median(fa4_e4[box][wm]):.2f}")
