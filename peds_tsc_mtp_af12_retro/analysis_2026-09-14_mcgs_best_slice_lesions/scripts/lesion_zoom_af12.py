"""Zoom of the lateral-ventricle region on vendor slices 38 and 46 (SPE 53 and 45): GT and all ten AF12 arms.
Read-only on the Peds AF12 topic. Display orientation and crop copied from Peds_TSC_MTP_3D/scripts/survey/lesion_zoom.py."""
import json, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro")
FIG = Path("/home/congcongliu/figs/peds_tsc_af12_lesion")
rows = json.load(open(R / "logs/summary.json"))

def lam_file(c, key, prefix):
    lam = [r for r in rows if r["method"] == key and r["contrast"] == c][0].get("lam")
    for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
        m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
        if m and abs(float(m.group(1)) - float(lam)) < 1e-12:
            return p
    raise FileNotFoundError((c, key, lam))

SLICES = [(53, 38), (45, 46)]                    # (SPE index, vendor slice)
CROP = (slice(90, 250), slice(70, 230))          # rows = RO, cols = PE after the 2:302 cut, as in the stage-1 zoom
for c, lab in (("T2s", "T2*w FA4/TE27.41"), ("T1w", "T1w FA16/TE3.25")):
    arms = [("GT (reference)", R / f"recon/gt_{c}_complex.npy"),
            ("CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
            ("CS-SENSE", lam_file(c, "cssense_o200", "cssense_o200")),
            ("Joint SENSE", lam_file(c, "jcssense", "jcssense_o200")),
            ("L1-ESPIRiT", lam_file(c, "l1espirit", "l1espirit")),
            ("ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
            ("3D-DIP", R / f"recon/dip_{c}_complex.npy"),
            ("3D-INR", R / f"recon/inr_{c}_complex.npy"),
            ("MC-INR", R / f"recon/mcinr_{c}_complex.npy"),
            ("GS-single", R / f"recon/gs_single_ar_{c}_mag.npy"),
            ("MC-GS", R / f"recon/gs_joint_ar_{c}_mag.npy")]
    vols = {name: np.abs(np.load(p, mmap_mode="r")) for name, p in arms}
    fig, axes = plt.subplots(len(SLICES), len(arms), figsize=(2.1 * len(arms), 2.35 * len(SLICES) + 0.6))
    for i, (spe, vend) in enumerate(SLICES):
        gt = np.asarray(vols["GT (reference)"][:, 2:302, spe])[CROP]
        lo, hi = np.percentile(gt, 1), np.percentile(gt, 99.5)
        for j, (name, _) in enumerate(arms):
            im = np.asarray(vols[name][:, 2:302, spe])[CROP]
            ax = axes[i, j]
            ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"vendor slice {vend}\n(SPE {spe})", fontsize=9)
    fig.suptitle(f"{lab}, retrospective AF 12: lateral-ventricle region, where the stage-1 visual survey saw dark foci "
                 f"(appearance of calcified subependymal nodules). Grey window from the reference, same for all panels. "
                 f"Visual check only, not a radiological reading.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG / f"peds_af12_{c}_lesion_zoom_vendor38_46.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"http://localhost:8765/peds_tsc_af12_lesion/{out.name}\n{out}")
