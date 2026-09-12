"""Comparison figure of one subject and echo: REF and every arm of the self-check (CG-SENSE and L1-wavelet at AF
8/12/16), three orthogonal planes through the heart-box centre, magnitudes (grey, [0, 1] of REF's P99.9) with the
error maps |rec - ref| / P99.9(ref) underneath (jet, fixed [0, 0.1]); body PSNR / SSIM and held-out NRMSE from
logs/selfcheck_<Vxx>.tsv in the titles. Coronal and sagittal planes are shown head-up."""
import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import LOGS, TOPIC, recon_dir, save_fig  # noqa: E402
from cardiac_fr.gauge import error_map, norm_scale  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject")
ap.add_argument("--echo", type=int, default=0)
ap.add_argument("--center", default=None, help="i,j,k; default = centre of gauge.json heart_box")
ap.add_argument("--name", default=None)
a = ap.parse_args()
s, e = a.subject, a.echo
gj = json.loads((TOPIC / "gauge.json").read_text())[s]
box = gj["heart_box"]
c = tuple(map(int, a.center.split(","))) if a.center else tuple((lo + hi) // 2 for lo, hi in box)
rows = {}
with open(LOGS / f"selfcheck_{s}.tsv") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        if int(r["echo"]) == e:
            rows[(r["method"], r["set"])] = r
arms = [("ref", "ref", "REF (CG, AF %.1f)" % gj["af_ref"])] + \
       [(m, f"af{af}", f"{lab} AF {af}") for m, lab in (("cg", "CG-SENSE"), ("l1", "L1-wavelet")) for af in (8, 12, 16)]
ref = np.abs(np.load(recon_dir(s) / f"ref_e{e}.npy"))
sc = norm_scale(ref)
vols = {t: np.abs(np.load(recon_dir(s) / f"{'ref' if t == 'ref' else m + '_' + t}_e{e}.npy")) for m, t, _ in arms}
planes = ((2, "transverse", (0, 1), False), (1, "coronal", (0, 2), True), (0, "sagittal", (1, 2), True))
fig, ax = plt.subplots(6, len(arms), figsize=(2.6 * len(arms), 15.5), squeeze=False)
for p, (fix, pname, (h, v), head_up) in enumerate(planes):
    for j, (m, t, lab) in enumerate(arms):
        x = vols[t]
        img = np.take(x / sc, c[fix], axis=fix).T
        err = np.take(error_map(x, ref), c[fix], axis=fix).T
        org = "upper" if head_up else "lower"
        ax[2 * p, j].imshow(img, cmap="gray", vmin=0, vmax=1, origin=org)
        im = ax[2 * p + 1, j].imshow(err, cmap="jet", vmin=0, vmax=0.1, origin=org)
        r = rows.get((m, t))
        title = lab if p else lab
        if r is not None and m != "ref":
            title += f"\nbody PSNR {float(r['body_psnr']):.2f}  SSIM {float(r['body_ssim']):.3f}\nheld-out NRMSE {float(r['heldout_nrmse']):.4f}"
        elif r is not None:
            title += f"\nheld-out NRMSE {float(r['heldout_nrmse']):.4f} (in-sample)"
        if p == 0:
            ax[0, j].set_title(title, fontsize=8)
        for a_ in (ax[2 * p, j], ax[2 * p + 1, j]):
            a_.set_xticks([])
            a_.set_yticks([])
    ax[2 * p, 0].set_ylabel(f"{pname}\naxis{fix} = {c[fix]}", fontsize=9)
    ax[2 * p + 1, 0].set_ylabel("|rec - ref| / P99.9", fontsize=9)
fig.subplots_adjust(left=0.04, right=0.95, top=0.9, bottom=0.02, wspace=0.08, hspace=0.12)
fig.colorbar(im, cax=fig.add_axes([0.96, 0.35, 0.01, 0.3]))
fig.suptitle(f"{s} echo {e} (TE {1.25 + 2.05 * e:.2f} ms), 224^3 reconstructions shown in the 220 mm FOV; "
             f"grey [0, 1] of REF's P99.9, error jet [0, 0.1]; planes through the heart-box centre {c}", fontsize=10)
save_fig(fig, a.name or f"{s}_compare_e{e}.png")
