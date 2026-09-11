"""Spec 5.7 self-check on V07: choose the L1 lambda on AF12 echo 0 (body PSNR vs REF), then CG-SENSE and
L1 at AF 8/12/16 on echoes 0 and 4. Every metric must get monotonically worse with AF."""
import csv
import json
import sys

import numpy as np

from common import LOGS, TOPIC, prepare_inputs, recon_dir, ref_lmax, run_many, save_fig, score

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr.gauge import error_map, monotone_worse  # noqa: E402

S, ECHOES, AFS = "V07", (0, 4), ("af8", "af12", "af16")
LAMS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
KEYS = (("body_psnr", True), ("body_ssim", True), ("body_hfen", False), ("heldout_nrmse", False))
ITERS = json.loads((LOGS / "checks_V07.json").read_text())["ref_iters"]
gj = json.loads((TOPIC / "gauge.json").read_text())
box = gj[S]["heart_box"]
scal = json.loads((recon_dir(S) / "ref_scaling.json").read_text())
LMAX = ref_lmax(S)                                  # lambda = lambda_rel x LMAX (spec 5.6, 2026-09-11)
REL = json.loads((recon_dir(S) / "ref_lambda.json").read_text())["lam_rel"]
for name in (*AFS, "heldout"):
    prepare_inputs(S, name, ECHOES)

run_many([dict(subject=S, name="af12", echo=0, reg="l1", lam=l * LMAX, lam_rel=l, iters=ITERS, tag=f"l1sweep_{l:g}",
               w=scal["0"]) for l in LAMS])
sweep = {l: score(S, f"l1sweep_{l:g}", 0, box)["body_psnr"] for l in LAMS}
best = max(LAMS, key=sweep.get)
gj["l1_lambda"] = best                              # relative to LMAX
gj["lmax_ref"] = LMAX
gj["ref_lam_rel"] = REL
gj["l1_lambda_sweep_body_psnr"] = {f"{l:g}": round(v, 3) for l, v in sweep.items()}
(TOPIC / "gauge.json").write_text(json.dumps(gj, indent=1))
print("L1 lambda sweep (body PSNR, AF12, echo 0):", gj["l1_lambda_sweep_body_psnr"], "-> best", best, flush=True)
if best in (LAMS[0], LAMS[-1]):
    print("NOTE: the best lambda is at the edge of the grid", flush=True)

run_many([dict(subject=S, name=a, echo=e, reg=r, lam=l * LMAX, lam_rel=l, iters=ITERS, tag=f"{m}_{a}", w=scal[str(e)])
          for e in ECHOES for a in AFS for (m, r, l) in (("cg", "l2", REL), ("l1", "l1", best))])
rows, verdicts = [], []
for e in ECHOES:
    rows.append({"echo": e, "method": "ref_in_sample", "set": "ref", "af": round(gj[S]["af_ref"], 3),
                 **score(S, "ref", e, box)})
    for m in ("cg", "l1"):
        per = {}
        for a in AFS:
            af_a = gj[S]["subsets"][a]["af_actual"]
            per[af_a] = score(S, f"{m}_{a}", e, box)
            rows.append({"echo": e, "method": m, "set": a, "af": round(af_a, 3), **per[af_a]})
        for k, hib in KEYS:
            verdicts.append({"echo": e, "method": m, "metric": k,
                             "monotone": monotone_worse({x: per[x][k] for x in per}, hib)})
with open(LOGS / "selfcheck_V07.tsv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
    wr.writeheader()
    wr.writerows(rows)
passed = all(v["monotone"] for v in verdicts)
(LOGS / "selfcheck_V07.json").write_text(json.dumps({"l1_lambda": best, "iters": ITERS, "verdicts": verdicts,
                                                     "pass": passed}, indent=1))
print((LOGS / "selfcheck_V07.tsv").read_text(), flush=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

si = int(json.loads((TOPIC / "orientation.json").read_text())["si_axis"])
fix = [i for i in range(3) if i != si][0]
c = (box[fix][0] + box[fix][1]) // 2
ref = np.abs(np.load(recon_dir(S) / "ref_e0.npy"))
sc = float(np.quantile(ref, 0.999))
tags = ["ref"] + [f"{m}_{a}" for m in ("cg", "l1") for a in AFS]
fig, ax = plt.subplots(2, len(tags), figsize=(2.2 * len(tags), 4.8), squeeze=False)
for j, t in enumerate(tags):
    v = np.abs(np.load(recon_dir(S) / f"{t}_e0.npy"))
    ax[0, j].imshow(np.take(v / sc, c, axis=fix).T, cmap="gray", vmin=0, vmax=1, origin="lower")
    im = ax[1, j].imshow(np.take(error_map(v, ref), c, axis=fix).T, cmap="jet", vmin=0, vmax=0.1, origin="lower")
    ax[0, j].set_title(t, fontsize=8)
    for a_ in ax[:, j]:
        a_.set_xticks([])
        a_.set_yticks([])
fig.colorbar(im, ax=ax[1, :].tolist(), shrink=0.8)
fig.suptitle(f"V07 echo 0 (TE 1.25 ms), plane axis{fix} = {c}; error = |rec - ref| / P99.9(ref), jet [0, 0.1]")
save_fig(fig, "V07_selfcheck_e0.png")
print("SELF-CHECK", "PASS" if passed else "FAIL", flush=True)
sys.exit(0 if passed else 1)
