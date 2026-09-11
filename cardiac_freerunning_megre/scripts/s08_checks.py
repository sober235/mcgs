"""Spec 5.6 checks on REF echo 0: iterations (0.6N vs N steps, body relative difference < 1 %) and -S
scaling (-w 2X vs -w X, body norm ratio within 1 %)."""
import argparse
import json
import sys

import numpy as np

from common import LOGS, prepare_inputs, recon, recon_dir, ref_lmax

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from diag_artefact.headonly import head_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject")
ap.add_argument("--ref-iters", type=int, default=100)
ap.add_argument("--lam-rel", type=float, default=None, help="REF lambda relative to the largest eigenvalue")
a = ap.parse_args()
s, n = a.subject, a.ref_iters
rd = recon_dir(s)
w = json.loads((rd / "ref_scaling.json").read_text())["0"]
prepare_inputs(s, "ref", [0])
short = int(round(0.6 * n))
lam = a.lam_rel * ref_lmax(s) if a.lam_rel is not None else 1e-3
recon(s, "ref", 0, "l2", lam, short, f"ref_it{short}", w=w, lam_rel=a.lam_rel)
recon(s, "ref", 0, "l2", lam, n, "ref_w2x", w=2 * w, lam_rel=a.lam_rel)
xn, xs, x2 = (np.load(rd / f"{t}_e0.npy") for t in ("ref", f"ref_it{short}", "ref_w2x"))
mag = np.abs(xn)
body = head_mask(mag / np.quantile(mag, 0.999))
it = float(np.linalg.norm((xn - xs)[body]) / np.linalg.norm(xn[body]))
ratio = float(np.linalg.norm(x2[body]) / np.linalg.norm(xn[body]))
rep = {"subject": s, "ref_iters": n, "lam_rel": a.lam_rel, "lam": lam, f"iter_rel_diff_{short}_vs_{n}": it, "iter_ok": it < 0.01,
       "scale_ratio_w2x_over_w1x": ratio, "scale_ok": abs(ratio - 1) < 0.01}
(LOGS / f"checks_{s}.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1), flush=True)
sys.exit(0 if rep["iter_ok"] and rep["scale_ok"] else 1)
