"""Spec 5.6 as revised on 2026-09-11 (user-approved): REF regularisation is lambda_rel x lambda_max of the REF
normal operator. Choose the smallest lambda_rel in {1e-3, 3e-3, 1e-2} for which REF echo 0 after 60 and 100
CG steps differs by < 1 % inside the body; write it to recon/<subject>/ref_lambda.json."""
import json
import sys

import numpy as np

from common import LOGS, prepare_inputs, recon, recon_dir, ref_lmax, ref_scaling

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from diag_artefact.headonly import head_mask  # noqa: E402

s = sys.argv[1]
rd = recon_dir(s)
prepare_inputs(s, "ref", [0])
w = ref_scaling(s, 0)
lmax = ref_lmax(s)
rows, chosen = [], None
for rel in (1e-3, 3e-3, 1e-2):
    xs = {}
    for it in (60, 100):
        recon(s, "ref", 0, "l2", rel * lmax, it, f"lamsel_{rel:g}_it{it}", w=w, lam_rel=rel)
        xs[it] = np.load(rd / f"lamsel_{rel:g}_it{it}_e0.npy")
    mag = np.abs(xs[100])
    body = head_mask(mag / np.quantile(mag, 0.999))
    diff = float(np.linalg.norm((xs[100] - xs[60])[body]) / np.linalg.norm(xs[100][body]))
    rows.append({"lam_rel": rel, "lam": rel * lmax, "iter_rel_diff_60_vs_100": diff})
    print(json.dumps(rows[-1]), flush=True)
    if diff < 0.01:
        chosen = rel
        break
rep = {"subject": s, "lmax": lmax, "candidates": rows, "lam_rel": chosen}
(rd / "ref_lambda.json").write_text(json.dumps(rep, indent=1))
(LOGS / f"lambda_select_{s}.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1), flush=True)
sys.exit(0 if chosen is not None else 1)
