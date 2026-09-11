"""Spec 5.7: held-out 5 % of REF (seed 0) and whole-shot subsets at AF 8, 12, 16 from the rest."""
import json
import sys

import numpy as np

from common import DERIVED_ROOT, TOPIC

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr.gauge import af, heldout_split, shot_subset  # noqa: E402

s = sys.argv[1]
ref = np.load(DERIVED_ROOT / s / "ref_ids.npy")
ho, pool = heldout_split(ref, 0.05, seed=0)
subs = {f"af{t}": shot_subset(pool, t) for t in (8, 12, 16)}
np.savez(DERIVED_ROOT / s / "gauge_split.npz", heldout=ho, pool=pool, **subs)
g = {"n_ref": int(ref.size), "af_ref": af(ref.size), "n_heldout": int(ho.size), "n_pool": int(pool.size),
     "seed": 0, "subsets": {k: {"n": int(v.size), "af_actual": af(v.size)} for k, v in subs.items()}}
p = TOPIC / "gauge.json"
gj = json.loads(p.read_text()) if p.exists() else {}
gj[s] = {**gj.get(s, {}), **g}
p.write_text(json.dumps(gj, indent=1))
print(json.dumps({s: g}, indent=1))
