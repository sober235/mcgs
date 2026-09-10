"""Spec 5.5: ESPIRiT coil maps from echo 0 on all steady non-SI spokes (brain-topic flags)."""
import json
import sys

import numpy as np

from common import DERIVED_ROOT, LOGS, bart, prepare_inputs, set_ids, three_plane, work_dir

s = sys.argv[1]
prepare_inputs(s, "r0", [0])
wd = work_dir(s)
bart.coil_maps(wd / "traj_r0", wd / "ksp_r0_e0", wd, DERIVED_ROOT / s / "sens", grid=112, calib=24,
               log=LOGS / f"bart_{s}_coilmaps.log")
sens = bart.read_cfl(DERIVED_ROOT / s / "sens")
rss = np.sqrt((np.abs(bart.read_cfl(wd / "coilimg")) ** 2).sum(-1))
support = rss > 0.05 * np.quantile(rss, 0.999)
sos = np.sqrt((np.abs(sens) ** 2).sum(-1))
rep = {"subject": s, "n_spokes": int(set_ids(s, "r0").size), "sens_shape": list(sens.shape),
       "support_frac": float(support.mean()),
       "sos_within_5pct_in_support": float(np.mean(np.abs(sos[support] - 1) < 0.05))}
(LOGS / f"coilmaps_{s}.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1), flush=True)
three_plane({"RSS coil images": rss, "|map 0|": sens[..., 0], "|map 8|": sens[..., 8],
             "|map 16|": sens[..., 16]}, f"{s}_coilmaps.png",
            title=f"{s} ESPIRiT maps (echo 0, {rep['n_spokes']} spokes)")
sys.exit(0 if rep["sos_within_5pct_in_support"] >= 0.95 else 1)
