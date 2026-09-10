"""Spec 5.1: read one subject's free-running 8-echo twix file into raw.h5 (+ noise.npy, meta.json)."""
import json
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import DERIVED_ROOT, N_COIL, N_ECHO, N_SAMP, N_SEG, N_SPOKE, RAW_ROOT, TOPIC  # noqa: E402
from cardiac_fr.twix import load_freerunning  # noqa: E402

KEYS = ("tProtocolName", "sKSpace.lBaseResolution", "sKSpace.lPartitions", "sKSpace.lRadialViews",
        "lContrasts", "alTR[0]", "alTE[0]", "alTE[1]", "alTE[7]", "adFlipAngleDegree[0]",
        "sRXSPEC.alDwellTime[0]")

s = sys.argv[1]
src = RAW_ROOT / s / "freerunning_8echo.dat"
out = DERIVED_ROOT / s
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
d = load_freerunning(src, n_echo=N_ECHO)
assert d.ksp.shape == (N_ECHO, N_SPOKE, N_COIL, N_SAMP), d.ksp.shape
assert d.noise.shape == (N_COIL, N_SAMP), d.noise.shape
assert (d.lin == np.arange(N_SPOKE)).all(), "Lin is not 0..21999 in acquisition order"
assert (d.seg == d.lin // N_SEG).all(), "Seg counter is not the interleave index"
si_flag = ((d.mask >> np.uint64(34)) & np.uint64(1)).astype(bool)
assert (si_flag == (np.repeat(d.lin, N_ECHO) % N_SEG == 0)).all(), "MDH bit 34 is not exactly the seg-0 spokes"
with h5py.File(out / "raw.h5", "w") as h:
    h.create_dataset("ksp", data=d.ksp, chunks=(1, 256, N_COIL, N_SAMP))
    for k in ("lin", "seg", "mask", "eco", "ts_ticks"):
        h.create_dataset(k, data=getattr(d, k))
np.save(out / "noise.npy", d.noise)
meta = {"subject": s, "source": str(src.resolve()), "protocol": d.protocol,
        "ksp_shape": list(d.ksp.shape), "n_adc_imaging": int(d.eco.size),
        "seconds": round(time.time() - t0, 1), "ascconv": {k: d.ascconv.get(k) for k in KEYS}}
txt = json.dumps(meta, indent=1)
(out / "meta.json").write_text(txt)
(TOPIC / "logs" / f"meta_{s}.json").write_text(txt)
print(txt, flush=True)
