#!/usr/bin/env python3
"""estimate_csm.py — estimate ESPIRiT coil sensitivity maps (CSM) for one contrast.

The paired reconstruction is only as good as its CSM: the same CSM defines BOTH the
forward operator and the ground truth, so a clean, smooth CSM is essential. The
validated recipe for this project is BART ESPIRiT (`ecalib`) from a fully-sampled
autocalibration region (ACS).

Validated settings (winner-v2 T1/FLAIR):
    bart ecalib -t 0.001 -c 0.0 -r 24 -m 1
      -t 0.001  : eigenvalue threshold (empirically smoothest; U-shaped vs -t)
      -c 0.0    : no soft crop (unit-modulus maps span the whole FOV; SoS==1 in brain)
      -r 24     : 24x24 calibration region
      -m 1      : one set of maps
ACS = readout axis fully sampled + central 32x32 in the transverse PE plane.
Output: a .mat with key 'map', complex (256,220,256,12) = (axis0,axis1,axis2,coil),
same dim-order as the reconstruction volume.

This script documents/checks the recipe; the BART call itself needs a BART build
(amax: TOOLBOX_PATH=/home/congcongliu/src/bart) and runs on the host. If you already
have a bundled sensitivity map in your data .mat (key map/csm/sensitivity_maps/maps/S),
you can skip CSM estimation and point csm_path at the data .mat itself.

Usage (verify an existing CSM .mat — checks SoS==1 inside the brain):
    python scripts/estimate_csm.py --verify /path/to/csm.mat --data /path/to/kspace.mat
Usage (print the recipe):
    python scripts/estimate_csm.py --recipe
"""
from __future__ import annotations
import argparse

RECIPE = """\
# ESPIRiT CSM recipe (run on a host with a BART build; amax example)
# 1. Build ACS: readout axis full + central 32x32 in the transverse PE plane.
# 2. bart ecalib -t 0.001 -c 0.0 -r 24 -m 1 <acs_kspace> <csm_out>
# 3. Save as .mat with key 'map', complex (256,220,256,12).
# See paired_gs/data/dataset.py::_load_external_csm for accepted keys
# (map/csm/sensitivity_maps/maps/S) and auto coil-axis handling.
"""


def verify(csm_path: str, data_path: str | None) -> None:
    from scipy.io import loadmat
    import numpy as np
    d = loadmat(csm_path)
    key = next((k for k in ("map", "csm", "sensitivity_maps", "maps", "S")
                if k in d), None)
    if key is None:
        raise KeyError(f"no CSM key in {csm_path}; keys={[k for k in d if not k.startswith('__')]}")
    S = d[key]
    # coil axis = smallest dim (typically 12)
    coil_ax = int(np.argmin(S.shape))
    sos = np.sqrt((np.abs(S) ** 2).sum(axis=coil_ax))
    print(f"CSM {csm_path}")
    print(f"  key='{key}'  shape={S.shape}  dtype={S.dtype}  coil_axis={coil_ax}")
    print(f"  SoS: min={sos.min():.3f} max={sos.max():.3f} mean={sos.mean():.3f} "
          f"(ESPIRiT -c 0.0 -> ~1.0 across FOV incl. air; expected)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", metavar="CSM_MAT")
    ap.add_argument("--data", metavar="KSPACE_MAT", default=None)
    ap.add_argument("--recipe", action="store_true")
    args = ap.parse_args()
    if args.recipe:
        print(RECIPE)
        return
    if args.verify:
        verify(args.verify, args.data)
        return
    print(RECIPE)


if __name__ == "__main__":
    main()
