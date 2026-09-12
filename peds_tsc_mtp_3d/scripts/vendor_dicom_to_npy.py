#!/usr/bin/env python
"""Export the vendor magnitude reconstructions of the MTP Cartesian scan (series M_802) to npy.

Runs in the nvgen environment (pydicom). One file per acquired contrast: data/vendor/M_FA{fa}_TE{te}.npy,
float32 (Z=80 slices ordered by increasing ImagePositionPatient z, rows=320 (RO), cols=300 (PE)).
No patient identifiers are copied; meta.json keeps only acquisition parameters.
"""
import os, glob, json, collections
import numpy as np, pydicom
SRC = "/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/dicom/mtp_cartesian_.7x.7x1_M_802"
OUT = "/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_3D/data/vendor"
os.makedirs(OUT, exist_ok=True)
groups = collections.defaultdict(list); meta = {}
for f in sorted(glob.glob(os.path.join(SRC, "*.dcm"))):
    d = pydicom.dcmread(f)
    key = (float(d.FlipAngle), float(d.EchoTime))
    groups[key].append((float(d.ImagePositionPatient[2]), d.pixel_array.astype(np.float32)))
    meta.setdefault(f"FA{int(key[0])}_TE{key[1]:.2f}", dict(
        FlipAngle=float(d.FlipAngle), EchoTime=float(d.EchoTime), RepetitionTime=float(d.RepetitionTime),
        Rows=int(d.Rows), Columns=int(d.Columns), PixelSpacing=[float(x) for x in d.PixelSpacing],
        SliceThickness=float(d.SliceThickness), SpacingBetweenSlices=float(d.SpacingBetweenSlices),
        ImageOrientationPatient=[float(x) for x in d.ImageOrientationPatient], MRAcquisitionType=str(d.MRAcquisitionType)))
for (fa, te), lst in groups.items():
    lst.sort(key=lambda t: t[0])
    vol = np.stack([a for _, a in lst], 0)
    name = f"M_FA{int(fa)}_TE{te:.2f}.npy"
    np.save(os.path.join(OUT, name), vol)
    meta[f"FA{int(fa)}_TE{te:.2f}"]["n_slices"] = int(vol.shape[0]); meta[f"FA{int(fa)}_TE{te:.2f}"]["file"] = name
    print(name, vol.shape, "z from %.1f to %.1f mm" % (lst[0][0], lst[-1][0]))
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
