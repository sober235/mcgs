#!/usr/bin/env python
"""Package the reference ("GT") multi-contrast dataset of the age-16 TSC exam into the project data root.

User decision 2026-09-12: the ground truth of every contrast is the ESPIRiT-SENSE combination of the
GRAPPA-completed k-space (first column of figures/<contrast>_sense.png). The package keeps, per contrast,
the completed multi-coil k-space, the ESPIRiT maps, that combined image, the masks and the metadata, in
the file layout of the retrospective AF12 topics (ksp_<c>.h5 dataset "kspace", csm_espirit_<c>.h5 dataset
"map", both (coil, RO, PE, SPE) complex64, DC-centred).

    ./py export_gt_dataset.py [--root /data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT]

Idempotent: a contrast is exported when its stage-1b outputs exist and the destination files are missing or
older than the sources. No patient identifiers are written.
"""
import argparse, hashlib, json, os, shutil, time

import numpy as np

TOPIC = "/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_3D"
DEFAULT_ROOT = "/data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT"
FILES = {"kspace_grappa.h5": "ksp_{c}.h5", "csm_espirit.h5": "csm_espirit_{c}.h5", "sense_grappa.npy": "gt_sense_{c}.npy",
         "masks.npz": "masks_{c}.npz", "meta.json": "meta_{c}.json"}

README = """# Peds_TSC_age16_MTP_GT — reference multi-contrast 3D dataset (UIH uMR 790, 3 T)

One exam of a 16-year-old (TSC), 2022-01-24, 3D multi-echo spoiled GRE "MTP": 5 echoes (TE 3.25 / 9.29 / 15.33 /
21.37 / 27.41 ms) x 2 flip angles (4 deg, 16 deg), TR 40.9 ms, 32 channels, 0.7 x 0.7 x 1.35 mm, FOV 224 x 210 mm.
Acquired prospectively at 2x (PPA along PE, 24-line ACS block, elliptical corner cut, 0.75 partial Fourier along
PE); there is no fully sampled acquisition. Provenance and every number: Results_20260908/Peds_TSC_MTP_3D/README.md.

## What "ground truth" means here (user decision 2026-09-12)

For each contrast the reference image is the ESPIRiT-SENSE combination of the GRAPPA-completed k-space:
    gt = sum_c conj(S_c) . ifft3c(ksp_c)        (S from `bart ecalib -t 0.001 -c 0.0 -r 24 -m 1`, unit SoS)
The acquired lines are the measurement, the odd PE lines inside the elliptical envelope are 3D GRAPPA
interpolations (kernel 4 PE x 5 RO x 5 SPE x 32 coils, Tikhonov 1e-3), and the partial-Fourier block
(PE 227..303) and the corners are zero by design (masks_<c>.npz: m_acq, m_grappa, m_zero, envelope). The
reference therefore carries 2x parallel-imaging noise and no partial-Fourier synthesis; retrospective
undersampling studies should define their masks relative to the acquired support (`envelope`) and treat
`m_zero` as known-unacquired for every method alike.

## Files (per contrast c = FA{4,16}_TE{3.25,9.29,15.33,21.37,27.41})

    ksp_<c>.h5            dataset "kspace"  (32, 320, 304, 104) complex64, axes (coil, RO, PE, SPE), DC at (160, 152, 52)
    csm_espirit_<c>.h5    dataset "map"     same layout, unit sum of squares over the whole FOV
    gt_sense_<c>.npy      (320, 304, 104) complex64, the reference image defined above
    masks_<c>.npz         m_acq, m_grappa, m_zero, envelope (PE x SPE bool), acs = [140, 164)
    meta_<c>.json         counts, kernel, timings, self-check numbers of the GRAPPA step
    index.json            list of contrasts with TE / flip angle, shapes, sizes and md5 sums

Image axes -> vendor DICOM (Z, rows, cols): SPE reversed and central 80 of 104, RO, PE cropped by 2 on each side.
The loader of the MC-GS framework reads these files directly (keys "kspace" and "map").
"""


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=DEFAULT_ROOT); ap.add_argument("--no-md5", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)
    index_path = os.path.join(args.root, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {"contrasts": {}}
    for c in sorted(os.listdir(os.path.join(TOPIC, "data"))):
        src = os.path.join(TOPIC, "data", c)
        if not c.startswith("FA") or not os.path.exists(os.path.join(src, "sense_grappa.npy")):
            continue
        entry = index["contrasts"].get(c, {})
        meta = json.load(open(os.path.join(src, "meta.json")))
        for s, d in FILES.items():
            sp, dp = os.path.join(src, s), os.path.join(args.root, d.format(c=c))
            if os.path.exists(dp) and os.path.getmtime(dp) >= os.path.getmtime(sp) and os.path.getsize(dp) == os.path.getsize(sp):
                continue
            t0 = time.time(); shutil.copy2(sp, dp)
            print(f"{c}: {s} -> {os.path.basename(dp)} ({os.path.getsize(dp) / 1e9:.2f} GB, {time.time() - t0:.0f} s)", flush=True)
        files = {}
        for s, d in FILES.items():
            dp = os.path.join(args.root, d.format(c=c))
            files[os.path.basename(dp)] = dict(bytes=os.path.getsize(dp), md5=None if args.no_md5 else md5(dp))
        entry.update(dict(flip_angle_deg=meta["flip_angle_deg"], te_ms=meta["te_ms"], echo=meta["echo"], shape=meta["shape"],
                          voxel_mm=meta["voxel_mm"], acs_lines=meta["acs_lines"], n_acquired=meta["n_acquired"], n_grappa=meta["n_grappa"],
                          n_zero_by_design=meta["n_zero_by_design"], acs_holdout_rel_error=meta["acs_holdout_rel_error"],
                          files=files, exported=time.strftime("%Y-%m-%d %H:%M")))
        index["contrasts"][c] = entry
        print(f"{c}: indexed ({len(files)} files)", flush=True)
    index["definition"] = "gt = sum_c conj(S_c) ifft3c(ksp_c); ksp = GRAPPA-completed prospective 2x k-space; S = ESPIRiT (ecalib -t 0.001 -c 0.0 -r 24 -m 1)"
    index["source_topic"] = TOPIC
    json.dump(index, open(index_path, "w"), indent=1)
    open(os.path.join(args.root, "README.md"), "w").write(README)
    print("index:", index_path, "contrasts:", sorted(index["contrasts"]))


if __name__ == "__main__":
    main()
