#!/usr/bin/env python3
"""enhancement_fidelity.py -- what each arm does to the enhancement, against the truth.
post - pre in white-matter units (each scan divided by its own white-matter median) averaged over the six reference regions
the prospective study defined on CS-SENSE (enhance_rois_cssense.npz: 病灶核心 / 核心外 0-3 mm / 正常白质 / 正常皮层 / 血管/静脉窦 /
脉络丛), plus the extracranial background of the post image (mean outside the head mask dilated by 6 voxels = 6 mm).
The truth row is the CS-SENSE pair itself; err_* = arm - truth.   -> logs/enhancement_fidelity.json"""
import json, os
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from arms import arm_files, LABEL

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
ROI_NPZ = Path(os.environ.get("RETRO_ROI", "/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz"))
KEYS = {"病灶核心": "lesion_core", "核心外 0-3 mm": "rim_0_3mm", "正常白质": "white_matter", "正常皮层": "cortex", "血管/静脉窦": "vessels", "脉络丛": "plexus"}


def wm_units(v, wm):
    return v / np.median(v[wm])


def main():
    with np.load(ROI_NPZ) as r:
        names = [str(n) for n in r["names"]]; roi = {KEYS[n]: r[f"roi{i}"] for i, n in enumerate(names)}
    wm = roi["white_matter"]
    with np.load(R / "sim/gt.npz") as g:
        gt = {s: np.abs(g[s]).astype(np.float32) for s in ("pre", "post")}
    p999 = float(np.quantile(gt["post"], 0.999))
    head = ndi.binary_dilation(gt["post"] > 0.05 * p999, iterations=6)
    vols = {"truth": gt}
    for key, label, p in arm_files("pre"):
        q = arm_files("post"); pp = dict((k, f) for k, _, f in q)[key]
        if p.exists() and pp.exists():
            vols[key] = {"pre": np.abs(np.load(p)).astype(np.float32), "post": np.abs(np.load(pp)).astype(np.float32)}
    rows = []
    for key, v in vols.items():
        d = wm_units(v["post"], wm) - wm_units(v["pre"], wm)
        row = {"method": key, "label": "truth (CS-SENSE)" if key == "truth" else LABEL[key],
               **{k: float(d[m].mean()) for k, m in roi.items()}, "background_post": float(wm_units(v["post"], wm)[~head].mean())}
        rows.append(row)
    truth = rows[0]
    for row in rows[1:]:
        for k in list(roi) + ["background_post"]:
            row["err_" + k] = row[k] - truth[k]
    json.dump({"units": "white-matter units (each scan / its own WM median); post - pre", "rows": rows}, open(R / "logs/enhancement_fidelity.json", "w"), indent=1)
    print(f"{'arm':18s}{'vessels':>9s}{'plexus':>9s}{'cortex':>9s}{'WM':>8s}{'core':>8s}{'rim':>8s}{'bg':>8s}")
    for r in rows:
        print(f"{r['label']:18s}{r['vessels']:>+9.3f}{r['plexus']:>+9.3f}{r['cortex']:>+9.3f}{r['white_matter']:>+8.3f}{r['lesion_core']:>+8.3f}{r['rim_0_3mm']:>+8.3f}{r['background_post']:>8.3f}")


if __name__ == "__main__":
    main()
