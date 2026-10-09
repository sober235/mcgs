#!/usr/bin/env python3
"""lesion_slice_allmetrics.py -- every slice through the lesion core (>= --min_core voxels, three axes): per arm, head-masked
slice PSNR / SSIM / NMSE / HFEN on both scans; reports the slices where MC-GS is first of the eleven on all four metrics on
both scans, and for the rest which metric/arm blocks it. Slice metrics as in fig_lesion_planes_zoom.py; slice HFEN =
||LoG(rec) - LoG(gt)||_head / ||LoG(gt)||_head with a 2-D LoG, sigma 1.5.
  env RETRO_TOPIC=<topic> python scripts/lesion_slice_allmetrics.py [--key gs_joint] [--out logs/lesion_slice_allmetrics.json]"""
import argparse, json, os
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity
from arms import arm_files

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
ROI = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--min_core", type=int, default=150)
    ap.add_argument("--mcgs_file", default="", help="override the MC-GS volume: pattern with {scan}")
    ap.add_argument("--out", default="logs/lesion_slice_allmetrics.json"); a = ap.parse_args()
    with np.load(ROI) as r:
        names = [str(n) for n in r["names"]]; core = r["roi%d" % names.index("病灶核心")] > 0
    D = {}
    for sc in ("pre", "post"):
        gt = np.abs(np.load(R / f"recon/gt_{sc}_complex.npy")); files = {lab: p for _, lab, p in arm_files(sc)}
        if a.mcgs_file:
            files["MC-GS"] = Path(a.mcgs_file.format(scan=sc))
        D[sc] = dict(gt=gt, gmax=float(gt.max()), head=gt > 0.05 * float(np.quantile(gt, 0.999)),
                     vols={lab: np.abs(np.load(p, mmap_mode="r")) for lab, p in files.items()})
    labels = list(D["pre"]["vols"])
    res = []
    for ax in range(3):
        for i in range(core.shape[ax]):
            if np.take(core, i, axis=ax).sum() < a.min_core:
                continue
            row = {"axis": ax, "index": i, "core": int(np.take(core, i, axis=ax).sum()), "scans": {}}
            for sc in ("pre", "post"):
                d = D[sc]; gt = np.take(d["gt"], i, axis=ax).astype(np.float64); hd = np.take(d["head"], i, axis=ax)
                lg = ndi.gaussian_laplace(gt, 1.5); m = {}
                for lab in labels:
                    v = np.take(d["vols"][lab], i, axis=ax).astype(np.float64)
                    _, sm = structural_similarity(v, gt, data_range=d["gmax"], full=True)
                    m[lab] = dict(psnr=20 * np.log10(d["gmax"]) - 10 * np.log10(np.mean((v[hd] - gt[hd]) ** 2)), ssim=float(sm[hd].mean()),
                                  nmse=float(np.sum((v[hd] - gt[hd]) ** 2) / np.sum(gt[hd] ** 2)),
                                  hfen=float(np.linalg.norm((ndi.gaussian_laplace(v, 1.5) - lg)[hd]) / np.linalg.norm(lg[hd])))
                best = {"psnr": max(m, key=lambda k: m[k]["psnr"]), "ssim": max(m, key=lambda k: m[k]["ssim"]),
                        "nmse": min(m, key=lambda k: m[k]["nmse"]), "hfen": min(m, key=lambda k: m[k]["hfen"])}
                row["scans"][sc] = dict(metrics=m, best=best, mcgs_all_first=all(b == "MC-GS" for b in best.values()))
            row["all_first"] = all(row["scans"][s]["mcgs_all_first"] for s in ("pre", "post"))
            row["blockers"] = [f"{s}:{k}->{b}" for s in ("pre", "post") for k, b in row["scans"][s]["best"].items() if b != "MC-GS"]
            j = row["scans"]; row["margin_single"] = {s: j[s]["metrics"]["MC-GS"]["psnr"] - j[s]["metrics"]["GS-single"]["psnr"] for s in j}
            res.append(row)
    json.dump(res, open(R / a.out, "w"), indent=1)
    ok = [r for r in res if r["all_first"]]
    print(f"{len(res)} lesion slices; MC-GS first on PSNR+SSIM+NMSE+HFEN on both scans: {len(ok)}")
    for r in sorted(ok, key=lambda r: -min(r["margin_single"].values()))[:10]:
        print("  axis %d idx %3d core %4d  margin over GS-single pre %+.3f post %+.3f" % (r["axis"], r["index"], r["core"], r["margin_single"]["pre"], r["margin_single"]["post"]))
    from collections import Counter
    print("blockers on the other slices:", Counter(b.split("->")[0] + "->" + b.split("->")[1] for r in res if not r["all_first"] for b in r["blockers"]).most_common(12))


if __name__ == "__main__":
    main()
