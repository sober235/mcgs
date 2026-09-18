#!/usr/bin/env python3
"""export_bundle.py -- the results bundle of this study: every registered volume (both scans), the simulation inputs
(mask, truth, meta), logs and figures, with MANIFEST.tsv and md5sums. --no_raw skips the ~26 GB of h5 / work_* k-space.
  python scripts/export_bundle.py [--no_raw]   -> export/ce_retro_af16_<YYYYMMDD>/"""
import argparse, hashlib, os, shutil, time
from pathlib import Path
from arms import arm_files, LABEL

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
ap = argparse.ArgumentParser(); ap.add_argument("--no_raw", action="store_true"); a = ap.parse_args()
out = R / f"export/ce_retro_af16_{time.strftime('%Y%m%d')}"; out.mkdir(parents=True, exist_ok=True)
rows = []


def put(src, rel, note):
    dst = out / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    rows.append((rel, note, dst.stat().st_size, hashlib.md5(dst.read_bytes()).hexdigest(), str(src)))


for scan in ("pre", "post"):
    for key, label, p in arm_files(scan):
        put(p, f"volumes/{scan}/{key}_{scan}_complex.npy", f"{label} reconstruction, gauge units (scale_{scan} in sim/gt.npz), layout (S-I 256, A-P 232, L-R 176)")
    put(R / f"recon/gt_{scan}_complex.npy", f"volumes/{scan}/gt_{scan}_complex.npy", "CS-SENSE truth in gauge units")
put(R / "sim/gt.npz", "sim/gt.npz", "truth volumes (raw units), mask2d, scale_pre/scale_post"); put(R / "sim/meta.json", "sim/meta.json", "sigma, kres, seeds")
for f in (R / "sim/mask").iterdir():
    put(f, f"sim/mask/{f.name}", "Poisson-disc AF16 mask")
if not a.no_raw:
    for f in list((R / "sim/gs").iterdir()) + [q for s in ("pre", "post") for q in (R / f"sim/work_{s}").iterdir()]:
        put(f, f"raw/{f.parent.name}/{f.name}", "synthetic k-space / maps / mask as the arms read them")
for f in sorted((R / "logs").glob("*.json")):
    put(f, f"logs/{f.name}", "score json")
for f in sorted((R / "figures").glob("*.p*")):
    put(f, f"figures/{f.name}", "figure")
with open(out / "MANIFEST.tsv", "w") as m:
    m.write("path\tnote\tbytes\tmd5\tsource\n"); m.writelines("\t".join(map(str, r)) + "\n" for r in rows)
with open(out / "md5sums.txt", "w") as m:
    m.writelines(f"{r[3]}  {r[0]}\n" for r in rows)
print(f"{len(rows)} files, {sum(r[2] for r in rows) / 2**30:.2f} GiB -> {out}"); print("verify: cd", out, "&& md5sum -c md5sums.txt | grep -c OK")
