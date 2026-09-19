"""Per-slice head PSNR / SSIM / NMSE of all ten Peds AF12 arms, every plane, both contrasts (read-only on the topic).
Metric definitions as the topic's compact three-plane figure: head = |GT| > 0.05 * P99.9 of the volume, restricted to the
central 80 partitions (SPE 12..91) like the primary table; PSNR at the volume max|GT|; SSIM = 7x7 uniform-window map
(data_range = max|GT|) averaged over the head pixels; NMSE = sum (rec-GT)^2 / sum GT^2 over the head pixels.
Slices kept: head area >= 25 % of the largest head area in that plane."""
import json, re, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity

R = Path("/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro")
OUT = Path(sys.argv[1])
rows = json.load(open(R / "logs/summary.json"))

def lam_file(c, key, prefix):
    lam = [r for r in rows if r["method"] == key and r["contrast"] == c][0]["lam"]
    for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
        m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
        if m and abs(float(m.group(1)) - float(lam)) < 1e-12:
            return p
    raise FileNotFoundError((c, key))

def arms_for(c):
    return [("cgsense", "CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
            ("cssense_o200", "CS-SENSE", lam_file(c, "cssense_o200", "cssense_o200")),
            ("jcssense", "Joint SENSE", lam_file(c, "jcssense", "jcssense_o200")),
            ("l1espirit", "L1-ESPIRiT", lam_file(c, "l1espirit", "l1espirit")),
            ("zsssdu", "ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
            ("dip", "3D-DIP", R / f"recon/dip_{c}_complex.npy"),
            ("inr", "3D-INR", R / f"recon/inr_{c}_complex.npy"),
            ("mcinr", "MC-INR", R / f"recon/mcinr_{c}_complex.npy"),
            ("gs_single_ar", "GS-single", R / f"recon/gs_single_ar_{c}_mag.npy"),
            ("gs_joint_ar", "MC-GS", R / f"recon/gs_joint_ar_{c}_mag.npy")]

G = {}
def init(c):
    gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float64)
    head = gt > 0.05 * np.quantile(gt, 0.999)
    head[:, :, :12] = False; head[:, :, 92:] = False
    G.update(c=c, gt=gt, head=head, gmax=float(gt.max()),
             vols={k: np.abs(np.load(p)).astype(np.float64) for k, _, p in arms_for(c)})

def one(job):
    axis, idx = job
    sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
    g2, m2 = G["gt"][sl], G["head"][sl]
    g = g2[m2]; out = {}
    for k, v in G["vols"].items():
        r2 = v[sl]; e = r2[m2] - g
        _, smap = structural_similarity(g2, r2, data_range=G["gmax"], full=True, win_size=7)
        out[k] = dict(psnr=20 * np.log10(G["gmax"]) - 10 * np.log10(float((e * e).mean())),
                      ssim=float(smap[m2].mean()), nmse=float((e * e).sum() / (g * g).sum()))
    return axis, idx, int(m2.sum()), out

result = {}
for c in ("T1w", "T2s"):
    init(c)
    area = {a: G["head"].sum(axis=tuple(x for x in range(3) if x != a)) for a in range(3)}
    jobs = [(a, i) for a in range(3) for i in range(G["gt"].shape[a]) if area[a][i] >= 0.25 * area[a].max()]
    with Pool(24) as pool:
        per = pool.map(one, jobs, chunksize=4)
    result[c] = [dict(axis=a, index=i, head_px=n, metrics=m) for a, i, n, m in per]
    print(c, "slices scored:", len(per), flush=True)
json.dump(result, open(OUT, "w"))
