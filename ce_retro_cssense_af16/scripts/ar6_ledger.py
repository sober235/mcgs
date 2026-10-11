#!/usr/bin/env python3
"""ar6_ledger.py -- phase 6 of the MC-GS autoresearch (2026-10-10/11): every scored probe against the MC-GS of record (i1)
and the fair GS-single, with the pre-registered reading (autoresearch_vdpois.md, phase 6):
  primary   MC-GS - fair single >= +0.5 dB head PSNR on both scans
  keep      both scans head PSNR >= i1 + 0.05, SSIM / NMSE / HFEN not worse on both scans, and |vessel| and |plexus|
            enhancement error not worse than i1 by more than 0.05 white-matter units
also the gap to ZS-SSDU (published size). Enhancement in white-matter units on the prospective study's CS-SENSE ROIs
(the same definition as enhancement_fidelity.py).
  python scripts/ar6_ledger.py [prefix ...]     # default prefixes ar_k (phase 6a)"""
import glob, json, os, sys
from pathlib import Path
import numpy as np

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois"))
ROI_NPZ = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz")
KEYS = {"正常白质": "wm", "血管/静脉窦": "vessels", "脉络丛": "plexus", "正常皮层": "cortex", "病灶核心": "lesion"}
I1 = "ar_i1_k30p90"; FAIR = {"pre": "ar_s_pre_wf00v", "post": "ar_s_post_wf00bv"}; ZS = {"pre": 35.319, "post": 37.527}
M = ("head_psnr", "head_ssim", "head_nmse", "head_hfen")


def row(key, scan):
    f = R / f"logs/{key}_{scan}.json"
    if not f.exists():
        return None
    return next(r for r in json.load(open(f))["rows"] if r["contrast"] == scan)


def enhancement(key, rois, gtw):
    vols = {}
    for s in ("pre", "post"):
        p = R / f"recon/{key}_{s}_complex.npy"
        if not p.exists():
            return None
        vols[s] = np.abs(np.load(p)).astype(np.float32)
    wm = rois["wm"]
    d = vols["post"] / np.median(vols["post"][wm]) - vols["pre"] / np.median(vols["pre"][wm])
    return {k: float(d[m].mean()) - gtw[k] for k, m in rois.items() if k in ("vessels", "plexus", "cortex", "lesion")}


def main():
    prefixes = sys.argv[1:] or ["ar_k"]
    with np.load(ROI_NPZ) as r:
        names = [str(n) for n in r["names"]]
        rois = {KEYS[n]: r[f"roi{i}"] for i, n in enumerate(names) if n in KEYS}
    with np.load(R / "sim/gt.npz") as g:
        gt = {s: np.abs(g[s]).astype(np.float32) for s in ("pre", "post")}
    dgt = gt["post"] / np.median(gt["post"][rois["wm"]]) - gt["pre"] / np.median(gt["pre"][rois["wm"]])
    gtw = {k: float(dgt[m].mean()) for k, m in rois.items()}
    keys = sorted({Path(f).name.rsplit("_", 1)[0] for p in prefixes for f in glob.glob(str(R / f"logs/{p}*_pre.json"))})
    base = {s: row(I1, s) for s in ("pre", "post")}
    e_i1 = enhancement(I1, rois, gtw)
    print(f"reference i1: pre {base['pre']['head_psnr']:.3f}  post {base['post']['head_psnr']:.3f};  fair single "
          f"{row(FAIR['pre'], 'pre')['head_psnr']:.3f} / {row(FAIR['post'], 'post')['head_psnr']:.3f};  ZS-SSDU {ZS['pre']} / {ZS['post']}")
    print(f"i1 enhancement error (WM units): " + "  ".join(f"{k} {v:+.3f}" for k, v in e_i1.items()))
    lines = []
    for k in [I1] + [k for k in keys if k != I1]:
        rp, rq = row(k, "pre"), row(k, "post")
        if rp is None or rq is None:
            continue
        dpre = {m: rp[m] - base["pre"][m] for m in M}; dpost = {m: rq[m] - base["post"][m] for m in M}
        mfair = (rp["head_psnr"] - row(FAIR["pre"], "pre")["head_psnr"], rq["head_psnr"] - row(FAIR["post"], "post")["head_psnr"])
        e = enhancement(k, rois, gtw)
        worse = lambda d: d["head_ssim"] < 0 or d["head_nmse"] > 0 or d["head_hfen"] > 0
        enh_ok = e is not None and all(abs(e[x]) <= abs(e_i1[x]) + 0.05 for x in ("vessels", "plexus"))
        keep = dpre["head_psnr"] >= 0.05 and dpost["head_psnr"] >= 0.05 and not worse(dpre) and not worse(dpost) and enh_ok
        verdict = "i1 (reference)" if k == I1 else ("KEEP" if keep else "drop")
        lines.append(f"| {k} | {rp['head_psnr']:.3f} ({dpre['head_psnr']:+.3f}) | {rq['head_psnr']:.3f} ({dpost['head_psnr']:+.3f}) | "
                     f"{dpre['head_ssim']:+.4f} / {dpre['head_nmse']:+.5f} / {dpre['head_hfen']:+.4f} | "
                     f"{dpost['head_ssim']:+.4f} / {dpost['head_nmse']:+.5f} / {dpost['head_hfen']:+.4f} | "
                     f"{mfair[0]:+.3f} / {mfair[1]:+.3f} | {rp['head_psnr'] - ZS['pre']:+.3f} / {rq['head_psnr'] - ZS['post']:+.3f} | "
                     + (f"{e['vessels']:+.3f} / {e['plexus']:+.3f}" if e else "n/a") + f" | {verdict} |")
    hdr = ("| probe | pre PSNR (Δ i1) | post PSNR (Δ i1) | pre ΔSSIM / ΔNMSE / ΔHFEN | post ΔSSIM / ΔNMSE / ΔHFEN | "
           "vs fair single | vs ZS-SSDU | enh. err vessels / plexus | reading |\n|---|---|---|---|---|---|---|---|---|")
    text = hdr + "\n" + "\n".join(lines)
    print(text)
    (R / "logs/ar").mkdir(exist_ok=True)
    (R / "logs/ar/ledger_phase6.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
