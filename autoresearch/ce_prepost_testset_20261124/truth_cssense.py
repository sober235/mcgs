#!/usr/bin/env python3
"""truth_cssense.py -- the complete image (truth) of one test subject: CS-SENSE of all acquired data, both scans.

The recipe of the record subject (Results_20260908/CE_pre_post_contrast), run by that topic's own code:
  * ce_tv_sense.py (ADMM, 200 outer iterations, 8 CG steps, mu 0.1, tik 0.01) and ce_common.py are copied to
    <subject>/truth/scripts with ONE line changed (TOPIC -> <subject>/truth), because ce_common reads the
    normalisation from TOPIC/classical/cssense_<scan>_meta.json; the sha256 of the originals is recorded.
  * normalisation = P99.9 of |ZF|, ZF = sum_c conj(S_c) ifft3c(y_c), every 7th voxel (ce_classical.py lines 93-94),
    computed here for this subject's scan.
  * 95 / 5 split of the acquired lines, seed 0 (ce_common.make_split, as ce_split.py).
  * lambda per scan: lowest held-out residual over 2.5e-4 ... 4e-3 (--mask fit); a minimum on an end of the tested
    range extends it by a factor 2 (at most 3 times per side) (ce_select_lambda.py); then --mask full at that lambda.
No truth or other reference enters. Existing results are kept (ce_tv_sense skips finished metas).

--csm_crop T: coil maps zeroed where the ESPIRiT eigenvalue map (ev.npy of prep_cases.py) is <= T, written once to
<subject>/work_<scan>_crop<T>/ (k-space and mask linked). Without it the uncropped maps of work_<scan> are used.

  gpu.py run --gpu ID -- @python autoresearch/ce_prepost_testset_20261124/truth_cssense.py --subject S --gpu ID
(ce_tv_sense.py sets CUDA_VISIBLE_DEVICES itself, so --gpu is the physical card.)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--subject", required=True)
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--csm_crop", type=float, default=None)
ap.add_argument("--scans", default="pre,post")
ap.add_argument("--outer", type=int, default=200)
a = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

import numpy as np  # noqa: E402
import torch  # noqa: E402

SRC = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/scripts")
OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_prepost_testset_20261124")
ART = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
GRID = [2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3]
MAX_EXT = 3
SUB = OUT / "subjects" / a.subject
T = SUB / ("truth" if a.csm_crop is None else f"truth_crop{a.csm_crop:g}")
log = lambda *x: print(*x, flush=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
scans = a.scans.split(",")


def workdir(scan):
    w = SUB / f"work_{scan}"
    assert (w / "prep.json").is_file(), w
    if a.csm_crop is None:
        return w
    wc = SUB / f"work_{scan}_crop{a.csm_crop:g}"
    if (wc / "crop.json").is_file():
        return wc
    assert not wc.exists(), f"{wc} incomplete; decide by hand"
    wc.mkdir()
    for f in ("ksp_full.npy", "mask2d.npy"):
        os.symlink(w / f, wc / f)
    keep = np.load(w / "ev.npy") > a.csm_crop
    s = np.load(w / "csm.npy", mmap_mode="r")
    o = np.lib.format.open_memmap(wc / "csm.npy", mode="w+", dtype=np.complex64, shape=s.shape)
    for c in range(s.shape[0]):
        o[c] = np.asarray(s[c]) * keep
    o.flush()
    (wc / "crop.json").write_text(json.dumps(dict(threshold=a.csm_crop, kept_fraction=float(keep.mean()),
                                                  source=str(w), sha256_csm=sha(wc / "csm.npy")), indent=2))
    log(f"[{scan}] maps cropped at eigenvalue {a.csm_crop:g}: kept {keep.mean():.4f} of the volume")
    return wc


# ---- per-subject copy of the two scripts, TOPIC redirected ----
(T / "scripts").mkdir(parents=True, exist_ok=True)
orig = {}
for f in ("ce_common.py", "ce_tv_sense.py"):
    txt = (SRC / f).read_text()
    orig[f] = sha(SRC / f)
    if f == "ce_common.py":
        old = 'TOPIC = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast")'
        assert txt.count(old) == 1
        txt = txt.replace(old, f'TOPIC = Path("{T}")')
    dst = T / "scripts" / f
    if dst.exists():
        assert dst.read_text() == txt, f"{dst} differs from the expected copy"
    else:
        dst.write_text(txt)
W = {s: workdir(s) for s in scans}
env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CE_SPLIT=str(T / "heldout_split" / "split_95_5_seed0.npz"),
           **{f"CE_WORK_{s.upper()}": str(W[s]) for s in scans})
os.environ.update({k: env[k] for k in env if k.startswith("CE_")})
sys.path.insert(0, str(T / "scripts"))
import ce_common as C  # noqa: E402

assert C.TOPIC == T and all(C.WORK[s] == W[s] for s in scans)
dev = torch.device("cuda:0")

# ---- split and normalisation ----
info = dict(subject=a.subject, csm_crop=a.csm_crop, originals_sha256=orig, scans={})
for s in scans:
    y, S, m2 = C.load_scan(s, dev)
    if not C.SPLIT.exists():
        fit, ho, nl, nho = C.make_split(m2.cpu())
        C.SPLIT.parent.mkdir(parents=True, exist_ok=True)
        np.savez(C.SPLIT, fit=fit.numpy(), ho=ho.numpy(), lines=nl, heldout_lines=nho, seed=0, holdout=0.05)
        log(f"split: {nl} lines, {nl - nho} fit, {nho} held out")
    d = np.load(C.SPLIT)
    assert ((d["fit"] | d["ho"]) == m2.cpu().numpy()).all() and not (d["fit"] & d["ho"]).any(), "split does not match this scan's mask"
    mp = T / "classical" / f"cssense_{s}_meta.json"
    if not mp.exists():
        zf = torch.zeros(y.shape[1:], dtype=y.dtype, device=dev)
        for i in range(0, y.shape[0], 14):
            zf += (S[i:i + 14].conj() * C.ifft3c(y[i:i + 14] * m2)).sum(0)
        scale = torch.quantile(zf.abs().flatten().float()[::7], 0.999).item()
        C.dump_json(dict(scale=scale, rule="P99.9 of |ZF|, every 7th voxel (ce_classical.py)", work=str(W[s])), mp)
    info["scans"][s] = dict(work=str(W[s]), coils=int(y.shape[0]), lines=int(m2.sum()), scale=C.classical_scale(s))
    log(f"[{s}] coils {y.shape[0]}, lines {int(m2.sum())}, scale {C.classical_scale(s):.4e}")
    del y, S, m2
    torch.cuda.empty_cache()
torch.cuda.empty_cache()
R = T / "tv_sense"


def run(scan, mask, lams):
    subprocess.run([PY, "-u", str(T / "scripts" / "ce_tv_sense.py"), "--mode", "cs", "--scan", scan, "--mask", mask,
                    "--lams", ",".join(f"{l:g}" for l in lams), "--outer", str(a.outer), "--gpu", str(a.gpu),
                    "--out", str(R)], env=env, cwd=str(T), check=True)


def heldout(scan, lam):
    return json.load(open(R / f"cssense_o{a.outer}_lam{lam:g}_fit_{scan}_meta.json"))["kres_heldout"]


for s in scans:
    lams = list(GRID)
    run(s, "fit", lams)
    ext = {"low": 0, "high": 0}
    while True:
        lams.sort()
        rows = {l: heldout(s, l) for l in lams}
        best = min(rows, key=rows.get)
        side = "low" if best == lams[0] else ("high" if best == lams[-1] else None)
        if side is None or ext[side] >= MAX_EXT:
            break
        new = float(f"{(best / 2 if side == 'low' else best * 2):.3g}")
        ext[side] += 1
        log(f"[{s}] minimum at the {side} end ({best:g}); adding lambda {new:g}")
        run(s, "fit", [new])
        lams.append(new)
    run(s, "full", [best])
    full = json.load(open(R / f"cssense_o{a.outer}_lam{best:g}_full_{s}_meta.json"))
    info["scans"][s].update(lam=best, at_edge=side, heldout={f"{l:g}": v for l, v in rows.items()},
                            kres_full=full["kres"], alpha_abs=float(abs(complex(full["alpha_re"], full["alpha_im"]))),
                            truth=str(R / f"cssense_o{a.outer}_lam{best:g}_full_{s}_complex.npy"))
    log(f"TRUTH {a.subject} {s}: held-out " + "  ".join(f"{l:g}:{v:.5f}" for l, v in rows.items())
        + f"  -> lambda {best:g}; refit on all lines kres {full['kres']:.5f}")
C.dump_json(info, T / "truth_selection.json")
C.dump_json(info, ART / f"truth_selection_{a.subject}.json")
log("DONE")
