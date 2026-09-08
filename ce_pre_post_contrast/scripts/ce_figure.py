#!/usr/bin/env python3
"""ce_figure.py -- horizontal comparison on the UIH prospective CE pair.

Columns : ZF(reference anchor) | CG-SENSE | CS-SENSE | ZS-SSDU | GS single | GS joint
Rows    : PRE image / PRE error-vs-y  /  POST image / POST error-vs-y

There is NO fully sampled ground truth for this prospectively undersampled acquisition,
so the error map is the image-domain footprint of the k-space data-consistency residual
against the MEASURED signal y:

    alpha = argmin_a || M .( a.A(x) - y ) ||_2          (best-fit complex scale)
    r     = M .( alpha.A(x) - y )                        (on-mask residual, 42 coils)
    e     = | sum_c conj(S_c) . ifft3c(r_c) |            (SENSE-combined residual image)

Every reconstruction is exported PRE hard-data-consistency; with hard DC the on-mask
residual would be identically zero and every one of these panels would be black.
"""
import os, sys, json, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=7)
ap.add_argument("--slice", type=int, default=170)   # mid-brain axial level (basal ganglia / ventricles)
ap.add_argument("--out", default="/home/congcongliu/figs/ce_pair")
args = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np, torch, torch.fft as tfft
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    try: font_manager.fontManager.addfont(_f)
    except Exception: pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

dev = torch.device("cuda:0")
W = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast")
RR = Path("/data0/congcong/code/NV-Generate-CTMR/recon_real")
WORK = {"pre": RR / "work_ce_t1_16887049370010", "post": RR / "work_ce_t1_16887049370060"}
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)

def _c(fn, x, d): return tfft.fftshift(fn(tfft.ifftshift(x, dim=d), dim=d, norm="ortho"), dim=d)
def fft3c(x):  return _c(tfft.fftn,  x, (-3, -2, -1))
def ifft3c(x): return _c(tfft.ifftn, x, (-3, -2, -1))

# column spec: (label, path-template or None for ZF)
COLS = [
    ("ZF\n(reference anchor)",      lambda s: W / f"classical/zf_{s}_complex.npy"),
    ("CG-SENSE",                    lambda s: W / f"classical/cgsense_{s}_complex.npy"),
    ("CS-SENSE\n(ADMM-TV)",         lambda s: W / f"classical/cssense_{s}_complex.npy"),
    ("ZS-SSDU",                     lambda s: W / f"classical/zsssdu_{s}_complex.npy"),
    ("3D-DIP\n(ConvDecoder)",       lambda s: W / f"dipinr/dip_{s}_complex.npy"),
    ("3D-INR\n(hash coord-MLP)",    lambda s: W / f"dipinr/inr_{s}_complex.npy"),
    ("GS single\n(each scan alone)", lambda s: W / f"gs_single_{s}/gs_single_{s}_raw_complex.npy"),
    ("GS joint\n(shared geometry)", lambda s: W / f"gs_joint/gs_joint_{s}_raw_complex.npy"),
]
CHUNK = 6
res = {}
for scan in ("pre", "post"):
    log(f"=== {scan} ===")
    y = torch.from_numpy(np.load(WORK[scan] / "ksp_full.npy")).to(dev)
    S = torch.from_numpy(np.load(WORK[scan] / "csm.npy")).to(dev)
    m3 = torch.from_numpy(np.load(WORK[scan] / "mask2d.npy")).to(dev) > 0.5
    y = y * m3
    NC = y.shape[0]; G = tuple(m3.shape)

    def AH_k(k):
        acc = torch.zeros(G, dtype=torch.complex64, device=dev)
        for i in range(0, NC, CHUNK):
            acc += (S[i:i+CHUNK].conj() * ifft3c(k[i:i+CHUNK] * m3)).sum(0)
        return acc
    zf = AH_k(y)
    ref_peak = torch.quantile(zf.abs().flatten().float()[::7], 0.999).item()
    log(f"  ref_peak (P99.9 |ZF|) = {ref_peak:.4e}")

    for label, pth in COLS:
        p = pth(scan)
        if not p.exists():
            log(f"  MISSING {p}"); continue
        x = torch.from_numpy(np.load(p)).to(dev)
        if tuple(x.shape) != G:                       # GS renders in (X, Z, Y)
            assert tuple(x.shape) == (G[0], G[2], G[1]), f"{p}: shape {tuple(x.shape)} vs {G}"
            x = x.permute(0, 2, 1).contiguous()
        num = torch.zeros((), dtype=torch.complex64, device=dev)
        den = torch.zeros((), device=dev)
        for i in range(0, NC, CHUNK):
            km = fft3c(S[i:i+CHUNK] * x) * m3
            num = num + (km.conj() * y[i:i+CHUNK]).sum(); den = den + km.abs().square().sum()
        alpha = num / den.clamp_min(1e-30)
        e = torch.zeros(G, dtype=torch.complex64, device=dev)
        nr = 0.0; dr = 0.0
        for i in range(0, NC, CHUNK):
            km = fft3c(S[i:i+CHUNK] * x) * m3
            r = (alpha * km - y[i:i+CHUNK]) * m3
            e += (S[i:i+CHUNK].conj() * ifft3c(r)).sum(0)
            nr += float(torch.linalg.vector_norm(r).item() ** 2)
            dr += float(torch.linalg.vector_norm(y[i:i+CHUNK]).item() ** 2)
        kres = (nr / dr) ** 0.5
        kho = None
        mp = p.with_name(p.name.replace("_complex.npy", "_meta.json"))
        if mp.exists():
            kho = json.load(open(mp)).get("kres_heldout")
        res[(scan, label)] = dict(
            img=((alpha * x).abs() / ref_peak).cpu().numpy(),
            err=(e.abs() / ref_peak).cpu().numpy(),
            kres=kres, kho=kho)
        log(f"  {label.splitlines()[0]:<12s} kres={kres:.5f}  |alpha|={abs(alpha.item()):.4f}")
    del y, S, m3
    torch.cuda.empty_cache()

json.dump({f"{s}|{l.splitlines()[0]}": v["kres"] for (s, l), v in res.items()},
          open(W / "kres_summary.json", "w"), indent=2)

# ---------------- slice pick ----------------------------------------------------------
have = [l for l, _ in COLS if ("pre", l) in res and ("post", l) in res]
assert have, "nothing to plot"
key = next((l for l in have if l.startswith("CG-SENSE")), have[0])
if args.slice >= 0:
    sl = args.slice
else:                                   # fallback: strongest post-pre enhancement, brain levels only
    a = res[("pre", key)]["img"]; b = res[("post", key)]["img"]
    d = np.clip(b - a, 0, None) * (a > 0.08)
    prof = d.reshape(d.shape[0], -1).sum(1)
    lo, hi = int(0.55 * d.shape[0]), int(0.90 * d.shape[0])
    sl = lo + int(prof[lo:hi].argmax())
log(f"slice = {sl} / {res[('pre', key)]['img'].shape[0]}")

ROWS = [("pre",  "img", "平扫 PRE"),
        ("pre",  "err", "平扫  |error vs y|"),
        ("post", "img", "增强 POST"),
        ("post", "err", "增强  |error vs y|")]

def draw(vmax_err, fname, note):
    ncol = len(COLS)
    fig, axes = plt.subplots(4, ncol, figsize=(2.35 * ncol, 4 * 2.85))
    h = None
    for ri, (scan, kind, rlab) in enumerate(ROWS):
        for ci, (label, _) in enumerate(COLS):
            ax = axes[ri, ci]; ax.set_xticks([]); ax.set_yticks([])
            if (scan, label) not in res:
                ax.text(.5, .5, "n/a", ha="center", va="center", transform=ax.transAxes); continue
            d = res[(scan, label)]
            im = np.rot90(d[kind][sl])
            if kind == "img":
                ax.imshow(im, cmap="gray", vmin=0, vmax=0.9)
            else:
                h = ax.imshow(im, cmap="jet", vmin=0, vmax=vmax_err)
                ax.set_xlabel(f"kres = {d['kres']:.4f}" + (f"\nheld-out {d['kho']:.4f}" if d.get("kho") else ""), fontsize=8, labelpad=2)
            if ri == 0: ax.set_title(label, fontsize=10)
            if ci == 0: ax.set_ylabel(rlab, fontsize=11)
    fig.suptitle("UIH 3D T1 GRE 平扫/增强配对 — 前瞻欠采 AF 5.33，无全采 GT\n"
                 "error map = |对测量信号 y 的 k 空间残差的 SENSE 合成图|"
                 f"   (jet, 固定 0–{vmax_err:g}{note})", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    cax = fig.add_axes([0.28, 0.012, 0.44, 0.011])
    fig.colorbar(h, cax=cax, orientation="horizontal").set_label("|error| / ref peak", fontsize=9)
    q = OUT / fname
    fig.savefig(q, dpi=150, bbox_inches="tight"); plt.close(fig)
    log(f"saved {q}")

draw(0.10, "ce_pair_methods_errormap.png", "")
draw(0.02, "ce_pair_methods_errormap_zoom.png", "，5× 放大")

log("kres table")
log(f"{'method':<26s} {'pre':>9s} {'post':>9s} {'pre_ho':>9s} {'post_ho':>9s}")
for label, _ in COLS:
    if ("pre", label) in res:
        a, b = res[('pre', label)], res[('post', label)]
        f = lambda v: f"{v:9.5f}" if v else "        -"
        log(f"{label.replace(chr(10), ' '):<26s} {a['kres']:9.5f} {b['kres']:9.5f} {f(a.get('kho'))} {f(b.get('kho'))}")
