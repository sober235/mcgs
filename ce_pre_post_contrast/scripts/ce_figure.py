#!/usr/bin/env python3
"""ce_figure.py -- horizontal comparison on the UIH prospective CE pair.

Columns : ZF | CG-SENSE | CS-SENSE | Joint SENSE | L1-ESPIRiT | ZS-SSDU | 3D-DIP | 3D-INR | MC-INR | GS-single | MC-GS
          (11 columns since 2026-09-15; the 09-08 8-column version is kept as *_8col_20260908.png)
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
ap.add_argument("--gpu", type=int, default=7, help="-1 = CPU")
ap.add_argument("--slice", type=int, default=170)   # mid-brain axial level (basal ganglia / ventricles)
ap.add_argument("--out", default="/home/congcongliu/figs/ce_pair")
ap.add_argument("--cache", default="/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/figures/cache/errormap_res.npz",
                help="images, error maps (float16) and kres; reused while every input volume is unchanged")
ap.add_argument("--recompute", action="store_true")
ap.add_argument("--drop", nargs="*", default=["L1-ESPIRiT"],
                help="column labels to leave out; default drops L1-ESPIRiT (user, 2026-09-15: L1-SPIRiT is the L1 arm). '--drop' with no labels renders every column")
ap.add_argument("--suffix", default="", help="appended to the output file names")
args = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu >= 0 else ""
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

dev = torch.device("cuda:0" if args.gpu >= 0 else "cpu")
if args.gpu < 0:
    torch.set_num_threads(16)
W = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast")
RR = Path("/data0/congcong/code/NV-Generate-CTMR/recon_real")
WORK = {"pre": RR / "work_ce_t1_16887049370010", "post": RR / "work_ce_t1_16887049370060"}
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)

def _c(fn, x, d): return tfft.fftshift(fn(tfft.ifftshift(x, dim=d), dim=d, norm="ortho"), dim=d)
def fft3c(x):  return _c(tfft.fftn,  x, (-3, -2, -1))
def ifft3c(x): return _c(tfft.ifftn, x, (-3, -2, -1))

# column spec: (label, path-template). Lambdas of the 09-14 arms come from the held-out selection.
SEL = {fam: json.load(open(W / "logs/ext_20260914" / f"selection_{fam}.json"))["best"]
       for fam in ("cs_pre", "cs_post", "joint", "l1_pre", "l1_post")}
COLS = [
    ("ZF",                          lambda s: W / f"classical/zf_{s}_complex.npy"),
    ("CG-SENSE",                    lambda s: W / f"classical/cgsense_{s}_complex.npy"),
    ("CS-SENSE",                    lambda s: W / f"tv_sense/cssense_o200_lam{SEL['cs_' + s]:g}_full_{s}_complex.npy"),
    ("Joint SENSE",                 lambda s: W / f"tv_sense/jsense_o200_lam{SEL['joint']:g}_full_{s}_complex.npy"),
    ("L1-ESPIRiT",                  lambda s: W / f"l1espirit/l1espirit_i100_lam{SEL['l1_' + s]:g}_full_{s}_complex.npy"),
    ("ZS-SSDU",                     lambda s: W / f"classical/zsssdu_{s}_complex.npy"),
    ("3D-DIP",                      lambda s: W / f"dipinr/dip_{s}_complex.npy"),
    ("3D-INR",                      lambda s: W / f"dipinr/inr_{s}_complex.npy"),
    ("MC-INR",                      lambda s: W / f"mcinr/mcinr_full_{s}_complex.npy"),
    ("GS-single",                   lambda s: W / f"gs_single_{s}/gs_single_{s}_raw_complex.npy"),
    ("MC-GS",                       lambda s: W / f"gs_joint/gs_joint_{s}_raw_complex.npy"),
]
# L1-SPIRiT (2026-09-15): inserted after L1-ESPIRiT once its held-out selection exists (ce_orchestrate.py plan spirit)
SP = {sc: json.load(open(W / "logs/ext_20260914" / f"selection_spirit_{sc}.json")) for sc in ("pre", "post")
      if (W / "logs/ext_20260914" / f"selection_spirit_{sc}.json").exists()}
if len(SP) == 2:
    COLS.insert([l for l, _ in COLS].index("L1-ESPIRiT") + 1,
                ("L1-SPIRiT", lambda s: W / f"spirit/spirit_ks{SP[s]['ksize']}_i150_lam{SP[s]['lam']:g}_full_{s}_complex.npy"))
COLS = [c for c in COLS if c[0] not in set(args.drop)]
# held-out kres shown under each error map: only numbers whose fit never saw the 383 held-out lines
# (kres_summary_20260914.json; heldout4_scores.json once the 95% runs of the other four columns exist)
HO = {}
for r in json.load(open(W / "kres_summary_20260914.json"))["rows"]:
    if r["heldout_pre"] is not None:
        HO[("pre", r["method"])], HO[("post", r["method"])] = r["heldout_pre"], r["heldout_post"]
if (W / "heldout4_scores.json").exists():
    for label, per in json.load(open(W / "heldout4_scores.json")).items():
        for sc, v in per.items():
            HO[(sc, label)] = v["kres_heldout"]
for sc, v in SP.items():
    HO[(sc, "L1-SPIRiT")] = json.load(open(W / f"spirit/spirit_ks{v['ksize']}_i150_lam{v['lam']:g}_fit_{sc}_meta.json"))["kres_heldout"]
CHUNK = 6
cache = Path(args.cache)
colsig = lambda sc, pth: json.dumps([str(pth(sc)), os.path.getmtime(pth(sc))])
stored = {}
if cache.exists() and not args.recompute:
    with np.load(cache) as c:
        stored = {k: c[k] for k in c.files if k not in ("sig", "meta")}
res = {}
for sc in ("pre", "post"):
    for label, pth in COLS:
        k = f"{sc}|{label}"
        if k + "|img" in stored and "sig|" + k in stored and str(stored["sig|" + k]) == colsig(sc, pth):
            res[(sc, label)] = dict(img=stored[k + "|img"].astype(np.float32), err=stored[k + "|err"].astype(np.float32),
                                    kres=float(stored["kres|" + k]), kho=HO.get((sc, label)))
todo = {sc: [(l, pth) for l, pth in COLS if (sc, l) not in res] for sc in ("pre", "post")}
log(f"cache {cache.name}: {2 * len(COLS) - sum(map(len, todo.values()))} of {2 * len(COLS)} panels reused")
for scan in [sc for sc in ("pre", "post") if todo[sc]]:
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

    for label, pth in todo[scan]:
        p = pth(scan)
        if not p.exists():
            log(f"  MISSING {p}"); continue
        x = torch.from_numpy(np.load(p)).to(dev)
        if tuple(x.shape) != G:                       # GS renders in (X, Z, Y)
            assert tuple(x.shape) == (G[0], G[2], G[1]), f"{p}: shape {tuple(x.shape)} vs {G}"
            x = x.permute(0, 2, 1).contiguous()
        # sums in float64 (float32 accumulation over ~4e8 samples shifts kres by ~2e-4 on CPU; 2026-09-15)
        num = 0j; den = 0.0
        for i in range(0, NC, CHUNK):
            km = (fft3c(S[i:i+CHUNK] * x) * m3).to(torch.complex128)
            yy = y[i:i+CHUNK].to(torch.complex128)
            num += complex((km.conj() * yy).sum().item()); den += float(km.abs().square().sum().item())
        alpha = torch.tensor(num / max(den, 1e-30), dtype=torch.complex64, device=dev)
        e = torch.zeros(G, dtype=torch.complex64, device=dev)
        nr = 0.0; dr = 0.0
        for i in range(0, NC, CHUNK):
            km = fft3c(S[i:i+CHUNK] * x) * m3
            r = (alpha * km - y[i:i+CHUNK]) * m3
            e += (S[i:i+CHUNK].conj() * ifft3c(r)).sum(0)
            nr += float(r.to(torch.complex128).abs().square().sum().item())
            dr += float(y[i:i+CHUNK].to(torch.complex128).abs().square().sum().item())
        kres = (nr / dr) ** 0.5
        kho = HO.get((scan, label))
        res[(scan, label)] = dict(
            img=((alpha * x).abs() / ref_peak).cpu().numpy(),
            err=(e.abs() / ref_peak).cpu().numpy(),
            kres=kres, kho=kho)
        log(f"  {label.splitlines()[0]:<12s} kres={kres:.5f}  |alpha|={abs(alpha.item()):.4f}")
    del y, S, m3
    torch.cuda.empty_cache()

if any(todo.values()):
    cache.parent.mkdir(parents=True, exist_ok=True)
    allsig = dict(COLS)
    for (sc, l), v in res.items():
        k = f"{sc}|{l}"
        stored[k + "|img"], stored[k + "|err"] = v["img"].astype(np.float16), v["err"].astype(np.float16)
        stored["kres|" + k], stored["sig|" + k] = np.array(v["kres"]), np.array(colsig(sc, allsig[l]))
    np.savez(cache, **stored)
    log(f"cached to {cache}")
json.dump({f"{s}|{l.splitlines()[0]}": v["kres"] for (s, l), v in res.items()},
          open(W / "kres_summary_errormap.json", "w"), indent=2)   # kres_summary.json keeps the 09-08 values

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
            im = d[kind][sl]                                  # (A-P, L-R): anterior (eyes) up (2026-09-15, was rot90)
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

draw(0.10, f"ce_pair_methods_errormap{args.suffix}.png", "")
draw(0.02, f"ce_pair_methods_errormap_zoom{args.suffix}.png", "，5× 放大")

log("kres table")
log(f"{'method':<26s} {'pre':>9s} {'post':>9s} {'pre_ho':>9s} {'post_ho':>9s}")
for label, _ in COLS:
    if ("pre", label) in res:
        a, b = res[('pre', label)], res[('post', label)]
        f = lambda v: f"{v:9.5f}" if v else "        -"
        log(f"{label.replace(chr(10), ' '):<26s} {a['kres']:9.5f} {b['kres']:9.5f} {f(a.get('kho'))} {f(b.get('kho'))}")
