#!/usr/bin/env python3
"""fig_pipeline_diagram.py -- structure diagram of THIS run's reconstruction pipeline.

Differs on purpose from the repo's docs/forward_graph.png, which draws an earlier variant:
that figure shows a FLAIR-private residual branch and isotropic TV plus a T1-guided dtv term.
Here the contrast channel is T2, the private-residual branch is OFF (num_private_points = 0),
lambda_dtv = 0, and the regulariser is edge-weighted TV with a Charbonnier eps.

Figure conventions: ASCII math only (no exotic Unicode -> no tofu boxes), Chinese in Noto CJK.
Layout is in points on a fixed-size canvas so a box never has to guess how tall its text is.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

R = Path("/data0/congcong/code/GS/Results_20260908/T1T2_AF12_retro")
F = Path("/home/congcongliu/figs/t1t2_af12")
F.mkdir(parents=True, exist_ok=True)

C = {"data": "#4C72B0", "geom": "#3F7F4F", "t1": "#4C9F70", "t2": "#C44E52",
     "op": "#7D6BA8", "loss": "#D2691E", "phys": "#8A6D52", "grey": "#585858"}

W, H = 2000.0, 1150.0          # canvas in "points"; 1 unit = 1 pt at dpi 100 with figsize W/100
fig = plt.figure(figsize=(W / 100, H / 100))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ax.set_facecolor("white")

TT, TB, LS = 15, 12.0, 1.5      # title size, body size, line spacing


def box(x, y, w, h, title, body, color, fs_t=TT, fs_b=TB, tc="white"):
    """(x, y) is the TOP-LEFT corner. Title then body, block-centred in the box."""
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=3,rounding_size=10",
                                fc=color, ec=color, lw=1.0, zorder=2))
    lines = body.split("\n") if body else []
    hb = len(lines) * fs_b * LS
    block = fs_t * 1.5 + (8 + hb if lines else 0)
    top = y - (h - block) / 2.0
    ax.text(x + w / 2, top - fs_t * 1.1, title, ha="center", va="center",
            fontsize=fs_t, color=tc, fontweight="bold", zorder=3)
    if lines:
        ax.text(x + w / 2, top - fs_t * 1.5 - 8, body, ha="center", va="top",
                fontsize=fs_b, color=tc, zorder=3, linespacing=LS)


def arrow(p0, p1, color="#333333", ls="-", lw=2.0, rad=0.0, z=1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", linestyle=ls, lw=lw, color=color,
                                 mutation_scale=17, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3))


ax.text(W / 2, H - 30, "MC-GS joint reconstruction of a 3D T1 + T2 pair  --  retrospective AF 12",
        ha="center", va="center", fontsize=25, fontweight="bold")
ax.text(W / 2, H - 62, "vol1_huangyuwei,  256 x 256 x 176,  26 coils   |   shared Gaussian geometry, "
                       "per-primitive dual complex density   |   one mask for both contrasts",
        ha="center", va="center", fontsize=14, color="#444444")

# ------------------------------------------------------------------ row 1: data preparation
y1, h1 = H - 95, 165
xs = [30, 425, 820, 1215, 1610]
bw = 360
box(xs[0], y1, bw, h1, "raw data  (BART CFL)",
    "MC_Jia/Train_SpiritDiffusion_CFL\nvol1_huangyuwei,  t1_fsp + t2\n(256, 256, 176, 26) complex64\n"
    "readout = axis0,  PE = (ky, kz)\nDC exactly on the array centre", C["data"])
box(xs[1], y1, bw, h1, "ESPIRiT maps  S_m",
    "bart ecalib -t 0.001 -c 0.0\n            -r 24 -m 1\nSoS == 1 over the whole FOV\n"
    "(expected for -c 0.0, not a bug)\n38 min + 45 min, CPU only", C["data"])
box(xs[2], y1, bw, h1, "ground truth  x_GT",
    "built from the FULL k-space\nx_GT = sum_c conj(S_c) ifft3c(y_c)\n\n"
    "the same maps define the forward\noperator: self-consistent gauge", C["data"])
box(xs[3], y1, bw, h1, "retrospective mask  M",
    "vdPoisMex 256 x 176, ONE mask\naccelx 3 / accely 4 / calib 24\n3847 / 45056 = 8.538 %\n"
    "AF 11.71 over the full PE plane\nreadout fully sampled", C["data"])
box(xs[4], y1, bw, h1, "measured data  y_m",
    "y_m = M . F3c . (S_m . x_GT)\n\nT1  AF 11.71 vs its own support\nT2  AF  9.89 vs its own support\n"
    "(T2 support = 77.71 % ellipse)", C["data"])
for i in range(4):
    arrow((xs[i] + bw, y1 - h1 / 2), (xs[i + 1], y1 - h1 / 2))

# ------------------------------------------------------------------ row 2: the model
y2 = y1 - h1 - 74
ax.text(W / 2, y2 + 30, "both data-consistency terms back-propagate into the SAME geometry  ->  "
                        "T2's lines constrain T1's null space and vice versa",
        ha="center", va="center", fontsize=14.5, style="italic", color="#222222")

hg = 266
box(30, y2, 340, hg, "shared Gaussian geometry  G",
    "{ mu_n ,  Sigma_n = R(q_n) diag(s_n^2) R^T }\n\n600 000 primitives, fresh init\n"
    "(the 576902 dual-warm cloud is a\ndifferent FOV and does not register)\n\n"
    "ONE geometry for BOTH contrasts\n= the shared-anatomy prior", C["geom"])

hr = 105
rows = [(y2 - 10, "T1", C["t1"]), (y2 - 161, "T2", C["t2"])]
cols = [(430, 330, "density  alpha_n^(%s)", ["per-primitive complex amplitude",
                                             "independent of the T1 amplitude"]),
        (800, 330, "render  x_%s", ["x_m = sum_n alpha_n^(m) g_n",
                                    "tile voxelizer, max_radius 20"]),
        (1170, 330, "forward  A_%s", ["A_m = M_m . F3c . (S_m . x_m)",
                                      "T2 also carries its shutter"]),
        (1540, 430, "DC loss  L_%s", ["|| A_m x_m - y_m ||^2, frequency weighted",
                                      "centre 0.3 / mid 1.0 / high 2.2"])]
for ri, (yr, tag, col) in enumerate(rows):
    for ci, (x, w, tfmt, bodies) in enumerate(cols):
        box(x, yr, w, hr, tfmt % tag, bodies[ri], col, fs_t=14, fs_b=11)
    for ci in range(len(cols) - 1):
        x0 = cols[ci][0] + cols[ci][1]; x1 = cols[ci + 1][0]
        arrow((x0, yr - hr / 2), (x1, yr - hr / 2), color=col, lw=1.8)
    arrow((370, y2 - hg / 2 + (40 if ri == 0 else -40)), (430, yr - hr / 2),
          color=C["geom"], lw=1.8, rad=0.12 if ri == 0 else -0.12)

# gradient path from both losses back into the shared geometry, routed through the corridor
# between the two contrast rows so it never crosses a box
yc = y2 - 138
for xd in (1755,):
    arrow((xd, y2 - 115), (xd, yc), color="#111111", ls=(0, (5, 4)), lw=1.8)
    arrow((xd, y2 - 161), (xd, yc), color="#111111", ls=(0, (5, 4)), lw=1.8)
arrow((1755, yc), (372, yc), color="#111111", ls=(0, (7, 5)), lw=2.4)
ax.text(1040, yc + 5, "d(L_T1 + L_T2) / d(mu, Sigma)", ha="center", va="bottom",
        fontsize=12.5, color="#111111", style="italic",
        bbox=dict(fc="white", ec="none", pad=1.5))

# ------------------------------------------------------------------ row 3: physics / reg / scoring
y3 = y2 - hg - 44
h3 = 248
box(30, y3, 620, h3, "measured-zero region  Z    (T2 only)",
    "The elliptical vendor shutter is a MEASURED zero, not missing data.\n"
    "Z is fitted from the ACQUIRED lines alone; the fit then uses\n"
    "mask_fit = M or Z, with y set to 0 on Z.\n\n"
    "estimated ellipse = 21.51 % of the PE plane (measured shutter 22.29 %)\n"
    "294 / 306 sampled zero lines explained,  0 nonzero lines swallowed\n"
    "T1 has no shutter, so known_zero is off there; prewhiten off on both.",
    C["phys"], fs_t=15, fs_b=11.5)

box(690, y3, 590, h3, "regulariser  R_geo    (frozen working point)",
    "edge-weighted TV with a Charbonnier eps\n\n"
    "w   = clamp( 1 / (1 + (|grad u|_sg / kappa)^2), w_floor, 1 )\n"
    "R   = lambda_geo * sum_axes  w * sqrt(g^2 + eps^2)\n\n"
    "lambda_geo 0.10   eps 0.04   kappa_q 0.85   w_floor 0.10\n"
    "warmup 500   4000 iters   seed 42\n"
    "lambda_dtv = 0,  no private-residual branch",
    C["grey"], fs_t=15, fs_b=11.5)

box(1320, y3, 650, h3, "scoring    (one gauge for every method)",
    "hard data consistency, then conjugate-SENSE combine\n"
    "  k_dc = mask_fit . y_dc + (1 - mask_fit) . A(x)\n"
    "  x_dc = sum_c ifft3c(k_dc) . conj(S_c)\n\n"
    "whole@max   PSNR / SSIM / NMSE / HFEN\n"
    "head-only   |GT| > 0.05 * P99.9(|GT|)\n"
    "plus sharp, grain, e_unobs, and the ZF anchor",
    C["grey"], fs_t=15, fs_b=11.5)

arrow((200, y3), (200, y2 - hg), color=C["phys"], lw=1.8)
arrow((985, y3), (985, y2 - 250), color=C["grey"], lw=1.8)
arrow((1645, y3), (1755, y2 - 248), color=C["grey"], lw=1.8, rad=-0.08)

# ------------------------------------------------------------------ row 4: arms
y4 = y3 - h3 - 46
ax.text(W / 2, y4 + 14, "arms compared on exactly the same data, mask, maps and gauge",
        ha="center", va="center", fontsize=16, fontweight="bold", color="#111111")
arms = [("ZF", "zero-filled SENSE combine", "#8C8C8C"),
        ("CG-SENSE", "Tikhonov normal equations\ntik 0.01,  60 iterations", "#8C8C8C"),
        ("CS-SENSE", "ADMM isotropic 3-D TV\nlam 1e-3, mu 0.1, 35 x 8", "#8C8C8C"),
        ("MC-GS single", "the same contrast in BOTH\nchannels = that scan alone", C["t1"]),
        ("MC-GS joint", "T1 in plain, T2 in contrast\none shared geometry", C["t2"])]
aw, gap = 370, 15
for i, (t, b, col) in enumerate(arms):
    box(30 + i * (aw + gap), y4, aw, 118, t, b, col, fs_t=16, fs_b=11.5)

out = F / "mcgs_t1t2_af12_pipeline.png"
fig.savefig(out, dpi=110, facecolor="white")
fig.savefig(R / "figures/mcgs_t1t2_af12_pipeline.png", dpi=110, facecolor="white")
print("wrote", out)
