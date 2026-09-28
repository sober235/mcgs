"""Three summary charts of the widen loop and the ceiling probes (jiangyizhou T1 + FLAIR), every number read from the
runs' score json (head-only metrics, the scorer the AF topics register).

  A  widen_margin_dumbbell   MC-GS - GS-single per AF: published / fair start / fair after the loop
  B  widen_ceilings          gain over the fair GS-single of every oracle probe at AF 15.19 (T1 target, FLAIR target)
  C  widen_asym_trend        joint gain on FLAIR as FLAIR gets sparser (dPSNR and HFEN change, two charts, one axis each)

Writes PNG (200 dpi) and PDF (TrueType fonts) to figures/widen/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

W = Path("/data0/congcong/code/GS/mcgs_widen_wt/results_arc")
H = Path("/data0/congcong/code/GS/mcgs_hparam_wt/results_arc")
A = Path("/data0/congcong/code/GS/mcgs_af16_wt/results_arc")
A12 = Path("/data0/congcong/code/GS/mcgs_af12_wt/results_arc")      # the AF12 line (its registered arm is P14)
OUT = Path(__file__).resolve().parents[1] / "figures/widen"
OUT.mkdir(parents=True, exist_ok=True)

# reference palette (dataviz skill), validated: blue/orange and blue/red pass every light-mode check
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"


def m(root, exp, ch, key="head_psnr"):
    return json.load(open(root / f"{exp}_score.json"))[ch][key]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def save(fig, name):
    for ext, kw in (("png", dict(dpi=200)), ("pdf", {})):
        fig.savefig(OUT / f"{name}.{ext}", facecolor=SURF, bbox_inches="tight", **kw)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ---------------------------------------------------------------- A: the three margins per AF
AFS = [(12, 11.71), (16, 15.19), (20, 18.56), (24, 24.00)]   # label = PE-plane points / mask points (user convention)
REG = {12: ("p3_e0001_ptv001_full4k", "p3_e0001_ptv001_singleT1_full4k", "p3_e0001_ptv001_singleFLAIR_full4k"),
       16: ("p3_r3_e0001_ptv003_full4k", "p3_r3_singleT1_full4k", "p3_r3_singleFLAIR_full4k"),
       20: ("af20_p3_joint_full4k", "af20_p3_singleT1_full4k", "af20_p3_singleFLAIR_full4k"),
       24: ("af24_p3_joint_full4k", "af24_p3_singleT1_full4k", "af24_p3_singleFLAIR_full4k")}
REG_ROOT = {12: A12, 16: A, 20: A, 24: A}
# fair start = the published recipe with GS-single at lambda_edge 0 (AF12: the P14 recipe, re-run in this tree)
START = {12: ((W, "wd_af12_p14_j"), (W, "wd_af12_p14_s_t1_e0"), (W, "wd_af12_p14_s_fl_e0b")),
         16: ((H, "hp_n36"), (H, "ca_s_t1_edge0"), (H, "ca_s_fl_edge0")),
         20: ((A, "af20_p3_joint_full4k"), (W, "wd_af20_s_t1_e0"), (W, "wd_af20_s_fl_e0_r")),
         24: ((W, "wd_af24_j_e001_r"), (W, "wd_af24_s_t1_e0_r"), (W, "wd_af24_s_fl_e0_r"))}
rows = []
for af, lab in AFS:
    j, st, sf = REG[af]
    RR = REG_ROOT[af]
    pub = (m(RR, j, "plain") - m(RR, st, "plain"), m(RR, j, "contrast") - m(RR, sf, "plain"))
    (rj, ej), (rt, et), (rf, ef) = START[af]
    start = (m(rj, ej, "plain") - m(rt, et, "plain"), m(rj, ej, "contrast") - m(rf, ef, "plain"))
    # fair = the best single run at this AF (rule A); at AF12 the P14-settings single (rotation_lr 0.008) beats the
    # one-recipe single on FLAIR, so it is the baseline there (see scripts/wd_ledger.py, which picks the same)
    best_t1 = max(m(W, e, "plain") for e in [f"wd_v_af{af}_s_t1"] + (["wd_af12_p14_s_t1_e0"] if af == 12 else []))
    best_fl = max(m(W, e, "plain") for e in [f"wd_v_af{af}_s_fl"] + (["wd_af12_p14_s_fl_e0b"] if af == 12 else []))
    now = (m(W, f"wd_v_af{af}_j", "plain") - best_t1, m(W, f"wd_v_af{af}_j", "contrast") - best_fl)
    rows.append((lab, pub, start, now))
    print(f"AF {lab:5.2f}  published {pub[0]:+.3f}/{pub[1]:+.3f}  fair start {start[0]:+.3f}/{start[1]:+.3f}  "
          f"fair now {now[0]:+.3f}/{now[1]:+.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True, facecolor=SURF)
for ci, (ax, cname) in enumerate(zip(axes, ("T1", "FLAIR"))):
    style(ax)
    for xi, (lab, pub, start, now) in enumerate(rows):
        ax.plot([xi, xi], [start[ci], now[ci]], color=BLUE, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([xi], [pub[ci]], marker="D", s=46, color=MUTED, edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.scatter([xi], [start[ci]], marker="o", s=70, facecolor=SURF, edgecolor=BLUE, linewidth=2, zorder=4)
        ax.scatter([xi], [now[ci]], marker="o", s=70, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=5)
        ax.annotate(f"{now[ci]:+.3f}", (xi, now[ci]), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"AF {lab:.2f}" for lab, *_ in rows])
    ax.set_xlim(-0.5, len(rows) - 0.3)
    ax.axhline(0, color=BASE, linewidth=0.8, zorder=1)
    ax.set_title(cname, fontsize=11, color=INK, loc="left")
axes[0].set_ylabel("MC-GS − GS-single, head PSNR (dB)", fontsize=9.5, color=INK2)
axes[0].set_ylim(-0.01, 0.2)
h = [plt.Line2D([], [], marker="D", color=MUTED, linestyle="", markersize=6, label="published (GS-single carries the common-edge prior)"),
     plt.Line2D([], [], marker="o", markerfacecolor=SURF, markeredgecolor=BLUE, markeredgewidth=2, color=BLUE,
                linestyle="", markersize=7, label="fair baseline, published recipe (start)"),
     plt.Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=7, label="fair baseline, after the loop")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.06),
           labelcolor=INK2)
fig.suptitle("Joint margin over single-contrast GS, jiangyizhou T1 + FLAIR", x=0.07, ha="left", fontsize=12, color=INK)
fig.text(0.07, 0.905, "fair = GS-single at λ_edge 0 with its own best per-channel settings (user choices A / (1)); "
         "published recipe = phase 3 at AF 15.19-24, P14 at AF 11.71",
         fontsize=8.5, color=INK2)
fig.subplots_adjust(top=0.8, bottom=0.18, wspace=0.08)
save(fig, "widen_margin_dumbbell")

# ---------------------------------------------------------------- B: ceilings at AF 15.19
S_T1, S_FL = m(W, "wd_v_af16_s_t1", "plain"), m(W, "wd_v_af16_s_fl", "plain")
T1_ITEMS = [("MC-GS, this recipe (FLAIR partner at AF 15.19)", m(W, "wd_v_af16_j", "plain")),
            ("oracle: FLAIR partner fully sampled", m(W, "wd_o_ffull_j", "plain")),
            ("oracle: FLAIR fully sampled, coupling at maximum", m(W, "wd_o_ffull_js", "plain")),
            ("oracle: edge weights from the true FLAIR", m(W, "wd_o_s_t1_gfl", "plain")),
            ("oracle: edge weights from the true T1 (own)", m(W, "wd_o_s_t1_gself", "plain")),
            ("oracle: own true edges, TV x3", m(W, "wd_o_s_t1_gself_geo03", "plain")),
            ("oracle: directional TV, true FLAIR guide, λ 0.05", m(W, "wd_o_s_t1_dtvp005_gfl", "plain")),
            ("oracle: directional TV, true FLAIR guide, λ 0.15", m(W, "wd_o_s_t1_dtvp015_gfl", "plain"))]
FL_ITEMS = [("MC-GS, this recipe (T1 partner at AF 15.19)", m(W, "wd_v_af16_j", "contrast")),
            ("oracle: T1 partner fully sampled", m(W, "wd_o_t1full_j", "contrast")),
            ("oracle: T1 fully sampled, coupling at maximum", m(W, "wd_o_t1full_js", "contrast")),
            ("oracle: edge weights from the true T1", m(W, "wd_o_s_fl_gt1", "plain")),
            ("oracle: edge weights from the true FLAIR (own)", m(W, "wd_o_s_fl_gself", "plain")),
            ("oracle: directional TV, true T1 guide, λ 0.10", m(W, "wd_o_s_fl_dtvp010_gt1", "plain"))]
XMIN, XMAX = -0.95, 1.38
fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), facecolor=SURF,
                         gridspec_kw=dict(height_ratios=[len(T1_ITEMS), len(FL_ITEMS)], hspace=0.42))
for ax, (cname, base, items) in zip(axes, (("T1 target", S_T1, T1_ITEMS), ("FLAIR target", S_FL, FL_ITEMS))):
    style(ax)
    ax.grid(axis="y", visible=False)
    names = [n for n, _ in items]
    vals = [v - base for _, v in items]
    ys = list(range(len(items)))[::-1]
    for y, v in zip(ys, vals):
        shown = max(v, XMIN)
        ax.barh(y, shown, height=0.56, color=BLUE if v >= 0 else RED, zorder=2)
        if v >= XMIN:                       # value outside the bar end
            ax.annotate(f"{v:+.3f}", (shown, y), xytext=(5 if v >= 0 else -5, 0), textcoords="offset points",
                        ha="left" if v >= 0 else "right", va="center", fontsize=8.5, color=INK)
        else:                               # bar clipped at the axis: the value goes inside, white on the fill
            ax.annotate(f"{v:+.3f} (bar clipped)", (XMIN, y), xytext=(6, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8.5, color="white", fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=8.5, color=INK)
    ax.set_xlim(XMIN, XMAX)
    ax.axvline(0, color=INK2, linewidth=0.9, zorder=3)
    for xv, lab in ((0.3, "0.3 dB: closed below"), (1.0, "1 dB: worth a design")):
        ax.axvline(xv, color=MUTED, linewidth=0.9, zorder=1)
        ax.text(xv + 0.01, ys[0] + 0.55, lab, fontsize=7.5, color=INK2, va="bottom")
    ax.set_ylim(-0.6, len(items) - 0.2)
    ax.set_title(f"{cname}: gain over the fair GS-single ({base:.3f} dB head PSNR)", fontsize=10.5, color=INK,
                 loc="left", pad=16)
axes[1].set_xlabel("head PSNR gain over the fair GS-single (dB), AF 15.19, 4000 it, seed 42; "
                   "the 0.3 / 1 dB lines were fixed before the runs", fontsize=9, color=INK2)
axes[0].text(0.0, 1.32, "Ceilings: how much can a second contrast add, even when it is known exactly?",
             transform=axes[0].transAxes, fontsize=12, color=INK, ha="left")
save(fig, "widen_ceilings")

# ---------------------------------------------------------------- C: asymmetric acceleration
XS = [15.19, 26.68, 43.32]
S = [(W, "wd_v_af16_s_fl"), (W, "wd_a_s_fl32"), (W, "wd_a_s_fl64")]
J16 = [(W, "wd_v_af16_j"), (W, "wd_a_j_t16_fl32"), (W, "wd_a_j_t16_fl64")]
JF = [(W, "wd_o_t1full_j"), (W, "wd_a_j_tfull_fl32"), (W, "wd_a_j_tfull_fl64")]


def gains(js, key):
    out = []
    for (rj, ej), (rs, es) in zip(js, S):
        a, b = m(rj, ej, "contrast", key), m(rs, es, "plain", key)
        out.append(a - b if key == "head_psnr" else 100.0 * (a / b - 1.0))
    return out


series = [("T1 partner at AF 15.19", BLUE, J16), ("T1 partner fully sampled (oracle)", ORANGE, JF)]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), facecolor=SURF)
for ax, key, ylab in ((axes[0], "head_psnr", "FLAIR head PSNR gain (dB)"),
                      (axes[1], "head_hfen", "FLAIR head HFEN change (%)")):
    style(ax)
    for name, col, js in series:
        g = gains(js, key)
        ax.plot(XS, g, color=col, linewidth=2, solid_joinstyle="round", solid_capstyle="round", zorder=3)
        ax.scatter(XS, g, s=64, color=col, edgecolor=SURF, linewidth=2, zorder=4)
        ax.annotate(f"{g[-1]:+.2f}" + ("" if key == "head_psnr" else " %"), (XS[-1], g[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
        print(f"{key} {name}: " + "  ".join(f"{x:.2f}x {v:+.3f}" for x, v in zip(XS, g)))
    ax.set_xticks(XS)
    ax.set_xticklabels([f"{x:.2f}x" for x in XS])
    ax.set_xlim(12, 49)
    ax.axhline(0, color=BASE, linewidth=0.8, zorder=1)
    ax.set_xlabel("FLAIR acceleration (PE-plane points / mask points)", fontsize=9, color=INK2)
    ax.set_ylabel(ylab, fontsize=9, color=INK2)
axes[0].set_ylim(0, 0.32)
axes[1].set_ylim(-9, 0.5)
axes[1].invert_yaxis()
h = [plt.Line2D([], [], color=c, linewidth=2, marker="o", markersize=6, label=n) for n, c, _ in series]
fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.07),
           labelcolor=INK2)
fig.suptitle("Where the joint model starts to pay: FLAIR accelerated harder, T1 as partner", x=0.07, ha="left",
             fontsize=12, color=INK)
fig.text(0.07, 0.9, "gain = MC-GS minus GS-single on the same FLAIR mask; HFEN axis inverted, so up = lower HFEN = better edges",
         fontsize=8.5, color=INK2)
fig.subplots_adjust(top=0.8, bottom=0.25, wspace=0.28)
save(fig, "widen_asym_trend")
