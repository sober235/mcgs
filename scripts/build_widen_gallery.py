"""Local viewing page for the widen loop's figures (served by the 8765 http.server; the figures carry the subject's name,
so they stay on this server). Writes ~/figs/widen_mcgs/index.html: one tab per acceleration factor with a head-metric
table (MC-GS and the fair GS-single from this loop's score files, the best non-GS arm from the topic's summary.json,
L1-ESPIRiT excluded as the topics do), the T1 and FLAIR figures, the two asymmetric-probe comparisons and the summary charts.
"""
import html
import json
from pathlib import Path

W = Path("/data0/congcong/code/GS/mcgs_widen_wt/results_arc")
A12 = Path("/data0/congcong/code/GS/mcgs_af12_wt/results_arc")     # the AF12 line's registered P14 runs
R = Path("/data0/congcong/code/GS/Results_20260908")
FIGS = Path.home() / "figs/widen_mcgs"
NON_GS = {"cgsense": "CG-SENSE", "cssense": "CS-SENSE", "cssense_o200": "CS-SENSE (TV, converged)",
          "jcssense": "Joint SENSE", "spirit": "L1-SPIRiT", "zsssdu": "ZS-SSDU", "dip": "3D-DIP", "inr": "3D-INR",
          "mcinr": "MC-INR", "ploraks": "P-LORAKS"}      # P-LORAKS is a counted arm at AF12 only
AFS = [(12, "11.71"), (16, "15.19"), (20, "18.56"), (24, "24.00")]


def sc(exp, ch):
    d = json.load(open(W / f"{exp}_score.json"))[ch]
    return d["head_psnr"], d["head_ssim"]


def af_table(af):
    rows = json.load(open(R / f"T1FLAIR_AF{af}_v3_jiangyizhou/logs/summary.json"))
    out = []
    for c, jch in (("T1", "plain"), ("FLAIR", "contrast")):
        mc = sc(f"wd_v_af{af}_j", jch)
        gs = sc(f"wd_v_af{af}_s_{'t1' if c == 'T1' else 'fl'}", "plain")
        cand = [r for r in rows if r["contrast"] == c and r["method"] in NON_GS]
        best = max(cand, key=lambda r: r["head_psnr"])
        out.append((c, mc, gs, (NON_GS[best["method"]], best["head_psnr"], best["head_ssim"])))
    return out


ORDER = ["cgsense", "cssense", "cssense_o200", "jcssense", "spirit", "ploraks", "zsssdu", "dip", "inr", "mcinr"]
REF = {12: ("gs_single_p14", "gs_joint_p14", "P14, AF12 line"), 16: ("gs_single_p3", "gs_joint_p3", "phase 3"),
       20: ("gs_single_p3", "gs_joint_p3", "phase 3"), 24: ("gs_single_p3", "gs_joint_p3", "phase 3")}


def sc3(exp, ch):
    d = json.load(open(W / f"{exp}_score.json"))[ch]
    return d["head_psnr"], d["head_ssim"], d["head_hfen"]


def full_table(af):
    """every arm of the topic (L1-ESPIRiT retired, as in the topics) + this loop's two GS arms; published GS arms as
    reference rows; best of each counted column in bold (ZF and the reference rows are not counted)"""
    rows = json.load(open(R / f"T1FLAIR_AF{af}_v3_jiangyizhou/logs/summary.json"))
    by = {(r["method"], r["contrast"]): (r["head_psnr"], r["head_ssim"], r["head_hfen"]) for r in rows}
    body = [(NON_GS[mth], by[(mth, "T1")], by[(mth, "FLAIR")], "arm") for mth in ORDER if (mth, "T1") in by]
    body.append(("GS-single (fair: lambda_edge 0, own best settings)", sc3(f"wd_v_af{af}_s_t1", "plain"),
                 sc3(f"wd_v_af{af}_s_fl", "plain"), "arm"))
    body.append(("MC-GS (this recipe)", sc3(f"wd_v_af{af}_j", "plain"), sc3(f"wd_v_af{af}_j", "contrast"), "arm"))
    rs, rj, rname = REF[af]
    ref = [("ZF (zero-filled, reference)", by[("zf", "T1")], by[("zf", "FLAIR")], "ref"),
           (f"GS-single, published ({rname}, with the common-edge prior)", by[(rs, "T1")], by[(rs, "FLAIR")], "ref"),
           (f"MC-GS, published ({rname})", by[(rj, "T1")], by[(rj, "FLAIR")], "ref")]
    counted = [b for b in body if b[3] == "arm"]
    best = {}
    for ci in (1, 2):
        for k, better in ((0, max), (1, max), (2, min)):
            best[(ci, k)] = better(b[ci][k] for b in counted)
    trs = []
    for b in body + ref:
        cells = []
        for ci in (1, 2):
            for k, fmt in ((0, "{:.3f}"), (1, "{:.4f}"), (2, "{:.4f}")):
                v = fmt.format(b[ci][k])
                cells.append(f"<td><b>{v}</b></td>" if b[3] == "arm" and b[ci][k] == best[(ci, k)] else f"<td>{v}</td>")
        cls = ' class="ref"' if b[3] == "ref" else (' class="ours"' if b[0].startswith("MC-GS") else "")
        trs.append(f"<tr{cls}><td>{html.escape(b[0])}</td>{''.join(cells)}</tr>")
    return ("<table><thead><tr><th rowspan=2>method</th><th colspan=3>T1 (head)</th><th colspan=3>FLAIR (head)</th></tr>"
            "<tr><th>PSNR</th><th>SSIM</th><th>HFEN</th><th>PSNR</th><th>SSIM</th><th>HFEN</th></tr></thead><tbody>"
            + "".join(trs) + "</tbody></table>")


def p14_note(af):
    """AF12 only: the AF12 line's own tuned arm (P14, registered in the topic) beside this loop's one recipe"""
    if af != 12:
        return ""
    j = json.load(open(A12 / "p3_e0001_ptv001_full4k_score.json"))
    st = json.load(open(A12 / "p3_e0001_ptv001_singleT1_full4k_score.json"))["plain"]["head_psnr"]
    sf = json.load(open(A12 / "p3_e0001_ptv001_singleFLAIR_full4k_score.json"))["plain"]["head_psnr"]
    return (f'<p class="note">Reference, the AF12 line\'s own tuned arm (P14, registered in the topic, rotation_lr 0.008): '
            f'MC-GS {j["plain"]["head_psnr"]:.3f} / {j["contrast"]["head_psnr"]:.3f}, its GS-single (with the common-edge '
            f'prior) {st:.3f} / {sf:.3f}, published margin {j["plain"]["head_psnr"] - st:+.3f} / '
            f'{j["contrast"]["head_psnr"] - sf:+.3f} (T1 / FLAIR head PSNR).</p>')


def fig_name(af, c):
    """AF12 carries one more counted arm (P-LORAKS): its figure with that column when rendered (--with_ploraks)"""
    p = f"af{af}_{c}_ploraks.png"
    return p if af == 12 and (FIGS / p).exists() else f"af{af}_{c}.png"


def img(name, alt):
    p = FIGS / name
    if not p.exists():
        return f'<p class="pending">{html.escape(alt)}: not rendered yet</p>'
    return (f'<figure><a href="{name}" target="_blank"><img loading="lazy" src="{name}" alt="{html.escape(alt)}"></a>'
            f'<figcaption>{html.escape(alt)} (click for full size)</figcaption></figure>')


tabs, panes = [], []
for i, (af, act) in enumerate(AFS):
    tid = f"af{af}"
    tabs.append((tid, f"AF {act}x"))
    trs = []
    for c, mc, gs, (bn, bp, bs) in af_table(af):
        trs.append(f"<tr><td>{c}</td><td><b>{mc[0]:.3f}</b> / {mc[1]:.4f}</td><td>{gs[0]:.3f} / {gs[1]:.4f}</td>"
                   f"<td>{html.escape(bn)} {bp:.3f} / {bs:.4f}</td><td>{mc[0] - gs[0]:+.3f}</td><td>{mc[0] - bp:+.3f}</td></tr>")
    panes.append((tid, f"""
<h2>nominal AF {af}, actual {act}x (PE-plane points / mask points, both contrasts read one mask)</h2>
{full_table(af)}
<p class="note">MC-GS − GS-single: {" / ".join(f"{c} {mc[0] - gs[0]:+.3f}" for c, mc, gs, _ in af_table(af))} dB;
MC-GS − best non-GS: {" / ".join(f"{c} {mc[0] - bp:+.3f} ({bn})" for c, mc, gs, (bn, bp, bs) in af_table(af))} dB.
Bold = best of the counted arms in that column (HFEN: lower is better); grey rows are references, not counted.</p>
<p class="note">MC-GS = phase-3 recipe + phase TV 0.01 + T1 wavelet 0.18 + dtv 0.01 + guided edge / phase TV 0.25, lambda_edge 0.001.
GS-single = the same per-channel settings at lambda_edge 0 (its own best). Head-only metrics, 3.6M Gaussians, 4000 iterations, seed 42.</p>
{p14_note(af)}
{img(fig_name(af, "T1"), f"T1, AF {act}x, all arms")}
{img(fig_name(af, "FLAIR"), f"FLAIR, AF {act}x, all arms")}"""))
for lvl, act in ((32, "26.68"), (64, "43.32")):
    tid = f"asym{lvl}"
    tabs.append((tid, f"GS-only analysis: FLAIR {act}x"))
    s = sc(f"wd_va_s_fl{lvl}", "plain") if (W / f"wd_va_s_fl{lvl}_score.json").exists() else None
    rows = ""
    if s:
        for lab, exp in (("MC-GS, T1 partner at AF 15.19", f"wd_va_j_t16_fl{lvl}"),
                         ("MC-GS, T1 partner fully sampled (oracle)", f"wd_va_j_tfull_fl{lvl}")):
            if (W / f"{exp}_score.json").exists():
                j = sc(exp, "contrast")
                rows += f"<tr><td>{lab}</td><td>{j[0]:.3f} / {j[1]:.4f}</td><td>{j[0] - s[0]:+.3f}</td></tr>"
        rows = (f"<tr><td>GS-single</td><td>{s[0]:.3f} / {s[1]:.4f}</td><td>—</td></tr>" + rows)
    panes.append((tid, f"""
<h2>FLAIR at nominal AF {lvl}, actual {act}x; T1 as the partner (probe mask, not a topic mask)</h2>
<table><thead><tr><th>arm</th><th>FLAIR head PSNR / SSIM</th><th>vs GS-single</th></tr></thead><tbody>{rows}</tbody></table>
<p class="note"><b>Incomplete as a comparison:</b> only the two GS arms were run at this probe mask; the classical and learned
arms exist at AF 11.71 / 15.19 / 18.56 / 24.00 only. Shown as an analysis of the GS model, not as a method comparison.</p>
{img(f"asym_flair_actual{act}x_compare.png", f"FLAIR {act}x: GS-single vs MC-GS")}"""))
tabs.append(("summary", "GS-only analysis: charts"))
panes.append(("summary", f"""
<h2>analysis charts of the GS model (MC-GS against GS-single; every number read from the score files)</h2>
<p class="note">These charts study the joint model against its single-contrast counterpart (margins, oracle ceilings,
asymmetric acceleration); by construction they contain only the GS arms. The method comparison with every arm is in the AF tabs.</p>
{img("widen_margin_dumbbell.png", "fair vs published margin per AF")}
{img("widen_ceilings.png", "oracle ceilings at AF 15.19")}
{img("widen_asym_trend.png", "joint gain as FLAIR gets sparser")}"""))

buttons = "".join(f'<button data-t="{t}"{" class=on" if k == 0 else ""}>{html.escape(lab)}</button>'
                  for k, (t, lab) in enumerate(tabs))
body = "".join(f'<section id="{t}"{"" if k == 0 else " hidden"}>{p}</section>' for k, (t, p) in enumerate(panes))
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MC-GS widen loop</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#2c2c2a;--acc:#3987e5}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}
nav{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px;position:sticky;top:0;background:var(--bg);padding:6px 0;z-index:1}}
nav button{{border:1px solid var(--line);background:none;color:var(--ink);padding:6px 12px;border-radius:6px;cursor:pointer;font:inherit}}
nav button.on{{border-color:var(--acc);color:var(--acc);font-weight:600}}
h1{{font-size:18px;margin:0 0 8px}} h2{{font-size:15px;margin:10px 0}}
table{{border-collapse:collapse;margin:6px 0 4px;font-variant-numeric:tabular-nums}}
th,td{{border-bottom:1px solid var(--line);padding:4px 10px;text-align:left}} th{{color:var(--ink2);font-weight:500}}
.note,.pending,figcaption{{color:var(--ink2);font-size:12.5px}}
tr.ref td{{color:var(--ink2)}} tr.ours td:first-child{{font-weight:600}}
figure{{margin:14px 0}} img{{max-width:100%;height:auto;display:block;border:1px solid var(--line)}}
</style></head><body>
<h1>MC-GS against every comparison method, jiangyizhou T1 + FLAIR, AF 11.71 / 15.19 / 18.56 / 24.00</h1>
<nav>{buttons}</nav>{body}
<script>
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{{
 document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));
 document.querySelectorAll('section').forEach(s=>s.hidden=s.id!==b.dataset.t);}});
</script></body></html>"""
(FIGS / "index.html").write_text(page)
print("wrote", FIGS / "index.html")
for af, act in AFS:
    for c, mc, gs, (bn, bp, bs) in af_table(af):
        print(f"AF {act} {c}: MC-GS {mc[0]:.3f}  GS-single {gs[0]:.3f} ({mc[0] - gs[0]:+.3f})  best non-GS {bn} {bp:.3f} ({mc[0] - bp:+.3f})")
