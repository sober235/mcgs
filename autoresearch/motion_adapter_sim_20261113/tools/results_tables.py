#!/usr/bin/env python3
"""Local helper (this Mac): the Markdown tables of RESULTS.md, printed from summary.json (the committed copy of the
server's eval/summary.json). No number in those tables is typed by hand.

    tools/results_tables.py [summary.json] > tables.md
"""
import json
import pathlib
import sys

E = pathlib.Path(__file__).resolve().parents[1]
S = json.loads(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else E / "summary.json").read_text())
CONDS = ("C0", "E1a", "E1b", "E2", "E3", "E4a", "E4b", "E4c")
ARMS = ("A0", "A1", "A2", "A3", "A23", "O", "I-T1", "I-T2")
NAME = {"A0": "A0 无适配器", "A1": "A1 刚体", "A2": "A2 刚体 + 非刚体", "A3": "A3 刚体 + 分段", "A23": "A23 刚体 + 非刚体 + 分段",
        "O": "O 真值变换", "I-T1": "I-T1 独立 T1", "I-T2": "I-T2 独立 T2"}


def f(x, n=3, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def ok(p):
    return "—" if p is None else ("通过" if p else "**不通过**")


def rows_of(af):
    return {(r["cond"], r["arm"]): r for r in S[af]["table"] if not r.get("tag") and r["seed"] == 42}


def image_table(af):
    t, d = rows_of(af), S[af]["delta"]
    out = ["| 条件 | 组 | T1 PSNR | T2 PSNR（插回全部采样线） | T2 PSNR（只插回参考段） | T2 PSNR（原始渲染） | T2 SSIM | 归一化 Δ T1 | 归一化 Δ T2 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for c in CONDS:
        for a in ARMS:
            r = t.get((c, a))
            if r is None:
                continue
            t1, t2 = r.get("T1") or {}, r.get("T2") or {}
            dl = d.get(c, {}).get(a, {})
            out.append(f"| {c} | {NAME[a]} | {f(t1.get('head_psnr'))} | {f(t2.get('head_psnr'))} | "
                       f"{f(t2.get('head_psnr_refdc'))} | {f(t2.get('head_psnr_raw'))} | {f(t2.get('head_ssim'), 4)} | "
                       f"{f(dl.get('T1'), 3, True)} | {f(dl.get('T2'), 3, True)} |")
    return "\n".join(out)


def zero_table(af):
    out = ["| 组 | ΔT1（对 A0） | ΔT2（对 A0） | 平移估计与自带偏移之差 mm | 旋转估计 ° | 非刚体脑内最大 mm | 分段位姿最大 ° / mm |", "|---|---|---|---|---|---|---|"]
    for a, z in S[af]["zero"].items():
        seg = "—" if z.get("seg_max_deg") is None else f"{f(z['seg_max_deg'])} / {f(z['seg_max_mm'])}"
        out.append(f"| {NAME[a]} | {f(z['dpsnr_T1'], 3, True)} | {f(z['dpsnr_T2'], 3, True)} | {f(z['t_minus_native_mm'])} | "
                   f"{f(z['rot_deg'])} | {f(z.get('nonrigid_brain_max_mm'))} | {seg} |")
    return "\n".join(out)


def criteria_table(af):
    c = S[af]["criteria"]
    out = ["| 判据 | 条件 / 组 | 项目 | 数值 | 阈值 | 结论 |", "|---|---|---|---|---|---|"]

    def line(crit, where, what, it):
        if it is None or it.get("value") is None:
            return
        thr = it["threshold"]
        out.append(f"| {crit} | {where} | {what} | {f(it['value'])} | {it['sense']} {thr} | {ok(it['pass'])} |")
    s1 = c["S1"]
    for k, what in (("psnr", "A1–A3 对 A0 的最差 PSNR 差，dB"), ("translation", "平移估计与影像法估计之差，mm"),
                    ("rotation", "旋转估计，°"), ("nonrigid", "非刚体脑内最大位移，mm"),
                    ("segments_deg", "A3 分段位姿最大值，°"), ("segments_mm", "A3 分段位姿最大值，mm")):
        line("S1", "C0", what, s1.get(k))
    for cond, v in c["S2"].items():
        if not isinstance(v, dict):
            continue
        line("S2", f"{cond} / {v['arm']}", "T2 归一化 Δ，dB", v["delta_T2"])
        line("S2", f"{cond} / {v['arm']}", "T1 归一化 Δ，dB", v["delta_T1"])
        line("S2", f"{cond} / {v['arm']}", f"脑内平均 TRE（{'差分' if v['tre'].get('basis') == 'differential' else '绝对'}），mm", v["tre"])
    s3 = c["S3"]
    line("S3", "E3 / A2", "T2 病灶对比度恢复（最差的一个）", s3.get("recovery"))
    line("S3", "E3 / A2", "病灶质心偏移（最大），mm", s3.get("centroid"))
    line("S3", "E3 / A2", "串扰超出容差的量（最大）", s3.get("crosstalk"))
    for cond, v in c["S4"].items():
        r, sg = v["recovered"], v["segments"]
        if r.get("value") is None:
            continue
        line("S4", f"{cond} / {v['arm']}", "收回的 T2 损失比例", r)
        if sg.get("rot_err_deg_max") is not None:
            out.append(f"| S4 | {cond} / {v['arm']} | 分段位姿最大误差，° / mm | {f(sg['rot_err_deg_max'])} / {f(sg['t_err_mm_max'])} | "
                       f"<= {sg['threshold'][0]} / {sg['threshold'][1]} | {ok(sg['pass'])} |")
    return "\n".join(out)


def s4_table(af):
    out = ["| 条件 | 组 | A0 在 C0 | A0 在该条件 | 该组（只插回参考段） | 收回比例 |", "|---|---|---|---|---|---|"]
    for cond, v in S[af]["criteria"]["S4"].items():
        r = v["recovered"]
        if r.get("value") is None:
            continue
        out.append(f"| {cond} | {v['arm']} | {f(r['a0_c0'])} | {f(r['a0'])} | {f(r['arm_psnr'])} | {f(r['value'])} |")
    return "\n".join(out)


def motion_table(af):
    out = ["| 运行 | 旋转误差 ° | 脑内平均 TRE mm（绝对） | 脑内平均 / 最大 TRE mm（差分） | 非刚体 RMSE mm（差分） | 分段位姿最大误差 ° / mm |",
           "|---|---|---|---|---|---|"]
    for exp, m in S[af]["motion"].items():
        name = exp.replace(f"ma_{af}_", "").replace("_", " / ")
        diff = "—" if m.get("tre_mean_mm_differential") is None else f"{f(m['tre_mean_mm_differential'])} / {f(m['tre_max_mm_differential'])}"
        seg = "—" if m.get("seg_rot_err_deg_max") is None else f"{f(m['seg_rot_err_deg_max'])} / {f(m['seg_t_err_mm_max'])}"
        out.append(f"| {name} | {f(m.get('rot_err_deg'))} | {f(m.get('tre_mean_mm_absolute'))} | {diff} | "
                   f"{f(m.get('nonrigid_rmse_mm_differential'))} | {seg} |")
    return "\n".join(out)


def lesion_table(af):
    out = ["| 组 | 病灶 | 所在对比度 | 直径 mm | 对比度恢复 | 质心偏移 mm | 另一对比度上的串扰 | 串扰容差 |", "|---|---|---|---|---|---|---|---|"]
    for a in ("A0", "A2", "O", "I-T1", "I-T2"):
        for L in S[af]["lesions"].get(a, []):
            if L.get("contrast_recovery") is None and L.get("crosstalk") is None:
                continue
            out.append(f"| {a} | L{L['label']} | {L['contrast']} | {L['diameter_mm']:.0f} | {f(L.get('contrast_recovery'))} | "
                       f"{f(L.get('centroid_shift_mm'))} | {f(L.get('crosstalk'), 3, True)} | {f(L.get('crosstalk_tol'))} |")
    return "\n".join(out)


for af in S:
    n = len(rows_of(af))
    print(f"<!-- {af}: {n} runs -->\n")
    for title, fn in (("判据", criteria_table), ("图像指标", image_table), ("零运动检查（C0）", zero_table),
                      ("扫描内运动：收回的损失", s4_table), ("运动恢复", motion_table), ("病灶（E3）", lesion_table)):
        print(f"#### {af.upper()}：{title}\n\n{fn(af)}\n")
