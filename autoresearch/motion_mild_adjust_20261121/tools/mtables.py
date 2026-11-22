#!/usr/bin/env python3
"""The Markdown tables of RESULTS.md, printed from summary.json (the committed copy of the server's
eval/summary.json). No number in those tables is typed by hand. Standard library only: runs on this Mac.

    tools/mtables.py [summary.json] > tables.md
"""
import json
import pathlib
import sys

M = pathlib.Path(__file__).resolve().parents[1]
ARMS = ("B0", "B1", "B2", "B12", "B123", "Ow", "Op", "I-T2", "I-T2+w")
CONDS = ("C0", "E2", "Ta", "Tb", "Tc", "E2Tb", "TbL")
NAME = {"B0": "B0 未改动的 MC-GS", "B1": "B1 配准位姿（M1）", "B2": "B2 回波链权重（M2）", "B12": "B12 M1 + M2",
        "B123": "B123 M1 + M2 + M3", "Ow": "Ow 真值权重", "Op": "Op 真值状态、位姿和权重", "I-T2": "I-T2 独立 T2",
        "I-T2+w": "I-T2+w 独立 T2 + M2"}
ORDER = {"il": "交错", "kz": "线性"}


def f(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def ok(p):
    return "—" if p is None else ("通过" if p else "**不通过**")


def line(out, crit, where, what, it, n=3):
    """One criteria row from an item dict(value, threshold, sense, pass); nothing when the item is absent."""
    if it is None:
        return
    out.append(f"| {crit} | {where} | {what} | {f(it['value'], n)} | {it['sense']} {it['threshold']} | {ok(it['pass'])} |")


def criteria_table(S):
    c = S["criteria"]
    out = ["| 判据 | 条件 / 组 | 项目 | 数值 | 阈值 | 结论 |", "|---|---|---|---|---|---|"]
    for key, a in c["S1"]["arms"].items():
        order, arm = key.split(":")
        where = f"{order} / {arm}"
        if a["dT1"]["value"] is not None:
            line(out, "S1", where, "T1 相对基线的差，dB", a["dT1"])
        line(out, "S1", where, "T2 相对基线的差，dB", a["dT2"])
        line(out, "S1", where, "权重为 1 的回波链比例", a["frac_w1"])
        line(out, "S1", where, "状态数", a["n_states"], 0)
    for arm in ("B1", "B123"):
        a = c["S2"]["arms"][arm]
        line(out, "S2", f"E2 / {arm}", "刚体部分的差分配准误差（脑内平均），mm", a["tre_rigid"])
        line(out, "S2", f"E2 / {arm}", "含非刚体的差分配准误差（脑内平均），mm", a["tre_full"])
        line(out, "S2", f"E2 / {arm}", "T1 归一化 Δ，dB", a["delta_T1"])
        line(out, "S2", f"E2 / {arm}", "T2 归一化 Δ，dB", a["delta_T2"])
    for cond, a in c["S3"]["conds"].items():
        where = f"{cond} / B123"
        if a["skipped"]:
            out.append(f"| S3 | {where} | B0 相对 C0 只损失 {f(a['recovered']['loss'])} dB，问题在该幅度下不显著 | — | — | 不计入 |")
            continue
        line(out, "S3", where, "收回的 T2 损失比例", a["recovered"])
        line(out, "S3", where, "受污染回波链的检出率", a["recall"])
        line(out, "S3", where, "干净回波链的误降率", a["false_rate"])
        line(out, "S3", where, "状态数（阈值 = 真值）", a.get("n_states"), 0)
        line(out, "S3", where, "状态位姿最大旋转误差，°", a.get("pose_deg"))
        line(out, "S3", where, "状态位姿最大平移误差，mm", a.get("pose_mm"))
    s4 = c["S4"]
    line(out, "S4", "TbL / B123", "病灶对比度恢复与 B0 之差（最差的一个）", s4.get("recovery_vs_B0"))
    line(out, "S4", "TbL / B123", "串扰超出容差的量（最大）", s4.get("crosstalk"))
    out.append("")
    out.append("| 判据 | 总结论 |")
    out.append("|---|---|")
    for k, name in (("S1", "S1 零运动"), ("S2", "S2 扫描间"), ("S3", "S3 扫描内（交错顺序）"), ("S4", "S4 病灶")):
        out.append(f"| {name} | {ok(c[k]['pass'])} |")
    return "\n".join(out)


def rows_of(S):
    return {(r["order"], r["cond"], r["arm"]): r for r in S["table"]}


def image_table(S):
    t = rows_of(S)
    out = ["| 顺序 | 条件 | 组 | T1 PSNR | T2 PSNR（插回全部采样线） | T2 PSNR（评估用） | T2 PSNR（原始渲染） | T2 SSIM |",
           "|---|---|---|---|---|---|---|---|"]
    for order in ("il", "kz"):
        for cond in CONDS:
            for arm in ARMS:
                r = t.get((order, cond, arm))
                if r is None:
                    continue
                t1, t2 = r.get("T1") or {}, r["T2"]
                out.append(f"| {order} | {cond} | {NAME[arm]} | {f(t1.get('head_psnr'))} | {f(t2.get('head_psnr'))} | "
                           f"{f(t2.get('head_psnr_eval'))} | {f(t2.get('head_psnr_raw'))} | {f(t2.get('head_ssim'), 4)} |")
    return "\n".join(out)


def zero_table(S):
    c = S["criteria"]["S1"]
    out = ["| 顺序 | 组 | ΔT1，dB | ΔT2，dB | 权重为 1 的回波链比例 | 状态数 | 结论 |", "|---|---|---|---|---|---|---|"]
    for key, a in c["arms"].items():
        order, arm = key.split(":")
        out.append(f"| {ORDER[order]} | {NAME[arm]} | {f(a['dT1']['value'])} | {f(a['dT2']['value'])} | "
                   f"{f(a['frac_w1']['value'])} | {f(a['n_states']['value'], 0)} | {ok(a['pass'])} |")
    r = c.get("registration")
    if r:
        out.append("")
        out.append(f"无运动数据上的配准结果（只报告）：平移与数据自带偏移之差 {f(r['t_minus_native_mm'])} mm，旋转 {f(r['rot_deg'])}°。")
    return "\n".join(out)


def inter_table(S):
    c = S["criteria"]["S2"]["arms"]
    out = ["| 组 | 刚体部分误差 mm | 含非刚体误差 mm | 只用刚体部分对整个真值场的误差 mm | 控制网格脑内最大位移 mm | 归一化 Δ T1 | 归一化 Δ T2 |",
           "|---|---|---|---|---|---|---|"]
    for arm in ("B0", "B1", "B123"):
        a = c[arm]
        out.append(f"| {NAME[arm]} | {f(a['tre_rigid']['value'])} | {f(a['tre_full']['value'])} | "
                   f"{f(a['tre_full'].get('rigid_alone'))} | {f(a['tre_full'].get('nonrigid_brain_max_mm'))} | "
                   f"{f(a['delta_T1']['value'])} | {f(a['delta_T2']['value'])} |")
    return "\n".join(out)


def intra_table(S):
    c, w, st = S["criteria"]["S3"]["conds"], S["weights"], S["states"]
    out = ["| 条件 | B0 相对 C0 的 T2 损失 dB | B2 收回 | B12 收回 | B123 收回 | Ow 收回 | Op 收回 | B123 检出率 | B123 误降率 | B123 状态数 / 真值 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for cond, a in c.items():
        o = a["others"]
        name = next((r["exp"] for r in S["table"] if (r["order"], r["cond"], r["arm"]) == ("il", cond, "B123")), None)
        s = st.get(name) or {}
        ww = w.get(name) or {}
        states = "—" if not s else f"{s['n_states']} / {s['n_true']}"
        out.append(f"| {cond} | {f(a['recovered'].get('loss'))} | {f(o.get('B2'))} | {f(o.get('B12'))} | "
                   f"{f(a['recovered']['value'])} | {f(o.get('Ow'))} | {f(o.get('Op'))} | {f(ww.get('recall'))} | "
                   f"{f(ww.get('false_rate'))} | {states} |")
    return "\n".join(out)


def kz_table(S):
    out = ["| 条件 | B0 相对 C0 的 T2 损失 dB | B123 − B0，dB | Ow − B0，dB | Op − B0，dB | 是否无损害（≥ 阈值） |", "|---|---|---|---|---|---|"]
    for cond, a in S["criteria"]["kz"].items():
        harm = "—" if a["no_harm"] is None else ("是" if a["no_harm"] else "**否**")
        out.append(f"| {cond} | {f(a['loss'])} | {f(a['value'])} | {f(a['others'].get('Ow'))} | {f(a['others'].get('Op'))} | {harm} |")
    return "\n".join(out)


def lesion_table(S):
    out = ["| 组 | 病灶 | 所在对比度 | 直径 mm | 对比度恢复 | 质心偏移 mm | 另一对比度上的串扰 | 串扰容差 |", "|---|---|---|---|---|---|---|---|"]
    for arm in ("B0", "B123", "I-T2"):
        for L in (S.get("lesions") or {}).get(arm, []):
            if L.get("contrast_recovery") is None and L.get("crosstalk") is None:
                continue
            out.append(f"| {arm} | L{L['label']} | {L['contrast']} | {L['diameter_mm']:.0f} | {f(L.get('contrast_recovery'))} | "
                       f"{f(L.get('centroid_shift_mm'))} | {f(L.get('crosstalk'))} | {f(L.get('crosstalk_tol'))} |")
    return "\n".join(out)


def render(S):
    parts = [f"<!-- {S.get('n_runs')} of {S.get('n_formal')} formal runs; missing: {S.get('missing') or 'none'} -->"]
    for title, fn in (("判据", criteria_table), ("图像指标", image_table), ("零运动检查（C0）", zero_table),
                      ("扫描间（E2）", inter_table), ("扫描内运动，交错顺序：收回的损失", intra_table),
                      ("线性顺序（只报告）", kz_table), ("病灶（TbL）", lesion_table)):
        parts.append(f"#### {title}\n\n{fn(S)}")
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    print(render(json.loads(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else M / "summary.json").read_text())), end="")
