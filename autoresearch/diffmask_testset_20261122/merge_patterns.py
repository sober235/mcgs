#!/usr/bin/env python3
"""Three sampling patterns at AF 16 on the six test scans: variable-density Poisson, uniform density, pseudo-radial.

Reads two stored tables and recomputes nothing:
  - this study's table.json                    (uniform density, pseudo-radial)
  - ../t1flair_af_testset_20261121/table.json  (the AF study; its AF 16 cells are the variable-density Poisson pattern)
Both tables come from the same statistics code (collect_stats.py was ported from that study), same head mask rule and
same eleven methods. Only cells in which all eleven methods are present are counted; a pattern with missing scans is
marked INTERIM. Standard library only, so it runs on the local Mac:

  /usr/bin/python3 autoresearch/diffmask_testset_20261122/merge_patterns.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWN = HERE / "table.json"
AF = HERE.parent / "t1flair_af_testset_20261121" / "table.json"
PATS = ("vdpoisson", "uniform", "radial")
PATZH = {"vdpoisson": "变密度泊松", "uniform": "均匀密度", "radial": "伪径向"}
METRICS = ("head_psnr", "head_ssim", "head_nmse")
MNAME = {"head_psnr": "PSNR", "head_ssim": "SSIM", "head_nmse": "NMSE"}
CONTRASTS = ("T1", "FLAIR")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load():
    own, af = json.load(open(OWN)), json.load(open(AF))
    assert own["labels"] == af["labels"], "the two studies list different methods"
    cells = []
    for c in af["cells"].values():
        if c["af"] == 16:
            cells.append(dict(c, af="vdpoisson"))
    cells += [c for c in own["cells"].values() if c["af"] in ("uniform", "radial")]
    return cells


def summary(cells):
    out = {}
    for pat in PATS:
        for con in CONTRASTS:
            sel = [c for c in cells if c["af"] == pat and c["contrast"] == con]
            ok = [c for c in sel if c["complete"]]
            row = {"scans_complete": sorted(c["scan"] for c in ok), "n_complete": len(ok), "n_scans": 6,
                   "missing": {c["scan"]: c["missing"] for c in sel if not c["complete"]}}
            for m in METRICS:
                row[m] = {"first": sum(c["first"][m] for c in ok),
                          "not_first": {c["scan"]: c["rank"][m] for c in ok if not c["first"][m]}}
            for comp in ("GS-single", "strongest"):
                d = [c["diffs"][comp]["head_psnr"] for c in ok]
                row["psnr_vs_" + comp] = ({"mean": sum(d) / len(d), "min": min(d), "max": max(d),
                                           "positive": sum(x > 0 for x in d)} if d else None)
            out[f"{pat}:{con}"] = row
    return out


def write_md(s):
    n = {p: s[f"{p}:T1"]["n_complete"] for p in PATS}
    L = ["# 三种采样方式（AF16，测试集 6 次扫描）合并表", "",
         "本表由 `merge_patterns.py` 从两份已存的统计表合并，不重新计算任何分数。"
         "变密度泊松来自加速倍数研究（`t1flair_af_testset_20261121/table.json`）的 AF16 格子。", "",
         "完整扫描数：" + "，".join(f"{PATZH[p]} {n[p]} / 6" for p in PATS) + "。"]
    if min(n.values()) < 6:
        L += ["", "**中间结果：** " + "、".join(PATZH[p] for p in PATS if n[p] < 6) +
              " 的扫描还不齐，下面的计数只含已完整的扫描，不能作为最终结论。"]
    L += ["", "## MC-GS 排第一的扫描数", "", "| 采样方式 | 对比度 | 完整扫描 | PSNR | SSIM | NMSE |", "|---|---|---|---|---|---|"]
    for p in PATS:
        for con in CONTRASTS:
            r = s[f"{p}:{con}"]
            k = r["n_complete"]
            L.append(f"| {PATZH[p]} | {con} | {k} / 6 | " + " | ".join(f"{r[m]['first']} / {k}" for m in METRICS) + " |")
    L += ["", "## 头部 PSNR 差值（MC-GS 减对方，dB）", "",
          "| 采样方式 | 对比度 | 对 GS-single：均值（最小；正数个数） | 对最强的非 GS 方法：均值（最小；正数个数） |", "|---|---|---|---|"]
    for p in PATS:
        for con in CONTRASTS:
            r = s[f"{p}:{con}"]
            k = r["n_complete"]
            f = lambda d: "—" if d is None else f"{d['mean']:+.3f}（{d['min']:+.3f}；{d['positive']} / {k}）"
            L.append(f"| {PATZH[p]} | {con} | {f(r['psnr_vs_GS-single'])} | {f(r['psnr_vs_strongest'])} |")
    L += ["", "## MC-GS 不是第一的列", ""]
    any_ = False
    for p in PATS:
        for con in CONTRASTS:
            for m in METRICS:
                for scan, rk in s[f"{p}:{con}"][m]["not_first"].items():
                    L.append(f"- {PATZH[p]} {con} {MNAME[m]}：{scan} 第 {rk}")
                    any_ = True
    if not any_:
        L.append("- 无")
    miss = {k: v["missing"] for k, v in s.items() if v["missing"]}
    if miss:
        L += ["", "## 还缺方法的格子", ""]
        for k, v in miss.items():
            p, con = k.split(":")
            for scan, mm in v.items():
                L.append(f"- {PATZH[p]} {con} {scan}：缺 {'、'.join(mm) if mm else '（未列出）'}")
    L += ["", "## 数据来源", "", f"- 本研究 `table.json` sha256 `{sha(OWN)[:16]}…`",
          f"- 加速倍数研究 `table.json` sha256 `{sha(AF)[:16]}…`", ""]
    (HERE / "table_3patterns.md").write_text("\n".join(L), encoding="utf-8")


def main():
    s = summary(load())
    json.dump({"sources": {"own": sha(OWN), "af_study": sha(AF)}, "summary": s},
              open(HERE / "table_3patterns.json", "w"), indent=1, ensure_ascii=False)
    write_md(s)
    print((HERE / "table_3patterns.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
