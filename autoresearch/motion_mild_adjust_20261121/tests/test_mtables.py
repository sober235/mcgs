"""The Markdown tables of RESULTS.md are generated from the summary (tools/mtables.py), never typed."""
import importlib.util
from pathlib import Path

import mevaluate as ev
from test_criteria import good

HERE = Path(__file__).resolve().parents[1]


def _mtables():
    spec = importlib.util.spec_from_file_location("mtables", HERE / "tools" / "mtables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _summary():
    s = good()
    s["criteria"] = ev.criteria(s)
    s["recovered"] = {}
    for r in s["table"]:
        if r["cond"] != "C0":
            s["recovered"].setdefault(f"{r['order']}:{r['cond']}", {})[r["arm"]] = ev.recovered(
                s["table"], r["order"], r["cond"], r["arm"], base="I-T2" if r["arm"].startswith("I-") else "B0")
    s.update(n_runs=len(s["table"]), n_formal=43, missing=[])
    return s


def test_tables_have_the_numbers_of_the_summary():
    text = _mtables().render(_summary())
    for title in ("判据", "图像指标", "零运动", "扫描间", "扫描内", "线性顺序", "病灶"):
        assert title in text, title
    assert "| S1 | il / B2 | T2 相对基线的差，dB | -0.010 | >= -0.05 | 通过 |" in text
    assert "| S3 | Tb / B123 | 收回的 T2 损失比例 | 0.615 | >= 0.5 | 通过 |" in text
    assert "| il | Tb | B0 未改动的 MC-GS | 35.790 | 35.000 |" in text
    assert "None" not in text and "nan" not in text.lower()


def test_tables_mark_failed_and_skipped_items():
    s = _summary()
    for r in s["table"]:
        if (r["order"], r["cond"], r["arm"]) == ("il", "Ta", "B0"):
            r["T2"].update(head_psnr=36.25, head_psnr_eval=36.25)
        if (r["order"], r["cond"], r["arm"]) == ("il", "Tc", "B123"):
            r["T2"].update(head_psnr=35.30, head_psnr_eval=35.30)
    s["criteria"] = ev.criteria(s)
    text = _mtables().render(s)
    assert "**不通过**" in text
    assert "不计入" in text                                   # the skipped condition says why
