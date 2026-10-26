"""Close the bounded metric ablation and retain a separate three-metric incumbent."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


summary_path = HERE / "summary.json"
all_path = HERE / "all_results.json"
summary = json.loads(summary_path.read_text())
results = json.loads(all_path.read_text())
assert all(digest(Path(p)) == h for p, h in summary["source_sha256"].items())
formal = results["formal25"]
best = formal[0]
head_order = sorted(formal, key=lambda t: min(c["margin"] for c in t["whole_head_comparisons"].values()), reverse=True)
head_best = head_order[0]
post_wins = [p for p in best["planes"] if all(p["comparisons"]["post:" + m]["strict_win"] for m in ("psnr", "ssim", "nmse"))]
assert len(post_wins) == 1
assert not any(t["passing_planes_by_scan"]["pre"] for group in results.values() for t in group)
assessment = {
    "summary": {"path": str(summary_path), "sha256": digest(summary_path)},
    "all_results": {"path": str(all_path), "sha256": digest(all_path)},
    "formal_best": best["name"], "score_six": best["score_six"],
    "formal_passing_planes": best["passing_planes_both_scans"],
    "formal_pre_all_three": best["passing_planes_by_scan"]["pre"],
    "formal_post_all_three": best["passing_planes_by_scan"]["post"],
    "post_witness": post_wins[0],
    "whole_head_best_by_six_margin": head_best["name"],
    "whole_head_best_metrics": head_best["whole_head_metrics"],
    "whole_head_best_comparisons": head_best["whole_head_comparisons"],
    "whole_head_strict_winning_formal_trials": sum(all(c["strict_win"] for c in t["whole_head_comparisons"].values()) for t in formal),
    "scope": "Retrospective re-evaluation only; no new training, no GPU. Keep original four-metric ledger and model untouched. Future selection follows the separately versioned three-metric contract.",
}
incumbent = {
    "status": "retrospective_three_metric_incumbent_target_unmet",
    "exp": best["name"], "score": best["score_six"],
    "strict_success": False, "strict_planes": 0, "eligible_planes": 59,
    "metric_contract": {"path": str(HERE / "metric_contract_v1.json"), "sha256": digest(HERE / "metric_contract_v1.json")},
    "validated_assessment": {"path": str(summary_path), "sha256": digest(summary_path)},
    "source_objective": {"path": best["source_objective"], "sha256": best["source_objective_sha256"]},
    "legacy_four_metric_incumbent_preserved": str(ROOT / "development_incumbent.json"),
    "claim_boundary": "One case/one seed adaptive development; one enhanced plane wins all three, no plain plane and no paired plane wins. Existing checkpoint/recipe requires its normal validation before any future numerical continuation.",
}
for name, data in (("assessment.json", assessment), ("development_incumbent_three_metrics_v1.json", incumbent)):
    p = HERE / name
    assert not p.exists()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
lines = [
    "# 去除 HFEN 后的重新评估", "",
    "仅保留 PSNR、SSIM、NMSE，复核25次已完成正式实验与65个已有历史重建；病例、参考、十个完整对照、59个切片与归一化均固定，没有重新训练。指标区域仍为病灶所在切片的头部掩膜，并非病灶ROI。", "",
    f"按同切片/双扫描/三指标的最差标准化差距重新排序，第一名改为 `{best['name']}`，S6={best['score_six']:.12f}；旧第一名g1的S6={formal[1]['score_six']:.12f}。排名变化不代表已经胜出。", "",
    "25次正式实验和65个历史方案均没有双扫描三指标同时全胜的切片。所有方案的平扫三指标全胜均为0/59。g1与g2各自在同一个增强切片（axis1/index129）达到三指标全胜，但对应平扫未达标。", "",
    "## 新第一名在最接近达标的双扫描切片上", "",
    "axis1/index130，core255。每个单元格比较的是同一扫描/同一指标下十个对手的最佳值；这里均为完整ZS-SSDU。", "",
    "| 扫描 | 指标 | MC-GS g2 | 最佳对照 | MC-GS减对照 |", "|---|---|---:|---:|---:|",
]
for condition, c in best["best_plane"]["comparisons"].items():
    scan, metric = condition.split(":")
    lines.append(f"| {'平扫' if scan == 'pre' else '增强'} | {metric.upper()} | {c['candidate']:.8f} | {c['best_opponent']:.8f} | {c['candidate']-c['best_opponent']:+.8f} |")
lines += ["", "PSNR与NMSE在固定参考/区域下反映等价的平方误差排序，不能当作两个独立发现。新的主要差距是平扫平方误差：该切片PSNR差约0.147dB，NMSE高约3.44%。", "",
          "## 增强单扫描的局部胜出", "", "axis1/index129；下面三项均严格超过全部十个对照，对应平扫仍落后。", "",
          "| 指标 | MC-GS g2 | 最佳对照 |", "|---|---:|---:|"]
for condition, c in post_wins[0]["comparisons"].items():
    if condition.startswith("post:"):
        lines.append(f"| {condition.split(':')[1].upper()} | {c['candidate']:.8f} | {c['best_opponent']:.8f} |")
lines += ["", "## 整脑指标另行评价", "",
          f"按整脑六个标准化差距选出的第一名为 `{head_best['name']}`，与切片目标选出的g2不同。25次正式实验中，没有一个方案在整脑六项条件上同时超过全部对照。下表固定使用这一整脑方案，不能混合不同方案的最优值。", "",
          "| 扫描 | 方法 | PSNR↑ | SSIM↑ | NMSE↓ |", "|---|---|---:|---:|---:|"]
for scan in ("pre", "post"):
    for label, values in (("MC-GS lowreg0.75", head_best["whole_head_metrics"][scan]), ("ZS-SSDU", summary["whole_head_opponents"]["ZS-SSDU"][scan])):
        lines.append(f"| {'平扫' if scan == 'pre' else '增强'} | {label} | {values['psnr']:.5f} | {values['ssim']:.6f} | {values['nmse']:.8f} |")
lines += ["", "## 结论与下一步", "", "去掉HFEN改变了候选排序，也显示出增强的局部三指标优势，但尚未消除平扫差距。后续候选选择采用单独保存的三指标合同，旧四指标评分与已完成记录保留。新的数据一致性校正尚处CPU修复/审查，不能把它当作此次重评的已完成结果。", "",
          "这是同一病例/同一种子的适应性开发重评，不是独立测试集结果；65个历史重建不计为新训练实验，保留其既有来源限制，未重新计算历史整脑或ROI增强指标。", "",
          "完整排序见 [ranking.csv](ranking.csv)，全59切片/六项原值见 [all_results.json](all_results.json)，源文件前后SHA256与口径见 [summary.json](summary.json)。"]
report = HERE / "REPORT.md"
assert not report.exists()
report.write_text("\n".join(lines) + "\n")
assert all(digest(Path(p)) == h for p, h in summary["source_sha256"].items())
print(json.dumps({"report": str(report), "formal_best": best["name"], "whole_head_best": head_best["name"], "post_witness": [post_wins[0]["axis"], post_wins[0]["index"]], "all_source_hashes_unchanged": True}, ensure_ascii=False, indent=2))
