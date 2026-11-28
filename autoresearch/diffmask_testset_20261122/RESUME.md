# diff-mask 测试集研究：断点记录（2026-11-28 06:25 PST）

时间为本机时钟（PST）。

## 已完成

- 均匀密度、伪径向两种采样方式，AF16，6 次测试扫描，11 个方法：全部跑完并统一重新评分，12 / 12 个格子完整（`RESULTS.md` 第 1–7 节）。
- 变密度泊松（AF16）作为第三种采样方式并入：取自加速倍数测试集研究的最终统计表（24 / 24），`merge_patterns.py` → `table_3patterns.md` / `.json`，`RESULTS.md` 第 8 节。
- 量尺核对 `gauge_check.py`：132 个变密度泊松重建体积用本研究的评分重算，与对方表中的数字最大差值为 0（`gauge_check.json`）。
- 6 次扫描的切面对比图：脚本 `fig_testset_record_format.py`，预览和记录在 `figures/slices_preview/`；全分辨率文件在服务器 `codex_artifacts/mcgs_diffmask_testset_20261122/figures_slices_article_inbrain/`。
- 本研究在服务器上没有正在运行的作业，不占 GPU。

## 注意

- `RESULTS.md` 第 8 节是把 `merge_patterns.py` 的结果手工接在后面的。重跑 `collect_stats.py --pipeline` 会重新生成 `RESULTS.md` 并去掉第 8 节，之后要重跑 `merge_patterns.py` 并把第 8 节重新接上。
- “径向”= 2026-09 的伪径向掩膜（用户 2026-11-12 定的调参病例径向一行沿用该结果；本研究按此立项）。

## 还没有得到用户确认的事

- 切面图取的是头部面积最大的横断面（偏低），在这些切面上伪径向 T1 的 MC-GS 排第 2–3；是否换层面由用户定。
- 切面图只画了均匀密度和伪径向两种方式；是否补上变密度泊松一行由用户定。
