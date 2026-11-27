# diff-mask 测试集研究：断点记录（2026-11-27 06:09 PST，服务器断电前）

时间为本机时钟（PST）。本文件供服务器重新开机后接着做。

## 已完成（不需要重跑）

- 均匀密度、伪径向两种采样方式，AF16，6 次测试扫描，11 个方法：全部跑完并统一重新评分，12 / 12 个格子完整（`table.json`、`stats.json`、`table.md`、`RESULTS.md`、`figures/testset_summary.*`）。
- 6 次扫描的切面对比图（文章版格式，放大框和箭头都在脑内）：脚本 `fig_testset_record_format.py`，预览和记录在 `figures/slices_preview/`；全分辨率 PNG、600-dpi PDF 和数据在服务器 `codex_artifacts/mcgs_diffmask_testset_20261122/figures_slices_article_inbrain/`（32 个文件，校验值已核对）。
- 三种采样方式合并脚本 `merge_patterns.py` 和中间表 `table_3patterns.md` / `.json`（变密度泊松只有 4 / 6 次扫描，是中间结果）。
- 本研究在服务器上没有任何正在运行的作业，不占 GPU。所有结果都在服务器磁盘上，断电不影响。

## 开机后要做的

1. 等 diff-af 会话通知它的统计表到 24 / 24（断电前：对比方法 183 / 192 行，vol2 还缺 3D-DIP AF24、3D-INR、MC-INR；它的表不是最终版，AF16 格子现在不能当最终结果用）。
2. 取到 diff-af 的最终 `t1flair_af_testset_20261121/table.json` 后，在本机运行
   `/usr/bin/python3 autoresearch/diffmask_testset_20261122/merge_patterns.py`，得到三种采样方式的最终合并表。
3. 把合并结果写进 `RESULTS.md`，提交并推送到 GitHub 分支 `diff-mask`（需用户同意）。

## 还没有得到用户确认的事

- “径向”按 2026-09 的伪径向掩膜处理，待确认。
- 切面图取的是头部面积最大的横断面（偏低），在这些切面上伪径向 T1 的 MC-GS 排第 2–3；是否换层面由用户定。
