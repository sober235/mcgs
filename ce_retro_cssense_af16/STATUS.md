# STATUS

新会话先读本文件，再读 README，关键数字先重跑。

`$TOPIC` = `/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16`，`$WT` =
`/data0/congcong/code/GS/mcgs_ce_retro_wt`，`$PY` = `/home/congcongliu/anaconda3/envs/GSTMI/bin/python`。

## 已完成且已验证

Task 15 只做四件事：`readme_tables.py`、`export_bundle.py`、`README.md`、`STATUS.md`。都已写完、检查过、提交。

两个脚本没有跑真数据，理由见"关键决定的为什么"。检查方式：

语法检查：
```
cd $WT/ce_retro_cssense_af16/scripts
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -c "import ast; ast.parse(open('readme_tables.py').read()); ast.parse(open('export_bundle.py').read())"
```
输出：两行 `... parses OK`。

`export_bundle.py --help`（确认不碰任何目录）：
```
cd $WT/ce_retro_cssense_af16 && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/export_bundle.py --help
```
输出：`usage: export_bundle.py [-h] [--no_raw]`，退出码 0，没有在任何地方新建 `export/` 子目录。

两个脚本各自用手搭的几字节假 topic（不是真数据）跑通了一遍完整流程：`readme_tables.py` 的排序、加粗最优列、
MC-GS−GS-single 表、强化保真表；`export_bundle.py` 的 `--no_raw`、`MANIFEST.tsv`、`md5sums.txt`，并且
`md5sum -c md5sums.txt` 全部校验通过。这两遍手动验证已经原样写成了 `scripts/tests/test_readme_tables.py` 和
`scripts/tests/test_export_bundle.py`。

全量测试：
```
cd $WT/ce_retro_cssense_af16 && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest scripts/tests -q
```
输出（2026-09-18 00:41 跑的，29 = 之前 25 个 + 这次新加的 4 个）：`29 passed in 159.00s (0:02:39)`

已核对过的历史数字（来自 `sim/meta.json`、`sim/mask/mask_af16.json`、`logs/check_sim.json`、
`trainer/_trainer_e778bf2/results_arc/smoke_joint.log`、`trainer/_trainer_e778bf2/SOURCE.txt`；不是这次跑的，是
Task 3/5 留下的，这次只是原样抄进了 README，没有编数字）：

- mask：seed 0、acc 3.883、原始 2555 点 trim 3 个到 2552 点、AF 16.00、ACS 24×24、无椭圆快门。
- sigma pre 1.36686 / post 1.44523；kres 复现 0.184868 / 0.149690（参考 0.18487 / 0.14969，差 < 2e-6）。
- WM SNR pre 34.445 / post 33.098。
- check_sim 三道门全过：伴随相对误差 3.81e-7 / 3.84e-7；CG-SENSE 无噪头内 PSNR 69.19 / 75.91 dB；sigma
  写入值与 meta 相对偏差 4.27e-5 / 1.64e-5。
- MC-GS 冒烟跑（360 万点、42 线圈、50 步）峰值显存 43.8 GB。
- trainer 快照 commit e778bf2，CUDA 编译产物 `tile_voxelize_cuda_0141c525_7aedd3b0.so` 是从
  `mcgs_af16_wt/paired_gs/gaussian/cuda_ext/build/` 拷贝来的，不是本地重编的。

## 待用户拍板

- `export_bundle.py` 不带 `--no_raw` 会拷 `sim/gs` + `sim/work_pre` + `sim/work_post` 的原始 k 空间，大约 14 GB。
  Task 16 打包时带不带这部分，我没有替你定，默认行为是"带"（不加 `--no_raw` 才会拷）。
- 工作树里有两个没提交的脚本：`scripts/_chain.sh`、`scripts/_cs_chain_pre.sh`。注释写着"controller helper, not
  part of the plan"，看起来是手动调度 GPU 队列用的临时脚本。这次 commit 我特意没把它们加进去（见下）。要不要正式
  收进 git，还是保持未跟踪、跑完就删，你定。
- L1-SPIRiT 核 5 在 42 线圈、ACS 24×24 下标定方程欠定（400 个方程 < 1050 个未知数，design §5 已知风险）。等网格
  结果出来，如果核 5 真的发散或明显更差，是写进 README 里但两个核都保留，还是干脆从正式表里去掉核 5，还没定。

## 下一步

1. 等 GPU 队列跑完剩下的臂。截至 2026-09-18 00:42 的快照：`recon/` 里有 ZF、CG-SENSE、CS-SENSE 全部 7 个 λ、
   L1-SPIRiT 大部分（ks3 六个 λ 全、ks5 才 3 个）、Joint SENSE 一个 λ、ZS-SSDU、3D-DIP（仅 pre）；GS-single、
   MC-GS、3D-INR、MC-INR、DIP-post 还没出现在 `recon/` 下。这是个会变的快照，不是事实，重开会话请重新
   `ls $TOPIC/recon` 和 `$TOPIC/logs/orch/jobs`。
2. `$PY scripts/extract_gs_volumes.py --exp ... --key ... --map ...`，把 GS 的 npz 结果搬进 `recon/`（三次调用，
   joint / single_pre / single_post，命令见 README §6）。
3. `$PY scripts/summarize.py` → `$PY scripts/enhancement_fidelity.py` → `$PY scripts/find_best_slice.py` →
   `$PY scripts/best_slice_mcgs.py`。
4. `$PY scripts/readme_tables.py`，输出原样粘进 `README.md` §4，替换掉 `NOT_RUN_YET`。
5. `$PY scripts/fig_three_planes_compact.py` 和 `$PY scripts/fig_enhancement.py` 出图，按 README §5 的两行格式
   （URL + 绝对路径）填进去。
6. `$PY scripts/export_bundle.py [--no_raw]` 打包。
7. 通读 README/STATUS，确认没有遗漏的 `NOT_RUN_YET`。

具体命令都在 `README.md` §6，是照现有脚本的 argparse 核实过的，不是照 spec 空想的。

## 已知的坑与别重做什么

- `logs/summary.json` 现在只有 4 行（zf/cgsense × pre/post），是旧快照。直接拿它跑 `readme_tables.py` 会在
  MC-GS−GS-single 那一步因为找不到 `gs_joint`/`gs_single` 崩掉（`StopIteration`）。等 `summarize.py` 重跑完再跑。
- `sigma_from_residual` 的 `n` 曾经少算了读出轴，会把 sigma 按 `sqrt(读出维度)` 的倍数高估。已经在 Task 5 修过，
  这次没有碰 `sim_cssense_af16.py`。完整推导和两个不同网格下的具体倍数都在 README §7；别在别的脚本里重犯同一个
  错误：用二维 mask 花式索引四维张量 `(coil, readout, ky, kz)` 会把读出轴留下来，不会消掉。
- `check_sim.py` 的伴随恒等式在线圈覆盖低的地方是奇异的，Task 5 已经把这个门限制到 `sum|S_c|^2>0.25` 的头内
  体素。这是预先授权的适配，不是把门槛改松；这次没有改这部分代码。
- design 文档 §9 写 trainer 快照在 `scripts/_trainer_e778bf2/`，实际在 `trainer/_trainer_e778bf2/`
  （`test_deps.py`、`run_gs.sh` 都指向后者）。README 已经把这个偏差写清楚了，以后按实际路径找，不要照 spec 目录树。
- 别碰 GPU 作业。这次全程只读了 `nvidia-smi` 和日志，没有杀任何进程，也没有改 `_chain.sh`/`_cs_chain_pre.sh`。
- zsh 下变量要加引号，`$VAR:more/path` 会把冒号后面的东西吃掉。

## 关键决定的为什么

- 为什么两个脚本不跑真数据：`logs/summary.json` 现在只有 4 行，离全部 40 多行还差得远，跑 `readme_tables.py`
  会直接崩；`export_bundle.py` 跑起来要拷十几到几十 GB，而且大部分臂还没产出。任务里明确说了不许跑真数据，只做
  `--help`/import 级检查。
- 为什么用子进程而不是 `import` 来测试这两个脚本：`readme_tables.py` 和 `export_bundle.py` 都没有
  `if __name__ == "__main__"` 保护，模块级代码一 `import` 就整个跑一遍。同一个 pytest 进程里 `import` 会把
  `arms.py` 的模块级 `R`（第一次 import 时就定死，之后不会因为换了 `RETRO_TOPIC` 而重新计算）缓存住，可能污染
  同一进程里其它也 `import arms` 的测试（`test_summary.py` 就是）。子进程之间不共享 `sys.modules`，不会互相
  传染。这不是凭空加的顾虑：先手动搭了两个几字节的假 topic，把两个脚本当命令行工具跑通了一遍确认没问题，再照着
  手动验证的样子写成 pytest。
- 为什么 `git add` 只加了这四个文件，没有照抄任务里给的 `git add ce_retro_cssense_af16`：工作树里还有两个未
  跟踪的调度脚本（`_chain.sh`、`_cs_chain_pre.sh`），跟这次的 README/STATUS/两个脚本没关系，commit message 也
  只提了这四样东西。把不相关的文件混进同一条 commit 既不诚实，也不是这次任务范围内的事，交给你决定要不要收。
- 为什么 mask 图给了真链接、另外两张图写 `NOT_RUN_YET`：mask 图是 Task 3 就跑完并经你审批过的产物（design §3
  的审批门），是既成事实；三面图和强化图现在 `figures/` 目录里根本不存在，写真链接就是编数字。
- 为什么 trainer 路径按实际位置写而不是照抄 spec 目录树：spec 第 9 节的目录树和实际落地（`test_deps.py`、
  `run_gs.sh` 都验证过）不一致，README 的职责是让下一个会话少走弯路，照抄一个会读失败的路径没有意义。两处
  都在 README 里写清楚了，没有回头去改 spec 文档本身。
