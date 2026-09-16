# 交接 — gauss 掩膜调参循环（分支 `autoresearch-gauss-af16`，worktree `mcgs_gauss_wt`），2026-09-16

新会话先读 `CLAUDE.md` 顶部一节 → `STATUS.md` → `docs/superpowers/records/2026-09-16-gauss-loop.md` → `autoresearch_gauss.md`。

## 结论

目标（用户 09-16）：MC-GS 在 Gaussian 掩膜上超过所有对比方法 0.5 dB，GS-single 算对比方法且停在冻结配方。
**没达到。** 48 发 1500 步筛选 + 6 发 4000 步（4 个晋级 + 2 个种子重复）把参数空间封口。两个工作点，六格全第一：

| 4000 步 | T1 | FLAIR | 对冻结 GS-single | 对同配方 GS-single |
|---|---|---|---|---|
| 父配方（三期冻结） | 34.909 | 34.317 | +0.138 / +0.023 | — |
| rsw = 父 + rotation_lr 0.008 + scale_lr 0.004 + w_floor 0.05 | 34.923 | 34.362 | +0.152 / +0.068 | 未跑 |
| rswt = rsw + 矢量 edge-TV | 34.976 | 34.353 | +0.205 / +0.059 | +0.135 / +0.058 |

门槛 34.771 / 34.294，目标 35.271 / 34.794，还差约 0.3 / 0.44 dB。剩下的不是参数，是耦合机制（改代码）。

## 可粘贴的核对

```
cd /data0/congcong/code/GS/mcgs_gauss_wt; PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
cut -f1,7,8,9,10,11,25 autoresearch_gauss.tsv | grep -E "^exp|g_r2_|g_r3_" | column -t -s $'\t'
PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/gauss_judge.py          # 全部筛选对 control 的六格差与判定
PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/bar_gauss.py            # 重建门槛（只读 gauss 课题）
```

## 待拍板
1. 引用 rsw 还是 rswt。
2. 要不要为调参后的 MC-GS 建正式课题目录并出对比图（体积在 `results_arc/g_r2_rsw*_full4k_volumes.npz`）。
3. 要不要开耦合机制这条线（另立分支、改代码），还是接受现有余量。

## 坑
见 `STATUS.md` 第 4 节（tmux 服务器别按 pid 杀、队列原子写、cycle.py 的 warmup 守卫、init_mode 只有 head/intensity、warmup 是死旋钮、内存看门狗、卡 3/4/7）。

## 会话边界
本分支与 tag `handoff/2026-09-16-gauss-loop` 推 origin（私有仓库），不合 main（跨血脉禁止合并）。
