# T1FLAIR_AF16_hparam_v3_jiangyizhou — MC-GS 超参数敏感性：共享 Gaussian 数 N_s 与私有比例 N_p/N_s

2026-09-16。代码在 worktree `/data0/congcong/code/GS/mcgs_hparam_wt`（分支 `hparam-sensitivity`，自 `origin/autoresearch-jyz-af24` 5ea6189 切出），
设计 `docs/superpowers/specs/2026-09-16-hparam-sensitivity-design.md`，计划 `docs/superpowers/plans/2026-09-16-hparam-sensitivity.md`，
记录 `docs/superpowers/records/2026-09-16-hparam-sensitivity.md`。

## 什么数据

jiangyizhou 的 3D T1 + FLAIR，与 `T1FLAIR_AF16_v3_jiangyizhou` 完全相同：`data/` 里五个文件是指向该课题 `data/` 的符号链接
（k 空间、ESPIRiT 图、房屋 vd-Poisson 掩膜 `mask_af16_3d.h5`，md5 d08179b8e74f35846455983dbe299294，2966 点；
按整平面点数 ÷ 采样点数的口径 T1 15.19×、FLAIR 16.49×）。MC-GS 用三期冻结配方（`run_hp_mcgs.sh` 的配方块与 `run_af24_mcgs.sh`
逐字相同，已 diff 核对）：P6 冷启动、edge-TV λ_geo 0.10 / κ_q 0.85 / w_floor 0.10 / ε 0.04、kspace_high 2.2、warmup 500、
小波 db4 最细一层 T1 0.12 / FLAIR 0.10、共同边缘 0.001、相位 TV 0.03，4000 步，seed 42（`deterministic=False`，与所有注册运行相同）。

## 协议：只变一个旋钮

| 组 | 变量 | 取值 | 旋钮 |
|---|---|---|---|
| 1 容量 | 共享 Gaussian 数 N_s | 0.9M / 1.8M / 3.6M / 7.2M（N_p = 0） | `--set gaussian.initial_num_points=N`；`adaptive_control` 未开，训练中点数不变 |
| 2 私有比例 | N_p / N_s，N_s = 3.6M | 0 / 5% / 10% = 0 / 180k / 360k | `--set gaussian.num_private_points=N`；私有集是单独一套 Gaussian，**只渲染进 FLAIR 通道**，几何与幅值自有，只受 FLAIR 数据项训练，初始化 = 共享云等步长子集、幅值 ×0.1（trainer 默认，与三期 `p3_r2_priv03` 相同） |

0% 与第一组的 3.6M 是同一次运行（`hp_n36`），它同时是回归 anchor：复现注册的 `gs_joint_p3`（头内 PSNR 35.706 / 34.928），容差 0.02 dB。

用户 09-16 拍板：N_s 网格以配方为中心（收到的设计写 50k–400k，比工作点低 9–72 倍：体积 256×256×176 = 11.5M 体素，3.6M ≈ 0.31 个/体素）；
发表的 MC-GS 保持 N_p = 0，第二组是"往上加私有集有没有用"的消融，不是"去掉私有分支"；在 AF16 做而不是新抽 R=6 掩膜；
不报 Lesion CNR（三个 T1+FLAIR 受试者都是志愿者，无病灶），第二组同样报 PSNR / SSIM / NMSE。

## 口径

头内（|GT| > 0.05·P99.9|GT|）PSNR / SSIM / HFEN / NMSE 为主表，全体积 PSNR / NMSE 与复数口径头内 PSNR 作脚注；
`arc/cycle.py` 的打分 json 没有 head NMSE，`scripts/hp_summarize.py` 用存盘体积在同一头模上重算。

## 结果（完整表 `logs/hp_summary.md`，csv `logs/hp_summary.csv`）

| exp | N_s | N_p/N_s | T1 头内 PSNR / SSIM / HFEN / NMSE | FLAIR 头内 PSNR / SSIM / HFEN / NMSE | 对 3.6M（PSNR，dB） | 用时 | 显存 |
|---|---|---|---|---|---|---|---|
| hp_n09 | 0.9M | 0 | 35.357 / 0.9376 / 0.1142 / 0.01404 | 34.135 / 0.9165 / 0.1176 / 0.01647 | −0.349 / −0.793 | 47 min | 32.3 GB |
| hp_n18 | 1.8M | 0 | 35.592 / 0.9391 / 0.1133 / 0.01330 | 34.664 / 0.9227 / 0.1129 / 0.01458 | −0.114 / −0.264 | 53 min | 32.5 GB |
| **hp_n36** | **3.6M** | 0 | **35.706 / 0.9396 / 0.1139 / 0.01295** | 34.928 / 0.9261 / 0.1121 / 0.01372 | anchor：对注册值 0.000 / 0.000，PASS | 66 min | 32.9 GB |
| hp_n72 | 7.2M | 0 | 35.670 / 0.9392 / 0.1148 / 0.01306 | **34.980 / 0.9271** / 0.1126 / **0.01356** | −0.036 / +0.052 | 82 min | 33.8 GB |
| hp_p05 | 3.6M | 5% | 35.706 / 0.9396 / 0.1139 / 0.01295 | 34.933 / 0.9262 / 0.1122 / 0.01370 | +0.000 / +0.005 | 66 min | 33.0 GB |
| hp_p10 | 3.6M | 10% | 35.708 / 0.9396 / 0.1140 / 0.01295 | 34.938 / 0.9263 / 0.1122 / 0.01369 | +0.002 / +0.010 | 68 min | 33.0 GB |

全体积 PSNR（T1 / FLAIR）：0.9M 38.951 / 37.867，1.8M 39.159 / 38.337，3.6M 39.268 / 38.585，7.2M 39.222 / 38.628，5% 39.268 / 38.591，10% 39.270 / 38.598。
复数口径头内 PSNR：3.6M 32.497 / 31.883，7.2M 32.465 / 31.942，10% 32.499 / 31.893。用时是墙钟（一卡一训练，A100 80 GB 上独占）。

## 读法（spec §8 预先登记的判据）

- **第一组。** 0.9M → 3.6M 三项指标单调改善（头内 PSNR T1 +0.349、FLAIR +0.793 dB；SSIM、HFEN、NMSE 同向）。3.6M → 7.2M **不是饱和而是取舍**：
  T1 −0.036 dB（超出 0.02 dB 的判据），FLAIR +0.052 dB，SSIM / NMSE 同样一降一升。这与三期记录里 3.6M / 4.8M / 6.0M 的曲线
  （T1 35.706 / 35.697 / 35.675，FLAIR 34.928 / 34.962 / 34.977）是同一条线：T1 在 3.6M 最高，FLAIR 随容量继续涨。
  回答"是否依赖特定数量"：1.8M 到 7.2M 之间 T1 变动 ≤ 0.11 dB、FLAIR ≤ 0.27 dB；0.9M 明显欠容量（FLAIR 掉 0.8 dB）。
- **第二组。** 5% / 10% 对 0%：+0.000 / +0.005、+0.002 / +0.010 dB，SSIM / HFEN / NMSE 到小数第三、四位不动 → **null**，与三期
  `p3_r2_priv03`（0.3M 私有，+0.005 / +0.015）一致：共享云已经装下 FLAIR 需要的一切，私有集加的是参数不是信息。
  PSNR / SSIM / NMSE 不能支持"私有集保留对比度特异信息"的说法，那需要病灶数据，本课题没有。
- 图：`figures/hp_sensitivity_psnr_ssim.{png,pdf}`（左列 N_s，右列 N_p/N_s，上行头内 PSNR、下行头内 SSIM，T1 蓝圆、FLAIR 橙方，
  灰竖线 = 发表配方；`scripts/hp_fig_curves.py`，调色板经 dataviz 校验器六项全过）。

## 复现

```
cd /data0/congcong/code/GS/mcgs_hparam_wt
# 一次运行（EXP 必须以 hp_ 开头；SAVE_VOLUMES=1 存体积）
SAVE_VOLUMES=1 ./run_hp_mcgs.sh hp_n36 1 4000 --set gaussian.initial_num_points=3600000
# 或者整条队列（只用空卡，一卡一训练，持共享卡锁）
setsid nohup bash scripts/hp_queue.sh scripts/hp_queue.txt > results_arc/hp_queue.log 2>&1 < /dev/null &
# 汇总与图
env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/hp_summarize.py
env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/hp_fig_curves.py
```

`scripts/` 是 worktree 里同名文件的快照；`logs/` 有六个运行的训练日志、两个运行器日志、`hp_runs.log` 与汇总表；`mcgs/` 有六个打分 json。
存盘体积（6 × 0.71 GB）留在 worktree 的 `results_arc/`（gitignored），没有打包。

## 坑

- `run_hp_mcgs.sh` 里的 `TORCH_EXTENSIONS_DIR` 对这套代码无效：体素化 CUDA 扩展按源码哈希编进各 worktree 自己的
  `paired_gs/gaussian/build/`，worktree 之间不共享缓存。
- 卡锁 `Cardiac_FreeRunning_MCGS/work/.gpulock.<卡>` 与 gauss 循环、diff-mask、心脏三个调度器共用；本课题的 `hp_launch_one.sh` 持锁到运行结束。
- 两个运行器（卡 1、2 与卡 3、4）同时消费同一份队列，靠"有输出就跳过"不重复；运行器在所有本课题训练退出后自行退出。
- 冒烟 `hp_smoke_n09` / `hp_smoke_p05`（60 步）只是可行性检查，不是结果。
- 一次 4000 步在独占的 A100 上 47–82 min；AF24 课题日志里的 6–7.5 h 是四个作业共卡时的数字。
