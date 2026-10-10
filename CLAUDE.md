# mcgs worktree `mcgs_ce_mask_wt`(分支 `ce-af16-mask-audit`)— 新会话入口

平扫 / 增强(CE)回顾性 AF16 线,vdPoisMex 掩膜:十一臂对比、MC-GS autoresearch(第 1–5d 阶段)、对比方法逐臂审计(2026-10-09/10)。

> **⏸ 2026-10-10 08:56 起暂停中,不占任何 GPU、没有进程在跑。** 剩下的只有收尾(纯 CPU):恢复步骤在
> `ce_retro_cssense_af16/STATUS.md` 第一节,推荐的一条命令是带 `WAIT="dip_sup_b96_tv0.03_pre"` 重起 `scripts/cmp_final_chain.sh`。

## 先读什么(按顺序)

1. `ce_retro_cssense_af16/STATUS.md` 顶部一节(最新在上)。
2. `ce_retro_cssense_af16/CMP_ARMS_AUDIT.md`:对比方法审计,逐臂诊断、先登记的选择规则、每个探针的数、全表。
3. `ce_retro_cssense_af16/autoresearch_vdpois.md`:MC-GS 的 autoresearch,定稿 `ar_i1_k30p90`(34.690 / 36.940)。
4. `ce_retro_cssense_af16/VDPOIS_LINE.md`、`MASK_SWAP.md`:09-30 登记的十一臂线,以及为什么换掩膜。

上一会话的结论只当线索;关键数字先用下面的命令重跑再动手。

## 数据与课题目录(服务器,不进 git)

- `N=/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois`:全部臂的运行(`recon/`、`logs/`),对比方法审计的队列日志在 `logs/cmp/`。
- `..._vdpois_ar2`:MC-GS 定稿的确认目录(10-09)。
- `..._vdpois_cmp`:对比方法审计的确认目录(`scripts/cmp_final_chain.sh` 建,10-10)。

## 环境与命令

- `PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python`;一律 `env PYTHONNOUSERSITE=1 PYTHONPATH= RETRO_TOPIC=$N RETRO_TRAINER=$N/trainer/_trainer_e778bf2_ar`。
  MC-INR 走 `scripts/run_mcinr.py`,它在 `GR_pt1.10` 环境里调 `/data0/congcong/code/GS/mcinr_niessen2025`。
- 测试:`cd ce_retro_cssense_af16 && env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= $PY -m pytest scripts/tests -q`(31 passed,约 3 分钟)。
- 每臂按自己的规则选赢家:`cd $N && env RETRO_TOPIC=$N PYTHONNOUSERSITE=1 PYTHONPATH= $PY <worktree>/ce_retro_cssense_af16/scripts/cmp_select.py`。
- GPU 队列:`scripts/cmp_queue.py --jobs <tsv> --lanes <cards>`(固定车道,不自己挑卡,启动后改清单无效)。

## 规矩

- 提交作者 `Congcong Liu <sober235@users.noreply.github.com>`,提交者邮箱 liuc23539@gmail.com;不写任何 AI 署名行。
- 本分支不合进 main(mcgs 各线不交叉合并);交接打 `handoff/YYYY-MM-DD-...` 标签;push 只在用户点头后。
- 共享机器:一卡一个训练;跑前 `nvidia-smi`。zsh 不对 `$VAR` 分词,env 赋值要写开。
- 任何情况下不删数据和代码(钩子会拦);需要删的列清单交用户。
