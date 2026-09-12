# STATUS — 2026-09-12 09:20（分支 `autoresearch-jyz-af12`，worktree `mcgs_af12_wt`，未推送）

## 1. 已完成且已验证
- **jyz T1+FLAIR AF12 autoresearch 循环（09-11 19:40 起）**：从用户暂定的最优 12 倍结果 P6（36.753 / 36.165）出发，
  四次晋级到 **P10 = P6 + 小波先验改形（db4 基、只罚最细一层、T1 0.12 / FLAIR 0.10）**，4000 步头内 PSNR
  **36.944 / 36.268**（对 P6 +0.191 / +0.103），头内 SSIM 0.9503 / 0.9397，头内 HFEN 0.0947 / 0.0958，锐度 0.858 / 0.823。
  对手 λ 已在 jyz 上重扫（L1-ESPIRiT T1 λ0.002 = 36.426、FLAIR λ0.004 = 35.757，每格取所有 λ 最好值），
  六格余量 PSNR +0.518 / +0.511、SSIM +0.0030 / +0.0030、HFEN +0.0087 / +0.0042，全部为正。种子 43 复现到 0.002 / 0.001 dB。
  ```
  cd /data0/congcong/code/GS/mcgs_af12_wt && SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh <EXP> <GPU> 4000
  # results_arc/ar5_best_C10_T112_full4k_score.json: T1 head_psnr=36.944 head_ssim=0.9503 head_hfen=0.0947
  #                                                  FLAIR head_psnr=36.268 head_ssim=0.9397 head_hfen=0.0958
  ```
- 账本 63 行（`autoresearch_jyz_af12.tsv`，`scripts/autoresearch_jyz_af12_view.py` 按当前对手线重读六格）；
  程序与逐轮日志 `autoresearch_jyz_af12.md`；总结 `docs/superpowers/records/2026-09-12-autoresearch-jyz-af12.md`。
- P10 已登记进课题表（`Results_20260908/T1FLAIR_AF12_v3_jiangyizhou`，行名 gs_joint_ar），README 加了 λ 重扫一节；
  课题脚本新增 add_gs_arm.py、complex_gauge.py、fig_three_planes_compact.py（橙黄标注紧凑三面图，移植自 AF16），均已镜像。
- 附加行：两种子平均 36.990 / 36.295（2 倍算力）；GS-single T1 36.897（joint +0.047）；sym4 ≈ db4；
  复数口径（|rec−GT|）L1-ESPIRiT 反超 0.03 / 0.07 dB（logs/complex_gauge.json）。

## 2. 待用户拍板
- 尺度学习率 0.005 是否叠上 P10：4000 步 36.955 / 36.316，最小余量 +0.529，但 T1 HFEN 滑 0.0002（超地板），按规则未采纳。
- 课题主推行：单种子 P10 还是两种子平均。
- 是否推分支到 origin（本地未推）；`main` 与本血脉无共同祖先，不合并。
- 是否再跑一个留出受试者（vol2/5/6/7 或 tangbing）。

## 3. 下一步
- 在跑（约 10:00 前落地）：P10@6000 步（3 号卡，1.5 倍算力行）、GS-single FLAIR@4000（4 号卡）。落地后：登记单臂、
  刷新课题表、出紧凑三面图（`fig_three_planes_compact.py --contrast T1/FLAIR --slices 174 110 47 --mask_error`）、
  把最终表与图写进 record 第 7 节和课题 README、更新本文件、推分支。

## 4. 已知的坑与别重做
- 1500 步筛选看不出容量与尺度学习率在 4000 步的 T1 过拟合（都是 T1 换 FLAIR 的 reshape）；这两类只在晋级预算判。
- 小波强度在 3 层 db2 下上下都是 reshape；只有先改形（db4、1 层）之后 T1 才能吃到 0.12（0.14 锐度门不过）。
- 已证否/null：λ_geo 0.07/0.13、κ_q 0.90、w_floor 0.2、eps 0.03、warm-up、密度学习率、init_head_frac、k 空间频带权重、
  小波 4 层、db6、wav_eps。
- 打分脚本 score_x2.py 在 CPU 上会被内存看门狗杀，须 CUDA_VISIBLE_DEVICES 指到一张卡。
- 运行器现在烤入 P10；改配方只能靠 commit（程序规则）。别在 0–2 号卡放东西（AF16 第二阶段），6/7 与别人共享。

## 5. 关键决定的"为什么"
- 对手 λ 在本受试者重扫、每格取所有 λ 最好值：和循环同等特权，差距才算数。
- 六格不倒退规则优先于最小余量：用户要的是"所有指标拉开"，平滑换 PSNR 的旋钮一律拒绝。
- 晋级预算保持 4000：6000 步只 +0.007 / +0.014，算力 1.5 倍，留给最终配方单列一行。
- 终配方取 db4 而非 sym4：两者在地板内等价，终配方三件套已在 db4 上跑，不值得重跑。
