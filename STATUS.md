# STATUS — 2026-09-12（worktree `mcgs_af16_wt`，分支 `autoresearch-jyz-af16-phase2`）

本工作树只承载 jiangyizhou AF16 一条线，**已定格（用户 2026-09-12 拍板）：不再跑，代码与参数和 AF12 线隔离。**

## 1. 已完成且已验证
- 一期 P6（magnitude 口径主推）：4000 步头内 PSNR 35.552 / 34.807，对 L1-ESPIRiT +0.343 / +0.304，头内 SSIM、HFEN 也第一；
  记录 `docs/superpowers/records/2026-09-11-autoresearch-jyz-af16.md`，交接 `docs/handoff/2026-09-11-autoresearch-jyz-af16.md`。
- 二期 p6ptv（P6 + 相位 TV 0.05）：复数口径也第一（32.322 / 31.851 对 L1 的 32.201 / 31.718），锐度 0.876 / 0.837，
  magnitude 35.311 / 34.740（比 P6 低 0.24 / 0.07）；记录 `docs/superpowers/records/2026-09-12-autoresearch-jyz-af16-phase2.md`，
  交接 `docs/handoff/2026-09-12-autoresearch-jyz-af16-phase2.md`。账本 `autoresearch_af16_p2.tsv` 20 行。
- 课题 `Results_20260908/T1FLAIR_AF16_v3_jiangyizhou`：gs_joint_ar / gs_single_ar（P6）、gs_joint_p2 / p2b（二期）已登记，图在 figures/。

## 2. 待用户拍板
- 对外引用 P6（magnitude，惯例）还是 p6ptv（双口径、更锐、T1 magnitude 低 0.24）。
- 是否做留出受试者（从未跑过）。

## 3. 下一步
- 无（已定格）。若重开：不带相位 TV 的 AF12 小波改形迁移测试（用户 09-12 已拒绝）。

## 4. 已知的坑
- 每卡只放一个训练；运行器只认 `p2_` 名字、无 CONFIG 覆盖（隔离设计）；两条 AF16 分支不要互相合并；引用数字必须说明口径。

## 5. 为什么定格
- magnitude 目标一期已达，复数目标二期已达；剩下的杠杆属于 AF12 线，用户要求两线可分。
