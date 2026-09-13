# STATUS — 2026-09-12 10:05（分支 `autoresearch-jyz-af12`，worktree `mcgs_af12_wt`）

本工作树只承载 jiangyizhou AF12 autoresearch 一条线。先读 `docs/handoff/2026-09-12-autoresearch-jyz-af12.md`（五段交接），
再读 `autoresearch_jyz_af12.md`（程序、规则、逐轮日志、verdict）和账本 `autoresearch_jyz_af12.tsv`；英文总结在
`docs/superpowers/records/2026-09-12-autoresearch-jyz-af12.md`。

## 一句话
P10 = P6 + 小波先验改为 db4、只用最细一层、T1 0.12 / FLAIR 0.10，4000 步头内 PSNR 36.944 / 36.268，对重扫过 λ 的所有对手在
头内 PSNR/SSIM/HFEN 六格全部第一、所有完整脑层全部第一；对 P6 +0.191 / +0.103 dB。已登记进 jyz AF12 课题表（gs_joint_ar）。

## 待拍板
尺度学习率 0.005（+0.011 / +0.048，T1 HFEN −0.0002）、sym4、6000 步是否纳入；CLAUDE.md 的"最优 12 倍"节是否改指 P10；
是否做留出受试者检验。

## 二期（2026-09-12 下午起，分支 `autoresearch-jyz-af12-phase2`，同 worktree）
用户拍板：当前 AF12 结果 = T1 全 AF12 的 P10 紧凑图 + FLAIR 设计 B 的 P10 紧凑图（CLAUDE.md "Current AF 12 result"）；
设计 B 从此是工作设置，二期循环直接在设计 B 下调参（runner `autoresearch_jyz_af12_p2.sh`，程序 `autoresearch_jyz_af12_p2.md`，
账本续用 `autoresearch_jyz_af12.tsv`，行前缀 `p2_`）。杠杆 = 配置旋钮 + 分对比度 λ_geo（commit 30f8513，单测 4 个，默认路径逐位不变）；
边际项打包（scale lr 0.005 + 6000 步）作为一个候选按规则 7 判。第一轮三条链 15:59 在 2/3/5 号卡启动，见程序的 Round 0。

## 二期结果（2026-09-13 04:16 循环闭合，分支 `autoresearch-jyz-af12-phase2`）
**P13 = P10 + `gaussian.rotation_lr=0.008`**（出厂默认 0.0003，其余一字未改），runner `autoresearch_jyz_af12_P13.sh`。
设计 B 下头内 PSNR/SSIM/HFEN：T1 36.969 / 0.9504 / 0.0948（全 AF12 口径），FLAIR 36.683 / 0.9444 / 0.0904（AF11 掩膜）；
对 P10 的增益 T1 +0.025、FLAIR +0.046 dB，其余五格持平或更好，锐度不变；对最强对手余量 +0.543 / +0.926 dB；
两对比度三方向所有完整脑层全部第一；两种子平均 37.005 / 36.703。27 个实验、13 个跑满 4000 步。
先读 `docs/handoff/2026-09-13-autoresearch-jyz-af12-phase2.md`，再读 `autoresearch_jyz_af12_p2.md` 与
`docs/superpowers/records/2026-09-13-autoresearch-jyz-af12-phase2.md`。
待拍板：打包候选（scale lr 0.005 + 6000 步）按规则判为 reshape 未并入，可人工推翻；`gaussian.scale_lr` 已标定成"FLAIR 旋钮"
（0.003/0.005 买 FLAIR +0.023/+0.05 dB，花 T1 第四位小数），是否启用由用户定；未做留出检验。

## 三期结果（2026-09-13 收尾，分支 `autoresearch-jyz-af12-phase3`）
用户口径：**GS-single 算对比方法**。据此二期留下的开口是 FLAIR（比自家单臂低 0.037 dB）。三期从并行的 AF16 线借了两个项并在本数据上重新标定，
得到 **P14 = P13 + 共同边缘 0.001 + 相位 TV 0.01**：六格全部超过所有臂（含 GS-single），T1 37.015/0.9516/0.0928、FLAIR 36.360/0.9415/0.0935，
余量 +0.150/+0.036，对 L1-ESPIRiT +0.589/+0.602 dB；联合减单臂从 +0.066/−0.038 变成 **+0.150/+0.036**；种子地板 0.0037/0.0026。
**留出检验通过**（预先写定判读规则）：P14 一字未改跑 huangyuwei，FLAIR 联合减单臂 +0.030，对该课题最强经典臂 +0.670/+0.668 dB。
这是本线第一次有留出证据。先读 `docs/handoff/2026-09-13-autoresearch-jyz-af12-phase3.md`。
待拍板：容量 4.8M 的 4000 步验证（1500 步是 keep，FLAIR +0.051、T1 持平）；是否把对外口径从 P13 换成 P14；P14 的复数口径未测。
⚠️本期两次"命令没按预期执行"：pgrep 用 ERE（`\|` 不匹配）、zsh 不对 `$var` 分词——两次都用同一个坏构造去验证而误判成功。验证要换工具。

## MC-INR（Niessen et al. 2025 复现）并入 AF12 课题（2026-09-13 晚，用户要求）
论文多对比度 INR 的复现（仓库 `/data0/congcong/code/GS/mcinr_niessen2025`，结果课题 `T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou`）
已登记为本课题的**对比方法** `mcinr`。它跑的就是本课题的 AF12 掩膜、k 空间和 ESPIRiT 图，重建体软链进 `recon/`，
并用课题自己的 `t1t2_common.Pair.score` **重新打分**（`scripts/score_mcinr_arm.py` → `logs/mcinr_{T1,FLAIR}.json`），
与对方自报数字九项指标最大相对偏差 **1.9e-7**。
- 头内 PSNR/SSIM/HFEN：T1 32.192 / 0.8407 / 0.1778，FLAIR 33.042 / 0.8696 / 0.1391 —— 两个对比度都排 **8/12**，
  高于 3D-INR、3D-DIP、P-LORAKS、ZF，低于 CG-SENSE（−1.32 / −0.92 dB）、L1-ESPIRiT（−4.23 / −2.72）、MC-GS P14（−4.82 / −3.32）。
- **只看头内**：它往空气里填噪声，T1 整卷 PSNR 30.151 比零填充（30.240）还低。
- **不动任何余量**：P14 六格的 binding arm 加它之前之后都是 GS-single，结论原样成立。
- 图：交付的 P14 那对按同切片同版式重渲成十方法版（后缀 `_p14_mcinr`，九方法的 `_p14` 原样保留）；两张渲染日志与九方法版逐字节一致。
- `summarize.py` / `results_af12_md.py` / 三个 figure 脚本都加了这一列。
- ⚠️**这一行对 MC-INR 并不公平，README 已写明**：经典臂的 λ 和所有 MC-GS 配方都在本受试者上按真值选过，
  ZS-SSDU/3D-DIP/3D-INR 用留出 k 空间早停，而 MC-INR 用论文原配置、在这里一个参数都没选。但这解释不了它的位置——
  整轮 λ 重扫对经典臂最多值 +0.092 dB，而没有 λ 可调的 CG-SENSE 仍高它 1.323 / 0.916 dB。
- ⚠️**检验的是方法不是协议**：并行会话 2026-09-13 22:49 在 mcinr 仓库提交了保真度自审
  （`mcinr_niessen2025/docs/2026-09-13-fidelity-review.md`）：代码逐行对齐，协议没对齐（缺互补采样、两次扫描而非一次、
  评估掩膜与归一化不同）。所以这行只能读成"在本课题共用掩膜、两次扫描的设置下低于 CG-SENSE"，不能读成"论文没复现"。
  那份自审列的补救实验都没跑。AF16 尚未登记。
