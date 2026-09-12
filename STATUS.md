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
