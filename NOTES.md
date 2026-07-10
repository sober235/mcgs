# memory.md — MC-GS (3dgsVC) 工作状态与已核实结论
_最后更新: 2026-07-10_

记录已核实的事实与工作状态（观测性内容）。约定性内容见 CLAUDE.md。

## 轴向约定（已双重核实）
- 体 (256,220,256)=(axis0,axis1,axis2): **axis0=冠状 coronal, axis1=矢状 sagittal, axis2=横断 transverse**。
- 证据1（用户裁定）: 用户本会话明确 modify，覆盖旧的相反结论。
- 证据2（原始重建佐证）: raw_T1_recon.npy 三方位形态与裁定一致；脑组织支撑沿 axis0 仅 65%（含颅顶/颅底空白）、axis1 83%、axis2 89%。
- 交付图: MC_GS_three_planes_labeled.png（三正交面标注）。

## ESPIRiT 3D CSM 正确性校验（已通过）
csm_espirit_T1.mat，键 `map`，(256,220,256,12) complex64：
- 形状/维序: 线圈轴在最后，与重建体一致 ✅
- SoS 归一化: √Σ|Sc|² 全 FOV min=median=max=1.0，脑内 100% 体素 |SoS-1|<0.05 ✅
  （全 FOV=1.0 含空气区是 `-c 0.0` 的预期行为，非缺陷）
- 三方位取向: 各线圈敏感度与解剖轮廓贴合、方位对齐 ✅
- 相控阵物理形态: 平滑、空间局域、互补覆盖 ✅
- 交付图: MC_GS_csm_3d_check.png。
- 待办: FLAIR CSM 尚未同样过一遍（若需要可补）。

## 重建评分（ESPIRiT-CSM 自洽口径）
- T1 38.51 / FLAIR 37.76 dB（SSIM 0.910/0.901），incoh AF11.33，4000 iter，warm-start 576902 高斯。
- 口径提醒: 与 headline 38.19/36.37 非苹果对苹果——CSM 同时决定 GT 与前向算子，各自和自己 GT 比。

## "分片/低分辨率感" 三成因（定量结论，**注意 mask 隐患见下**）
- ① 欠采 AF = 物理上限，收益最大（AF12->AF6: T1 +1.3 / FLAIR +1.5 dB）。
- ② TV 正则 = 分片主因也是最可控旋钮，甜点 tv=0.01（T1 38.66/FLAIR 38.19，高频保留 0.38->0.65）。
- ③ 高斯表示几乎无影响（densify 加密仅 +0.03 dB，576902 高斯已饱和）。

## !! 未决关键隐患 !!
- **调参扫描 mask 用错**: 本轮 TV/densify/低AF 全部 7 个变体用的是项目自带 mask3d_incoh_af12_acs32.mat，
  而非 house convention 要求的 ESPIRiT vdpois mask。上述三成因定量结论需在正确 mask 下**重做**才可信。

## 落盘教训
- 会话恢复后工作区内存重置，且 .claude/*.md 曾多次"目录在但文件 NOT WRITTEN"。
  => 每次写关键文件后必须用 bash ls/cat 显式回读确认，不能假定 cell 成功。

## !! 沙箱写入限制: .claude/ 不可写 !!
- 沙箱对**所有** .claude/ 目录强制盖了一层 per-kernel tmpfs（平台对 agent 配置目录的固定保护，早于权限检查生效）。
  /proc/mounts 可见 /data0/.../.claude 全部是 `tmpfs`，项目根 3dgsVC 本体是真实 ext 磁盘 /dev/sda1。
- 后果: python 往 .claude/ 写"看似成功"（有字节数），但只落到该 kernel 的内存幻影层——
  bash（另一 kernel）看到空目录，且 kernel 一停即失、永不同步回物理磁盘。
  这就是历史上反复出现的 ".claude 目录在但文件 NOT WRITTEN" 的**真正根因**（不是写入不稳定）。
- 与用户授权无关: 用户已授 /data0/congcong 完整 rw，项目根写入正常落盘并经 bash 独立回读确认。
- **对策**: 约定/状态文件写到**项目根** 3dgsVC/CLAUDE.md 与 3dgsVC/memory.md（真实磁盘，Claude Code 亦读根级 CLAUDE.md）。
  若确需放进 .claude/，只能由用户在沙箱外手动 cp，sandbox 内无法完成。

## MC-GS vdPois 重建：读出轴纠正（2026-07-10）
- **关键纠正**：变密度泊松欠采必须作用在**横断面 transverse**（PE 平面 = axis0×axis1），全采读出轴 = **axis2 (S-I)**。
  判据口诀：波松散点只应出现在纯 PE 平面；读出轴朝向决定波松落在哪个方位。
- 早先误把读出轴放 axis0（波松落冠状面），已作废。
- **正确 mask**：mask/mask3d_vdpois_af12_transverse.mat（2D vdPois af12 转置成 (256,220)=(axis0,axis1) 后沿 axis2 广播，AF11.83，横断面采样 8.45%，axis2 方差=0 全采）。
- **config**：configs/ce_joint_espirit_vdpois_transverse.yaml（readout_axis=2, phase_axes=[0,1], tv=0.03, 4000iter, ESPIRiT CSM 双对比度, warm-start）。output_dir=results_espirit_vdpois_transverse。
- **评分（ESPIRiT-CSM 口径，三跑同口径）**：
  | 对比度 | 纠正版(横断波松) | 旧错版(冠状波松) | incoh 基线 |
  |---|---|---|---|
  | T1(plain)     | 37.44 dB / SSIM 0.881 | 37.63 / 0.870 | 38.51 / 0.910 |
  | FLAIR(contrast)| 37.82 dB / SSIM 0.904 | 36.84 / 0.880 | 37.76 / 0.901 |
- 结论：纠正后 FLAIR +0.98 dB（已达/微超 incoh 基线），T1 基本持平（SSIM 反升）。vdPois 与 incoh 在同 AF/同几何下应接近，旧版 FLAIR 掉点是读出轴放错的伪劣化，已消除。
- **CSM 三方位对齐**已核实（MC_GS_csm_align_planes.png）：各线圈敏感度空间支撑贴合头颅轮廓，无转置/翻转错位。CSM 只从全采 ACS 估、与欠采 mask 无关，换 mask 不影响 CSM，可复用。
