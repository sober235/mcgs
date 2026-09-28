# 多部位 3D 多对比 AF16 验证:总体安排、公共框架与膝(SKM-TEA)设计

- 日期:2026-09-28
- 分支:`nonbrain-infra`(从 `widen-mcgs-af12-16-20-24` 尖端 2eef592 切出),工作树 `/data0/congcong/code/GS/mcgs_nb_infra_wt`
- 范围:本 spec 覆盖子项目 0(公共框架)和子项目 1(膝)。肝、乳腺、胸腹各自另写 spec。
- 状态:五节设计已逐节经用户确认(2026-09-28),待用户审阅本文件。

## 0. 目标

MC-GS 目前只在脑 T1w+FLAIR 上验证过。用户要求在膝、肝、乳腺、胸腹四个部位做 3D 多对比验证,
加速倍数全部固定为 AF=16,各数据集的参数必须独立,互不影响,也不能影响脑课题的参数和结果。

## 1. 总体安排

### 1.1 子项目与顺序(用户 2026-09-28 选定方案 A 与此顺序)

| 顺序 | 子项目 | 数据 | 采集 | 状态 |
|---|---|---|---|---|
| 0 | 公共框架 | 无 | 无 | 本 spec |
| 1 | 膝 | SKM-TEA(本机 `/data2/congcong/data/FM_data/SKM-TEA/`) | 3D 笛卡尔 qDESS 双回波 | 本 spec |
| 2 | 肝 | PT-fNAV(Zenodo 8338128) | 3D 径向 phyllotaxis,8 回波,自由呼吸 | 下载中;另写 spec |
| 3 | 乳腺 | fastMRI Breast | stack-of-stars,造影前段 + 造影后段 | 需用户本人签 breast 的 DSA;另写 spec |
| 4 | 胸腹 | MR-linac 5D(Zenodo 21278895) | 3D 笛卡尔 bSSFP | 下载中;另写 spec |

下载(用户 2026-09-28 批准,只下载与校验 md5,不解压不处理):根目录 `/data2/congcong/data/nonbrain_public/`,
脚本 `scripts/zenodo_fetch_loop.sh` 与 `scripts/launch_both.sh`,每个数据集目录有 `manifest.txt`、`MANIFEST.md5`、`logs/`,
完成后写 `download_status.txt`。开始时实测约 3.6 MB/s,总量 159.13 GB。

### 1.2 后续子项目开工前要用户决定的事(记录在此,不在本 spec 内解决)

- 径向数据上 AF=16 的定义。探针结果:按 Nyquist(3D 全直径径向 πN²/2,stack-of-stars πN/2,出处 Schauman 等 MRM 2026,PMC12746353),
  肝 N=192 时 Nyquist 57,906 条、已采 9,328 条(R 6.21),AF16 保留 3,619 条(已采的 38.8%);乳腺 N=320 时面内 Nyquist 503 条、已采 288 条,AF16 保留 31 条。
- 没有全采样时的参考图:肝和乳腺只能用全部已采数据的重建作参考(不是真值);胸腹任何运动态都不是全采样,附带参考图是模型输出。
- 肝 8 个回波选哪两个,或扩展到 M 个对比度。
- 胸腹的第二个对比度:发布版只有 bSSFP;论文说每人还采了 OPRA 扰相 GRE(TR/TE 3.4/1.6 ms,FA 10°,60 s),未发布。
- 乳腺:本人签 DSA(knee/brain 的授权不覆盖 breast);造影后段信号随时间变化;两段之间可能有位移。

探针全文:`/home/congcongliu/outputs/nonbrain_raw_survey_2026-09-28/probe_2026-09-28.md`。

## 2. 参数隔离规则

### 2.1 分支与目录

- `nonbrain-infra` 只放共享改动(第 3 节)与各子项目 spec。
- 每个数据集一条分支、一个工作树,从 `nonbrain-infra` 切出:膝为 `nb-knee-skmtea`,工作树 `/data0/congcong/code/GS/mcgs_nb_knee_wt`;
  以后依次为 `nb-liver-ptfnav`、`nb-breast-fastmri`、`nb-thorax-mrlinac`。
- 合并方向只有 `nonbrain-infra` → 数据集分支。数据集分支之间永不合并,也不合回 widen 或 main。
- 膝的课题目录:`/data0/congcong/code/GS/Results_20260908/KNEE_SKMTEA_AF16/`(`data/ recon/ scripts/ logs/ figures/ README.md`),
  仓库里镜像不含体积的部分。

### 2.2 参数独立

- 膝的 spec 目录 `specs/knee_skmtea/` 放完全展开的 yaml:MC-GS 联合臂、GS-single 的 E1 与 E2、各对比方法的 λ 网格与冻结值、打分设置、
  受试者名单与数据路径。不继承、不 include 任何别的文件。
- 每个受试者一份展开后的 spec(由模板加受试者名单生成并入库);测试钉住"除数据路径外与模板完全相同"。
- 所有代码默认值都显式写进 spec:先用 `run_spec.py --audit` 短跑,列出训练器读到默认值的全部键及其值,补进 spec。
  此后任何运行只要读到 spec 里没有的键就中止。
- 每个运行的 json 记 spec 的 sha256、git commit(含未提交改动标记)、数据文件 md5 和完整展开配置。
- 调参只用膝的 2 名验证集受试者。起点可参考脑 AF16 的数值,最终值只按膝的验证集选;选定后冻结进 spec,测试集只跑冻结值。
- 对比方法的代码复制一份到膝课题的 `scripts/`,不从任何别的工作树 import(脑课题的 `t1t2_common.py` 现在从 `mcgs_af16_wt` import,
  该工作树 2026-09-28 检出的是 `gif-gaussian-to-images`;膝不重复这个问题)。
- 结果只写膝工作树和膝课题目录;运行名一律以 `knee_` 开头。
- 膝分支上有测试钉住共享文件的 md5:脑的配置、`run_*_mcgs.sh`、`arc/cycle.py`、`paired_gs/trainers/joint_trainer.py`、
  `diag_artefact/headonly.py`、`metrics.py` 必须与 2eef592 一致。

## 3. 公共分支 `nonbrain-infra` 的改动

### 3.1 现有文件只改一处

`paired_gs/data/dataset.py:372`:回顾性真值归一化调用 `torch.quantile(ground_truth_mag, p)`。元素数超过 2^24(16,777,216)时
torch 报错,改为此时用 numpy 分位数,写法照同文件 :306–313 已有的兜底。脑体积 256×256×176 = 11,534,336 < 2^24,走原分支,结果逐位不变。

训练路径上其他 quantile 调用已查过(2026-09-28):`joint_trainer.py:273, 568–569` 先按步长降采样;`joint_trainer.py:706`、
`gaussian_model.py:500, 550` 按高斯个数计算;`init_cloud.py` 降采样到约 200 万;`edge_tv.py` 用 numpy。膝的体积不会触发它们。
打分里的 `cycle.py:132, 246` 不被新入口使用。

### 3.2 新增文件(脑的流程不使用)

- `paired_gs/utils/strict_config.py`:`StrictConfig`,递归包装嵌套字典。`.get(键, 默认值)` 或 `setdefault` 在键缺失时记录;
  严格模式下立即抛错。键齐全时读出的值与普通字典完全相同。
- `arc/run_spec.py`:严格入口。
  - 命令行只有 `--spec PATH --exp NAME --gpu ID`,另有 `--audit N`(只跑 N 步,列出读到默认值的键和值,不打分不存体积)。
  - 在进程内完成:读 spec → 校验运行名前缀和输出目录(必须在 spec 的 `output.results_root` 下)→ 设种子(取自 spec)
    → 用 `StrictConfig` 构造 `JointTrainer` → 训练 → 用 `spec_scoring` 打分 → 存体积与 json。
  - json 字段:spec sha256、git commit 与是否 dirty、数据文件 md5、完整展开配置、全部指标、显存、耗时。
- `arc/spec_scoring.py`:复制 `cycle.py` 的打分函数(`score_contrast` 及其依赖的 hfen、sharp、grain、头部指标计算),改为由 spec 的
  `scoring` 节控制:解剖掩膜阈值(脑为 0.05·P99.9)、是否计算白质颗粒指标 N(膝不算)、元素数超过 2^24 时用 numpy 分位数。
  `cycle.py` 与 `diag_artefact/headonly.py` 不改。
- `scripts/regress_brain_pin.sh`:`cycle.py --from_ckpt /data0/congcong/code/GS/mcgs_hparam_wt/results_arc/hp_n36/hp_n36/checkpoints/latest.pth`,
  参数取 `hp_n36_score.json` 记录的原 argv,运行名另起;期望 head PSNR 35.706153 / 34.928080,容差 1e-4 dB。约 4 分钟 GPU。

### 3.3 新增测试

- `StrictConfig`:缺键中止;键齐全时与字典读值一致;嵌套访问与 `setdefault` 都被跟踪。
- quantile 分支:元素数 ≤ 2^24 的输入走原分支,输出逐位相同;超过 2^24 的输入不再报错,且与 numpy 分位数一致。
- 打分等价:`spec_scoring` 在脑的打分设置下,对同一 checkpoint 给出与 `cycle.py` 完全相同的数。需要 GPU 和脑数据,按现有 skipif 惯例标注。
- 共享文件 md5 清单(见 2.2 最后一条)。

### 3.4 合并条件

每次公共改动合入数据集分支前:现有 223 个测试(`--ignore=tests/smoke_test.py --ignore=tests/test_score_edge.py`)加新测试全部通过;
重打分检查在 1e-4 dB 以内。训练器、体素器、损失未改动时不重训脑锚点;改动时重训 hp_n36 锚点(约 65 分钟,35.706 / 34.928,±0.02 dB)。

### 3.5 本阶段不做

编码算子接口与 NUFFT(到肝的子项目再做,可按文件取心脏分支 `cardiac-phase2` 的 `forward/radial.py`,该分支与 widen 同属框架血脉,
merge-base cf9aa20);按轴的初始尺度或按轴的 TV 权重(若验证集上看出 z 方向吃亏,再在公共分支加,默认关闭);M>2 个对比度。

## 4. 膝:数据处理

### 4.1 数据事实(2026-09-28 核实)

- 155 个 `MTR_*.h5`,`kspace` 形状 (512 x, 512 ky, 160 kz, 2 echo, coil) complex64;第 0 轴 x 已在图像域(混合空间)。
  轴向:SI(读出)、AP(ky)、LR(kz,矢状层面)。另有 `maps`、`target`、`masks`。
- kz 只有中间 80 层(索引 40–119)有数据(ZIP2 零填充);读出 416/512 个频点非零。
- ky×kz 平面(512×80 = 40,960 点)中椭圆支撑 31,576 点(77.1%),四角为零。
- 外周 ky 行:偶数行实采,奇数行是 ARC 合成(用对侧行线性预测,奇数行 NRMSE 0.000–0.011,偶数行 0.23–0.42;MTR_005、MTR_006,两回波,三个 x 位置)。
  中心全采带 ky 244–267(24 行;带内奇数行残差 0.04–0.21,带外 0.003–0.015;MTR_005、MTR_001)。
  kz 两个奇偶组互相预测 NRMSE 0.29–0.45,kz 全部实采。16 线圈病例 MTR_001 规律相同。
- 数据集自带的 `poisson_16.0x` 掩膜是 416×80(2,070 点),与 k 空间网格不符,不使用。
- 官方划分 `/data2/congcong/data/FM_data/SKM-TEA_ltr/annotations/v1.0.0/{train,val,test}.json`:86 / 33 / 36 人,均为不同受试者、timepoint 0。
  验证集 30 人为 `HD TR Knee PA`(8 线圈)、3 人 `GEM Flex LG Full`;测试集 34 人 8 线圈、2 人 16 线圈,34 人为 160 层、各 1 人为 156 与 144 层。
- 分割 `/data2/congcong/data/FM_data/SKM-TEA_ltr/segmentation_masks/raw-data-track/MTR_*.nii.gz`(155 个);组织:半月板、ACL、PCL、股骨软骨、髌骨软骨、胫骨软骨。
- 探针读过 MTR_001、MTR_005、MTR_006 的 k 空间做格式核查(奇偶行、支撑、中心带),未做任何重建或调参;其中 005、006 属于测试集,不因此排除。

### 4.2 受试者

- 合格条件:h5 线圈维为 8(`HD TR Knee PA`)、`matrix_shape` 为 [512, 512, 160]、主 h5 完整可读。
- 调参 2 人:从验证集合格者中抽取;报告 5 人:从测试集合格者中抽取。
- 抽取方法:合格 `scan_id` 升序排列,`numpy.random.default_rng(20260928).choice(n, k, replace=False)`,验证集与测试集各自独立抽取。
- 名单、种子和合格者数量写进 `specs/knee_skmtea/subjects.yaml`。

### 4.3 转换(每人每回波)

1. 读 `kspace[:, :, 40:120, e, :]`。
2. 沿第 0 轴做居中正交 FFT(`fftshift ∘ fft(norm="ortho") ∘ ifftshift`),与加载器的 `fft3c` 约定一致,得到 (kx, ky, kz, coil)。
3. 转置为 (coil, kx, ky, kz) = (8, 512, 512, 80),存 h5 键 `kspace`。配置:`readout_axis: 0`,`phase_axes: [1, 2]`,`presampled: false`。
4. 转换检查:在 160 层上用数据集自带 `maps` 做零填充 SENSE 合成,与 `target` 的相关系数 ≥ 0.99,之后再裁到 80 层。
5. 重建网格 512×512×80,体素 0.3125×0.3125×1.6 mm。

### 4.4 线圈图

`bart ecalib -r 24`(1 个 map),在 E1 的转换后 k 空间上估计,两个回波和全部方法共用。中心 24×24 标定块
(ky 244–267 × kz 28–51)全部位于实采区。检查:支撑内 Σ|S|² ≈ 1。存 `data/<MTR_ID>/csm.h5`。

### 4.5 AF16 掩膜

- 实采集合 A = 椭圆支撑 ∩ ({偶数 ky} ∪ {ky ∈ [244, 267]}),约 16,700 点。
- 掩膜点数正好 2,560,全部属于 A,包含中心 24×24 全采块,沿读出方向恒定。
- 生成:`vdPoisMex(512, 80, 512, 80, accelx, accely, 24, 1, 1)`(椭圆开);中心带外落在奇数 ky 上的点移到相邻偶数行(两侧都已被占则丢弃);
  照 AF24 课题 `prep_02_mask_pool_af24.m` 的办法先拟合 accelx 再建抽签池,挑出点数正好 2,560 的一张。
- 两个回波、全部 7 名受试者共用这一张。对每名受试者检查掩膜点都落在其椭圆支撑内。
- 存 `data/mask_af16_3d.h5` 与二维图案 `data/mask_af16_2d.mat`,md5 写入课题 README 与 `subjects.yaml`。vdPoisMex 按时间取种子,掩膜文件即唯一真源。
- 报告三种 AF 口径:相对整平面 40,960 / 2,560 = 16.00(用户定义,标题所用);相对椭圆支撑 12.33;相对实采集合约 6.5。

### 4.6 参考图与 known_zero

- 参考图:加载器按回顾性模式,用数据集的完整 k 空间(外周奇数行为 ARC 合成)与线圈图做 SENSE 合成。论文写明参考图外周含 ARC 补全数据。
- 输入:只含实采点(掩膜 ⊂ A),不含 ARC 合成数据(用户 2026-09-28 选定)。
- known_zero(椭圆四角作为已知零)是 MC-GS 与 GS-single 的可调项,在验证集上按规则 A 决定。

### 4.7 存放

`Results_20260908/KNEE_SKMTEA_AF16/data/<MTR_ID>/{ksp_E1.h5, ksp_E2.h5, csm.h5}`,每人约 4 GB,7 人约 28 GB(/data0 剩约 4.3 TB)。

## 5. 膝:实验与评估

### 5.1 方法

ZF(参照列)、CG-SENSE、CS-SENSE、CS-SENSE(TV 收敛版,ADMM 外迭代 200)、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、
GS-single(E1、E2 各一个)、MC-GS。P-LORAKS 是脑 AF12 课题的对比臂,AF16 不跑。对比方法实现复制自脑 AF16 课题
`Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts/`;MC-INR 来自 `/data0/congcong/code/GS/mcinr_niessen2025`,环境 GR_pt1.10。

### 5.2 调参(验证集 2 人)

- 经典方法(CS-SENSE 收敛版、Joint SENSE、L1-SPIRiT 的 ksize × λ):扫网格,按 2 人两回波解剖区 PSNR 均值选;
  最优点必须位于网格内部,落在边缘就向外扩网格。CS-SENSE(默认设置)与 CG-SENSE 按脑课题的固定设置。
- 扫描特异的学习类方法(ZS-SSDU、3D-DIP、3D-INR、MC-INR):沿用各自的留出集早停,不看真值;结构配置照脑课题。
  3D-DIP 若在验证集上显存超过 80 GB,停下来问用户,不自行改方法。
- MC-GS 与 GS-single:规则 A。通道内旋钮两个臂对称地扫、各取最好;跨对比度耦合项只属于 MC-GS;GS-single 的 λ_edge = 0。
  起点取脑 AF16 的数值,高斯数按膝的体积重定。预算上限 40 次运行、约 80 GPU·h。
  选择标准:2 人两回波解剖区 PSNR 均值,平手看 SSIM。
- 选定值冻结进 spec,记录 sha256。

### 5.3 报告(测试集 5 人)

每个方法用冻结 spec 各跑一次。看到测试结果后不再调参。

### 5.4 指标

- 主终点(每个回波):解剖区 PSNR、SSIM、HFEN、NMSE。解剖区 = |GT| > 0.05·P99.9,各回波用各自的 GT;PSNR 峰值取 max|GT|,幅值口径;
  SSIM 为 SSIM 图在解剖区内的均值;HFEN 为 LoG σ1.5。
- 附带:全体积 PSNR/SSIM/NMSE;复数口径。
- 统计:每人数值;5 人均值 ± 标准差;MC-GS 对每个方法的配对差值与"5 人中赢几人"。n=5,不报显著性检验。
- 次要终点 T2:
  - 用 qDESS 解析模型(Sveinsson 等 2017)由 E1/E2 计算 T2;公式与固定参数(如 T1、扩散系数)与 SKM-TEA 发布 T2 图所用的 DOSMA qDESS 实现一致,
    实施时从 DOSMA 源码核对后写入 `specs/knee_skmtea/scoring.yaml`;TR、TE、翻转角、扰相梯度面积与时间取自 `all_metadata.csv`。
  - 重建为 80 层,先沿 kz 零填充插到 160 层(与数据集的 ZIP2 流程相同),再与自带分割对齐;标签含义按 SKM-TEA 的 DATASET.md 核对。
  - 在股骨、胫骨、髌骨软骨与半月板内,比较各方法相对参考图的 T2 平均偏差与逐体素 MAE。

### 5.5 图

全部方法入图,主视图矢状位;误差图 jet、固定 [0, 0.1];软骨处放大框;面板内橙色指标;正式图 200 dpi PNG 加矢量 PDF,标题写 AF 16.00×。

### 5.6 算力与调度

测试集约 100–150 GPU·h,调参约 80–120 GPU·h。只用空卡,每卡一个任务;启动前查 `nvidia-smi --query-compute-apps`;
长任务用 bash 脚本 `setsid nohup … &` 脱离会话;只停自己的进程(按 PID)。2026-09-28 空卡为 3–7 号,约 3–5 天。

## 6. 测试、验证与交接

### 6.1 膝分支测试

1. 转换:混合空间 → k 空间 → 混合空间逐位一致;160 层自带线圈图重建与 `target` 相关 ≥ 0.99。
2. 掩膜:点数 2,560;每名入选受试者的每个掩膜点都属于其实采集合 A;含中心 24×24;沿读出恒定;md5 与记录一致;三种 AF 口径为 16.00、12.33、约 6.5。
3. 前向模型:膝几何上 SENSE 前向与伴随的内积检验;支撑内 Σ|S|² ≈ 1。
4. spec 完整:联合臂、单臂 E1、单臂 E2 的 `run_spec --audit` 读到默认值的键数为零。
5. 隔离:膝 spec 文件哈希钉住;共享文件 md5 与 infra 尖端一致;静态扫描膝的脚本,不得出现指向其他工作树的 import 或 sys.path。
6. T2:与 DOSMA 或手算样例一致。

### 6.2 回归

infra 合入膝分支时:223 个旧测试、infra 新测试、脑重打分检查(35.706153 / 34.928080)全部通过。

### 6.3 交接

膝分支有自己的 CLAUDE.md 入口段与 STATUS.md(已验证、待拍板、下一步、坑、为什么五段);记录放 `docs/superpowers/records/`;课题目录有 README。
提交者为仓库身份 `Congcong Liu <liuc23539@gmail.com>`,提交信息不留 AI 痕迹。会话边界为提交加 tag;推送到 GitHub 前先问用户。

## 7. 用户决定(2026-09-28)

| 决定 | 内容 |
|---|---|
| 总体方案 | 方案 A:公共分支 + 每个数据集一条分支和一个工作树;数据集之间永不合并 |
| 顺序 | 公共框架 → 膝 → 肝 → 乳腺 → 胸腹 |
| 下载 | 现在就在后台下载肝(45.00 GB)与胸腹(114.13 GB)到 /data2,只下载不处理 |
| 加速倍数 | 全部数据集固定 AF=16 |
| 膝的输入与参考 | 掩膜只落在实采点;参考图用数据集的完整 k 空间,论文写明外周含 ARC 补全 |
| 膝的受试者 | 验证集 2 人调参,测试集 5 人报告 |
| 膝的 T2 | 加,作为次要终点 |
| 实现方案 | A:共享代码最小改动 + 新严格入口;NUFFT 推迟到肝 |
| 第一至五节设计 | 逐节确认,无修改 |
