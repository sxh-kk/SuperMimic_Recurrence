# Ollie 空中人板相对参考修复

日期：2026-10-23。当前推荐候选是 **video_timed_v1**：它修正了脚落到板下方的运动学目标和过早腾空的板高度先验，保留已验证的准备段。它尚未通过真实物理执行；连续原生 PD 对照仍然失败。

本记录中的视频时间、物理时间始终分开。`t_video = 1.5 × t_physics` 是沿用的假设，不能据此声称原片实测慢放倍率为 1.5。

## 1. 输入与问题定位

源参考：`data/ollie/self_collision_corrected/joint_reference_50hz.npz`，SHA256：

```text
f6150d1217a4cf4952c3c03e601df4abe20b649ce03abb6f6b4c208d894f8dbc
```

人体与板的来源分别是 `logs/ollie_gvhmr/ollie/body_world_raw.npz`、`logs/ollie_gmr/common_world/g1_robot_50hz.npz`、`logs/ollie/board_reference_v3/board_reference_50hz.npz`。相机使用 `data/ollie/common_world.json` 中固定的 camera→Z-up 变换；没有把逐帧人体反推出的外参当成相机运动。

物理 3.20s、视频约 4.80s 的板局部坐标说明了问题：

- 原 GVHMR 人体左右脚关节 Z 为 `+0.08495 / +0.14964m`，Y 为 `+0.11337 / +0.13000m`。
- 原 GMR 机器人脚 site Z 已为 `−0.08969 / −0.01461m`。
- 加上站姿插值平移与关节平滑的 pose prior 后，机器人脚 site Z 变成 `−0.31068 / −0.23933m`。
- 最终旧 IK/手臂修复参考的左右脚 site Z 为 `−0.30338 / −0.23203m`，Y 为 `+0.32496 / +0.33977m`。

`contact_ik.py` 在没有接触标注的空档仍插值 `root_translation_calibration`，并未直接把它清零。问题来自 GMR 的人体/根部缩放、随时段变化的单目深度偏差、板的强倾斜，以及 stance 外回到 pose prior 的共同作用。该帧插值平移约 `[0.00045, 0.28336, −0.08795]m`；它在板法向上的投影会进一步把脚移到板下。仅冻结最后一帧 stance 的额外 IK 小平移不能解释或消除整个偏差。

准备段原人体脚相对板还有约 `−0.33m` 的 Y 深度偏差，而空中该偏差明显改变，故不存在已被验证的统一常量平移。GMR 的 pelvis/腿尺度为 0.866，滑板保持真实模型尺寸，不能把两者混合后的位置当作三维观测真值。

原始图像核对也有不确定性：该帧 SMPL-X 踝点到 2D 检测踝点的距离约 11.2/17.0px，检测置信度约 0.913/0.908。SMPL-X 脚关节与 G1 足底 site 不是同一解剖标记，二者数值对照只用于定位整体偏移。

逐阶段机器记录：`logs/ollie/air_repair/retargeting_stage_diagnosis.json`。主脚本也会输出原人体投影、原/新机器人板局部脚位置和来源 hash。

## 2. 原片时序纠正

已目视核对 `logs/ollie/source_takeoff_review.png` 与 `logs/ollie/source_landing_review.png`：

- 视频 4.337s 仍整板在地；4.421/4.504s 抬头但尾部仍近地；4.587/4.671s 尾尖仍近地；4.754s 才明显整板离地。
- 视频 5.171s 仍空中；5.255s 低侧轮很靠近地面，接触有歧义；5.338s 四轮已明显接地，5.422s 稳定在地。

旧板拟合在物理 2.8475s，即视频约 4.271s，就给出全板包络高于地面 5mm，属于错误的腾空时间先验。它不能证明该时段无外力。

因此，**`joint_v3` 中强制物理 2.86..3.60s 全段重力抛物线的假设已撤回**。该窗包含真实尾部推地及落地，水平加速度可以来自地面支持，不能只为压低 COM 残差牺牲图像证据。

推荐修复使用物理 **3.16..3.48s**，即视频 **4.74..5.22s**，作为保守的空中核心重力先验。边缘时刻仍不确定；这不是由仿真接触力测出的飞行窗。

## 3. 新分支如何改动

入口：`scripts/video/repair_ollie_air_reference.py`。默认启用原片支持/落地时序，默认 COM 核心窗为 3.16..3.48s。

推荐候选采用根部平移拟合和板 Z 时序修正：

- 物理 **0..2.76s 的完整 92 维状态与速度逐值不变**；额外保留 2.78s 的位置节点，以保持保护端点的中心差分速度。
- 机器人 29 个关节位置/速度、根部旋转/角速度完全不变。滑板 XY、旋转、轮角和轮速完全不变。
- 只改变机器人根部 XYZ 与线速度，以及板 Z 与线速度。
- 板在物理 3.12s 节点前贴合其真实完整碰撞包络最低点，覆盖原片约 4.671s 尾部近地证据；物理 3.56s 节点起恢复几何落地。
- 保留物理 **3.16..3.40s** 的板位置插值节点，完整覆盖原始 **114..122 帧**的 9 份较可靠图案观测。因此这 9 帧的新旧投影增量严格为 0px，原有角点 RMSE 仍为 2.199px。
- 空中脚目标只受宽松横向包络和单侧上半空间屏障约束。屏障使用 7 个真实足底胶囊与板抬起的鼻/尾上包络，不只检查中心平面。没有把脚高度设成板面等式，没有把空中脚焊在板上。
- 接触、阶段和上游置信度数组保持不变。`unknown=-1` 仍是未知；新的几何屏障不产生“实测接触”或力标签。保留的置信度也不代表修复结果的新精度认证。

二次目标同时约束相对几何、根部速度/加速度和全系统质量加权 COM。空中先验是水平近直线、竖直重力 9.81m/s²。目标通过 Cholesky 白化后求同一个凸二次问题，白化不改变约束或目标。

## 4. 推荐候选的实测结果

候选：`data/ollie/air_relative_video_timed_v1/joint_reference_50hz.npz`，SHA256：

```text
941f748cf3b13edbcf60dd20f2db41810cee2cbd1e9b4dccba1e3b1d4b5a71f5
```

机器结果与曲线位于 `logs/ollie/air_repair/video_timed_v1/`：`result.json`、`geometry_before.npz`、`geometry_after.npz`、`root_translation_repair.npz`、`kinematic_repair_diagnostics.png`。

改动幅度：

- 根部平移最大 **0.59905m**；绝对根部速度峰值 **2.451→3.150m/s**，加速度峰值 **70.419→66.486m/s²**。
- 腿部及全部关节改动为 **0rad**，板旋转改动为 **0rad**，板 XY 改动为 **0m**。
- 板 Z 最大改动 **0.17187m**。从尾部近地到首个保留 PnP 节点的短过渡仍出现板速度峰值 **5.903m/s**、加速度峰值 **131.334m/s²**；这是未解决的动力学/深度过渡问题，不能把它称为物理可行。
- 物理 3.20s 的新左右脚 site 板局部坐标约为 `[0.22834,−0.00801,0.13713]`、`[−0.20301,0.00680,0.20848]m`。旧的板下 23–30cm 和侧偏约 33cm 已消除，但并不意味着空中两脚已接触板面。

完整碰撞审计使用真实启用的碰撞几何，每 2.5ms 采样，共 **2057** 个规定姿态样本：

- 最大自碰撞深度 **1.47425mm**，与原手臂修复参考一致。
- 机器人与板的最大穿透 **0mm**。
- 地面最大穿透 **0.10774mm**，来自板支撑包络节点之间的姿态插值；不是严格零穿透。
- 没有超过 2mm 的接触穿透。未关闭碰撞或缩小几何。

这项审计是对规定姿态插值的几何检查，**不是把参考状态写入真实回合后声称执行成功**。

板时序核对：视频 4.337..4.671s，完整最低包络约在地面；视频 5.255s 离地约 40.5mm，5.338s 约 0.34mm，5.422s 约 0.062mm。接地支持的是实际最低几何，不能一概把全部支撑样本叫尾尖触地。

物理 3.16..3.48s 核心窗的全系统 COM：

- 修复前二次加速度约 `[-1.082,−5.261,−13.051]m/s²`；修复后约 `[0.0066,0.0437,−9.8598]m/s²`。
- 水平直线/竖直重力弧的 XYZ 残差 RMSE 从 `[5.05,31.71,28.98]mm` 变为 `[0.545,0.548,0.725]mm`。
- 边缘敏感性仍存在：3.14..3.42s 的修复后 XYZ 残差约 `[3.33,5.08,4.36]mm`；更宽且包含推地/落地的窗口不适用“全程自由落体”验收。

COM 接近重力只是必要的运动学检查。它没有证明板单体动力学、两脚冲量、关节力矩或起跳接触可行；单目板尺度、相机近固定假设、板旋转与脚高差仍有不确定性。

继承的顶层 `qa/config` 描述源 contact IK 处理，修复后的指标以 `air_relative_repair` 和本节机器结果为准。新根部速度峰值 3.150m/s 已高于源 IK 的 3m/s 参数；该分支没有把它隐瞒成仍满足原运动学速度上限，也没有修改任何物理关节/扭矩限位。

## 5. 真实原生 PD 对照

实际回合位于 `logs/ollie/air_repair/video_timed_v1/native_reference_pd/`。一次初始化，随后只施加 29 关节目标，每控制步 8×2.5ms；没有 root/板状态覆盖、额外外力或中途 reset。

整段运行 5.14s，结果 **FAILED**：

- 身体首次触地约 **1.035s**。
- 最大实际穿透 **27.900mm**，没有合格全板腾空或板尾触地事件。
- 最大完整板最低点高度仅 **0.693mm**，末端未稳定站板。

根部与板参考位置不会直接驱动被动板或机器人自由根部。由于这次修复没有改变关节目标，原生 PD 的 qpos/qvel 与另一个同关节参考对照逐值相同；修复参考本身不会自动变成可执行控制。下一步需要从真实准备段末态连续续接，再评估学习控制或完整反馈控制，而非重置到空中目标。

## 6. 教师数据迁移与 hash 更正

原教师为 `logs/ollie_quasistatic/corrected_static_final_v2/feedback_0.45/teacher.npz`，SHA256 `becf8edd26596b6ae858418e3a5f20f1f8bea5ab42c89c862d29d1347dd1d738`。

**正式迁移副本**为 `logs/ollie/air_repair/video_timed_v1/teacher_migrated_v2.npz`，SHA256：

```text
c953f231a5d0d47ce7d850f71ddf01ba64807e5b0ec78f102a50ad86cf964a8c
```

139 帧、物理 0..2.76s 的新旧参考完整状态 hash 相同：

```text
5f6eed90f41315c2d47754a480113d80cf4657cf892d4fb04c83051daad7aad3
```

迁移只更新顶层/元数据 reference hash、相关来源记录和 **15 个 future goal 向量**。所有实际 qpos/qvel、state/next_state、action/raw_action、1104 个 substep states/controls、时间轴和关节顺序均逐值不变。新的 future goals 与新参考按原 `step+1+[0,5,10]` 索引生成。

旧的首次副本 `teacher_migrated.npz`、hash `e9cca929…` 漏更新了顶层 reference hash，虽元数据和 future goals 正确，仍构成 provenance 矛盾；它已撤回使用且未被覆盖。root 的一致性门槛实际拒绝了旧副本，随后才绑定正式 v2。

证据：`teacher_migration_v2_provenance.json`。另外已验证：人为把受保护前缀首帧根部改动 1mm，迁移入口会拒绝，且不会写出教师文件。

有限动态准备段的原始稳定余量、实际关节软限位偏差等限制仍全部继承，见前序审计。它不是完整 Ollie 教师。由于未来目标改变，旧 BC/latent 不能被当作等价结果沿用；需要明确重训及真实连续筛选。

## 7. 重跑与交付

以下命令必须使用新的输出目录，入口会拒绝覆盖已有参考：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python scripts/video/repair_ollie_air_reference.py \
  --out data/ollie/air_relative_video_timed_rerun_01/joint_reference_50hz.npz \
  --log-dir logs/ollie/air_repair/video_timed_rerun_01 \
  --board-timing-prior --com-window 3.16 3.48 --com-sigma .003 \
  --teacher logs/ollie_quasistatic/corrected_static_final_v2/feedback_0.45/teacher.npz
```

依赖为项目现有的 NumPy、SciPy、MuJoCo、Matplotlib；不安装新依赖、不启用 GPU、不修改物理模型。当前原片时序先验明确限定于 scale=1.5、50Hz 分支。

默认推荐不启用 `--move-board`。该选项仅用于显式的联合 XYZ 假设对照，会记录 9 帧真实重投影增量。错误宽窗 `joint_v3` 的板平移达 0.698m、投影增量最大 23.60px，只作为撤回假设的记录保留，不能替换本节主候选。

本批源文件仅新增入口与本记录；原参考、原教师、已绑定候选均未覆盖。验证包括 Ruff、编译/CLI 检查、2057 样本全几何审计、前缀/hash/动作逐值校验、错误迁移负例，以及真实原生 PD 回合。所有 MP4/模型/机器数组仍留在 ignored 数据与日志目录。

冻结前又在 `default_validation` 新目录完整执行默认入口，重新生成的参考 SHA256 仍严格等于 `941f748…`，同时确认新教师顶层/元数据 hash 一致、前缀完整状态 hash 一致。该复验没有覆盖 root 已绑定的候选或正式教师副本。
