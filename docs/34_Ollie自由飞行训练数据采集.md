# Ollie 自由飞行与碰撞训练数据采集

## 实际结果与范围

新增 `scripts/local/collect_ollie_air_dataset.py`，已实际采集 **64 个0.5 s原生 MuJoCo 回合、1600 个50 Hz状态转移**，用于补充 [离线世界模型预训练](33_Ollie真实数据世界模型预训练.md) 缺少的自由下落、板落地和机器人碰撞数据。

推荐数据：`logs/ollie_air_augmentation_04/transitions.npz`。

SHA-256：`ce1b0ecacce3894d281c7005eb7080fafedc1d9adb91b12f14cac9c2ab5c736c`。

这是**训练专用 augmentation**：初始高度、速度和控制均为人工采样，不是视频观测，不是原片重建、真实策略评测或成功示范。没有渲染成 Ollie 视频，没有把它替代原片连续物理评测。本批尚未用于新的 WM 或 actor 训练。

全部64段均先处于真实离地状态，随后发生真实板地面碰撞，首次板触地时间范围 **0.2000～0.4075 s**；47段也发生机器人触地。这里的碰撞后动态样本不表示已经恢复站立或完成落板。

## 合法初始化与真实演化

姿态源为已审查的 `data/ollie/self_collision_corrected/joint_reference_50hz.npz`，SHA-256 `f6150d1217a4cf4952c3c03e601df4abe20b649ce03abb6f6b4c208d894f8dbc`。它提供运动学姿态种子，**不是重新包装成实际执行的前缀末状态**。

尝试从 frame `[0,40,80,100,110,128,138]` 抽样，随后：

- 机器人和板围绕同一板中心施加共享世界 yaw、XY 平移；共同整体抬高 **0.4～0.8 m**，不改变两者的初始相对比例。
- 机器人关节位置加标准差0.01 rad小扰动，按真实关节目标范围留0.005 rad边距裁剪；轮角随机初始化，轮 hinge 仍无驱动。
- 采样真实根线速度：共同 XY 基量在 ±0.4 m/s，Z基量在 -1～0.6 m/s，再加0.12 m/s独立扰动；世界角速度共同基量在 ±1 rad/s，再加0.25 rad/s扰动，转为各 root 的真实 body frame。
- 关节初始角速度在 ±0.6 rad/s，轮角速度在 ±8 rad/s。完整实际 qpos/qvel、源帧和空间平移均保存到 manifest。

初始化后按原模型全部启用的碰撞对检查：最大穿透≤**2 mm**、无机器人/板触地、整板最低包络离地>0.1 m。实际接受样本的初始最大穿透 **1.958 mm**，最小整板离地约0.4 m。该检查覆盖当前碰撞模型，不是可视网格任意表面的连续几何证明。

每段只初始化一次，之后连续执行25个20 ms动作、每动作保持八个0.0025 s物理子步。目标是在初始关节位置附近的随机平滑正弦加小扰动，按真实目标范围裁剪，只写 **29 个 G1 位置驱动器**。

没有根或板姿态回写、回合内重置、QP点力、外部广义力或 body force；重力保持 `[0,0,-9.81] m/s²`，原 actuator gain、力矩范围、摩擦、碰撞和关节模型均未修改。随机初速是训练初态，不是仿真中持续施加的推力。

## 实际关节角与数据质量筛选

早期01/02诊断批虽所有**目标角度**均在原范围内，但碰撞时的**实际关节角**可越过 MuJoCo软限位约 **0.29639 rad**，主要涉及左右踝 pitch/roll。不能把目标裁剪误当作实际关节角绝不越界，也不能称这些数值有限的软约束越界为 NaN/数值重置。

因此增加独立数据质量阈值：逐物理子步实际关节角超过原范围 **0.05 rad** 时拒绝整段，同时保留真实目标与力矩限位。没有为满足过滤修改物理模型或把状态裁回范围。

严格03批在512次尝试上限内只得57/128合格段，入口正确拒绝导出不完整数据。没有放宽阈值凑数；随后将推荐批目标明确改为64段，并提供可记录的 `--max-attempt-factor`。

最终04共 **568 次尝试**：

- 64段接受。
- **158** 次初始几何不合格，未进入物理演化。
- **337** 段因实际关节软限位越界>0.05 rad拒绝。
- **9** 段因动态穿透>30 mm拒绝。
- **0** 次非有限状态或时间重置等 `physics_numerical_failure`。

接受数据的逐子步最大穿透 **27.021 mm**、最大实际软限位越界 **0.049821 rad**，有1134个动作元素触发真实目标范围裁剪。这是明确的模拟数据质量范围，**不等于硬件机械止挡保证**，也不符合原片教师的10 mm穿透验收。没有任何一段被标记为成功教师。

过滤会改变分布，必须保留偏差：最终源帧计数为0:8、40:12、80:14、100:9、110:16、138:5，最深蹲 frame128 无入选。不能据此声称覆盖了全部深蹲空中动力学，也不能把128个较松批与64个严格批混成同一推荐数据源。

## 标准接口与独立核验

NPZ包含规范的 `state/next_state (1600,92)`、真实 `action/ctrl (1600,29)`、float64副本、`episode_id`、执行动作前的 `time_s`、`control_dt=0.02`、`schema=supermimic.ollie.entity_state.v1`、`robot_joint_names` 和 `board_joint_names`。没有 `expert_label`。

`verified_teacher=False`，source/metadata明确标注训练 augmentation、合成初态、非视频观测。manifest另存姿态源 hash、物理模块/板XML hash、MuJoCo版本、实际初态、原关节/驱动力矩范围、所有接受/拒绝原因和质量阈值。不完整采集以后会保存 `incomplete_attempts.json` 且不导出 transitions。

已直接使用 `fit_ollie_wm_dataset.py` 的标准 reader，**无需 schema 推断**读取本批，原始保存精度下每回合 next-state 链误差为 **0**，actual action与ctrl逐元素相等。整回合20%留出可形成 **51 train / 13 heldout episodes**，对应 **918 / 234 个H8窗口**；不能将相邻子段随机拆分冒充独立留出。

另新建原生模型，仅依据 manifest 初态和保存的实际动作独立回放 episode0/63，各201个物理采样、所有控制边界状态最大误差均为 **0**，没有外力。检查保存在 `logs/ollie_air_augmentation_04/validation.json`。

## 复现与后续使用

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python scripts/local/collect_ollie_air_dataset.py \
  --reference data/ollie/self_collision_corrected/joint_reference_50hz.npz \
  --out-dir logs/new_ollie_air_augmentation \
  --episodes 64 --max-attempt-factor 16 \
  --height 0.4 0.8 --max-penetration-m 0.03 \
  --max-angle-overshoot-rad 0.05 --seed 0
```

这段采集只用CPU原生物理，没有GL或GPU训练。已完成实际采集、标准读取、整回合划分、两段独立动作回放、Ruff和编译检查。原PD/DAgger源数据、原参考、前缀教师与评测视频未改动。

后续可将04的 `transitions.npz` 作为单独源加入离线 WM 入口，继续保留每源独立整回合留出，并分开报告air源与原片准备源的1/8步误差。数据增加不自动意味着模型误差下降；仍需实际训练/诊断，更不能将训练合成空中初态当作原片从地面起跳的成功证据。
