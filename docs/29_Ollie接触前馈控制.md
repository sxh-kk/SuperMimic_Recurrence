# Ollie 原片准备/预蹲接触前馈控制

## 结果与适用范围

已在 [手臂自碰修复参考](25_Ollie手臂自碰修复.md) 上，得到一个实际连续执行 **0～2.76 s**、通过指定准备/预蹲门槛的前缀教师。控制器为**接触准静态前馈加真实接触力门控的踝反馈**，没有训练网络，不是 SuperMimic/WM 学习策略，也未完成整段 Ollie。

推荐后续 BC 输入只有本轮最终版本：

`logs/ollie_quasistatic/corrected_static_final_v2/feedback_0.45/teacher.npz`

验收版本为 `stance_upper_deck_support_v2`。它包括中心/尾/鼻**上表面向上支持力 >1 N** 的全窗及末 300 ms 逐脚占比，排除轮轴、底面、侧面仅有法向力的假支持。全部相关统计来自原生 MuJoCo 已求解的实际接触力。

这只是**单一初态、有限时间窗、余量较窄的前缀教师**。最小 upright 为 0.85331，距离 0.85 门槛仅 0.00331；最大 root-board XY 距离为 0.19577 m，距 0.20 m 门槛约 4.23 mm。终点 robot 线速度 **0.80594 m/s**、角速度 **1.97618 rad/s**，板线速度 **0.51365 m/s**；并未静止停在深蹲，也不保证继续执行不摔倒。后续蹬伸必须从这个真实末状态连续衔接，不能重新初始化成更有利的起跳姿态。

## 共享准静态接口

新增 `src/supermimic/ollie/quasistatic.py`，原片与 generic 物理基线可以共用，不依赖 GPU：

```python
from supermimic.ollie.quasistatic import ContactStatics

solver = ContactStatics(model, index, friction=0.9)
solution = solver.solve(data, initial=previous_solution)
target, raw_target = solver.pd_target(qref29, qvel_ref29, solution)
```

`solve()` 只读输入 `data`，在内部 scratch data 上求解。`solution` 返回规范 G1 顺序的 `tau_nm (29,)`、`forces_local (12,3)`、成功状态、完整平衡残差、摩擦余量、秩、支撑点投影量与迭代次数。解出的接触力**绝不写入仿真**。

准静态默认零速度，建立 `h=B*tau+Jᵀ*f`，其中 `h=qfrc_bias-qfrc_passive`；本模型零速度下的被动力为零。29 个关节力矩按真实 dof/actuator 名称地址映射，浮动根、板和四个被动轮均无驱动。

支撑模型有每脚四个近似足底矩形点和四个轮地面点。足板力按同一点的 `(J_foot-J_board)` 成对作用；前八个 force 使用板局部坐标，最后四个使用世界坐标。轮的最低点按真实圆柱包络计算。足底点投影到平板上表面，受 flat patch 范围约束；本次最大投影约 **7.05 mm**，包含对参考小间隙和近似足底矩形的处理。这是假设可建立的支撑模型，不是视频实测力或精确接触互补解。

求解约束真实 actuator 力矩范围、法向非负和保守摩擦棱锥，以标准化力矩及小权重接触力正则为目标，用 SciPy SLSQP 解算。通过秩分析移除冗余等式后，仍检查全部 45 维原方程。不能仅凭优化器的成功消息判断平衡成功；`solution.success` 还要求完整残差 <1e-5、摩擦余量合格。

## 真实控制与数据接口

新增 `scripts/local/track_ollie_quasistatic.py`。在原参考 139 个姿态上求静力前馈后，执行 138 个 20 ms 动作，每个动作在 0.0025 s 物理步长下保持八个子步：

```text
ctrl_raw = qref + (tau + kv*qvel_ref)/kp + feedback
ctrl = clip(ctrl_raw, 原关节目标范围并留0.005rad边距)
```

没有 ±0.35 rad 人造残差限幅。真实关节位置目标范围、驱动力矩限位、PD 增益、质量、摩擦、重力均保持原模型。只写 29 个机器人 PD 目标；每回合只初始化一次，之后没有根/板位姿回写、外力、接触力注入或回合内重置。

踝反馈每 20 ms 读取真实状态和实际足板支持力，仅在有向上支撑的脚上启用，采用正方向的世界踝轴投影。误差项包含相对于参考的根倾角/角速度、机器人 COM 相对板的位移及相对 XY 速度。COM 速度由真实 Jacobian 计算，扣除板对应点的速度，没有把运动板当作静止地面。

原片的显式双脚 stance 先验为视频 0～4.15 s；当前最后样本是视频 4.14 s，即物理 2.76 s。脚本拒绝跨过该有限准备窗继续硬套双脚/四轮静力假设。

数据包含 `state/next_state (138,92)`、真正执行且经过真实目标限位的 `action (138,29)`、全部 139 个边界状态、1104 个子步状态/动作、qpos/qvel、前馈与反馈分量和实际支持力。`time_s` 为执行动作**前**的物理时间；future goals 仍为 `round(time_s/0.02)+1+[0,5,10]` 对应的完整 92 维参考，并保存规范 joint names、board joint names、参考 SHA-256。

参考 SHA-256 为 `f6150d1217a4cf4952c3c03e601df4abe20b649ce03abb6f6b4c208d894f8dbc`。只有全部门槛通过的完整前缀才输出 `teacher.npz`；失败增益和诊断分支只保存 `unverified_candidate.npz`，不能混入 BC 正样本。

## 实际 2.76 s 对照

139 个准静态解全部通过，完整 45 维平衡残差最大约 **8.88e-14**。

无反馈时，首次机器人触地 **2.3475 s**，最小 upright 0.1664，足板漂移和根高均失败。只有静力方程成立不足以得到动态轨迹。

最终反馈尺度 0.45 的真实结果：

- 无机器人触地，最小 upright **0.85331**，最大穿透 **7.901 mm**，初始穿透仅浮点误差。
- 全窗左/右脚上表面向上支持 >1 N 占比 **99.819% / 99.457%**；末 300 ms **100% / 99.167%**。
- 四轮同时接地占比 **99.095%**；足底 site 相对板的最大高度误差 **4.559 mm**，最大平面漂移 **27.745 mm**。
- 参考根高度跟踪 RMSE **19.616 mm**、最大误差 **68.567 mm**；root-board XY 最大 **195.768 mm**。
- 无外力、无中途重置、无板驱动；终点动量和窄余量限制见开头。

门槛固定为：upright≥0.85、无机器人触地、穿透≤10 mm、逐脚真实接触支持占比≥80%、四轮同时接地占比≥80%、足底高度误差≤25 mm、平面漂移≤40 mm、根高 RMSE≤80 mm且最大误差≤160 mm、root-board XY≤200 mm、初始穿透≤2 mm、无外力且静力解全合格；新增上表面向上 >1 N 的全窗与末 300 ms 逐脚占比也均须≥80%。

尺度 0.5 可不触地且其他多数门槛通过，但最小 upright 0.7874，未合格；0.55～0.7 等对照同样因倾斜未过。0.25 在 2.7325 s 触地。不是增益越大越好，也没有通过盲 CEM 追加预算来产生该教师。

参考本身此窗最小 pelvis upright **0.82589**。0.85 是本次教师的工程验收取舍，**不表示原参考的正常前倾就是错误**。额外绝对直立目标对照反而在约 2.19～2.22 s 触地，未采用，没有为满足验收阈值偷偷改动参考姿态。

## 固定初始/深蹲姿态复核

均从修复后真实姿态初始化一次、零速度，使用对应静力前馈保持 1 s，不施加计算出的接触力。

初始姿态通过固定姿态门槛：最小 upright **0.99525**，无触地，峰穿透 **2.690 mm**，双脚上表面支持均约 **99.501%**，末窗均为 100%。腰 pitch 所需 `tau/kp≈-0.53170 rad`。

最深蹲为 frame128、物理 2.56 s。腰 pitch 所需约 **-0.94792 rad**。实际保持 1 s 无触地、峰穿透 **5.741 mm**、双脚支持均约 99.501%，但最小 upright **0.84433**，未通过 0.85 门槛。修复前 37.54 mm 的手腕穿腿初始碰撞已消除，仍不能把静力数值解当成动态平衡证明。

## 动作范围与 BC 可达性

最终真实 `action-reference` 最大值为腰方向 **1.02208 rad**，在现有腰上限 1.1 内。与原 `data/ollie/residual_limits.json` 比较，有五个 joint 超出策略残差范围：

- `left_hip_pitch_joint`：峰值 0.82537 rad，旧范围 0.75。
- `right_ankle_pitch_joint`：0.78353，旧范围 0.7。
- `right_ankle_roll_joint`：0.38487，旧范围 0.35。
- `left_shoulder_roll_joint`：0.51337，旧范围 0.35。
- `right_shoulder_roll_joint`：0.42607，旧范围 0.35。

只有两个动作元素触发真实机械目标限位裁剪；没有为策略可达性裁掉教师残差。实际 29 维峰值、signed min/max、raw 峰值与规范 joint names 已保存到每个结果 JSON，推荐教师目录的 `residual_envelope.json` 另含完整命名数组与输入校验信息。BC 必须显式调整、保存策略每关节残差范围或另选可达教师，不能静默截断这些目标；本脚本没有修改策略范围。

## 独立验证与惯性项诊断

独立重新求解并重跑同参数时，state/action/future-goals/time 数组逐元素一致。再新建原生模型，**只回放已保存的 138 个实际动作**，1104 个子步的 92 维状态最大误差为 **0**，真实上表面支持和其他门槛重复通过。最终目录 `independent_action_replay.json` 保存该检查；数据形状、连续 transition、pre-action 时间、future-goal 索引、关节名字与参考 hash 也已实际核验。

作为未采用的诊断，另试 `M*qacc_ref+h-qfrc_passive` 和原参考速度。139 个样本均未通过全部 45 维平衡校验，最大残差约 1012.77，真实控制约 1.5～1.9 s 触地，没有产生教师。不能只看 SLSQP 在所保留等式上的“success”消息。

该运动学参考给四轮 q/qd/qdd 固定为零，而板和机器人有非零水平加速度；接触/尺度优化也会造成有限差分加速度尖峰。四个抽查姿态即使撤去力矩/摩擦边界、仅做全矩阵最小二乘，仍存在最大约 0.030～0.374 的平衡分量残差，说明不能直接将整份参考的速度/加速度当作可实现的刚体逆动力学要求。这是当前支撑模型与参考运动学不相容的证据，具体误差来源仍需区分；不能把全部失败归因于某一个滤波参数。诊断在 `logs/ollie_quasistatic/inertial_infeasibility.json`，不阻塞已验证的准静态路线。

## 复现入口

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python scripts/local/track_ollie_quasistatic.py \
  --reference data/ollie/self_collision_corrected/joint_reference_50hz.npz \
  --out-dir logs/ollie_quasistatic/new_quasistatic_run \
  --feedback-scales 0 0.45
```

默认会同时做两个固定姿态 1 s 复核。可用 `--inertial` 复现未通过的惯性项诊断；`--absolute-upright` 是额外未采用的直立目标对照。正式实现、真实 CPU 求解/rollout、独立动作回放、Ruff、编译检查均已完成。后续完整蹬伸、踩尾、腾空、接板和落地需要连续实际控制与独立验收，本轮结果不能外推为整段成功。
