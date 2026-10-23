# Ollie 准备到预蹲的补偿控制试验

## 结论

本轮没有得到通过完整准备/预蹲硬门槛的 2.76 s 教师。单末端 15 维补偿、两末端 30 维补偿以及仅腰 pitch 扩大负向范围的对照，均在物理约 2.005 s 首次发生机器人触地。所有数据均标记 `verified_teacher=False`，只导出 `unverified_candidate.npz`，没有把失败搜索作为 BC 正样本。

接触逆静力审查提供了更具体的方向：初始姿态可用关节前馈改善，1 s 真实恒姿控制通过；最深蹲所需的部分 PD 目标补偿超出本轮搜索范围，且**参考本身已有手/腕与腿的严重自碰**。继续单纯增加 CEM 预算缺乏依据，下一步应先修复参考自碰，再重新检查受力和预蹲控制。之前的足板 QA 通过不等于全身碰撞 QA 通过。

## 正式复现入口和控制约束

新增 `scripts/local/fit_ollie_stance.py`，只用 CPU 原生 MuJoCo rollout，不训练网络，也不修改训练或评测模块。

输入为 extended s=1.5 参考 `data/ollie/contact_extended/joint_reference_50hz.npz`，早期支点为 `logs/ollie_stance_bias/limit025_gated/result.json` 的 29 维固定偏置。参考与根/腿/板轨迹没有改动。

- 每个候选只在开始初始化一次；之后只写 29 个机器人位置目标。
- 物理步长 0.0025 s，控制周期 0.02 s，每个动作保持八个子步；总长 2.76 s，共 138 个动作。
- 物理 0～1.34 s 保持早期偏置，然后在 1.34～2.4 s 平滑过渡到后期偏置，之后保持至 2.76 s。
- 一末端版本优化腿与腰 15 维，末端时刻为 2.4 s；两末端版本优化 30 维，末端时刻为 1.87、2.4 s。
- 主试验后期偏置为 ±0.35 rad；对照仅将腰 pitch 下界改为 -0.65 rad，其他边界仍为 ±0.35 rad。真实关节/力矩限位、驱动器增益和摩擦均未改。
- 板保持被动，不施加外力、额外根控制、板位姿回写或 weld/equality。

搜索使用 64 候选×40 代 CEM、八个 elite、随机种子 1。目标包括直立、跟随参考根高度、足板相对运动、根到板水平距离。预蹲允许根自然降低，没有使用“根高度必须大于 0.7 m”的站立目标来阻止蹲下。

最终候选必须同时满足：upright≥0.85、足板 site 高度误差≤2.5 cm、平面漂移≤4 cm、root-board 水平距离≤20 cm、根高度跟踪 RMSE≤8 cm且最大误差≤16 cm；然后再由独立逐子步 native 接触记录确认双脚足板法向力接触占比均≥80%、无机器人触地、最大穿透≤10 mm、没有外力。数值连续也必须通过。

## 实际 CEM 运行结果

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python scripts/local/fit_ollie_stance.py \
  --out-dir logs/ollie_stance_teacher/knots1_limit035

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python scripts/local/fit_ollie_stance.py \
  --out-dir logs/ollie_stance_teacher/knots2_limit035 --knots 2

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python scripts/local/fit_ollie_stance.py \
  --out-dir logs/ollie_stance_teacher/knots1_waist065 --waist-pitch-lower -0.65
```

单末端主试验耗时约 20.6 s：最优候选首次机器人触地 2.005 s，左右足板接触力占比约 72.4%/73.5%，最大穿透约 18.9 mm，最终已翻倒。

两末端 30 维试验耗时约 22.6 s：代价有所改善，但首次触地仍为 2.005 s，左右足板接触占比约 75.3%/74.4%，最大穿透约 15.8 mm，足板高度误差约 17.4 cm，仍不合格。

腰 pitch 下界 -0.65 的对照耗时约 22.3 s，本次最优解没有利用放宽的下界，得到与单末端主试验相同的最优失败轨迹。因此不能把该对照说成“已经证明较大腰补偿有改善”，也不能据此认为所有更大范围都无效。

一个额外限制是：原固定偏置在 1.34 s 虽然 upright 约 0.983，根线速度已约 **0.319 m/s**、角速度约 **0.707 rad/s**；到 1.7 s 根线速度约 1.095 m/s。短窗没有摔倒不代表静止平衡，后续平滑前馈要同时处理已有动量；缺少反馈是本基线的实质限制。

## 数据接口与有效范围

脚本保存每控制边界的真实 92 维状态和真实目标动作：

- `state/next_state (N,92)`；`action (N,29)` 为真正执行、按真实关节目标范围裁剪后的 G1 规范顺序动作。
- `time_s (N,)` 为动作执行**前**的物理时间；`episode_id (N,)` 标记回合。
- `reference_goal_indices = round(time_s/0.02)+1+[0,5,10]`，末端 clamp；`future_goals (N,3,92)` 保存完整参考目标。
- `board_joint_names`、`joint_names`、`reference_sha256`，以及全部边界状态、qpos/qvel、每子步实际 control/state 和原始动作。
- `verified_teacher` 与 metadata 明确区分有效教师和失败候选。

只有完整硬门槛通过时才创建 `teacher.npz`。本次三组只有 `unverified_candidate.npz` 和 `result.json`、`native_contacts.csv`，**不得把这些失败候选直接喂入 BC 作为正样本**。此前 1.34 s 的站稳数据仍只在其原验证范围和原参考来源内有效，不能重标成已通过 2.76 s extended 预蹲的数据。

## 接触逆静力两姿态审查

临时 CPU 诊断留在 `logs/ollie_stance_teacher/quasistatic/solve_two.py`，结果为同目录 `result.json`、`pose_000.npz`、`pose_128.npz`。没有将求解的接触力施加到仿真。

在零速度的初始姿态和最深蹲姿态，使用模型 `h=qfrc_bias`，建立 `h=B*tau+J_contact^T*f`。每脚用四个近似上表面支撑点，板通过四个轮的地面点支撑；足板力成对等大反向进入机器人与板的 Jacobian，地面只作用于轮。浮动根、板与被动轮没有驱动力。

约束包括真实驱动力矩范围、接触法向非负和保守摩擦棱锥；以标准化关节力矩和轻度接触力正则为二次目标，通过 SciPy SLSQP 求解。冗余的被动轮平衡行经秩分析移除后，仍对全部 45 维原方程检查残差。这个模型假设参考中的小接触缝隙闭合、支撑可建立，未求解碰撞互补条件，不能称完整逆动力学。

初始姿态：受力平衡最大残差约 **6.0e-14**，摩擦约束满足。最大 `tau/Kp` 补偿为腰 pitch **-0.53170 rad**，左右髋 pitch 约 +0.219/+0.170，左右踝 pitch 约 +0.297/+0.289。把全部 29 个关节的补偿只转换成 PD 目标、保持正常重力和真实限位，恒姿连续 1 s：最小 upright **0.99524**，无机器人触地，最大穿透约 **2.691 mm**，双脚足板力接触占比均约 **99.5%**。它支持“物理前馈比单纯参考角更有用”，但不是动态预蹲成功。

最深蹲为物理 **2.56 s**、视频 **3.84 s**、参考根高约 **0.6430 m**。受力平衡数值残差约 1.4e-14，但需要腰 pitch `tau/Kp≈-0.96183 rad`、左髋 pitch约 +0.55589、左右踝 pitch约 +0.48684/+0.48031。它们超出本次主搜索 ±0.35 及腰负向 -0.65 的补偿范围；这不是力矩或关节角限位被放宽，而是低 Kp 下所需的目标偏置量较大。

该深蹲恒姿真实 PD 1 s 没有碰地，但 min upright 约 0.8495，记录最大穿透 37.54 mm，未通过门槛。进一步定位发现，**37.54 mm 在仿真开始前的原参考 qpos 中已经存在**：`left_thigh_collision`—`left_wrist_collision` 37.542 mm、`left_shin_collision`—`left_wrist_collision` 27.118 mm、`right_thigh_collision`—`right_hand_collision` 11.223 mm。该参考姿态不是几何合法的静力起姿，不能把数值受力平衡解当作物理可行性证明。

## 后续顺序

先建立只修手臂自碰的新参考分支，保持根、腿、板不变，检查全部 self/board/ground 几何，再复核 COM 和接触目标。不能关闭碰撞来制造教师成功。

在合法参考上，优先用接触静力/准静力前馈构造有物理依据的初始补偿，并加入真实状态/速度反馈，重新验证完整预蹲。若最终仍未稳定，应保留失败结果，BC 只使用已经实际通过门槛的阶段，不能靠改 metadata 或截取未验证时段硬造成功。
