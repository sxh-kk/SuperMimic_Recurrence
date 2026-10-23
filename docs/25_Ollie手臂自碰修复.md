# Ollie 手臂自碰修复与全身几何复核

## 实际结果

已生成独立参考分支 `data/ollie/self_collision_corrected/joint_reference_50hz.npz`，只修改 G1 的 14 个手臂关节位置及对应速度。原机器人根、腿、腰的全部状态和板的全部 21 维状态逐元素保持，extended 与默认参考均未替换。

在原生 MuJoCo 模型全部启用的碰撞对上，258 个 50 Hz 样本的最大自碰穿透从 **67.310 mm** 降至 **1.474 mm**。进一步按每控制区间八个子步检查 2057 个插值姿态，自碰最大值仍为 **1.474 mm**，没有大于 2 mm 的自碰接触。

**全身连续插值几何尚未全部通过。** 足板在 50 Hz 样本上最大穿透为 1.235 mm，但区间内仍有最大 **6.990 mm** 的足板穿透。该问题在修复前后完全一致；涉及保留不动的腿和板，不能通过本分支的手臂自由度修复。因此输出明确记录：

- `self_geometry_under_2mm=True`：原始样本及本次 2.5 ms 加密采样中的自碰均通过。
- `sampled_geometry_under_2mm=True`：258 个原始参考姿态的全部启用碰撞对通过。
- `all_geometry_under_2mm=False`：包含插值区间后，足板仍有超过 2 mm 的残差。

本结果是**接触感知运动学参考修复**，未执行真实控制、未重新训练策略，不代表预蹲或 Ollie 物理成功。2.5 ms 检查也是离散检查，不构成任意连续时刻的严格几何证明。

## 输入、输出与修改范围

输入为 [预蹲接触扩展参考](22_Ollie预蹲接触参考对照.md)，保留原视频人工接触先验、板 v3、50 Hz 时间轴和 s=1.5 假设：

- 输入：`data/ollie/contact_extended/joint_reference_50hz.npz`，SHA-256 `0174586f94e68d35eaa63fd02743291e34db5660f2401d9e2241abe0968a7387`。
- 输出：`data/ollie/self_collision_corrected/joint_reference_50hz.npz`，SHA-256 `f6150d1217a4cf4952c3c03e601df4abe20b649ce03abb6f6b4c208d894f8dbc`。
- 时轴：258 帧，物理时间 0～5.14 s，`time_video=1.5*time_physics`。
- 状态顺序：`robot_state[:,22:36]` 为 14 个手臂位置；仅其速度 `[:,57:71]` 由修复后角度的时间差分重新生成。
- `robot_state[:,:22]`、`robot_state[:,36:57]`、`board_state`、contact/phase/confidence 及时间数组保持原值。没有通过重写接触标签隐藏几何问题。

最大单关节位置改动 **0.17810 rad**，全段全部手臂角度改动 RMS **0.01968 rad**。最大相邻帧角速度 **11.0349 rad/s**，低于 12 rad/s 限制；真实手臂关节角限位误差为 0。没有修改真实驱动力矩限位、PD 增益、碰撞掩码、质量、摩擦或关节结构。

## 修复方法与数值核对

新增 `scripts/video/repair_ollie_self_collisions.py`，复用 `compile_scene()` 的真实机器人与被动板模型，使用规范 G1 顺序的 qpos/dof 地址映射。

每个启用的 MuJoCo 接触采用有符号间隙 `d` 与接触法向 `n`，在接触点计算两个 body 的平移 Jacobian：

```text
∂d/∂q_arm ≈ n · (J_body2 - J_body1)[:, arm_dofs]
```

约束目标保留 3 mm 间隙，并用带真实关节角边界、相邻帧角速度边界的有界最小二乘求解。每步最多 0.15 rad，回溯检查实际更新后的碰撞，不能只凭线性方程预测宣称修复成功。手臂轨迹先做一帧尺度的高斯平滑，目标同时保留原姿态邻域与相邻帧平滑；优化不是全局最优或精确最小改动证明。

第一版仅约束 50 Hz 样本时，加密检查仍发现 3.575 mm 帧间自碰。因此正式版本将当前帧两侧各七个中间姿态也加入约束，插值关节 Jacobian 按对应端点权重缩放。根和板四元数采用 MuJoCo `mj_differentiatePos/mj_integratePos` 的最短旋转插值，hinge 关节线性插值。默认最多四次交替方向扫描，本次实际两次即通过自碰门槛。

另在原参考 frame124、128、219 的 12 个严重自碰接触上，以 `mj_geomDistance`、步长 1e-6 rad 做中央差分。解析接触 Jacobian 与数值差分最大绝对差为 **1.41e-10 m/rad**，确认了法向与两 body 差分的符号。诊断留在 `logs/ollie/self_collision_corrected/contact_jacobian_check.json`。

## 全身几何与足板复核

所有碰撞分类都来自同一原生模型 `mj_forward()` 的实际接触列表，覆盖模型当前启用的 self、robot-board、ground、board-internal 和 other 对。没有关闭模型原有碰撞；不把未启用的可视网格当作已检查的碰撞表面。

258 个参考样本上：

- self：修复前最大 **67.310 mm**，修复后 **1.474 mm**；大于 2 mm 的接触事件从 150 个变为 0。此处按逐帧接触条目计数，不是独立碰撞次数。
- robot-board：修复前后均 **1.235 mm**。
- ground：仅浮点误差量级，board-internal 与 other 未发现穿透。
- 左右真实足底 sites 的世界位置最大变化 **0 m**。由于根/腿/板没有改变，原受约束足底间隙 0.820 mm、anchor 平面误差 0.142 mm 等运动学足板 QA 保持。

2057 个 2.5 ms 插值姿态上：

- self：修复前有 1206 个大于 2 mm 的接触条目；修复后为 0，最大 1.474 mm。
- robot-board：修复前后均有 15 个大于 2 mm 的接触条目，最大 6.990 mm。
- 最严重残差位于参考 frame 坐标 **175.625**，物理 **3.5125 s**、视频 **5.26875 s**，`left_foot3_collision` 与 `board_nose`。
- ground 最大约 **0.000312 mm**，board-internal 和 other 无穿透。

上述足板残差不能由“采样点通过”推定不存在。后续若要构造更严的连续运动学参考，须建立另一个允许腿/板接触校正或提高参考时域分辨率的分支，同时重新检查原视频证据、接触释放和飞行时间，不能把插值穿透简单当作足板焊接条件。

## 总质心与时间尺度复核

按模型所有 body 的 `body_mass*xipos` 加权重算 G1+板总 COM，修复导致总 COM 最大位置变化 **3.572 mm**。已重新运行 [时间尺度审计](19_Ollie物理时间尺度审计.md)，结果存于 `logs/ollie/self_collision_corrected/timing_audit/`，其参考 SHA-256 与上述新输出一致。

相同视频 4.71～5.26 s 窗口、19 个插值采样中：

- 修复前展示竖直加速度 **-4.66184 m/s²**，修复后 **-4.66607 m/s²**。
- s=1.5 时对应的物理加速度由 **-10.48914** 变为 **-10.49865 m/s²**。
- 条件重力等效倍率由 **1.45063** 变为 **1.44997**。
- 二次拟合 RMSE 由 **28.885** 变为 **28.819 mm**，最大残差仍约 **68.552 mm**。

手臂修复没有实质改变前一版的时间尺度结论。边界扰动诊断的条件倍率范围仍很宽（约 1.096～1.861，非置信区间）；不能从这种经过单目估计、缩放、接触校正的非抛物线轨迹反证或验证视频慢放率。保留 s=1.5 工程基线，不用本次修复暗改时轴。板 0.61 m 峰高和所有原始板轨迹均未改变。

## 复现与后续验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python scripts/video/repair_ollie_self_collisions.py \
  --reference data/ollie/contact_extended/joint_reference_50hz.npz \
  --out data/ollie/self_collision_corrected/joint_reference_50hz.npz

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python scripts/video/audit_ollie_timing.py \
  --reference data/ollie/self_collision_corrected/joint_reference_50hz.npz \
  --out-dir logs/ollie/self_collision_corrected/timing_audit
```

脚本拒绝输出覆盖输入路径。输出同名 JSON 包含优化参数、前后全部分类最大穿透、加密检查、限位/速度/COM 诊断；`*_arm_diagnostics.npz` 保存原始和修复手臂角、前后 COM、两套时间轴。`JointReference.load()`、实际 CPU 修复、加密碰撞审查、时间审计、脚本帮助、Ruff 与编译检查均已执行。

下一步可在此合法性改善后的参考上重新评测 Reference PD、固定偏置及修正的平衡反馈，重新计算深蹲逆静力种子。本轮没有把旧参考上的站稳偏置、失败 CEM 数据或未训练 actor 改标成新参考的成功教师；动态控制、BC/WM 学习和完整 Ollie 验收仍需独立完成。
