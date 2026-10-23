# Ollie 真实动作干预配对数据

## 本批结果与用途

新增 `scripts/local/collect_ollie_action_branches.py`，仅使用原生 CPU MuJoCo，从同一实际仿真状态复制分支，逐关节施加正负位置目标扰动，已采集 **295个回合、2360条50 Hz实际转移、145对正负动作干预**。没有进行新的 WM 或 actor 训练。

数据目录为 `logs/ollie_action_branches_01/`，推荐转移文件为 `transitions.npz`，SHA-256：

`f3dd1fdd5be54deb3df2721134ac0fde387afd7dc0ed8fa33615334c8e64fbaf`。

这批数据针对一个明确问题：原有 on-policy 数据中状态和动作强相关，WM 可能依靠状态推进而忽略动作。这里固定初态和其余动作，只改变一个真实执行目标，直接测量动力学差异。它是**训练专用干预数据，不是成功示范、恢复站立证据或 Ollie 成功结果**。`verified_teacher=False`。

## 实际状态来源与精确重放

来源为 `logs/ollie_actor_video_timed_01/native_0000/rollout.npz`。源回合是原片时序分支的实际连续控制评测，本身判定失败；没有将参考运动学姿态冒充实际状态。

源参考 SHA-256 为 `941f748cf3b13edbcf60dd20f2db41810cee2cbd1e9b4dccba1e3b1d4b5a71f5`，源策略 checkpoint SHA-256 为 `63f7d35ef9b48b962d9a5f8efe51770020a8503063ab17ed6e6a79bce22af239`。源文件 hash、完整原评测 metadata、物理模块和板 XML hash 均写入本批 `manifest.json`。

采集器从源回合真实 `qpos/qvel` 首帧出发，按源 evaluator 相同的初始目标与 `mj_forward` 初始化，随后执行**已保存的 actual control**，每20 ms目标保持八个2.5 ms物理子步。每子步的 `mj_step → mj_forward` 顺序与源评测相同，不重新计算策略、不调用 expert label。

在 physical **0.5、1.5、2.5、2.7、2.9 s**，对应控制边界25、75、125、135、145，保存完整 `MjData` 副本。为核对最后一个 H8，实际连续重放至3.06 s。全部控制边界的 `qpos/qvel` 与原记录最大误差为 **0**。

每个初态用 **`mujoco.mj_copyData`** 复制，包含接触求解器 warmstart；每条干预开始前重新从原 snapshot 复制完整数据，不从92维状态冷启动。复制后到首次控制之间不再调用 `mj_forward`、reset 或 pose write。

初态 fingerprint 明确覆盖 integration state、qpos、qvel、ctrl、qacc、qacc_warmstart、活动 efc_force，以及 contact geom/position/frame/distance；每个分支复制后均逐一核对。`snapshots.npz` 留存这些诊断字段，**不是可独立恢复完整 MjData 的序列化文件**。要复现原求解器状态，必须按源动作链重新重放，再用完整 `mj_copyData` 分支，不能仅加载这里的92维初态。

每组未干预 baseline 后续八步也与源记录逐控制边界比较，五组误差全部为 **0**。因此本批没有把重置差异或接触求解器冷启动差异当作动作作用。

## 干预的定义与动作裁剪

每个 snapshot 产生59个回合：一个 baseline，加上规范 G1 顺序的29个关节各 `-0.05/+0.05 rad`。每回合八个控制、0.16 s。

- 八步 baseline 目标均取源文件中原来执行的完整29维向量。
- 干预回合在这八个向量的**同一个关节**上持续增加同一正负偏置；另外28个目标与 baseline 逐元素一致。
- 干预后仍按原关节范围、每边0.005 rad目标余量裁剪。未改变原驱动增益、力矩范围、软限位、重力、碰撞或摩擦。
- 动作以真实裁剪后的 `action/ctrl` 保存，另外保存逐步29维 `actual_delta_from_baseline`，不把请求偏置误当成一定施加成功。
- 八步动作序列冻结，不因为状态变化重新计算 policy。这是固定序列的动作因果实验，不是闭环策略对比。

实际有八个干预回合、32个目标元素触发裁剪。episode259 的 `right_ankle_roll_joint` 负向干预在首个20 ms被完全裁掉，因而首步没有动力学差异；后面控制边界仍有有效干预。不能把这种零实际动作差异解释为物理系统或 WM 不响应动作。

除分支复制这一次初始化外，回合内只写29个机器人目标，没有 root/board 的 qpos/qvel 回写、reset、QP外力或板驱动。原始 MuJoCo 时间保留取样时的绝对值，没有在分支开始把 `data.time` 清零。

## 接触、穿透和实际关节质量

这些 snapshot 来自**已承重的实际软接触状态**，不是无碰撞的参考 reset。其初始最大接触穿透约0.130～0.467 mm；2.5 s 的实际机器人关节已存在约0.03005 rad软限位越界。采集器保留这些状态和力，不修正姿态或把关节裁回范围。

在运行前固定的质量门槛为：所有启用接触对的动态穿透≤**30 mm**、实际关节角相对原软限位的越界≤**0.05 rad**、状态/接触力有限、时间连续、无外力。该门槛沿用训练 augmentation 的动态数据质量范围，**不等于原片教师的10 mm验收或运动学参考 reset 的2 mm门槛**，更不是硬件机械止挡保证。

本批295段全部通过，没有为补足数量放宽阈值。逐物理子步实际最大值为：

- robot self：**8.609 mm**。
- robot/board：**3.679 mm**。
- board/ground：**0.681 mm**。
- robot/ground 和 board self：本批无负间隙接触。
- 实际关节软限位越界：**0.049326 rad**，距离0.05阈值较窄。
- 最大实际 actuator force 绝对值：**54.697 N·m**；外部广义力与 body force 均为0。

穿透统计来自当前启用的 MuJoCo collision geometry 和接触候选，不代表所有视觉网格间距的几何证明，也不能因“通过数据质量筛选”称这些动态自接触为理想参考姿态。

`contacts.npz` 共记录 **19,175个物理采样**（每回合初态加64子步）、**340,002条接触记录**。每条保存 geom pair、位置、3×3接触坐标系、有符号间距、完整6维 `mj_contactForce` 和类别。每采样另存全部接触 pair signature、法向力>1 N的 pair signature、分类最大穿透、实际关节越界、最大实际驱动力及外力范数。接触坐标系第0行是 MuJoCo 从 geom0 指向 geom1 的 normal；力以 MuJoCo 本地接触坐标表达。

## 已测得的动作响应

相对各自冻结 baseline，290条非基准干预中，289条在 H1（20 ms）存在大于1e-8的 root/board 位置或速度响应，剩余一条即上述首步被完全裁掉的目标。到 H8（160 ms），290条全部存在可测响应。

H1 的差异中位数/最大值为：

- robot root 位置：**0.0351 / 0.2823 mm**。
- board root 位置：**0.00570 / 0.4110 mm**。
- robot root 线速度：**0.00264 / 0.02350 m/s**。
- board root 线速度：**0.000506 / 0.03261 m/s**。

H8 的差异中位数/最大值为：

- robot root 位置：**0.7691 / 6.3636 mm**。
- board root 位置：**0.5696 / 9.5460 mm**。
- robot root 线速度：**0.00724 / 0.05374 m/s**。
- board root 线速度：**0.00623 / 0.11591 m/s**。
- 29维关节位置差的 L2 范数：**0.04394 / 0.07506 rad**。

这里是实际位移/速度响应，不是 WM 预测精度、自动微分方向正确率或训练收益。部分 H1 板位置效应仅几微米，不宜用单一很小动作下的 root 位置差断言动作不可识别。H8则混合接触切换与累积动态，不能把它直接当作固定接触状态的局部 Jacobian。

## 文件接口与组隔离

`paired_branches.npz` 保留全部295回合（未来若有拒绝也保留，使用 `quality_valid` 区分）：

- `state (295,9,92)`、`action (295,8,29)`、`initial_state`、`endpoint_state_h1/h8`，均保留float64。
- `actual_delta_from_baseline (295,8,29)`、请求偏置、joint index/name、sign、baseline episode id、snapshot group/fingerprint。
- `time_s (8,)` 是0～0.14 s的相对**pre-action**时间；`source_time_s (295,8)` 是源原生绝对物理时间。
- `completed_controls`、`quality_valid`、`split`、标准 state schema、规范关节名、`control_dt=0.02`。

标准 `transitions.npz` 只输出通过质量门槛的完整回合：`state/next_state (2360,92)`、真实 `action/ctrl (2360,29)`、float64副本、episode/group/baseline id、逐步相对/绝对时间、实际目标偏差和源 fingerprint；不含 expert label。原数据版本、wxyz根四元数、SI单位、机器人29关节及被动四轮名字均显式校验。

`causal_effects.npz` 保存 branch/baseline 的规范化 state error vector、正负回合配对、正负实际动作差、全H8正负状态差，以及 H1/H8 位置和线速度差。字段 `baseline_difference_nse_vector` 保存的是**规范化有符号 error vector，并非已经平方平均的 NSE 标量**；不能按其字段名直接解释为训练 loss。

各正负分支只有起点是同一个状态。首个动作后状态已分叉，因此未来差分 loss 应从共同 `initial_state` 分别滚动两条实际动作序列，比较两个 endpoint 或整条响应；不能把后来不同状态的 `next_state` 差当作“同 state 的单步动作差分”。

本批按完整 **snapshot group** 留出，seed0 将 **2.9 s 的 group4全部59回合/472条转移**标为 heldout，其余四组236回合/1888条转移标为 train。baseline与58个兄弟干预不会跨集合。

**不能直接把本批交给旧 `fit_ollie_wm_dataset.py` 的 episode-only 随机划分**，否则同一个初态的不同动作分支可能进入两侧。本批没有修改该训练入口；后续训练须采用已保存的 group split，或实现等价整组划分，并单独评测动作差分响应。

五个取样点仍来自同一条原片失败回合，2.7/2.9 s尤其接近；该留出防止相同初态泄漏，但不是跨原片、跨独立回合或自由飞行分布的泛化证明。现有 [自由飞行 augmentation](34_Ollie自由飞行训练数据采集.md) 的覆盖限制仍需单独报告。

## 复现与验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python scripts/local/collect_ollie_action_branches.py \
  --source logs/ollie_actor_video_timed_01/native_0000/rollout.npz \
  --out-dir logs/new_ollie_action_branches \
  --snapshot-times 0.5 1.5 2.5 2.7 2.9 \
  --amplitude 0.05 --max-penetration-m 0.03 \
  --max-angle-overshoot-rad 0.05 --seed 0
```

输出目录必须不存在，避免覆盖原数据。`--amplitude 0.1` 可用于另一次更大干预实验，必须保存为新批并重新做质量筛选；本记录中的数字均只对应±0.05批。

采集入口已完成编译、Ruff和实际运行。数据检查包括逐回合 `next_state[k]==state[k+1]`、其他28目标逐元素等于基准、共同初态逐元素相同、实际时间保持、所有接触力有限、group不跨split、输出hash一致。

另用独立原生仿真循环重放全部 **295分支、18,880个物理子步**，所有控制边界92维状态最大误差为 **0**，全部接触 pair signature 和法向力>1 N的 pair signature 不一致数均为 **0**，逐子步无外力。该核验重新从源控制链构造完整 snapshot，没有调用采集器的控制推进函数或从状态92冷启动。结果保存在数据目录的 `validation.json`。

本批只提供数据和物理因果响应，不声称世界模型已经使用了动作或 actor 梯度已经可信。后续需要对照 [动作梯度审计](36_Ollie世界模型动作梯度审计.md)，检查预训练后模型是否恢复实际正负响应的方向和量级，再决定是否扩大策略训练预算。
