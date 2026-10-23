# Ollie 原生物理采样 MPC 准备试验

## 范围

**唯一固定配置完成0～2.76 s准备段，并通过原有实际质量门槛；独立保存动作回放的qpos/qvel/92维状态误差均为0。** 这是准备段教师探索，不是 SuperMimic 算法、世界模型梯度训练或已完成的 Ollie。末状态尚有上升和角动量，没有验证其后踩尾、腾空、接板、落地的连接。

独立入口为 `scripts/local/track_ollie_sampling_mpc.py`，没有修改文档48/52的 QP、共享静力求解器、训练入口或物理资产。主轨迹只初始化一次，每20 ms执行原29个关节位置目标，内部仍为8个2.5 ms原生物理子步。没有施加求出的接触力、板驱动、根/板状态回写或阶段重置。预测分支允许从主轨迹当前热状态另行展开，不能污染主轨迹。

参考是 `data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz`，SHA-256：

`38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7`

静力前馈来自 `logs/ollie_quasistatic/tail_upright_prep_01/contact_feedforward.npz`，SHA-256：

`08b18eb03c59cc30379dd328053298961e3dcd970693a775b856fd3bc34f5115`

入口严格检查该文件的参考SHA、139帧qpos逐值一致、静力帧求解成功。名义目标为 `qref+(tau+Kd*qdref)/Kp`，再截到原真实关节目标范围内侧5 mrad；求出的点力从不施加。静力前馈只有准备段，预测视界超过2.76 s后保持最后准备姿态/前馈，不接入尚未验证的空中参考。

## 首次控制之前确定的唯一配置

先测得64候选×16控制步×8物理子步约0.039 s。根据这个成本，在任何MPC主轨迹控制之前，将唯一配置确定为：

- CPU原生 `mujoco.rollout.Rollout(nthread=16)`；128候选，H32，即0.64 s真实预测。
- 每个控制周期3轮CEM，top8更新分布；5个时间结点、12腿+3腰共15维残差，初始标准差0.06 rad，随机种子53。
- 每轮包含静力名义基线和当前均值；其余候选独立采样、时间线性插值，均截到原真实目标限位。
- 下一轮更新均值/方差，方差范围0.005～0.06 rad；跨周期移位的是已评估胜出序列的残差。
- 实际只执行所有3轮中**真实模拟过且代价最低的序列首目标**，从不执行未经模拟的新CEM均值。

没有先跑64/H16的主轨迹再挑128/H32，也没有控制参数网格。按0.10→0.50→1.00→2.76 s门控顺序，前一段通过后才延长；每段都从同一初态、同一种子完整连续开始。所有实际控制试验合计预算不超过300 s。

## 分支一致性与传感器排错

保留原始场景作为主轨迹模型；预测模型仅添加只读位置、姿态和subtree COM传感器。质量、惯性、关节、接触参数和驱动数组逐值相等。分支初值使用 `FULLPHYSICS`（含时间、qpos、qvel）及实际 `qacc_warmstart`，不是只冷设qpos/qvel。

最终proof从已连续物理运行0.10 s的热状态开始：128×32×8分支，所有分支与原模型独立逐子步执行的FULLPHYSICS最大误差 **0**。根、板、双脚和COM的全部新增传感器与**子步前**独立FK最大误差 **0**；和子步后状态会有非零差，因此代价明确使用初态加返回states去掉末帧的移位方式，与传感器时刻对齐。主轨迹每个物理子步后重新forward，再进行实际接触质量评估。

最终单批benchmark约0.127～0.138 s（早一次冷批约0.469 s）。主执行每20 ms还与选中分支的第8个物理状态核对，任何误差超过1e-10立即拒绝。

保留了一个明确的实现排错分支：`logs/ollie_sampling_mpc_smoke_01/`。初版FRAMEQUAT使用`BODY`，MuJoCo返回**惯性主轴系**；板的惯性主轴和板体坐标不同，导致初态脚误差错报333 mm、板倾角代价错误。这个试验不作为有效控制结论。已改为`XBODY`并扩展上述全部姿态proof；没有改CEM配置或质量门槛，随后从头重做0.10 s。旧文件没有覆盖或删除。

## 代价和实际验收

每个真实预测物理子步计算：

- 双脚在板体坐标中的位置误差，XY尺度20 mm、Z尺度10 mm。
- 机器人COM相对板位置，尺度50 mm；pelvis相对板旋转，尺度0.15 rad。
- 全29关节位置/速度，尺度0.2 rad和2 rad/s；速度项权重0.25。
- 板的世界倾角，尺度0.1 rad。
- 板的世界XY参考位置/速度，尺度0.10 m和0.4 m/s，权重0.25，避免只跟踪相对站姿却人板共同滑走。
- upright<0.8、root低于参考0.16 m或实际软关节越界>0.05 rad的大罚项（系数1e4），以及小的动作变化代价（系数0.01）。

运行代价取均值，并给最后一个传感器时刻额外2倍代价。最后传感器时刻为视界结束前一个2.5 ms子步；没有冒称它是视界后状态。非有限结果和物理时间未完整推进的候选拒绝。

上述只是采样排序。最终仍按48/52原有400 Hz真实质量门槛验收：upright≥0.85、无身体触地、最大穿透≤10 mm、实际软关节越界≤0.05 rad、初始穿透≤2 mm、足高度误差≤25 mm/平面误差≤40 mm、root-board XY≤0.2 m、root高度RMSE≤80 mm/最大≤160 mm、两足真实上表面向上支持全段及末300 ms各≥80%、真实唯一4轮接地比例≥80%。足支持需>1 N，轮接触检测阈值沿用0.1 N。候选代价没有代替这些真实接触门槛。

## 已完成的门控结果

**0.10 s，`logs/ollie_sampling_mpc_smoke_02/`，通过。**

- 最小upright0.99813；穿透2.941 mm、实际软关节越界0。
- 足XY最大误差2.090 mm、Z2.016 mm；关节位置RMSE0.00986 rad。
- 双足支持95.12%，4轮接地82.93%；板XY位移3.166 mm、最大XY速度0.0809 m/s。
- CPU2.914 s，执行首控制预测误差0。

**0.50 s，`logs/ollie_sampling_mpc_pilot_01/`，通过。**

- 最小upright0.99784，无身体触地；穿透2.941 mm、实际软关节越界0.00589 rad。
- 足XY最大误差3.731 mm、Z2.016 mm；root高度RMSE1.244 mm。
- 关节位置RMSE0.02772 rad、最大0.14753 rad，速度RMSE0.28713 rad/s。
- 双足支持99.0%、末300 ms均100%；4轮接地96.52%。
- 板XY最大位移13.341 mm、最大XY速度0.1714 m/s；CPU14.403 s。

同参考的52 strong QP在0.50 s关节RMSE0.11542 rad、足平面误差20.704 mm；这里为0.02772 rad和3.731 mm。这个对照支持真实预测改善短段控制，但不据此推论完整Ollie可行。

**1.00 s，`logs/ollie_sampling_mpc_second_01/`，通过。**

- 最小upright0.99145，无身体触地；穿透2.941 mm、实际软关节越界0.00589 rad。
- 足XY最大误差5.603 mm、Z2.016 mm；root高度RMSE2.261 mm/最大6.677 mm。
- 关节位置RMSE0.03354 rad、最大0.14753 rad。
- 双足支持99.50%、末300 ms均100%；4轮接地98.25%。
- 板XY最大位移23.186 mm、最大XY速度0.2282 m/s；CPU27.670 s。

**2.76 s，`logs/ollie_sampling_mpc_prep_01/`，完整准备段通过。**

- 138个连续20 ms控制；最小upright **0.92158**，无身体触地。
- 最大穿透 **4.220 mm**，实际软关节越界 **0.01207 rad**；没有放宽原10 mm/0.05 rad门槛。
- 足XY最大误差 **11.843 mm**、Z **3.063 mm**；root-board XY最大0.14136 m。
- root高度RMSE **6.239 mm**、最大误差36.598 mm；关节位置RMSE **0.03662 rad**、最大0.26116 rad，速度RMSE0.47043 rad/s。
- 左/右脚实际上表面向上支持 **99.638%/99.729%**，末300 ms均100%；真实唯一4轮接地比例 **99.005%**。
- 板XY最大位移 **23.186 mm**，最大XY速度 **0.22818 m/s**。这些是真实世界位移/速度，没有只靠相对人板误差判通过。
- CPU控制计算 **78.762 s**。四个有效控制回合合计123.75 s，另保留坐标排错回合3.28 s；没有追加参数搜索或飞行控制。

52 strong QP在同参考下1.475 s身体触地；这次原生采样完成了准备段。两者是不同工程控制器，不能将此结果归为SuperMimic训练收敛。

## 独立重放与末状态余量

完整实际轨迹 `actual_rollout.npz` SHA-256：

`559fdfb376120a1529bbe5e3de8f0537103c7f4001a4ed3934e5be7152b9b96e`

审计另用未加传感器的原模型，从唯一保存初态开始重放138个实际目标，未冷置中间状态。qpos、qvel和state92最大误差均 **0**；1105条真实ContactObserver的全部原字段逐值相等。独立使用板体旋转矩阵计算脚位置，足Z/XY最大误差为3.062696/11.843479 mm，与主日志一致。实际目标没有越界，实际力矩没有超过原cap，最高达到cap的83.96%。所有状态/动作有限，全部准备门槛再次通过。

保存的获选计划首动作与实际动作逐值相同，每个首控制末状态与实际qpos/qvel均误差0。完整回合前1.00 s的state/action/qpos/qvel与先前1.00 s回合逐值相同，没有为延长回合更换种子、姿态或控制参数。

原始审计在 `independent_replay_audit.json/.npz`，包含逐物理子步实际力矩、机器人/板/系统COM速度以及末状态。**2.76 s（按1.5时标对应原视频4.14 s）不是静止交接点**：

- root世界速度 `[+0.02068,−0.08801,+0.33171] m/s`；同帧参考root速度为`[−1.3160,−0.7749,+1.0005] m/s`。通过位置门槛不等于速度精确匹配原参考。
- 机器人COM世界速度 `[+0.01033,−0.01548,+0.28200] m/s`；整个机器人+板系统COM为`[+0.00077,−0.01459,+0.26470] m/s`。
- root世界角速度范数 **0.71251 rad/s**；pelvis upright **0.92441**，高于0.85门槛0.07441，同帧参考0.92473；与参考完整旋转误差0.14413 rad。
- 板世界速度约`[−0.14545,−0.00098,−0.00001] m/s`，板角速度约0.00123 rad/s；末双脚向上支持211.23/194.08 N、4轮实际接地。
- 末root高度比参考低 **36.60 mm**；末关节速度误差RMSE **1.7733 rad/s**。该状态没有证明能实现原参考要求的下一段冲量。

因此不能把通过准备门槛写成“可直接拼接起跳教师”。后续必须从这个实际末状态和完整控制前缀产生热状态，重新验证起跳连接；不能把机器人/板回写到参考2.76 s或把末速清零。

## 实际准备视频

视频只渲染不可变的实际qpos，没有重新仿真或参考ghost。原视频对照按假设的1.5时标同步，右侧标注`PREFIX PASS`和`no Ollie`。

- [实际准备视频](/home/zxc/SuperMimic/logs/ollie_sampling_mpc_prep_01/video_01/rollout.mp4)：800×600、50 fps、139个实际姿态（0～2.76 s）；SHA-256 `a183debe7cb0b2a73020eb0dfacb309e1e9d0b16262d99f25ab616e86128c2e5`。
- [原视频与实际准备对照](/home/zxc/SuperMimic/logs/ollie_sampling_mpc_prep_01/video_01/comparison_preparation.mp4)：1280×480、24 fps，源视频0～4.14 s；SHA-256 `e1e19c5c04f011b84dddf40d79f520ba671713dfc80fa10b4148d1758c8201cb`。
- 同目录`render.json`、`video_audit.json`和`preview.png`记录全解码/视觉检查。视频末姿态显示时长会包括最后一帧，实际物理区间以139姿态之间的138×20 ms=2.76 s为准。

## 准备教师导出提案，尚未执行

现有`actual_rollout.npz`保持139个边界state和138个实际action。若后续决定将其作为**仅准备段**教师，可以另导出兼容既有初始化入口的文件：

- `state=state[:-1]`、`next_state=state[1:]`各138×92；`action`为138×29的绝对实际PD目标，不能导出残差均值代替它。
- `time_s=0.02*arange(138)`、`joint_names`和`board_joint_names`用现有严格顺序；另保留`terminal_state`和原qpos/qvel。
- 既有actor接口的`reference_offsets=[0,5,10]`，`reference_goal_indices=min(arange(1,139)[:,None]+offsets,len(reference)-1)`，`future_goals`取绑定SHA的参考状态float32。动作与下一目标的索引语义沿用原接口，不把MPC名义目标误当执行动作。
- `metadata_json`应携带本轮全部实际gates、`initializations=1`、`intermediate_resets=0`、`board_actuators=0`、`forces_applied=False`、原参考/前馈/轨迹SHA，以及`scope=preparation_prefix`和有效物理末时刻2.76 s。
- `verified_teacher`仍需单独决定，不能因导出自动改True；不得把末尾超出准备段的future reference当作已验证动态目标。需要明确准备网络的生效时段，并从实际末状态验证下一段。

本轮没有导出自动认证教师、BC训练或延长飞行。该提案与实际通过准备质量验收是两个独立结论。

## 输出与重跑

每段保存实际92维联合状态、29维实际执行目标、原模型qpos/qvel、全部400 Hz接触与关节误差、每轮128候选代价、选择编号以及获选的完整32步动作/256物理状态计划。日志中的`verified_teacher=False`、`ollie_success_certified=False`始终保留，准备段通过也不自动晋升为最终教师或完整技巧。

先验证，再按门控逐段运行，输出目录必须不存在：

```bash
env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked python scripts/local/track_ollie_sampling_mpc.py \
  --out-dir logs/ollie_sampling_mpc_proof_new --benchmark-only

env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked python scripts/local/track_ollie_sampling_mpc.py \
  --out-dir logs/ollie_sampling_mpc_smoke_new --seconds .1
```

只有该段通过，才用新的输出目录依次改为`.5`、`1`、`2.76`。不修改候选数、视界、轮数或代价来挑最好结果。
