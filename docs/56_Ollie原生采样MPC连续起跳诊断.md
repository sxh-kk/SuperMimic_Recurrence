# Ollie 原生采样 MPC 连续起跳诊断

**唯一固定配置已连续完成 5.14 s，严格结果为 FAILED。**准备前缀逐值保留，后续没有形成合格全板腾空，3.785 s 起身体触地、板曾倒置，最大穿透 14.35 mm。运行耗时 147.92 s；这说明计算与连续物理链路可执行，不能说明已经完成 Ollie。

## 目的与输入

本轮从 53 已验证的**真实准备轨迹**连续接到 5.14 s，研究原生物理预测能否在已知不一致的运动学目标附近找到较可行的控制。入口为 `scripts/local/continue_ollie_sampling_mpc.py`，只控制原 G1 的 29 个位置目标。它是 CEM-MPC 工程诊断，不是 SuperMimic 策略训练、世界模型优化或已完成 Ollie 的证明。

以下输入在首次延续控制前冻结：

- 53 实际前缀：`logs/ollie_sampling_mpc_prep_01/actual_rollout.npz`，138×29 实际目标，139 个 92 维边界状态。SHA256 `559fdfb376120a1529bbe5e3de8f0537103c7f4001a4ed3934e5be7152b9b96e`。
- 53 原准备参考：`data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz`，SHA256 `38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7`。
- 原静力前馈：`logs/ollie_quasistatic/tail_upright_prep_01/contact_feedforward.npz`，SHA256 `08b18eb03c59cc30379dd328053298961e3dcd970693a775b856fd3bc34f5115`。
- 54 **已明确失败的软名义目标**：`data/ollie/landing_ballistic_v1/failed_joint_reference_50hz.npz`，SHA256 `62177f3a1c842718bd2eaf824cafae3e5a320a24f434609a9e9c65b16891f560`。它有约 34 mm 板穿透、17 mm 地穿透，`eligible_training_reference=False`。使用它只为给采样提供软方向，不能当作已可执行参考、训练正样本或中途初始化。

`run` 模式锁定上述 SHA；未来参考必须保留 0–2.76 s 全部原准备参考状态。主轨迹从唯一原始初态开始，重放全部 138 个保存的真实目标；每个 20 ms 边界要求 qpos、qvel、state92 **逐值一致**。到达 2.76 s 后保留真实热状态和 solver warmstart，才开始规划剩余 119 个目标。没有把根、板、速度改成相位参考；真实末 root VZ=0.3317 m/s 与参考约 1.0005 m/s 的差异必须由控制面对。

## 首次实际延续前锁定的配置

- 原生 MuJoCo CPU 物理，16 个预测线程；实际控制 50 Hz，每控制 8 个 2.5 ms 子步。
- 128 候选，H48，即 0.96 s；每决策 3 轮，top8 CEM，6 个时间结点、15 个腿腰残差维，初始 σ=0.10 rad，σ 范围 0.005–0.10 rad，种子 56。
- 每轮包含新的零残差名义序列与当前均值序列；实际仅执行所有已模拟序列中胜者的首目标，不执行未评估的更新均值。
- 首次均值继承 53 最后**已评估的胜出目标计划**：移去已执行首步、末目标保持到 H48，再减去新名义目标并拟合六结点残差。它是 target 序列热启动，不是状态重置，也不假设旧预测在新时刻仍正确。
- 采用该热启动的原因在首次实际控制前确定：新名义首目标与末实际目标的腿腰最大差约 0.50575 rad；零均值、σ=0.10 的局部采样未必覆盖连续目标。并非试验后挑选更好初始化。
- 唯一完整回合，实际计算 wall 上限 300 s；完整 batch 物理加代价投影若大于 280 s，在主延续前拒绝。没有配置网格或改门槛来选择成功结果。

H48 在 2.76 s 首次规划时能看到 3.72 s，约覆盖源片首次落地后 0.16 s；仍不能在这一视界里证明末 300 ms 站稳。最终成功必须另按完整回合实际接触验收。

## 名义目标与终点延拓

```text
u = clip((t - 2.76) / (3.10 - 2.76), 0, 1)
fade = 1 - (10u^3 - 15u^4 + 6u^5)
target = q_ref + (Kd * qd_ref + fade * tau_static_last) / Kp
```

全部目标截在原关节目标区间内侧 5 mrad，原 Kp/Kd 和真实 actuator force caps 不变。静力接触力从不直接施加；板没有 actuator。quintic fade 是目标前馈的平滑退出，不是根或板的外力。

预测时间**严格超过 5.14 s**时保持最后参考姿态，但期望 joint qd、board world v 置零。这是恒姿态预测尾端的工程假设，避免“位姿不动但仍要求末速度”的矛盾；0–5.14 s 源参考、真实状态和最终质量标准均不改。

## 预测代价与接触任务

全部预测代价与传感器使用子步前时刻。`mujoco.rollout` 返回子步后 FULLPHYSICS，因此实际广义 q/qd 使用“初态 + returned states 去掉最后一帧”的移位数组；没有将子步后 q/qd 与子步前几何混用。

保留 53 的软跟踪：双脚 site 在板坐标中的误差（XY 20 mm、Z 10 mm）、机器人 COM 相对板（50 mm）、pelvis 相对板旋转（0.15 rad）、全 29 关节位置/速度（0.2 rad、2 rad/s，速度权重 0.25），以及动作变化小罚项。这里的 site 跟踪只是参考姿态软任务，不代表真实足底接触。

板改用其**目标四元数测地误差**，尺度 0.15 rad；绝对 Z 尺度 40 mm；世界 XY 尺度 0.10 m、三维线速度尺度 0.4 m/s（后二者权重 0.25）。完全移除了 53 “板始终直立”的常值倾角项，允许板按目标抬头。机器人 upright 的大罚阈值改为 0.2，避免把空中原参考俯仰与准备 upright≥0.8 混为硬约束；最终实际 Ollie 规则没有随之放宽。

新增原生 CONTACT 只读传感器：

- 全体 contact 的 `found + dist`，`mindist`、num=1，读取预测原生最深接触；深度超过 5 mm 的软成本为 `1000 * (max(depth-0.005,0)/0.005)^2`。
- pelvis 整个 subtree 对 ground 的 `found`，存在几何接触时加 10000 大罚。`found` 包含匹配到的 contact，不能当成已测得正接触力；实际验收仍用 ContactObserver 的力阈值。

CONTACT 数据来自物理碰撞流程，不额外注入力。只请求几何字段时传感器位于 POS 阶段；空槽为零，`found` 区分无接触与零距离。这些语义依据[官方 CONTACT 说明](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-contact)，并已用原生逐步接触数组独立核验。

源 49 的前脚 nose 拖动与后脚 catch 是额外几何软任务：前脚约 2.9474 s 开始靠近 nose，3.2811 s 后允许 center/nose；后脚约 3.2811–3.3089 s 进入 center/tail 回板候选。尺度 20 mm。每个有限盒取真实 top/upper-side 的五个投影见证点，求点到实际七个足胶囊的距离，允许鞋侧、鞋尖、盒边缘关系，不强制脚 Z 法向与板面平行。

**五个点不是完整 capsule-box 最短距离，也不是实际接触证明。**它们可能漏掉更好的接触位置；没有写 contact labels、没有施加对应力、没有把脚焊到板上。真正的穿透和身体触地由原生 contact 代价处理，最终支持由 400 Hz 实际 ContactObserver 处理。

## 已完成的前缀、分支与成本校验

`logs/ollie_sampling_mpc_continuation_preflight_04/` 保存的校验均未执行 2.76 s 后的主控制：

- 从唯一初态完整重放 138 实际目标，qpos/qvel/state 最大误差均 0；1105 条原 ContactObserver 字段逐值一致，热末态 fingerprint 保留。
- 添加只读传感器前后，468 个非 sensor/name 数字数组及 option 字段逐值相等。
- 从该真实热末态展开 128×H48×8 scratch 分支；首分支与未加传感器的原模型独立逐子步 FULLPHYSICS 最大误差 0，全部几何、CONTACT 最小距离/计数与身体触地计数误差 0。
- 由 foot body 只读位姿及原 geom 变换重建全部足胶囊，其中心/长轴与原生 FK 最大差 2.34e-15。没有用 foot site 替代胶囊几何。
- 主 MjData 的完整 integration/qacc/efc fingerprint 不变；正式 CEM RNG 与均值不被 benchmark 消耗。
- 原始循环实现完整代价曾测约 0.789 s，配合预测导致约 396 s 的预算投影，因此没有启动实际控制。改为批量相对变换、解析轴距，未改候选、视界、轮数、代价、400 Hz 采样或物理。
- `cost_equivalence.json/.npz` 对同一 128 分支验证：四组几何 gap 最大差 ≤1.67e-15 m，全代价数组差 0、argmin 与 finite mask 一致。没有冒称所有浮点中间量 bitwise 一致。
- CPU 释放后单 batch 完整 cost 约 0.299 s，物理加代价的 119×3 投影约 176.57 s；投影不是完成回合的性能保证，复杂接触可改变速度。

`failure_archive_tests.json` 另用 fake step 验证失败记录，不是物理控制试验：非法候选 cost 写 null+valid mask，只有有限候选能进入 elites；完整 8 步实际边界先归档再核预测；中途失败另存目标、已执行子步数、raw terminal qpos/qvel/time，不冒充完整 20 ms transition。三种失败路径都维持常规边界数组 `len(state)=len(action)+1`。

## 完整实际验收与输出

真实轨迹必须连续完成 5.14 s，完整 prefix 与延续都按 400 Hz 记录。使用未改的 `evaluate_ollie_policy.outcome`：`criteria_version=whole_board_geometry_top_support_v3`。包括尾接触先于持续至少 40 ms 的全板腾空、全板至少 15 mm 离地、板不倒置、身体不触地、最大穿透≤10 mm、初始≤2 mm、无外力，以及末 300 ms 双脚真实上表面支持和四轮接地/速度/姿态等原有门槛。预测代价较低不替代成功判定。

输出包括实际边界 qpos/qvel/state92、真实 action29、400 Hz contacts、候选选择、每个获选预测计划及完整 v3 结果。失败也保留实际末态；没有阶段复位、重跑挑选或自动认证 teacher。

## 唯一实际回合结果

实际目录 `logs/ollie_sampling_mpc_continuation_01/`，终点 5.140000000000104 s，wall 147.920 s。run 内再次完成独立 branch/sensor proof；首次实际控制前测得的完整 119×3 预算投影为 192.80 s，低于 280 s 前置资源门槛。

- 257 个实际目标（原 prefix 138 + 延续 119），258 个 50 Hz 边界状态；2057 条 400 Hz 接触记录。
- `full_reference_episode=True`、`numerically_continuous=True`、`numerical_failure=None`，没有 partial control。
- 初始穿透约 3.47e-18 m，无外力、无阶段重置、无根/板回写；119 次实际首控制与当次获选预测的末状态最大差为 0。
- **没有任何合格全板 flight window**，全板最低几何点的最大高度仅 **1.282 mm**。四轮最低点曾升高 22.862 mm，但此时板尾/板鼻等几何仍低，不能据四轮高度伪称全板腾空。
- 3.785 s 首次身体触地；最大穿透 **14.350 mm @3.8125 s**，超过原 10 mm 门槛。
- 板 upright 最低 **−0.378873 @3.9925 s**，倒置规则失败；机器人 upright 最低 0.661443 @4.145 s。
- 首次 nose-ground 3.935 s、tail-ground 3.940 s，均晚于身体触地。该迟到的尾接触没有接上合格腾空，不是成功踩尾 pop。
- 末 300 ms 左/右足真实上表面>1 N 支持比例 68.33%/100%，四轮实际接地比例 **0**，因此不是稳定站在板上。
- 最大实际软关节越界约 0.04955 rad，单独报告；没有拿它代替上述失败判据。

原 `whole_board_geometry_top_support_v3` 的六项失败为：尾接触先于合格 flight、全板 15 mm 离地、板不倒置、身体不触地、穿透≤10 mm、末 300 ms 站板稳定。完整时长、数值连续、初始≤2 mm、无外力规则通过。没有修改任何一个成功阈值，也没有因后段代价下降而改写结果。

## 产物一致性核验与边界

- `actual_rollout.npz` SHA256：`44467039d304d547af23557136ee96cbaeca347ba7ab0ab701564ea6658a0fcc`。state258×92、qpos258×47、qvel258×45、action257×29，全部有限。
- 与 53 原文件比较，前缀 state/qpos/qvel/action 四数组逐值相同。
- `selected_plans.npz` 保存 119×48×29 获选目标和 119×384×93 FULLPHYSICS；逐值核对其首 action 与真正延续 action 一致，第 8 物理状态的 qpos/qvel 与真正下一控制边界一致。
- `result.json` 保存完整 v3 判断，`contacts.json` 保存所有 400 Hz 实际记录，`choices.json` 保存三轮候选代价/valid mask/finite elites。`preflight.json` 保留冻结配置、输入 SHA、热末状态 fingerprint、传感器时序和 terminal-hold 假设；`output_hashes.json` 汇总产物 SHA。
- 2026-10-25 另用原始无新增传感器模型，从唯一原始初态连续重放全部 257 目标：qpos/qvel/state92 最大误差均为 0，2057 条 ContactObserver 全部原字段逐值相等，严格 v3 结果全部字段与原结果相同。`independent_replay_audit_retarget.json` 保留审计；没有重新搜索、训练或初始化中间相位。
- 力矩在每个 `mj_step` 后、额外诊断 `mj_forward` 前读取：执行力矩未超原 cap。准备段腿部峰值为 cap 的 35.59%，无腿≥95%；延续段仅左髋 pitch ≥95%累计 80 ms，峰值100%出现在4.18 s。不能把此次失败归结为腿部全程力矩饱和；准备段左腕 pitch 曾达到100%，不再沿用旧 post-forward 83.96% 作为执行阶段峰值。
- [完整原片／真实物理对照](/home/zxc/SuperMimic/logs/ollie_sampling_mpc_continuation_01/video_01/comparison_full.mp4)：1280×480、24 fps、187帧，SHA256 `fa8d3127a2de58c40a90cbfdddc656a60327ca0cb0cb1838aacbd51add1e0030`。右栏明确标记 FAILED，保留失败尾段。
- [实际物理原速](/home/zxc/SuperMimic/logs/ollie_sampling_mpc_continuation_01/video_01/rollout.mp4)：800×600、50 fps、258帧，SHA256 `a79fc66aa004e5a3d07f06fc224250fb0228161095ebc84497c3ccc286ce18c8`。只在独立 rendering data 中渲染保存 qpos，主状态文件 hash 不变；两视频完整解码及六时点画面人工检查通过。对照使用待验证的1.5时标，末实际姿态5.14 s对应视频7.71 s，末显示采样点最多保持40 ms。

这次固定预算的局部搜索没有找到完整可行 Ollie，亦没有证明所有控制或原片都不可行。已知不可行软名义目标、前缀末真实速度与目标差异、90 维有限采样以及接触模式变化均是限制。没有再跑配置，没有把该回合导出成成功 teacher，也没有自动开启新训练。

## 运行入口

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/local/continue_ollie_sampling_mpc.py \
  --mode benchmark \
  --reference data/ollie/landing_ballistic_v1/failed_joint_reference_50hz.npz \
  --out-dir logs/ollie_sampling_mpc_continuation_proof_new

# 只有首次实际配置与参考锁定且完整proof/成本门控通过后，使用新的目录执行。
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/local/continue_ollie_sampling_mpc.py \
  --mode run \
  --reference data/ollie/landing_ballistic_v1/failed_joint_reference_50hz.npz \
  --out-dir logs/ollie_sampling_mpc_continuation_new
```
