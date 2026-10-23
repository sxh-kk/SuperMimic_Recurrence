# Ollie 真实数据联合世界模型预训练

## 实际结果与边界

已新增离线入口 `scripts/local/fit_ollie_wm_dataset.py`，只使用真实执行的 **92 维状态、29 维动作、下一状态** 训练 `JointWorldModel`。没有使用反事实 `expert_label` 当作实际动作，没有训练 actor，也没有把失败回合改标为成功教师。

实际完成 500 updates，发现总体预测改善但原片准备段板姿态有回归后，又做一次同配置、同初始权重/数据划分的 1000 updates 有限对照，没有继续扩大预算。推荐供后续真实在线数据训练初始化的权重为：

`logs/ollie_wm_offline_02/model.pt`

SHA-256：`95d42ddbaefc5865b806419f439d788b34f86e3db71a43c568204468b1fff72b`。

总体留出集 H8（0.16 s）机器人根位置 RMSE 从 90.48 mm 降到 **49.82 mm**，板根从 42.81 mm 降到 **33.67 mm**，但仍是厘米级误差。部分单步板指标和准备段板旋转仍有退步。**不能据此认为 actor 梯度可靠、学会自由飞行/接触切换或完成 Ollie。** 数据主要覆盖贴地 PD 与准备段混合控制，自由飞行覆盖不足；该检查点只是实际数据上的动力学预训练起点。

## 数据与输入契约

三个源文件共 **7950 transitions、71 个完整回合**，源文件未修改：

- source0：`logs/ollie_pd_pilot_18/transitions.npz`，3630 条/33 回合；SHA-256 `34f8db0cd35312ce0a9c226fa9065080114b6f44788caef3928e827c49ad3c92`。
- source1：`logs/ollie_pd_pilot_19/transitions.npz`，3630 条/33 回合；SHA-256 `b7dc6c53be9cfd1edc88bd3f6da00fe4261854cff7c84a2f959676a76b838303`。
- source2：`logs/ollie_bc_prefix_03/dagger_actual_transitions.npz`，690 条/5 回合；SHA-256 `61938f679fe063a234d2ab99ff26b9cdfbc2b768f563b1b20d4f8123f8f025fc`。

PD18/19 内含 `supermimic.ollie.entity_state.v1`、机器人/板关节名字和 0.02 s control_dt，均与当前规范匹配。DAgger 文件没有保存这些 schema 字段；已读其本地生成脚本 `scripts/local/warm_start_ollie_policy.py`，确认 `expert.indices.pack()` 生成状态，`executed` 按 `index.robot_actuators` 写入真实控制，连续执行八个 0.0025 s 子步，NPZ `action` 保存的正是 executed。

因此 DAgger 契约是**根据本地生成脚本明确推断，不是文件内自证**。入口默认拒绝缺 schema 的任意文件，本次使用显式 `--infer-dagger-schema`，在 manifest 记录缺失字段、规范名字、推断依据和生成脚本 SHA-256。当前审核脚本 hash 为 `fa277ea4740838dc3e1f07eb9cbb9319bd7516c64c0ce4214a19f9fd5cfb5c4c`；它不是对历史生成版本的独立证明。

DAgger 的 `verified_teacher=False` 不妨碍动力学学习，因为下一状态确实由实际动作生成；但这些回合不能因此成为成功示范。`expert_label`、`future_goals`、`label_used` 不进入世界模型输入或目标。所有输入检查有限性、92/29 维度、wxyz 单位四元数、每回合从零开始的 50 Hz 时间，且 `next_state[t] == state[t+1]`。

原始保存精度下的链检查误差全部为 **0**；PD 源另用其 float64 副本复核，DAgger 使用本身 float64 数据。实际模型使用 float32，模型精度下的链误差同样为 0。补充检查记录在 `logs/ollie_wm_offline_02/source_precision_and_split_audit.json`。

## 整回合划分与防泄漏

先按完整回合内容进行去重分组，再分配 train/heldout；相同实际回放若出现于多个文件，会被分到同一侧。内容指纹使用模型精度的状态/实际动作，并规范化 quaternion 符号和负零。当前没有完全相同的整回合重复组。

固定 seed0、20% 源内留出：

- source0：26 个训练回合，7 个留出回合。
- source1：26 个训练回合，7 个留出回合。
- source2：4 个训练回合，1 个留出回合。

每个源均至少一个完整留出回合，窗口不会跨回合。训练原有 5880 个 H8 窗；额外删除与留出集内容完全重复的 **50** 个训练窗，以及训练集内部 **30** 个重复窗，最终 **5800 训练窗、1573 留出窗**。不是将同一条教师回放的相邻窗口随机拆到两侧。

完整回合编号、分组 fingerprint、源文件 hash 和划分全部在 manifest；留出原始 episode 为 source0 `[2,4,10,11,23,26,30]`、source1 `[5,7,8,16,28,30,32]`、source2 `[4]`。源文件之间相同的 episode_id 不表示同一回合。

训练按源均匀采样，再在该源非重复窗口中均匀采样，避免690条原片数据被两个较大 PD 源淹没。留出集按实际窗口数量汇总，同时分别报告每个源。留出用于本次诊断对照，属于 validation；没有额外独立任务成功 test，也不代表不同初态/物理参数上的分布外泛化。

## 模型与训练

配置和初始 model 权重来自 `logs/ollie_bc_prefix_03/best.pt`。该 BC 检查点的 `world_model_trained=False`；没有将其 policy 权重作为模型或动力学标签。

- 状态 92、canonical feature 94、delta 90、动作 29，四个被动轮保留真实角度/速度。
- 模型 hidden 为 `(512,512,512)`，共 634970 个参数，固定场景无额外 xi。
- H8 自回归展开，batch128，Adam lr2e-4，梯度范数裁剪1.0，seed0。
- 八个轮角/轮速误差项权重 **0.05**，其余82项权重1；未把未知参考轮运动当观测。
- 只学习实际状态转移，不读取参考轨迹，因此当前参考空中几何问题不会直接成为这里的监督目标。

首500步耗时约 **32.88 s**；第二次同初始1000步对照约 **61.98 s**，仅运行小型 GPU MLP，没有启动新的物理采集或 actor 优化。

## 实际留出指标

以下 NSE 指 `JointStateSpec.error_vector` 归一化后的平方误差。`raw_nse` 是**不加轮权重的90维 NSE**，不是未归一化米制 state MSE；weighted 指八个轮项乘0.05后按总权重归一。另报告根位置米制 RMSE 与旋转 rad RMSE，避免把不同量纲混成一个分数。

总体1573个留出窗，初始 → 500步 → 1000步：

- weighted 平均窗口 NSE：**0.822704 → 0.346805 → 0.276020**。
- H1 weighted NSE：0.072387 → 0.055403 → **0.050934**；H8：2.024973 → 0.763791 → **0.592551**。
- H1 raw NSE：0.086226 → 0.069417 → **0.064721**；H8：5.176527 → 3.203490 → **2.770667**。
- H1 robot root pose NSE：0.002928 → 0.002885 → **0.002598**；H8：1.487456 → 0.657883 → **0.504668**。
- H1 board root pose NSE：0.003039 → 0.003236 → **0.003072**；H8：0.768206 → 0.462156 → **0.340120**。
- H1 robot 位置 RMSE：2.621 → 2.562 → **2.431 mm**；H8：90.480 → 59.589 → **49.818 mm**。
- H1 board 位置 RMSE：1.964 → 2.130 → **2.374 mm**；H8：42.806 → 35.846 → **33.666 mm**。
- H8 robot 旋转 RMSE：0.23770 → 0.15896 → **0.14266 rad**；board：0.19688 → 0.15030 → **0.12599 rad**。

source2 原片准备段单独131个留出窗：1000步 H8 weighted NSE **0.427317→0.088572**，robot 位置 **25.139→8.448 mm**，board位置 **16.329→9.531 mm**。其 board root pose NSE 从0.017794先退至500步0.040425，再改善到1000步 **0.013402**，是做有限延长对照的具体原因。

仍保留回归：source2 H8 board 旋转 RMSE **0.001011→0.020995 rad**，单步 board 位置 **0.385→0.644 mm**；不能只报告总体改善而隐藏这些量。更多训练没有被当作必然解决方式，本次停止在1000步，后续优先补真实自由飞行/碰撞覆盖和实际在线数据。

## 检查点、复现与限制

`logs/ollie_wm_offline_02/model.pt` 含 model、optimizer 状态、learning/environment config、关节名字、updates 和 manifest hash，不含训练后的 policy。`manifest.json` 的 SHA-256 为 `e464db7f56849d0c6f2b00398a152387b6731e6971db9921c8fecbc0a5ef0e21`；500/1000 两次使用相同数据/划分/config，故数据 manifest 相同，实际更新数分别在 checkpoint 与 diagnostics 保存。

已用 CPU map_location 加载1000步检查点，按相同配置重新建立 `JointWorldModel` 严格加载全部权重，确认参数/真实输入前向结果有限、输出为92维、manifest hash匹配、三个源文件 hash 未变化。记录在 `load_compatibility_audit.json`。这核验了 `train_ollie.py --wm-init` 所需的 model/board names 接口，没有为此启动 actor 或物理环境。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python scripts/local/fit_ollie_wm_dataset.py \
  --datasets logs/ollie_pd_pilot_18/transitions.npz \
             logs/ollie_pd_pilot_19/transitions.npz \
             logs/ollie_bc_prefix_03/dagger_actual_transitions.npz \
  --config-checkpoint logs/ollie_bc_prefix_03/best.pt \
  --infer-dagger-schema --updates 1000 --batch-size 128 \
  --out-dir logs/new_ollie_offline_wm
```

后续可显式传 `--wm-init logs/ollie_wm_offline_02/model.pt` 作为在线模型学习起点，但必须继续在真实仿真上验证策略，限制不可靠区域中的 imagination 梯度。当前模型没有足够自由飞行/落板样本，不保证接触时刻、整板离地幅度、能量或控制可行性；后续训练专用 air reset 数据也必须与原片评测和成功示范严格区分。
