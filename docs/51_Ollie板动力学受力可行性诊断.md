# Ollie 板动力学受力可行性诊断

## 结论与范围

本轮新增 `scripts/local/audit_ollie_board_wrench.py`，只做真实 MuJoCo 惯性与有限几何条件下的**逐时刻必要受力平衡诊断**。没有施加求解接触力，没有改参考、板轨迹、质量、重力、力矩或接触参数，也没有运行策略训练。

正式检查了 50 最终候选 `air_contacts_v3`，同时保留 `tail_contact_upright_v4` 无空中脚板接触的对照。两者在本次检查窗的板状态相同，脚离板仍很远：物理时间 3.14–3.54 s 的连续 21 个 50 Hz 节点没有任何距离不超过 2 mm 的足板或板地候选。即使允许被动轮加速度自由变化、轮速作额外放松，内锥和外锥均不能解释这些节点的板运动。当前候选不能作为已具有可行传力路径的物理参考。

这不证明原视频无法完成 Ollie。50 的前脚拖板目标仍未实现，单目板轨迹和 50 Hz 导数也有误差。大加速度本身不能排除短时碰撞产生大力；此处关键证据是**当前几何没有对应接触，却需要非零外部合力**。有地接触的离地、落地边界还混合了差分与接触模式变化，不能据单节点宣布原片动力学不可能。

## 输入与冻结结果

- 准备姿态对照：`data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz`，SHA256 `38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7`。
- 50 最终几何安全但接触目标失败的候选：`data/ollie/air_contacts_v3/joint_reference_50hz.npz`，SHA256 `39cb87079fc94d0b40e8df65f54c37264770a65a14b3600271eb89101aeced54`。
- 正式受力结果：`logs/ollie/board_wrench_v4_aircontacts_v3_05/report.json`；简版为同目录 `summary.json`。每份输入各检查 3.10–3.58 s，间隔 20 ms，共 25 个节点。
- 50 独立导出的密集有限几何距离：`logs/ollie/air_contacts_v3/actual_finite_gaps_400hz.npz`，2057 个 2.5 ms 样本，含两只脚到 center/tail/nose 的距离与实际最近点。51 没有把接触请求标签当作实际接触。
- 数值护栏：`logs/ollie/board_wrench_selftest_04/validation.json`。

采用原生 MuJoCo 3.16.0 的 47 个 qpos、45 个 qvel、29 个 G1 actuator。板子系统包含 board free joint 的 6 个速度分量与 4 个无驱动轮 hinge；板总质量 2.18 kg。使用新编译的原始物理模型，没有 GPU、窗口或渲染上下文。

## 惯性方程与被动轮处理

在每个参考姿态赋值 qpos/qvel 后执行 `mj_forward`，通过 `mj_fullM` 取得真实质量矩阵。板与轮的 10 行方程为：

```text
(M qacc + qfrc_bias - qfrc_passive)[board6, wheel4] = J_contact^T f
```

板上没有 actuator。程序检查 actuator/equality/隐藏 control callback，外加力字段置零；求解得到的 f 只保存在报告里。原始 29 个位置控制器没有参与这次受力求解，机器人侧的力矩与平衡可行性因此也尚未证明。

轮 qdd 未观测。仅为组装右端把其已知部分写零，再通过真实 `M_BW` 与 `M_WW` 作 Schur 消元：

```text
r_bar = r_B - M_BW M_WW^-1 r_W
J_bar = J_B  - M_BW M_WW^-1 J_W
qdd_W = M_WW^-1 (J_W f - r_W)
```

**不约束轮 qdd 为零。**求解后重新验证全部 10 行，容差为 free joint 平移每轴 1e-3 N、转动每轴 1e-4 N·m、轮每轴 1e-6 N·m。真实模型方程未通过的结果不会标为平衡成立。

轮 qd 也未观测，因此报告分开保存全部轮 0、−20、+20 rad/s 的条件结果。另在实际模型 h 对四轮 qd 的仿射关系通过四组 ±20/±50 rad/s 原生核验后，给出无界轮速放松。正式窗口最大仿射核验残差为 8.9e-14；LP 的最终轮速仍会代回真实模型重算，而非只信仿射近似。这项放松是有意偏宽的必要性测试，部分最优残差解轮速达到约 1.93e5 rad/s，**没有观测依据，也不是可实现轮速轨迹**。即使该放松可行，也不能直接认证参考。

## 接触几何、法向与摩擦

程序只使用真实 contype/conaffinity 允许的足板、板地对，并记录每个候选的两个 geom 名、世界接触点、法向、距离、condim、五个 μ 参数和候选来源。正法向始终是**作用在板或轮上的方向**：板上表面的脚向下压板，地面向上推板；板侧、底部与轮侧候选另行标记，不伪装成上表面支撑。

候选由以下有限见证组成：

1. 原生 `data.contact` 的全部有效 manifold 点。
2. `mj_geomDistance` 对真实 foot capsule 与有限 board geom 的最近点，不使用无限延长的板平面。
3. 对平行 capsule/box 补充胶囊两端见证；只接受确实落在胶囊外边界的见证，避免把胶囊内部球面当作新外表面。
4. 板地使用真实盒顶点与圆柱最低支撑线的端点/中心。

距离带最大 2 mm，仅用于几何不确定性诊断，允许该带内可能闭合的接触传力；它不是传感器测得的接触，也没有建立互补条件。单个 `geomDistance` 最近点可能漏掉平行面的分布接触，上述补充改善但没有完整覆盖连续接触片全部极值。故“外锥不可行”的准确范围是**当前有限见证集合**，不能无限外推到所有连续接触分布。

摩擦系数取实际 contact 或按 MuJoCo 的 geom priority/逐项最大值规则混合；显式 pair 若存在则优先。当前模型没有显式 pair。实际脚板滑动 μ 为 0.9；板地同为 0.9。wheel-ground 的 condim=6，保留 torsional μ=0.001 m、rolling μ=0.00005 m；未把六维接触错误简化成三个力。脚板上表面及 tail-ground 是 condim=3。本次模型 `opt.cone=0`，报告保留这一枚举值。[官方接触参数说明](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#contact-parameters)

受力 LP 同时计算两种非负法向力锥：

- **inner**：单摩擦轴方向的非负射线组合，是椭圆锥内的保守多面体。
- **outer**：每个摩擦维独立允许 `−μ_i F_N ≤ f_i ≤ μ_i F_N`，包住椭圆锥的放松盒。

inner 失败不能证明真实椭圆锥失败。outer 可行也不自动证明椭圆锥可行；程序额外保存该力解的椭圆锥违反量。outer 若不可行，只能排除相同有限见证集合与相同导数假设下的真实锥。MuJoCo 的摩擦维度及椭圆/多面锥定义见[官方动力学说明](https://mujoco.readthedocs.io/en/stable/computation/index.html)。

## 正式结果与时间导数限制

3.14–3.54 s 两份参考均无候选接触。3.18–3.50 s 的连续 17 个节点，其保守 ±40 ms 位置差分范围内也没有板地接触。该内窗对“仅是跨离地或落地差分”的解释更有区分力，但仍不是高频实测加速度。

以下有效所需合力取实际 M/h、轮速零、轮 qdd 已消元；括号内为独立的板整体 COM 位置三点二差分 `m(a_COM−g)`，单位 N：

- 3.14 s：`[20.377, 10.717, 75.113]`；位置法 `[26.587, 10.609, 152.925]`，其 ±40 ms 窗仍含板地边界，不能作为稳态空中力估计。
- 3.18 s：`[17.371, 16.097, -138.093]`；位置法 `[25.585, 16.432, -6.071]`。
- 3.26 s：`[-3.301, 19.811, -5.822]`；位置法 `[0.681, 27.877, -0.219]`。
- 3.34 s：`[-14.227, 5.302, -10.663]`；位置法 `[-12.009, 9.149, -7.211]`。
- 3.50 s：`[2.324, 6.660, 110.781]`；位置法 `[2.382, 7.018, 116.621]`。

无接触内窗两种导数均要求非零合力；位置法合力范数范围 7.56–116.86 N，M/h 形式为 6.09–140.11 N。它们并不支持同一精确峰值：存储线速度已经是位置中心差分，继续作中心差分得到的是跨 ±40 ms 的宽二阶模板，而三点位置差分只跨 ±20 ms。位姿/四元数插值和稀疏节点也会改变峰值。尤其 3.18 s 的竖直力差异很大，**不能将其中某个数当成真实鞋板撞击力**。

角速度使用 MuJoCo free joint 的 body frame 分量；其 qacc 是这些 body 分量对时间求导，平移 qvel/qacc 为 world 分量。若换成 world 角加速度，必须先按根旋转转回 body frame，不能直接写入 qacc。本轮使用存储 qvel 的导数；存储板角速度来自前向四元数差分，存在约半个采样间隔的相位近似，转动受力结论亦受这一限制。

有地接触节点仅见地面对 tail 或前右轮向上的法向，没有足板接触：

- 3.10/3.12 s：各 1 个 tail-ground 点，normal `[0,0,1]`。
- 3.56/3.58 s：各 3 个 wheel-front-right-ground 见证，normal `[0,0,1]`，含原生 manifold 与有限圆柱支撑见证。

这些节点的 outer LP 同样没有完整平衡解。例：3.10 s、轮速零的最小加权残差解保留合力范数约 113.60 N、转矩范数 2.66 N·m；无界轮速放松可把合力拟合到近零，但留下 35.54 N·m 转矩。3.58 s 在无界轮速放松下合力近零，转矩仍约 10.38 N·m。这些数是同一**力与转矩加权 L1 折中解**的两个分量，不是分别优化得到的独立最小值；转矩用 0.4 m 特征长度折算成 N。边界差分、有限接触片遗漏和未知轮速限制使这些节点只适合定位问题，不能直接作为原片不可行的证据。

## 已实际通过的护栏

`--self-test` 不读取动作数据、不创建 GL/CUDA：

- 负间隙 `geomDistance` 法向与原生 contact 对齐最小值 `0.9999999999999262`，避免穿透情况下符号翻反。
- 无接触的原生自由下落返回可行；非零 body 角速度与 20 rad/s 轮速下，原生轮 qdd 约四轮均 `−5.89724 rad/s²`，Schur 恢复一致。错误固定轮 qdd=0 会留下约 `7.14e-4` 的惯性右端，正确消元后约 `2.33e-16`。
- `Jᵀ[force, torque]` 与原生 `mj_applyFT` 最大差 `5.55e-17`。
- 一个位于真实摩擦圆内、保守菱形外的合成力返回 inner 不可行、outer 可行，确认没有把 inner 失败冒充真实锥失败。
- Ruff 与 Python 编译检查通过。

## 复现

从仓库根目录执行。输出目录须不存在，避免覆写既有审计。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/local/audit_ollie_board_wrench.py \
  --self-test --out-dir logs/ollie/board_wrench_selftest_new

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/local/audit_ollie_board_wrench.py \
  --references \
    data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz \
    data/ollie/air_contacts_v3/joint_reference_50hz.npz \
  --times 3.10 3.12 3.14 3.16 3.18 3.20 3.22 3.24 3.26 3.28 3.30 \
    3.32 3.34 3.36 3.38 3.40 3.42 3.44 3.46 3.48 3.50 3.52 3.54 3.56 3.58 \
  --out-dir logs/ollie/board_wrench_audit_new
```

## 后续决策

当前应先解决参考的人板接触可达性，再在确有 front/nose/catch 几何见证的连续时间窗重跑本脚本。50 的安全几何候选没有修成这些接触，增加策略训练预算不能替代缺失的传力路径。

若未来接触与全身限位均可行，而 outer 放松在连贯内窗仍失败，再评估带像素误差预算、PnP 二解与时间正则的板物理重拟合；本轮没有动 9 个 PnP 观测、板状态或视频时轴。

下一项 landing-aware 联合 COM 候选可以先以真正无碰撞、限位内的回板站姿确定落地可达 COM 区间，再拟合起飞/落地边界与弹道，而非继续把旧空中 COM 全部硬固定。需要同时保护准备段、板观测证据，并重新审查人板接触分量及速度边界。这只是下一步提案，本轮尚未实现，也不能仅因全局 COM 弹道与几何通过就声称完整 Ollie 动力学可行。
