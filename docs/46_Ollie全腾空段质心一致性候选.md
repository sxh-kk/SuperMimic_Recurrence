# Ollie 全腾空段质心一致性候选

## 结果：联合节点条件通过，整体必要动力学条件失败

新增独立脚本 `scripts/video/repair_ollie_ballistic_reference.py`，完成CPU参考校正和独立核验。只改机器人根XYZ与根线速度，保留完整板状态、两条时轴、29个关节位置/速度、机器人根四元数/角速度及所有phase/contact/confidence标记。没有训练、没有物理执行、没有施加任何力或修改32/44/shared模块。

最终诊断候选为 `data/ollie/ballistic_tail_upright_candidate_v3/joint_reference_50hz.npz`，SHA-256：

`1611f169de8e5df7442aadccabd309f2ecbc33b9acd1f530ca9f774c68b997d8`。

它在**50 Hz节点及其中心有限差分**上修正了主腾空窗的联合COM弹道和边界突变，完整密集碰撞仍≤1.313 mm。但接触连通分量审计发现：人和板彼此脱离时，固定板自身COM和机器人自身COM都不满足独立弹道。因此状态明确为：

```text
candidate_status = GLOBAL_NODE_CONDITION_ONLY_COMPONENT_DYNAMICS_FAILED
global_com_correction_passed = true
all_sampled_component_necessary_conditions_passed = false
recommended_as_physical_reference = false
```

这里的`global_com_correction_passed`仅指下文明确范围内的联合COM节点条件，不能解释成连续动力学可行、成功起跳或可直接作为物理教师。**本批没有产出推荐的、全部必要条件合格的物理参考。**

## 输入与保留范围

先读取旧 `air_relative_video_timed_v1`（SHA `941f748cf3b13edbcf60dd20f2db41810cee2cbd1e9b4dccba1e3b1d4b5a71f5`）定位边界问题，正式候选输入采用已通过尾板接触/直立几何QA的新分支：

`data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz`，SHA-256 `38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7`。

板全部21维、原单目拟合/PnP来源和时间假设保留，未修改板以迁就COM。源参考的 **physical 0～2.90 s完整92维状态逐bit相同**；另保留2.92 s这个邻帧的根位移为零校正，使保护末端的中心有限差分不受后续修改影响。

独立检查还确认：完整板状态、机器人除根XYZ/线速度以外的所有字段，以及time/labels/confidence，均逐bit保持。没有更改慢放倍率、关节角、角速度、轮状态或脚接触标签，也没有将新参考绑定为旧教师的新成功轨迹。

## 用真实几何识别离地窗口

每个50 Hz区间按八子步检查，共 **2057个2.5 ms样本**。使用实际MuJoCo碰撞模型、启用的机器人和整板碰撞体，以primitive的完整最低支撑包络确定ground clearance；包括板盒子、倾斜尾/鼻和四轮圆柱，而非仅板root高度。

同时构造机器人、板、地面三个接触连通节点：

- ground连接由实际接触候选与各实体完整最低包络共同判断。
- robot/board连接除了实际`mj_contact`外，还通过启用几何的包围球筛选和 `mj_geomDistance` 查询≤2 mm正间隙，避免把ncon未包含的近接面武断判成完全脱离。
- **2 mm是重建几何不确定带，不是实测接触力或支持证明**。连边只保守地表示可能传力；无边表示在该几何带下没有传力路径。

当机器人和整板都高于2 mm且都无ground连接，自动得到主离地窗口 **3.1225～3.5450 s**，170个密集采样。未使用写死的“宽飞行相位”或仅3.16～3.48的内部拟合窗。

另外逐项报告>1 µm净空且无严格ground接触的窗口。固定板插值在准备/落地段存在若干很小的间隙，保护段最大约 **0.317 mm**，其中576个严格微间隙采样落在0～2.90 s保护范围。全片618个采样属于严格微间隙而未达到2 mm可靠分离标准。不能为修复这些不确定微隙偷改保护段或板，也不能把它们宣称已测得的地面支持。

主窗口在严格微隙口径下延续到 **3.5575 s**；下述缓冲已经覆盖它。若要求所有µm级间隙，包括整个保护段，都精确无支撑弹道，那么现有固定板/保护前缀约束本身就不相容，本候选不作这种认证。

## 整段COM与缓冲区的构造

全部35.521142 kg质量均纳入：机器人33.341142 kg，板和四轮共2.18 kg。逐帧用所有body的world inertial `xipos` 计算：

```text
C = sum(body_mass * xipos) / total_mass
C_new = C_source + robot_mass / total_mass * root_translation_delta
C_ballistic(t) = C0 + V0 * (t - t0) + 0.5 * [0, 0, -9.81] * (t - t0)^2
```

在完整离地窗两端先包含其相邻50 Hz节点，再各延拓一个节点，得到弹道节点缓冲 **3.10～3.58 s**。这样首末离地节点的中心差分也只读取弹道节点，不再在起飞边界突然接回原非弹道root。

场景中至少有板接近地面的两侧区间采用C1 cubic Hermite衔接：**2.92～3.10 s** 和 **3.58～3.78 s**。每侧附加零端点位置/速度的平滑XYZ bump，允许优化过渡几何，而不破坏弹道端的C1连接。下文接触图同时表明机器人自身在这两段不连接板，因此这些“场景有ground近接的桥段”**不能解释为机器人已经获得合法外力**；这也是整体失败原因。

每个离地窗只优化12个参数：COM起点/初速度六维和两个桥段bump各三维。目标最小化根位移、校正速度和新root曲率；几何约束包括：

- 足底所有真实capsule的最低支撑点保持在板上半空间，使用原参考已有间隙与实际翘尾最高包络形成单向下界，容差1 mm。
- 保留原纵向±0.39 m、单脚横向±0.16 m、双脚中心横向±0.08 m的宽松包络；源已超出时不把它暗中裁回。
- 机器人完整最低包络不穿地超过2 mm，根各轴校正不超过0.5 m。
- 若出现robot/board等接触穿透，用实际contact normal对整体机器人平移的距离Jacobian补充线性约束，重新求解；不关闭碰撞。

本次白化后的凸二次目标由SLSQP两次迭代收敛，线性约束最大残差约 `4.44e-16 m`。最大root位置改变量 **0.131785 m**，最大root线速度改变量 **2.82576 m/s**。这些都是参考目标的修改，不是实际机器人已经能够产生的运动。

## 节点弹道与边界有限差分结果

输入v4在physical 3.14/3.16/3.18 s的联合COM中心差分VZ约 **2.61139 / 1.77501 / 0.89852 m/s**，0.04 s内下降1.71286 m/s，远大于重力应产生的0.3924 m/s；XY同样突变。

最终候选同三节点变为：

- VZ：**1.43981 / 1.24361 / 1.04741 m/s**，每20 ms减少0.1962 m/s。
- VX：恒 **0.0478468 m/s**。
- VY：恒 **0.263071 m/s**。

21个可靠离地50 Hz节点的二阶中心差分加速度与 `[0,0,-9.81]` 最大分量误差低于 **3.24e-12 m/s²**。这个结果覆盖整个自动识别窗口及其边界，不是只对内部子窗拟合残差小。

根线速度仍按**新root位置的中心有限差分**保存，最大分量误差小于 `6.8e-15 m/s`。没有为让某个COM指标精确而把根线速度改成不属于根轨迹的量。

早期v1/v2试验曾补偿源固定joint qd/root omega造成的瞬时Jacobian COM速度偏差，以强制 `J_com*qvel` 等于目标COM差分速度；这会让根线速度偏离自己的位置差分，air段最大约0.128 m/s、过渡段约0.558 m/s。该做法已拒绝，**v1/v2不是最终候选**。

最终v3保留了输入qd/omega的离散一致性残差：air内 `J_com*qvel` 与COM中心差分速度最大XYZ偏差约 **0.05436 / 0.12001 / 0.06779 m/s**，与源相同。不能将“50 Hz位置差分满足重力”说成“所有瞬时速度字段已经严格连续动力学一致”。

## 密集几何与插值限制

全片2057个密集样本最大穿透：self **1.313 mm**、robot/board **0.178 mm**、ground **0.108 mm**，所有类别>2 mm计数均为0。足底上半空间约束违例约 `2.57e-16 m`；主离地窗两脚最低gap约 **24.5 / 57.0 mm**，没有把脚拉到板底或焊在板上。

碰撞QA使用现有标准插值：root/joint位置线性，四元数按MuJoCo位置积分的geodesic插值。因此它在节点之间**不是精确连续弹道**。相对解析COM弹道，缓冲区密集位置最大XYZ残差约 **0.403 / 1.234 / 1.218 mm**。不能把50 Hz节点的机器精度数字外推为2.5 ms连续加速度也精确等于g。

即使这部分离散误差很小，也不能覆盖接下来更严重的实体独立动力学失败。

## 接触连通分量揭示的不可修复项

对于无地面传力路径的组件，其COM必须各自弹道，不能只检查所有实体相加后是否弹道。使用2 mm保守连接图，结果为：

- 机器人从 **2.905～3.9775 s** 无robot/board或robot/ground可能连接。板尾早期仍接地并不能对已离板的机器人施力。
- 板在 **3.1225～3.5450 s** 既不接地也不接机器人，必须单独满足板COM弹道。
- 固定板自身COM在19个完整无连接差分stencil上，加速度相对g的最大XYZ残差为 **11.74 / 14.82 / 248.05 m/s²**。这个必要条件失败无法通过移动机器人根修复。
- 该板误差的等价未解释外力峰值约 **541 N**。它只是 `mass*(a-g)` 的缺失力诊断，**没有把这种力加入仿真**。
- 机器人校正后在51个完整无连接stencil上也失败，最大XYZ加速度残差约 **27.34 / 78.66 / 60.81 m/s²**，等价未解释外力峰值约 **2685 N**。

联合COM原先需要的等价缺失力峰值约3534 N，校正后可下降到数值零，但此时机器人被数学上移动去抵消固定板的非弹道运动，**没有真实接触路径传递这部分动量**。不能将总和正确误当作各实体正确。

组件必要条件诊断使用0.2 m/s²的显式工程加速度容差，只统计整个±20 ms stencil都无可能连接的节点，避免将真实接触/冲击跨窗差分误判为自由飞行外力。这里的残差远大于该容差。存在近接连边也仍不证明接触法向力、摩擦或受力可行。

因此在“板全state、关节轨迹和0～2.90前缀全固定”的范围内，本次只能给出联合COM节点校正候选，不能给出全部组件必要动力学一致的参考。下一步需要单独重审板起飞/落地边界与单目板轨迹的共同拟合，以及机器人脚真正离板/重新接板的时序；不能让控制训练去追一个依赖不存在接触力的目标。

## 验证与复现

最终候选已用新建的原生模型逐节点独立计算全部body质量加权COM，复核弹道差分；另使用原有 `Geometry.sample` 独立再查全片2057个密集碰撞样本，结果一致。逐bit保留检查、根位置/线速度关系和独立QA记录在：

`logs/ollie/ballistic_tail_upright_candidate_v3/independent_validation.json`。

同目录另有完整 `report.json`、修复前后密集 `before_samples.npz/after_samples.npz`，包含组件COM、Jacobian COM速度、脚底间隙、真实ground最低包络与接触图。脚本已通过Ruff、编译和实际CPU运行。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python scripts/video/repair_ollie_ballistic_reference.py \
  --source data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz \
  --preserve-through 2.90 --contact-band-m 0.002 --blend-seconds 0.20 \
  --out data/ollie/new_ballistic_candidate/joint_reference_50hz.npz \
  --log-dir logs/ollie/new_ballistic_candidate
```

输出和日志路径必须是新路径，不能覆盖源或已有分支。脚本执行成功表示候选与诊断文件写出，不表示 `recommended_as_physical_reference` 通过。本次没有进行策略评测、训练、教师迁移或物理成功声明。
