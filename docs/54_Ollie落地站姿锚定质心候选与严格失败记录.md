# Ollie 落地站姿锚定质心候选与严格失败记录

本轮只运行一个新的落地COM工程假设：把已验证蹲姿相对滑板的整体质心刚体搬到落地板姿态，以此与起跳后COM点和重力确定抛物线，再运行50相同的接触IK。结果仍 **FAILED**：几何上合法的落地锚点没有变成连续可用轨迹，所有回退档位都未通过。没有送51进行新wrench验算，没有训练。

## 固定来源与输出

- 新脚本：`scripts/video/repair_ollie_landing_ballistic.py`。
- base为50正式失败v3：`data/ollie/air_contacts_v3/joint_reference_50hz.npz`，SHA256 `39cb87079fc94d0b40e8df65f54c37264770a65a14b3600271eb89101aeced54`。
- 蹲姿来源：`data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz`，SHA256 `38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7`。
- 唯一正式输出：`data/ollie/landing_ballistic_v1/failed_joint_reference_50hz.npz`，SHA256 `62177f3a1c842718bd2eaf824cafae3e5a320a24f434609a9e9c65b16891f560`。
- 结果：`logs/ollie/landing_ballistic_v1/landing_ballistic_result.json`，状态 `FAILED_NOT_ELIGIBLE_FOR_WRENCH_OR_TRAINING`。

输出QA与 `eligible_for_new_wrench_audit` 均为false。目录内没有可被误认为合格结果的正式 `joint_reference_50hz.npz`。生产过程容器保留为 `producer50_before_landing_metadata.npz`，只供追溯，不能用于训练。

## 落地COM从哪里来

取v4第138帧，即phys2.76s整个机器人与板的质量加权COM；转换到当时板坐标，再用base第178帧、phys3.56s的板姿态搬回世界坐标：

```text
C_local = R_board138^T (C_stance138 − p_board138)
C_landing = p_board178 + R_board178 C_local
```

没有手填落地高度。得到：

- 原蹲姿世界COM：`[−0.0610316, −0.2660495, 0.5924813]m`。
- 板局部整体COM：`[−0.0058207, −0.0525531, 0.5182140]m`。
- 搬运后落地世界COM：`[−0.2552935, −0.6889294, 0.6009692]m`。

另将完整蹲姿广义位姿刚体搬到目标板上，使用原生MuJoCo FK验证：实际COM与上述公式相差3.14e−16m，自碰/板碰均0、地面只有约3.5e−18m数值误差。该目标因此有一个明确来源的合法静态几何姿态。

但这个锚点使用蹲姿当时的全部关节，包括手臂；正式候选仍逐值保留base各时刻的手臂q/qd。锚点成立不能证明固定另一套手臂动作后的连续接触IK一定可达，更不能证明真实物理稳定。

## 新抛物线与约束

保留base第155帧、phys3.10s的COM：`[−0.2602504, −0.7882118, 1.0465725]m`。将其与上述3.56s落地COM和重力 `[0,0,−9.81]m/s²` 联立，得到初始COM速度：

```text
v0 = [0.0107760, 0.2158312, 1.2875972]m/s
C(t) = C0 + v0(t−3.10) + 0.5g(t−3.10)^2
```

目标覆盖原来相同的25节点、frame155..179、phys3.10..3.58s，终端两个采样为3.56/3.58s，没有额外扩到3.60s。该曲线的时轴和落地锚点仍是工程假设，不能描述成视频实测COM。

新脚本在独立进程内复用50的同一生产流程，只替换25个 `target_com` 节点。50脚本字节SHA256保持 `668fe9dd050da99013b6d6b05df29583403b9ac7ef1b7cbf009b8feb11731287`，未改接触目标、权重、最小二乘、完整回溯merit、12rad/s或板轨迹。

最终输出保持：

- v4准备phys≤2.76s整个92维state逐值相同，2.78s位姿也相同。
- 板全部21维state逐值相同。
- 手臂q/qd逐值相同。
- 源contact标签逐值相同；没有因为请求接触就补1。
- 25节点新COM误差最大1.89e−15m；真实关节范围违规0、最大邻帧速度12rad/s。

速度由修改后的位姿重算，未补写COM速度。节点符合新抛物线只是必要位置条件。

## 实际严格结果

原始局部IK按2.5ms插值检查2057姿态：

- 自碰最大1.312586mm。
- 机器人-板最大34.245170mm，发生在3.55s右腿连杆支架与中心板之间。
- 地面最大17.196639mm，3.56s左足穿地。
- 前脚拖板、前脚landing、右脚catch均有未达有限上/边面目标的采样。即便某些样本距离很小，也伴随严重身体穿板，不能把它们计为合法接触成功。

相同的8档回退比例 `1, .75, .5, .25, .125, .0625, .03125, 0` 全部失败。特别是0%修正也必须重新投影到**新的**COM曲线，不能沿用旧COM参考的安全性：

- 0%档机器人-板穿透46.732505mm。
- 0%档地面穿透110.632712mm。

这说明把旧腿姿态连同根简单下移，并不是安全基线。脚本没有选择0%并声称通过，也没有放宽2mm阈值。没有任何回退被接受，故 `accepted_correction_weight=null`；0只是最后测试档位。最终保存的是原始失败IK，带失败文件名和QA=false，供复核其具体问题。

生成过程中曾有一个临时容器元数据把“最后测试权重0”当成“接受权重0”。本批在正式冻结前已明确更正为null，并记录 `output_is_raw_failed_ik=True`。更正仅涉及元数据，92维数值state逐值不变；前置临时hash保留在结果JSON，正式引用以上 `62177f...`。

## 结论边界

有来源且单姿态合法的landing COM目标，仍不足以让当前固定板轨迹、固定手臂、局部滑动/回板目标、连续关节速度与全身几何同时满足。本次没有证明原片无解，也不能把COM-free触点穿透解、或单独合法的落地锚点，直接升级成完整参考。

本批没有进行时间倍率网格、权重搜索、额外候选、外力注入、仿真weld或训练。由于未通过严格2057姿态/2mm/12rad/s/全部真实有限接触检查，明确不送51进行新wrench可行性验算。

## 后续控制诊断的使用边界

根代理计划在56中把这份已知失败NPZ仅作为soft nominal，研究原生物理下最接近它的可行行为。该计划必须从53合法初态重放实际prefix后连续规划，不能phase reset，并增加真实最深穿透成本，避免追踪穿板目标。即使这种控制诊断获得局部改善，也不会改变本54参考的QA=false或使它成为eligible训练参考。本批没有运行56、自动改QA或输出成功视频。

## 机器材料与重跑

`logs/ollie/landing_ballistic_v1/` 保存：

- `landing_ballistic_result.json`：来源hash、工程假设、全部严格结果与明确不合格状态。
- `landing_anchor_and_curve.npz`：刚体搬运蹲姿与25个COM目标节点。
- `com_node_comparison.csv`：目标与实际质量加权COM逐节点对照。
- `result.json`、`solver_frames.json`、`iteration_trace.json`：复用50生产流程的详细原始记录。
- `raw_local_ik_qpos.npz`、`surface_witnesses.json`、`after_samples.npz`：实际失败位姿和几何材料，不是物理回合或合格训练数据。

只允许新输出目录：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/video/repair_ollie_landing_ballistic.py \
  --out-dir data/ollie/landing_ballistic_new \
  --diagnostics logs/ollie/landing_ballistic_new
```

脚本绑定上述两个输入hash，避免误用其他候选。已实际完整运行这一候选并核对严格失败状态、源文件不变、前缀/板/手臂/标签不变量；Ruff格式与检查通过。
