# Ollie 采样MPC准备教师固定BC试验

## 实际结论

本轮只用文档55已认证的v4准备教师，完成一次固定的3000更新行为克隆试验。**六个候选的独立原生准备验收全部未通过；没有可晋升的策略。** 拟合误差很小仍不能保证闭环复现教师。原评分最低的3000更新候选仅作为FAILED诊断对象，额外完整执行5.14 s，仍没有踩尾或整板腾空，身体3.2275 s触地。

没有放宽upright≥0.85、root-board XY≤0.2 m等门槛，没有把原`best.pt`文件名当作通过认证，也没有追加训练、参数网格、静力DAgger、WM或BPTT。本轮只新增本记录，复用已提交训练和原生评估入口，未修改文档53/55对应脚本。

## 唯一配置与输入

- v4参考SHA：`38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7`；54参考仍失败，本轮没有采用。
- 教师：`logs/ollie_mpc_teacher_v4_02/teacher.npz`，SHA `274963f820973826cb993b9f239f979769e1d06d5a2e8e9b4ff6f3ae15d645e5`。
- 基础残差范围：`logs/ollie_bc_prefix_03/residual_limits.json`。复用入口自带的范围核对，**本轮没有任何关节需要扩展基础范围**；原机械限位、PD和tau cap未变。
- fresh policy、seed53，没有载入旧策略；教师均值只用于原入口已有的输出bias初始化。
- 138条准备教师数据，3000更新、每500更新保存/原生筛查，学习率入口参数0.001，沿用已提交脚本的cosine调度，DAgger轮数0。
- 网络训练使用4090，筛查使用原生CPU MuJoCo。总入口耗时 **12.715 s**（含6次CPU筛查），远低于300 s预算；GPU随后释放。checkpoint格式虽含未训练WM字段，`world_model_trained=False`，本轮没有WM优化或BPTT更新。

输出：`logs/ollie_bc_mpc_prep_v4_01/`。全部输入教师/原MPC source hash保持不变，所有failed候选保留。

## 独立准备验收方式

对6个`native_XXXXX/rollout.npz`逐一创建原模型，仅重放保存的138个实际`control`目标。每个2.5 ms子步重新forward，并按文档53同一准备门槛计算真实支持/穿透/实际角度、独立板体坐标FK足误差和root误差。每20 ms边界核对保存qpos/qvel/state92。

六次重放全部数值连续、qpos/qvel/state92最大误差均 **0**；实际目标与实际tau均没有超原限制。每个回合都是同一初态连续2.76 s，没有中途重置或外力。完整gate、误差、hash保存在各候选`independent_preparation_audit.json/.npz`，合并记录与选规位于：

[独立准备验收及选择](/home/zxc/SuperMimic/logs/ollie_bc_mpc_prep_v4_01/independent_preparation_selection.json)

原`evaluate`的`outcome=FAILED`用于完整Ollie标准，部分准备窗口天然没有完整参考或跳跃。因此本轮**没有直接用该字符串判准备成败**，而是另算全部400 Hz准备gates。

## 六个固定候选结果

以下“cost”均为未修改的原prefix screening cost；同一轮只按既有更新间隔生成这6个候选。

**500更新**：训练动作RMSE0.007384 rad，cost11.47171。独立最小upright **0.713010**，仅upright gate失败，首次低于0.85在2.5750 s；root-board XY最大0.08974 m，足平面误差13.407 mm。准备FAILED。

**1000更新**：训练动作RMSE0.003184 rad，cost11.41600。upright0.947072，但root-board XY最大 **0.327783 m**，仅root XY gate失败；足平面误差23.258 mm。准备FAILED。

**1500更新**：训练动作RMSE0.002064 rad，cost10.65328。upright0.938570，但root-board XY最大 **0.252460 m**，仅root XY gate失败；足平面误差17.166 mm。准备FAILED。

**2000更新**：训练动作RMSE0.001564 rad，cost10.43500。最小upright **0.817104**，仅upright gate失败，首次低于阈值在2.6925 s；足平面误差12.601 mm。准备FAILED。

**2500更新**：训练动作RMSE0.001296 rad，cost10.31192。最小upright **0.829453**，仅upright gate失败，首次低于阈值在2.7175 s；足平面误差12.053 mm。准备FAILED。

**3000更新**：训练动作RMSE **0.001145 rad**，cost **10.11657**。最小upright **0.8491103997<0.85**，只在最后2.76 s样本触发，但仍是失败，不能四舍五入判过。其余准备gates通过：root-board XY最大0.11782 m、足平面误差13.068 mm、足高度误差3.361 mm、穿透4.125 mm、实际软关节越界0.01216 rad；双脚向上支持99.729%/99.548%、4轮接地99.186%。准备FAILED。

## 选择规则与拟合误差的限制

规则先固定为：**只在全部独立准备gate通过的集合中选原cost最低者**。实际集合为空，故`selected_passing_update=null`。没有另存passing checkpoint或把failed标为verified。

原训练入口的`best.pt`按旧cost自动保存3000更新候选；这是排名文件，不是认证。一次完整诊断使用同权重的`bc_03000.pt`：

`734c4230e69a7feaa9b0192104709ea855ae2688259dc0cd82497260e7b2fcb1`

3000更新在**教师状态**上的动作RMSE为0.001145 rad；在实际闭环访问状态上的输出相对原教师同时间动作RMSE为 **0.008826 rad**。二者输入状态不同，不应混用。这个差异支持“低训练拟合误差不足以证明闭环成功”，但没有通过本轮实验分离状态分布变化、策略结构与接触敏感性的因果份额。

相较原MPC实际教师，BC没有保持其准备质量：教师最小upright0.92158、通过全部gate；本轮最优排名候选仅0.84911而失败。

## 一次完整5.14 s诊断

使用3000更新FAILED候选，从同一初态连续执行257个控制步，没有拼接MPC实际前缀、没有重置到参考或更换物理参数：

`logs/ollie_bc_mpc_prep_v4_01/full_diagnostic_03000/`

- 完整参考物理时长 **5.14 s**，无数值重置；独立保存控制重放qpos/qvel误差 **0**。
- 前2.76 s与原`native_03000`的state/qpos/qvel/control逐值相同。
- **FAILED**：身体首次触地 **3.2275 s**；最大穿透 **24.064 mm**、实际软关节越界 **0.06572 rad**、最小upright **−0.09979**。
- 有效tail contact窗口为空，没有≥15 mm且连续≥40 ms的整板腾空；整板最低高度峰值约0，仅浮点地面误差量级。
- 末300 ms双脚上表面支持均0，脚落地，板4轮仍接地；没有接板站稳。
- 完整ranking cost2.63448。部分回合评分有未完成参考的+10项，**不能与前缀cost10.11657直接比较宣称改善**。

这次完整评测没有掌握完整动作。没有同v4配置的新全程对照训练，因此也不把跨参考的旧倒地时间差写成可靠提升。

## 重跑固定试验

```bash
timeout 300s env -u PYTHONPATH -u PYTHONHOME OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked python scripts/local/warm_start_ollie_policy.py \
  --reference data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz \
  --teacher logs/ollie_mpc_teacher_v4_02/teacher.npz \
  --base-residual-limits logs/ollie_bc_prefix_03/residual_limits.json \
  --out-dir logs/ollie_bc_mpc_prep_v4_new \
  --updates 3000 --eval-every 500 --lr .001 --seed 53 --device cuda:0 --dagger-rounds 0
```

原脚本输出的排名结果仍需独立完整准备gate验收。不能在本轮结束后临时降低阈值，或继续训练直到找到一个通过者而省略中间失败。
