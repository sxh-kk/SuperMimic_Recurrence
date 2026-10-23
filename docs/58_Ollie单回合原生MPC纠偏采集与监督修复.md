# Ollie 单回合原生MPC纠偏采集与监督修复

## 结果

**唯一混合采集回合通过完整准备门槛，但固定1000更新后的4个策略全部失败，且比文档57退化。该训练分支已停止。** 没有继续采集、追加训练、调参、放宽门槛或晋升failed策略。

本轮是原生MPC专家在实际访问状态上的单回合DAgger式标签采集与监督拟合，不是旧静力DAgger路径，也不是WM/BPTT或Ollie成功。独立入口为`scripts/local/dagger_ollie_sampling_mpc.py`，文档53/55/57及旧DAgger拒绝保护均未修改。

## 固定输入与唯一配置

- v4参考SHA：`38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7`。
- 原教师v4_02 SHA：`274963f820973826cb993b9f239f979769e1d06d5a2e8e9b4ff6f3ae15d645e5`，原文件保持不变。
- 起始策略为57的`bc_03000.pt`，SHA `734c4230e69a7feaa9b0192104709ea855ae2688259dc0cd82497260e7b2fcb1`。保留原残差范围、网络及head契约，没有扩范围或重新初始化网络。
- 专家完全复用53原生MPC：128候选、H32、3轮、5结点/腿腰15维、top8、初始σ0.06 rad。seed58；Gaussian噪声使用独立RNG对象、同一固定seed58。
- 只有一条连续2.76 s混合轨迹：`actual_target=clip(0.75*expert_label+0.25*clipped_BC_target+noise)`，噪声σ0.002 rad。BC目标先按原生eval方式截到真实目标范围内侧5 mrad；混合后再次截到同一范围。
- 主轨迹仅一次初始化，每20 ms执行29个机器人目标、8个原2.5 ms物理子步，没有根/板回写、阶段重置、板驱动或外加力。

只有混合轨迹通过全部原400 Hz准备gates、完整独立重放、专家标签在现有残差范围内，才进入训练。失败即停止，不重跑采集。

## 实际动作、专家标签与反事实分支

每个控制周期，MPC从**当前实际混合轨迹的热状态**真正模拟候选，选出专家动作。另复制完整MjData、仅执行该专家目标8个物理子步，与其获选预测的首20 ms末状态核对；最大误差0。这是反事实分支，不能当作实际混合轨迹的下一状态。

数据保存于`logs/ollie_mpc_dagger_collect_01/mixed_transitions.npz`，SHA：

`418d4f025b254e6d06e71a67ec3999ab69b8112a601c47e552afa194e83f2866`

其中分别保存：

- 实际`state[138,92]`、`actual_action[138,29]`、`next_state[138,92]`和139个边界状态。
- `expert_label_action[138,29]`及`expert_counterfactual_next_state[138,92]`。
- BC原始/截断目标、实际Gaussian噪声、严格下一目标indices/future goals、原qpos/qvel、关节顺序和输入hash。

独立重放实际前缀，同时在每个真实热状态复制分支执行保存的专家标签：实际qpos/qvel最大误差0，反事实专家state92最大误差0。**138个actual next state全部与counterfactual next state不同**，两种转移没有混写。实际动作按混合公式重新计算误差0。

实际action相对expert label的RMSE0.01805 rad；BC截断目标相对expert label为0.07171 rad；实际噪声经验RMS0.001990 rad。数据的`verified_teacher=False`始终保留：混合控制通过准备门槛，不等于整串反事实标签已经组成一条已验证示范。审计为`independent_label_audit.json`。

## 唯一混合轨迹验收

采集CPU计算 **80.978 s**，未超过300 s预算，结束后释放16线程MPC资源。独立重放通过全部原门槛：

- 最小upright **0.92503**，无身体触地。
- 最大穿透 **4.366 mm**，实际软关节越界 **0.01265 rad**。
- 足高度/平面最大误差 **2.511/13.418 mm**；root-board XY最大0.14313 m。
- root高度RMSE **10.559 mm**、最大61.961 mm。
- 双脚向上支持均 **99.729%**，末300 ms均100%；唯一4轮接地 **99.186%**。
- 实际qpos/qvel/state92重放误差0；目标、实际tau和专家标签残差范围均通过。

没有因为混合采集成功就认证其后的策略或完整跳跃。

## 相对原教师的同时间偏移

仅使用已保存数组，比较0～2.74 s的138个动作开始状态，不作对齐、重拟合或新MPC探测：

- 机器人根位置XYZ分量RMSE **7.078 mm**，位置向量最大差33.869 mm。
- 板根位置XYZ分量RMSE **1.844 mm**，向量最大差11.614 mm。
- 机器人关节位置RMSE **0.03645 rad**；根旋转测地误差RMSE **0.03127 rad**。
- 根世界线速度分量RMSE **0.03640 m/s**，世界角速度分量RMSE **0.23143 rad/s**；关节速度RMSE **0.25988 rad/s**。
- 新expert label相对原教师**同一时间action**的RMSE **0.06702 rad**，最大绝对差0.44855 rad。

详见`same_time_teacher_comparison.json`。这些同时变化的量**不能分离CEM随机种子、实际状态反馈和未观测的计划热启动历史的因果贡献**，也不证明专家动作仅由当前网络输入唯一决定。

## 固定监督修复与4个候选

原138条教师样本与新138条实际状态/专家标签等权合并为276条。始终以专家标签为监督目标，实际混合动作只用于真实转移记录。由57的3000更新权重继续，fresh Adam、seed58、**固定学习率1e-4，无cosine**，仅追加1000更新，每250更新原生2.76 s评估及独立重放。训练/4次验收计算共 **8.648 s**，低于GPU60 s预算。

所有候选的独立保存控制重放qpos/qvel/state92误差均0。实际结果：

**追加250更新**：合并动作RMSE0.01613 rad，原prefix cost26.72423；最小upright−0.18335，身体2.5775 s触地，穿透12.854 mm、足平面误差252.6 mm。准备FAILED。

**追加500更新**：合并动作RMSE0.01023 rad，cost31.61982；最小upright−0.44746，身体2.3675 s触地，穿透19.151 mm、实际软关节越界0.12966 rad。准备FAILED。

**追加750更新**：合并动作RMSE0.00801 rad，cost27.28081；最小upright−0.08661，身体2.4175 s触地，穿透20.933 mm、实际软关节越界0.06551 rad。准备FAILED。

**追加1000更新**：合并动作RMSE0.00678 rad，cost28.36976；最小upright−0.18811，身体2.4125 s触地，穿透20.534 mm、实际软关节越界0.06366 rad。准备FAILED。

所有候选还存在足板、root或末段支持门槛失败，完整列表保留在`logs/ollie_mpc_dagger_fit_01/native_XXXXX/independent_preparation_audit.json`。通过集合为空，没有passing checkpoint；只按原cost选250更新作为FAILED完整诊断对象，未称作修复成功。

四个新checkpoint SHA依次为：

- 250：`ae0f37d4a1d96dd9d8e0dfdf2dc4db14a175456885567821e6b3fa03ff14c033`。
- 500：`c5b341c85108ccb4d076c9fe18751d0a6ee5cb04423c37d54a398faf096c2289`。
- 750：`b59b451c44232e2f54acb59f4d150a15d00093aaacdd56677502c557e5cdb6c5`。
- 1000：`844e5b4bf0d0b028e556f06cca88025b9e80606b0c75a2965199b67f8866b797`。

## 原/新样本组拟合误差

对已冻结权重只读重新计算两个固定样本组的动作RMSE，结果单位均为rad：

- 修复前：原教师 **0.001145**，新混合状态expert labels **0.071758**。
- 追加250：原教师 **0.014171**，新样本 **0.017881**。
- 追加500：原教师 **0.008240**，新样本 **0.011891**。
- 追加750：原教师 **0.006223**，新样本 **0.009467**。
- 追加1000：原教师 **0.005185**，新样本 **0.008071**。

新样本拟合改善，但原教师拟合变差，闭环准备也退化。该固定试验没有证明继续增加更新或改变样本权重就能解决问题，因此不追加搜索。只读结果保存在`independent_fixed_group_fit.json`。

## 一次完整诊断与停止

250更新FAILED候选完整原生执行5.14 s，仍FAILED：身体2.5775 s触地、最小upright−0.48507、穿透12.854 mm；没有有效tail窗口或整板腾空窗口。前2.76 s的state/qpos/qvel/control与其准备评估逐值相同。结果位于`logs/ollie_mpc_dagger_fit_01/full_diagnostic/`，没有为此再生成候选。

WM和BPTT更新数均0，真实动力学参数没有改变。此分支已停止，所有输入、混合数据与failed权重保留。

## 分阶段重跑入口

```bash
timeout 300s env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked python scripts/local/dagger_ollie_sampling_mpc.py collect \
  --out-dir logs/ollie_mpc_dagger_collect_new
```

只有脚本确认完整混合gate与残差范围都通过，才允许第二阶段；固定失败记录不触发自动重采集：

```bash
timeout 60s env -u PYTHONPATH -u PYTHONHOME OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked python scripts/local/dagger_ollie_sampling_mpc.py fit \
  --collection logs/ollie_mpc_dagger_collect_new \
  --out-dir logs/ollie_mpc_dagger_fit_new
```

Ruff和diff空白检查通过。这里给出的是固定试验入口，不代表本轮失败后仍会继续运行。
