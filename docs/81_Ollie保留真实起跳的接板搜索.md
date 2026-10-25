# Ollie保留真实起跳的接板搜索

2026-10-25。80完整64轮、95.124 s仍FAILED：虽然首+.60 s双足走廊满足比例约67.6%、机器人首次接地推迟到3.6775 s，真实整板净空峰仅.4944 mm，没有合格腾空。首次触地为左手，末300 ms左脚无上表面支持，不能称接板改善成功。严格v3原样保留。80完整257控制/258状态/2057 Observer及856条执行力已独立重放逐值一致；没有把预测动作当真实执行。

## 为什么继续修改选择规则

80把不可回收飞行降到与仅尾撞相同的1级。64轮共有69个raw2候选（其中64个是每轮保留的旧76），但满足完整足走廊的2级候选为0。于是连续代价较低的“不跳、减少侧移”动作胜出。它能证实目标间的取舍，却不是向完整Ollie的成功推进；旧76仍是实际起跳基线，没有被80取代。

81仍使用80的完整足走廊、有限盒接板间隙、晚启用接板、终端稳定等同一组连续代价，做以下固定修改，在任何本轮动态控制前写明：

- nominal直接是旧76原107个已真实执行目标，前150个控制仍是旧60相同真实前缀。**残差0就逐值重放已验证的真实65 ms低跳**，不再用13节点拟合原动作当nominal。
- 选择等级恢复原76真实物理事件等级；走廊可回收等级另存`recoverable_selection_grade80`和`recoverable_qualified_flight`，不再淘汰低跳。最终成功仍看原严格完整v3。
- 每轮更新均值和σ只用当前最高物理事件等级最多16个精英，不混入较低等级来补足16人。若只有1个，样本σ为0，按既定floor截到.025 rad；这会收缩探索，明确记录而不隐藏。
- seed81、初始σ=.10 rad、floor.025。256候选、64轮、13节点、15腿腰残差、P150/H107、300 s预算和16原生候选线程不变；小矩阵库单线程。原ABC和旧60接站姿仍作为单独已评估候选保留。

这是一次有明确失败依据的固定搜索规则修正，不是把80改标签变成功；80数据、视频与FAILED结果保留。仍无学习策略、板动作、外力、根/板回写、phase reset或物理参数变化。

## 公共协议兼容性

76的`run_experiment`增加可选`elite_selector(order, grades, costs, config)`，默认仍是原`order[:config.elites]`；仅新增返回非空、唯一、来自已评估有限候选且不超过16项的检查。76/80不传回调，旧默认路径保持。80实际运行的旧协议源码已按原SHA `b305c0b58c7069836a66f5c0faa6c207b6ce04f9634439fb430e6a6159e4118b`精确保存在`logs/ollie_recoverable_catch_01/as_run_protocol_80.py`，不将后来的接口增补冒充其运行源码。

## 验证和实际结果

无积分核验`logs/ollie_hop_preserving_static_01/result.json`通过：nominal/首prior逐值等于旧76全部107目标，首150原动作/原初态一致，拟合mean为全0，零残差重建和限位/主状态不变通过。最高grade池1/2/20条等4例不混级，非法grade和非有限成本过滤正确；有效/无效终点/后NaN原80等级[1,−1,−1]在81成为[2,−1,−1]，代价/terms/已有metrics均相同。入口SHA `eb91c88fa10b13f54eaa3919ae0964fcd44a53ef5fe7e59a3f9c154ffb67b80b`，新协议SHA `c786bb7f07dd0208339b73a32c13fbed50504bce8b303133a77031b517965fa5`。

原生预检`logs/ollie_hop_preserving_benchmark_01/`通过：首批rollout.756/cost.745 s、投影64轮96.09 s；856步fullphysics差0、原生执行力误差≤2.85e−14 N、真实尾Fn和足TOUCH差0。首批胜出40号已恢复raw2真实飞行，整板净空峰17.816 mm，但走廊仍未通过；原始事件与可回收代理分别记录。原source150与旧20控制回归逐值误差0。预算预检没有更新分布或执行主后缀。

先做无积分的nominal精确性、最高等级精英、有效/无效候选检查，再做原生完整batch预算和执行一致性预检，最后执行唯一完整已评估胜出后缀。命令如下，输出目录必须不存在：

```bash
env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --locked python scripts/local/optimize_ollie_hop_preserving_catch.py \
  --mode run --out-dir logs/ollie_hop_preserving_catch_01
```

## 正式完整结果

`logs/ollie_hop_preserving_catch_01/`唯一正式64轮完成，97.497 s；winner61/224、原raw2、cost5801.071069，原可回收代理为0。实际单回合257控制至5.14 s，258状态、2057 Observer、856条后缀执行力、actual_prediction_max_error=0、partial=null。**严格v3仍FAILED，仅剩无机器人触地与末300 ms稳定站板两项失败**。没有训练策略或晋升教师。

真实尾地窗3.2350 s单样本2.5 ms，执行尾Fn151.27 N；此时两脚真实normal均0。连续真实整板腾空3.2575～3.3250 s共**70 ms**，9几何全板最低点峰**22.7632 mm**，不是用尾碰时长、四轮高度或COM替代。原76为65 ms/23.4328 mm，不能说这次跳得更高。板未倒置、最大穿透9.49483 mm。首次人体触地为**右手 @3.705 s**（原76右脚3.385 s），说明已延缓倒地，仍未落稳。

末300 ms左脚真实上表面>1 N支持100%、右脚65.83%，四轮100%，机器人已倒地。普通足板接触左100%/右95.83%不能替代真实上表面支持，更不能替代站稳。完整实际关节最大软越界仍.02904 rad（含旧前缀）；不因后缀代价显示0而删除原完整范围。

`independent_replay_01/result.json`和`independent_force_audit.json`确认qpos/qvel/state92、2057条Observer、856条执行力及全部严格v3字段逐值一致，旧60前150和原53初态不变，原12°/μ=.6/29PD/neq0/无外力。执行力矩超限0，后缀腿峰61.93%上限、腿≥95%用时0，腕仍有100%。NPZ SHA `42424ce7f2e2909dc5216b3a72ad7b0ad61d30719ccfdfe1a5e3d5d7fa6b6465`；独立执行力审计SHA `8d618715236d0ef2a86b214a048f7b032f6e8782344ce03ac5138ece7fe05656`。

`independent_execution_01/foot_geometry_comparison_76_80_81.json`用同板作者坐标、真实七胶囊对三个有限盒比较：3.20 s左右足间隙从76的158.30/−.53 mm变成81的14.63/.049 mm，3.30 s从177.27/61.93 mm变成−.046/10.06 mm。81在3.2/3.3 s双足完整包络均留在走廊内，3.4 s左足略超；这是可见的足板协调进展，但没有覆盖一个完整合格16样本飞行窗，所以本winner的原代理仍0。不得把几个选定时刻的良好几何当成全窗通过。

64轮共有114次可回收飞行代理合格记录，原全局cost最优未选中它们。后续[82](82_Ollie可回收腾空候选完整验证.md)以精确RNG重建其中最低cost候选并单独完整验证，81原winner/FAILED归档不改写。

完整[原片/真实物理对照](/home/zxc/SuperMimic/logs/ollie_hop_preserving_catch_01/video_01/comparison_full.mp4)已导出并完整解码，258实际帧/187对照帧，保留倒地尾段与FAILED标签。trajectory hash不变检查见`video_01/video_audit.json`。
