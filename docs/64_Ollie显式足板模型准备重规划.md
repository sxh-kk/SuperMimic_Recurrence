# Ollie 显式足板模型准备重规划

2026-10-25。模型审计61发现脚priority覆盖板参数后，本轮固定采用**显式足板滑动摩擦μ=0.9**的工程假设。它恢复板XML原先声明的足板摩擦意图，但没有原视频实物测量，不能称为材料标定值。尾地、轮地、足地、接触柔顺性、几何/质量/惯量和原29个机器人电机均不变。

## 修改与兼容

`track_ollie_sampling_mpc.py`的`NativeSamplingMPC`增加可选`foot_deck_friction`关键字及CLI入口。主模型与添加只读传感器的预测模型都从同一个profile构造；proof增加42个显式pair的几何索引、condim、摩擦、solref/solimp及margin/gap逐值核对。默认None仍是原μ=0.6模型。

未把旧模型的2.60 s末态冷置到新模型。新控制从原始视频对应初始姿态、零速度开始，逐步用新物理重新规划。仍使用53相同的128候选、H32=0.64 s、3轮CEM、top8、5结点15腿腰维、σ=0.06 rad、seed53、16CPU线程；正常重力、50 Hz目标、每目标8×2.5 ms原生物理。只作用原29 PD目标，没有板驱动、外力、根/板回写或阶段重置。

参考保持`tail_contact_upright_v4` SHA256 `38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7`；静力前馈保持`08b18eb03c59cc30379dd328053298961e3dcd970693a775b856fd3bc34f5115`，仅用于名义目标，不直接施加接触力。

## 固定配置的0.5 s与2.60 s门控

先执行0.50 s，全部准备门槛通过后，才在相同配置/seed下从头执行2.60 s。两回合前0.50 s的state/qpos/qvel/action逐值相同，没有为长回合换参数。旧默认模型另做0.10 s回归，与53的原`smoke_02`四个数值数组逐值相同。

新完整准备目录`logs/ollie_mpc_explicit_mu09_prep_01/`：

- 130个真实目标、131个50 Hz边界状态、1041条400 Hz接触记录，完成2.60 s。
- 全部原准备质量门槛通过；无机器人触地，最小upright **0.90820**。
- 最大穿透 **3.178 mm**，最大实际软关节越界 **0.00859 rad**；原门槛不改。
- 脚高度/平面误差最大 **2.208/11.063 mm**；root高度RMSE **6.487 mm**。
- 左/右足实际板上表面支持均 **99.808%**，末300 ms均100%；真实四轮接地 **99.328%**。
- 板XY最大位移 **21.673 mm**，最大XY速度0.1674 m/s；不是只验证人板相对关系。
- 控制计算 **71.55 s**；每次实际首目标的8子步末状态与所选预测误差均0。

实际轨迹`actual_rollout.npz` SHA256为`c0263d1fbe71e235a04ec05b054e887b9354fc21ce645c52c0ea4db83ad15cc4`。结果明确`verified_teacher=False`、`ollie_success_certified=False`、`material_calibrated_from_real_board=False`。它只是新固定模型下的准备前缀，不是完整Ollie或SuperMimic结果。

## 独立重放与审计入口

另用未加预测传感器的新profile原模型重放130个保存目标：qpos/qvel最大误差均 **0**，1041条原Observer字段逐值相同；实际执行力矩在每个`mj_step`后、额外诊断forward前读取，未超过原cap。审计在`independent_audit_01/`，回归一致性在`code_regression.json`。

共享的`audit_ollie_saved_rollout.py`新增`--foot-deck-friction`及JSON源接触对照，并修正完整参考时长口径：不足默认5.14 s的准备/阶段片段，`full_reference_episode=False`，不会因保存的短回合已重放完就冒称完整原视频。原5.14 s完整回合口径不变。新增力矩记录不反馈到主状态。

```bash
env -u PYTHONPATH -u PYTHONHOME OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked python scripts/local/track_ollie_sampling_mpc.py \
  --seconds 2.6 --foot-deck-friction .9 \
  --out-dir logs/ollie_mpc_explicit_mu09_prep_new

env -u PYTHONPATH -u PYTHONHOME OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked python scripts/local/audit_ollie_saved_rollout.py \
  --rollout logs/ollie_mpc_explicit_mu09_prep_01/actual_rollout.npz \
  --out-dir logs/ollie_mpc_explicit_mu09_prep_01/audit_new \
  --control-dt .02 --control-order g1 --initial-control initial-pose \
  --foot-deck-friction .9 \
  --source-contacts logs/ollie_mpc_explicit_mu09_prep_01/contacts.json
```

Ruff、语法编译、默认0.10 s回归、新0.50 s前缀一致性、新2.60 s独立物理重放均通过。下一轮只能从该真实控制前缀连续接到起跳，不把62的静态IK或63的夹具组件当作实际机器人轨迹。
