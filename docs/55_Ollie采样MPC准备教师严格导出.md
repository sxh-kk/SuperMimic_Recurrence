# Ollie 采样MPC准备教师严格导出

## 已完成的结果与适用范围

文档53的完整准备轨迹通过重新创建的原生MuJoCo全回合重放和400 Hz质量验收，已**另行导出准备段教师**。原MPC日志的`verified_teacher=False`保持不变；新导出文件才设置`verified_teacher=True`，含义严格限定为：**单一初态、0～2.76 s连续准备控制已通过物理门槛**。

这不是稳态、扰动鲁棒性、起跳连接、完整Ollie或SuperMimic训练成功的认证。末机器人COM仍以0.2820 m/s上升、根世界角速度范数约0.7125 rad/s、板VX约−0.14545 m/s，不能将其当作可随意重置的静止初态。末速和姿态余量见文档53。

本轮**没有启动BC、DAgger、世界模型训练或飞行控制**。新参考54可能修改2.80～2.96 s的future goals；本文件仍明确绑定v4，不能静默换为54后使用旧hash。

## 文件与强绑定

导出入口为 `scripts/local/export_ollie_mpc_teacher.py`。有效文件：

- [准备教师](/home/zxc/SuperMimic/logs/ollie_mpc_teacher_v4_02/teacher.npz)，SHA-256 `274963f820973826cb993b9f239f979769e1d06d5a2e8e9b4ff6f3ae15d645e5`。
- [导出验收记录](/home/zxc/SuperMimic/logs/ollie_mpc_teacher_v4_02/result.json)。初次`v4_01`与最终原子发布版本`v4_02`的teacher文件SHA完全相同。
- [字段与拒绝检查](/home/zxc/SuperMimic/logs/ollie_mpc_teacher_export_audit_01/result.json)。

2026-10-25 修正力矩采样阶段后仅重导到`logs/ollie_mpc_teacher_v4_execution_tau_01/`，新teacher SHA256 `125d24e54bf5b6865fdc31871d795f1c230b772f300cf4ecf6f75c33c44e65a0`。新旧teacher除`metadata_json`外所有数组逐值相同，包含动作、状态和terminal integration；19项gate逐值相同且全通过。metadata将峰cap比例更正为1.0，并记录1104个执行阶段样本。旧文件保留，历史BC/DAgger仍绑定原SHA，没有重训或静默替换教师。

入口固定校验三个已明确接受的SHA：

- v4参考：`38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7`。
- 实际MPC轨迹：`559fdfb376120a1529bbe5e3de8f0537103c7f4001a4ed3934e5be7152b9b96e`。
- v4静力前馈：`08b18eb03c59cc30379dd328053298961e3dcd970693a775b856fd3bc34f5115`。

同时检查原输入的139×92状态、138×29实际目标、139×47 qpos、139×45 qvel、时间网格、关节顺序、MuJoCo版本、步长、唯一初始化、无板驱动/外加力等元数据。输出目录必须不存在。哈希或输入错误在导出前拒绝；实际质量gate失败时只写拒绝记录，不产生正式`teacher.npz`。发布先写临时文件，再确认原三个日志文件hash未改变，最后原子改名；不会改写原日志认证标记。

## 原生重放与19项验收

从绑定参考的唯一初态（qvel=0）开始，在原物理模型中仅执行保存的138个29维目标，保持原力矩cap、PD、真实关节范围、2.5 ms物理步长和20 ms控制步长。共1104物理子步，加初态为1105次接触检查，无中途重置。每个20 ms边界都核对原qpos/qvel/state92；每个2.5 ms子步核对真实ContactObserver原字段。

19项gate全部通过：完整时长、原生精确重放、全部接触原字段相同、无外加力、真实目标限位、真实tau cap、upright、无身体触地、穿透、实际软关节越界、初始穿透、足高、足平面误差、root-board XY、root高度RMSE/最大误差、两足全段向上支持、两足末300 ms向上支持、真实唯一4轮接地。

实际关键结果：

- qpos/qvel/state92最大重放误差均 **0**；1105条接触原字段逐值相等。
- 最小upright **0.92158**，无身体触地；最大穿透 **4.220 mm**、实际软关节越界 **0.01207 rad**。
- 独立FK足高度/XY最大误差 **3.063/11.843 mm**；root高度RMSE **6.239 mm**。
- 左右足支持 **99.638%/99.729%**，末300 ms均100%；唯一4轮接地 **99.005%**。
- 实际目标、实际tau均未超原限制。旧记录 **83.96%** 属于额外forward后诊断值；2026-10-25 独立执行阶段复核发现准备段左腕pitch曾到100%，腿峰35.59%。导出器已改为每次`mj_step`后、诊断forward前读取1104个实际力矩，19项门槛保持原值。旧教师文件与历史实验不改写，不据此重新训练。没有将“目标合法”替代“实际关节角合法”。

所有门槛与文档48/52/53相同，没有为导出放松质量。`verified_teacher=True`仅覆盖上述物理准备前缀；元数据明确`steady_state=False`、`robustness_verified=False`、`ollie_success_certified=False`。

## 教师张量与时间语义

文件保留实际物理数值精度，`state`、`next_state`、边界状态和实际action为float64；future goals为现有actor接口要求的float32。

- `state[138,92]=actual_boundary[:-1]`，`next_state[138,92]=actual_boundary[1:]`。
- `all_boundary_states[139,92]`完整保留0～2.76 s；`action[138,29]`是实际执行的绝对PD目标，并非CEM均值、名义曲线或残差。
- `time_s=0.02*arange(138)`对应每个动作的执行开始，`time_video_s`沿用v4的1.5时标。终态另存`terminal_state`。
- `reference_offsets=[0,5,10]`；`reference_goal_indices=min(arange(1,139)[:,None]+offsets,257)`。首目标是下一帧，最大goal index148（2.96 s），不是把当前帧误当下一目标。
- `future_goals=reference.state[indices].astype(float32)`，与绑定v4参考逐值相等；`joint_names`、`board_joint_names`严格沿用原顺序。
- 另存原始`qpos/qvel`、`terminal_integration_state`及其MuJoCo state spec，便于后续热状态审计。继续执行仍须使用匹配的原模型并验证接触/控制历史，不能只冷置92维状态冒充相同热状态。

字段roundtrip检查全部通过：state/next_state/边界/action与原实际轨迹逐值相同，goal indices和float32 future goals逐值相同。`time_s`与乘法网格逐值相同；与既有入口采用的`arange/50`最大浮点差4.44e-16 s，通过原1e-8 s时间校验。

元数据含`teacher_kind=native_sampling_mpc`、`scope=single_initial_state_preparation_prefix_only`、全部gates、输入文件SHA、`initializations=1`、`intermediate_resets=0`、`forces_applied=False`、`board_actuators=0`以及有效物理区间。还明确标记`future_reference_outside_preparation_verified=False`：2.76 s之后的goal作为接口数据存在，不代表该段已经完成动力学验收。

## DAgger保护与拒绝测试

现有 `warm_start_ollie_policy.py` 的DAgger专家仍是静力控制器，不能给原生采样MPC教师重新标注动作。仅新增一项窄保护：若教师元数据`teacher_kind`为`native_sampling_mpc`且`--dagger-rounds>0`，立即拒绝。旧教师不含这个kind，默认行为不变；`dagger-rounds=0`不受此保护影响，但本轮没有执行训练。

实际CLI负例已确认：使用真实v4教师、真实残差配置、`--dagger-rounds 1 --device cpu`，在创建训练输出目录和训练之前抛出明确错误。没有用静力专家冒充MPC专家。

另对复制输入追加字节进行哈希负例：更改参考或实际轨迹均拒绝，且没有创建输出目录。未修改原输入。Ruff及diff空白检查通过。

## 重跑严格导出

```bash
env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked python scripts/local/export_ollie_mpc_teacher.py \
  --out-dir logs/ollie_mpc_teacher_v4_new
```

该入口只接受本轮明确绑定的数据。若决定采用54参考，应先显式审查准备前缀一致性、future goals差异和hash绑定，再另行导出，不能直接重命名旧教师或修改其元数据。
