# GPU 显式足板模型与 SuperMimic 观测准备

2026-10-25。**本轮补齐GPU物理模型入口和真实几何净空，完成64/256环境固定控制重放；没有训练PPO，没有新的Ollie成功或合格教师。**65完整回合仍FAILED。

## 为什么需要这一步

65的实际受力诊断表明前脚持续压板，后脚踩尾与系统上升未同步；完整软目标还允许用“不起跳”换取较少跌倒。继续克隆该失败控制不会得到Ollie专家。

[TBS核验66](66_开源滑板控制实现核验.md)记录了另一项工作使用PPO参考模仿、准备课程和MuJoCo复验，但没有取得公开Ollie模型/动作/权重。我曾据此提出PPO教师分支，用户随即明确要求保持SuperMimic；**该分支已取消，从未启动PPO训练，也未创建或提交PPO实现/权重**。不能因为另一项工作用了PPO就替换本项目算法。

后续保持真实物理数据→人板联合世界模型→模型内BPTT更新策略。GPU接触测量用于数据标签、模型接触响应和梯度真实性核验，与PPO无关。开源滑板demo只借鉴建模和人板交互处理。

检查现有GPU采集器又发现：CPU准备模型已显式使用足板μ=0.9，`JointPhysicsEnv`仍只能编译默认μ=0.6模型。该不一致必须在新训练前解决。

## 代码改动

- `JointEnvConfig.foot_deck_friction`默认`None`，保持旧训练/评测的原模型行为；显式0.9透传同一`compile_scene`，产生42组足板pair。原PD、被动轮和其他物理参数不变。
- `diagnostics()`新增9个真实碰撞几何的最低点：5个box使用真实旋转矩阵和半尺寸，4个圆柱使用真实轴向、半宽和半径。返回`whole_board_geometry_clearance(N,9)`、`minimum_whole_board_clearance(N)`和`minimum_wheel_geometry_clearance(N)`。
- 旧`minimum_wheel_clearance`保持原值，并注明它是轮心减半径的legacy代理。新字段避免悄悄改变历史结果，也不能仅凭几何高度认证无接触腾空。
- 新增`scripts/local/benchmark_ollie_gpu_contact_profile.py`，仅重放固定准备控制，记录实际GPU接触材料、容量溢出、边界差异、几何公式和速度；不产生奖励、done或学习更新。

## 实际验证

静态检查通过。CPU原生128组随机板姿态及轮角中，box公式与8顶点、圆柱公式与支撑点的最大误差均2.22e−16 m；只读diagnostics调用未改变状态/原legacy值。默认模型0显式pair、新模型42显式pair。

两次GPU运行都从64的同一初始92维状态开始，输入原始固定控制，不搜索动作、不改材质。CPU对照使用同一可表示的float32初始位置/速度和相同float32控制，以便报告后端差异；它不是把原double实际轨迹声称为GPU逐值重放。源准备归档SHA256为`c0263d1fbe71e235a04ec05b054e887b9354fc21ce645c52c0ea4db83ad15cc4`。

1. `logs/ollie_gpu_mu09_benchmark_01/`：64环境，25步/0.5 s。42pair上传最大误差2.384e−8，实际GPU足板接触读到μ≈0.9、solref≈[0.02,1]；所有边界有限，50 Hz观测未见overflow。CPU/GPU根与板位置最大差0.0210 mm，机器人关节角0.000223 rad；9几何最小净空公式与同一GPU姿态的native FK差≤1.144e−8 m。纯GPU step约0.149 s，约10751环境控制步/s。
2. `logs/ollie_gpu_mu09_benchmark_02/`：256环境，130步/2.60 s。实际材料一致，所有边界有限、50 Hz无overflow，最小root upright≈0.907998。CPU/GPU根与板位置最大差30.206 mm，关节角0.048847 rad、关节速度0.879303 rad/s；重复GPU世界的92维最大差1.099（混合单位，不能称1.099 m），表明接触和float32后端误差会随开环积累。根相对板位置最大差30.390 mm。不能把CPU教师的开环控制直接当GPU精确成功。

第二次9几何公式与同一GPU姿态native FK差≤1.085e−8 m。纯GPU step约0.862 s，约38601环境控制步/s，初始化另约1.811 s；该吞吐排除了CPU复验、诊断、策略、reward和优化，不能直接换算完整PPO训练时长。Torch allocator仅记录很小的张量分配，**不含Warp自身显存，不能拿该值声称总显存只有KB/MB**。

GPU diagnostics当前只在50 Hz控制边界采样。CPU对照另有400 Hz原Observer，但不能用CPU记录认证GPU中间子步；尚未核验GPU400 Hz全部接触力/上表面支持/穿透和事件计数。因此本轮不晋升准备教师，也不启动以轮高度代理奖励的PPO。

源码冻结SHA256：`gpu_env.py`为`238404ede27394c80687c312873144eb6c974938e92cbd0d3507bdda93528343`；benchmark为`68967ecb55a81c126d1787336d59c51ddb0142b2a4cbbdd254199955c344be74`。每个运行目录保存manifest、边界数组、50 Hz GPU诊断/容量和400 Hz CPU接触记录，日志均不上传公开仓库。

复跑示例（新目录）：

```bash
env -u PYTHONPATH -u PYTHONHOME OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked python scripts/local/benchmark_ollie_gpu_contact_profile.py \
  --out-dir logs/ollie_gpu_mu09_benchmark_new --num-envs 256 --controls 130
```

## SuperMimic 训练前仍需完成

先把GPU真实400 Hz足板/尾地力、9几何净空、人体触地和上表面支持做成可核验的实际数据，再用于联合世界模型的辅助监督和动作响应检查。必须在留出的卸载/触尾样本上确认下一状态、接触变化和板响应方向；预测接触力的辅助head也不能未经真实动作干预验证就用作策略梯度。后续仍由SuperMimic模型内BPTT更新原策略，不接PPO runner/critic。

训练可以使用明确标记的合法初态课程，但最终验收始终从唯一原始初态、原29PD控制连续到5.14 s，无root/board回写、外力和phase reset。实际native严格v3保持不变。当前尚无PPO checkpoint，原一小时LAFAN1训练仍归档停止。
