# Ollie 本轮完整实际视频与独立结果核验

日期：2026-10-23。**本轮没有成功 Ollie。** 10 轮原型训练已经结束，实际执行了 70 次 BPTT 更新；按原生 CPU 回合筛选的最佳策略仍是 iteration 0 的 BC 初始化，含 1000 次 BC 更新、0 次 BPTT 更新。第 10 轮训练后的完整回合同样失败，且身体更早触地。

本记录只导出和核验原已保存的回合，没有为视频重新模拟，也没有从失败后板的运动中剪出“成功动作”。

## 1. 可以直接查看的视频

推荐先看最佳策略的三栏完整对照。左侧为原片，中间为明确标注 `Kinematic target ONLY` 的规定运动学目标，右侧为实际物理回合，全程标注 `FAILED`。

- [最佳策略：原片／运动学目标／实际失败回合](/home/zxc/SuperMimic/logs/ollie_actor_video_timed_01/final_video_export_02/selected_best/comparison_full.mp4)
- [最佳策略：完整实际物理视频](/home/zxc/SuperMimic/logs/ollie_actor_video_timed_01/final_video_export_02/selected_best/actual/rollout.mp4)
- [第 10 轮 BPTT70：三栏完整失败对照](/home/zxc/SuperMimic/logs/ollie_actor_video_timed_01/final_video_export_02/bptt70_iteration10/comparison_full.mp4)
- [第 10 轮 BPTT70：完整实际物理视频](/home/zxc/SuperMimic/logs/ollie_actor_video_timed_01/final_video_export_02/bptt70_iteration10/actual/rollout.mp4)

单栏实际视频均为 H.264/yuv420p、800×600、50fps、258 帧、5.16s，展示物理时刻 0..5.14s 的全部已保存姿态；最后一帧正常显示 20ms。

三栏视频均为 H.264/yuv420p、1920×524、24fps、188 帧、约 7.834s。顶部明确写明 `sim x1.5 slower (unverified assumption)`：按原片时间显示，仿真画面放慢 1.5 倍只是在沿用未核实的时间假设。末端增加约 0.1s 的静止展示，保证最终原片帧、参考姿态和真实末态均可见，没有补造运动。

## 2. 本轮训练完成状态与选择依据

源目录：`logs/ollie_actor_video_timed_01/`。完整 `result.json` 包含 10 条迭代记录，最终累计 163840 条物理转移，记录总耗时约 291.64s。

- iteration 0：BC1000、BPTT0，原生筛选 cost **2.886409**。
- iteration 1..3：BPTT 尚未更新，原生 cost 相同。
- iteration 4..10：累计 BPTT 10..70；实际原生 cost 均未优于初始化，约为 3.097..3.588。
- iteration 10：BPTT70，原生 cost **3.288982**。

`best.pt` 因此仍是 iteration 0。这里的“最佳”只表示本轮工程筛选 cost 最低，不表示物理验收成功。影片不会将它标为世界模型/BPTT 改善后的动作。

结论只覆盖本轮 10 轮研究原型。它不构成完整 SuperMimic 论文复现或完整 Ollie 成功声明。

## 3. 原生实际回合结果

实际输入分别来自原有 `native_0000/` 和 `native_0010/`。每条回合只初始化一次，从地面姿态和零初速度开始，随后只输出 29 个关节绝对 PD 目标，控制周期 20ms、每步 8 个 2.5ms 物理子步。原始结果记录了无中途 reset、无 root/板姿态覆盖、无被动板执行器、无额外施力。

### 最佳：iteration 0，BC1000 / BPTT0

- 身体首次触地：物理 **3.2575s**；按显示假设约为原片 **4.8863s**。
- 完整板碰撞包络最低点的全回合最高值：**2.04984mm**。
- 没有“完整板离地超过 5mm”的时间窗，也没有合格的连续 40ms 腾空窗。
- 没有板尾真实力接触事件。
- 最大实际穿透：**14.38620mm**，超过 10mm 验收门槛。
- 实际关节角最大软限位越界：**0.096010rad**；不能把合法控制目标偷换成实际关节角始终合法。
- 末 300ms 四轮力支持比例约 **73.33%**，左右脚上表面向上支持 >1N 的比例均为 **0%**。
- 原始结果及独立核验均为 **FAILED**。

### 训练后：iteration 10，BC1000 / BPTT70

- 身体首次触地：物理 **2.7400s**；显示假设下为原片 **4.1100s**。
- 完整板最低点最高值：**0.59002mm**，没有超过 5mm 的腾空窗，没有尾部真实力接触。
- 最大实际穿透：**12.57656mm**。
- 实际关节角最大软限位越界：**0.048342rad**。
- 末 300ms 虽然四轮力支持比例为 **100%**，左右脚上表面向上支持比例仍均为 **0%**，机器人已倒地；四轮在地不能代替站板落地成功。
- 原始结果及独立核验均为 **FAILED**。

两条回合中都不存在可解释为 Ollie 的合格完整板飞行窗。镜头中板的一端翘起、机器人摔倒后推动板、或者仅轮心上升，都不能替代完整包络、接触力和身体未触地条件。

## 4. 独立核验方法

没有调用 `mj_step` 重新生成任何物理回合。导出仅使用已保存的 258 帧 50Hz qpos；原力接触判断读取原始 `contacts.csv` 的 **2057 条 400Hz 子步记录**。

已检查：

- CSV 子步时间严格间隔 2.5ms，`CSV time[::8]` 与保存的控制边界时间逐值相同。
- 保存的 qpos/qvel 重新 pack 后与 92 维 state 逐值相同。
- 4 个独立轮几何的力接触布尔值之和，与 `unique_wheels_ground` 每个子步一致；使用原门槛 **0.1N**，没有用总接触点数代替轮数。
- 完整板包含中板、抬起的尾/鼻、前后轴、4 个圆柱轮，共 **9 个实际碰撞几何**。对全部 258 个保存姿态只做 FK，独立计算盒/圆柱最低点，与原 CSV 控制边界处的完整包络数值最大误差分别约 **5.55e−17m / 6.25e−17m**。
- 40ms 时间窗从原始 400Hz 力接触与最低点重新统计，没有从视频帧数反推成功。
- 脚支持读取原已保存的“真实上表面向上法向支持”字段，排除轴、底面和侧面。渲染过程新算的力没有参与任何验收。
- 两条回合的外加广义力及外加刚体力范数最大值均为 **0**。
- 导出前后原 rollout 的文件 hash 保持不变。

每个视频子目录下都有 `provenance_and_contact_audit.json`、`boundary_geometry_check.json`、`video_qa.json` 和推荐三栏成片对应的 `video_qa_full.json`。其中保留了原始回合、接触 CSV、检查点、参考和视频的完整 hash。

## 5. 真实准备段连续续接的五组对照

另行只读核对了 `logs/ollie_quasistatic/full_continuation_video_timed_02/`。五组均从同一真实准备段连续续接，前 138 个控制步、物理 0..2.76s 与教师实际状态最大误差均为 **0**；没有在起跳阶段重置到参考姿态。

- `immediate_reference`：身体 **3.1350s** 触地，完整板最低点最高 **9.42936mm**，最大穿透 **20.89433mm**。
- `fade016`：身体 **3.1375s** 触地，完整板最高 **0.59743mm**，最大穿透 **23.40389mm**。
- `fade032`：身体 **3.1500s** 触地，完整板最高 **0.53681mm**，最大穿透 **19.24506mm**。
- `pop03`：身体 **3.1475s** 触地，完整板最高 **0.72408mm**，最大穿透 **26.20404mm**。
- `pop06`：身体 **3.1600s** 触地，完整板最高 **0.02915mm**，最大穿透 **25.87018mm**。

五组均没有板尾真实力接触，全部为 **FAILED**。

特别注意：`immediate_reference` 有物理 **3.3925..3.4325s**、共 **42.5ms** 的“完整板离地 >5mm”窗口，但它发生在身体已经于 3.135s 触地之后，峰值又不足 15mm，且没有尾部接触。该段属于失败后板的运动，不能剪成 Ollie 成果。

这些连续对照说明真实准备段可以重放，但切换到后续动作尚未形成可执行起跳。它们没有把新运动学参考本身的腾空当作真实模拟结果。

## 6. 参考与检查点来源链

本轮参考为 `data/ollie/air_relative_video_timed_v1/joint_reference_50hz.npz`，SHA256：

```text
941f748cf3b13edbcf60dd20f2db41810cee2cbd1e9b4dccba1e3b1d4b5a71f5
```

导出目录各自保存了同 hash 的 `reference_snapshot.npz`。其修复范围、仍未解决的板速度/加速度及假设见记录 32；这里没有把它重新标成物理可行。

正式准备段教师为 `teacher_migrated_v2.npz`，SHA256 `c953f231a5d0d47ce7d850f71ddf01ba64807e5b0ec78f102a50ad86cf964a8c`。本轮没有使用顶层 hash 尚未更新的旧迁移副本。

原视频 SHA256：

```text
a7288f4abbccee27f756bb620c71826ba47b7e13f1786ac901eaa9a40002ddb2
```

最佳策略存在一个需要解释的文件别名：

- `best.pt` 文件 SHA256 为 `4992caf28b5eecdd6ad0530aed46dc36408a18c4224cd2f1b01c7e49c5b68245`。
- 原 `native_0000` 实际加载的是 `initial.pt`，SHA256 为 `63f7d35ef9b48b962d9a5f8efe51770020a8503063ab17ed6e6a79bce22af239`，与原回合记录严格一致。
- `best.pt` 在初始原生评测后保存，增加了排名元数据，因此不能谎称两个文件 hash 相同。已逐 tensor 检查其 actor 参数、完整 learning config、BC/BPTT 更新计数均与 `initial.pt` 完全相同，再据此选择原 `native_0000` 录像。

第 10 轮原回合加载的 `candidate.pt` 文件 SHA256 为 `83e88d56e3ec5ce3a91e1e9e1e12b917252c814a13b5fae1f3b07d136ecab42c`，与 `native_0010` 保存的检查点 hash 严格相同，无需别名推断。

## 7. 导出与画面 QA

使用既有 `evaluate_ollie_policy.render_actual` 读取保存 qpos，以 **CPU OSMesa** 渲染；既有 `export_ollie_comparison.py` 生成原片/规定目标/实际回合三栏。随后只增加明确的 BC/BPTT/FAILED 与倍率假设标题，并保留末端静止帧。没有修改训练、评测或对照导出源代码。

最终两条三栏视频 SHA256：

```text
selected_best/comparison_full.mp4
55f5a660c01b3b4626c72d78096dbbd21999a28e85c45a68949cccb27ef8a521

bptt70_iteration10/comparison_full.mp4
6ba4ff98f7590c3b7a6ccd7cb64e935ab13d78de5ebd18d111e07cd60c37ebd0
```

4 个推荐视频均已通过完整解码检查。已抽查起点、准备段末、摔倒后与最终帧，确认标签可读，原片与目标/实际画面分栏清楚，机器人摔倒后的完整行为仍在。每个子目录的 `qa/` 保留抽帧；`actual_04.png` 对应实际第 257 帧，即物理 5.14s 的最终保存状态。

本批唯一新增记录文档为本文件。README 与记录 09 留给主代理统一更新；原训练结果、权重、参考、教师及原始回合均未覆盖。
