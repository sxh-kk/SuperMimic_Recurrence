# Ollie提前纠正重心与全身落地

2026-10-25。83已完成真实102.5 ms整板飞行，但3.3975 s左脚有效触地，接板前双足同时top支持仍0，完整FAILED。3.16 s重心已接近板边，后续局部落地修复未成功。本轮保持83真实top/地接触、同窗50 ms、全29目标和原严格v3，将可改动作的起点提前。

## 预先固定的变化

新入口`../scripts/local/optimize_ollie_early_balance_landing.py`是83的薄封装，83源码保持SHA `cd564a077f23b6efdfdb5675cda58da85a0fe0c2efcf076cccd406dd9c79a81f`，top模块仍`48499f8c5333abc11f7991f9713207507c84fb8b2c2682d1e86943047ad92c66`。只改变连续前缀P145=2.90 s、余下H112目标、最后节点2.22 s和固定seed84；其余256×64、最多16最高等级精英、29残差维、σ=.10/floor.025、600 s预算、成本/权重/物理/PD和初态不变。原82全部112目标为nominal，零残差逐值重现实际低跳。

纯FK读取原82真实qpos/qvel并用原模型COM Jacobian核对：2.90 s robotCOM世界Vz+1.10495 m/s、重力线板Y−.03663 m，upright .99493；3.16 s为Vz−.20577/Y−.09264，upright .93437。提前不是修改保存state，而是在原连续实际回合中更早允许纠正关节。重力线更居中也不是动态可恢复或落地成功证书。

prior的相位公式仍相对分支起点：站姿+.12→+.60对应本轮绝对3.02→3.50，**已经跨起跳前，不叫after-air**；腰/肩方向候选也整体提前260 ms。wrapper为避免误称改其归档标签，动作数值/权重未另调。两者只是候选；exact82低跳和最高真实飞行等级优先，防止丢掉起跳的低代价动作替代它。

source仍是82已独立验证的SHA `5cf90dd936724fe920e15578b054724811c0368129fea253dfc6b6514e5c29f3`，复用逐数组相同的前170/171只读视图；主实际回合仍从原唯一初态逐145目标重放，不能设入2.90 s保存状态。旧60公共前缀本轮仅能附核前145，145之后开始新目标，不能再要求前150不变。

## 无积分审查与预检

`logs/ollie_early_balance_landing_review_01/review.json`确认源前145动作/146状态与旧60逐值一致，prefix-view数组/native时间一致，P+H=257与5.14终点、末控制5.12匹配；112×29 nominal/零残差精确重建，三prior限位通过，prior工厂前后主场景fingerprint不变。mock禁止mj_step且调用0，无新动作积分、训练或GPU。后续wrapper仅将相位误称的标签改为elapsed并加元数据，值和顺序不变。

先完整原生batch预算/执行阶段top与真实ground/原物理分支核验，再固定正式64轮、唯一实际完整后缀。独立全回合审计使用新v2，保留83旧审计源码hash，明确前145原82及old60公共前缀、896条本轮执行力、257控制/258状态/2057 Observer。实际结果和失败尾段如下完整保留。

```bash
env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --locked python scripts/local/optimize_ollie_early_balance_landing.py \
  --mode benchmark --out-dir logs/ollie_early_balance_benchmark_01
```

正式模式改run且使用新目录`logs/ollie_early_balance_landing_01/`。没有因局部物理候选自动晋升或训练SuperMimic actor。

## 完整结果：提前纠姿仍未完成落地

唯一正式64轮完成155.622 s，winner为round58/candidate45，cost33422.124275。主回合从原初态连续执行257个29维PD目标至5.14 s，保存258个边界、2057条400 Hz Observer和896条后缀执行力；actual_prediction误差0、partial=null。入口冻结SHA `2a2e03e0e4f735624b0c7d44438a5404cb0d67e47d2a7c5a1cc763f14c6e191b`，83入口和top模块保持上述原hash。仍为原12°板、默认足板μ=.6、无显式足板pair，原重力/质量/惯性/29PD不变，没有外力、板驱动、根/板状态回写或阶段reset。

原严格`whole_board_geometry_top_support_v3`结果为**FAILED**：

- 真实整板飞行3.2600～3.3325 s，共**75 ms**；全9个板碰撞几何最低点峰**24.2744 mm**。尾地主要窗口3.1950～3.2225 s，共30 ms，另有两个2.5 ms接触样本。
- 首次有效机器人触地为**3.405 s左脚**，具体为`left_foot4_collision`和`left_foot5_collision`。此时upright仍.98638，但root−board世界高度已降至.47252 m，直立朝向不能代替安全落脚和稳定承重。
- 全回合最大穿透**10.7569 mm**，超过原10 mm上限。最小robot upright .25580；原实际软关节越界最大.02904 rad另作诊断保留。
- 末300 ms左/右脚真实上表面支持>1 N均为**0%**，四轮接地87.5%，左脚触地53.33%；没有站稳。失败字段为身体触地、穿透超限及末段稳定站板三项。

本轮保留真实低跳，但没有把83的落地问题修好；不能从更早允许改动作或降低采样cost推断完整动作改善。探索raw grade2只记录真实飞行事件，不是合格教师或最终成功。**本轮actor更新0、SuperMimic训练0、无新策略检查点。**

## 独立重放与双阶段接触审计

`independent_replay_01/result.json`与`independent_force_audit.json`从唯一原初态重放全部257目标：qpos/qvel/state92边界、2057条Observer、896条后缀执行力各物理字段误差均0，全部strict判据与失败字段一致。累计边界时间相对理想20 ms时间轴仅有5.06e−14 s浮点误差。源82前145动作/状态、old60公共前145及原53初态逐值相同，真实执行力矩超限0。源actual归档SHA `0844dc7682e8adee517307f5251088615f72788419e94dd180d4580b3a37ce4e`。

`independent_top_stage_audit_01/result.json`独立将同一257目标在原模型与2453维只读观测模型连续重放：执行求解阶段和额外forward后阶段各2056个样本，完整物理、逐接触Fn/位置/法向/count、canonical top支持和真实ground判断误差全部0。单组实际最多14接触（容量32），robot-ground最多10（容量64），无overflow。审计脚本SHA `4292a9821b98fcdbdaa8a3aa4021de16ca8257659030650dd15391cd251af838`，没有新增动作搜索或训练。

两阶段都确认：**首40 ms飞行完成后、首次身体触地之前，双足同时真实top>1 N的最长窗口为0 ms；再要求upright≥.9和root−board高度≥.6 m也为0 ms。**执行阶段首触地的几何/区间起点为3.405 s，对应mj_step返回data.time3.4075 s；postforward在data.time3.405 s判定，同阶段分别与原Observer一致，没有跨阶段移位比较。

侧/底接触仍存在，执行阶段其法向Fn积分左/右分别14.5562/18.0977 N·s；它们不是向上承重冲量，不能当作top支持。额外forward后的重算力仅作该阶段诊断，不冒充实际执行冲量。

## 预检方向候选不能晋升

`logs/ollie_early_balance_benchmark_01/benchmark.json`首批完整候选rollout .928 s、cost1.940 s，投影64轮183.528 s，预算门通过。固定方向prior slot4 `waist_shoulders_com_direction_not_teacher`的首次身体触地指标为3.82 s，晚于正式winner；但其最大穿透**15.2805 mm**、最小upright−.33820，安全双足top窗口仍0。它只是预检中已评估的scratch候选，没有成为本轮主实际轨迹，不能只保留更晚触地时间而忽略几何违规、称为合格落地轨迹。此记录保留供后续改进完整回合的几何门槛和排序依据，不追认成功。

## 完整视频与抽帧

查看[原片/真实仿真完整FAILED对照](/home/zxc/SuperMimic/logs/ollie_early_balance_landing_01/video_01/comparison_full.mp4)，或[实际物理原速视频](/home/zxc/SuperMimic/logs/ollie_early_balance_landing_01/video_01/rollout.mp4)。实际258帧、对照187帧完整导出，保留5.14 s倒地终态；对照按用户估计1.5倍慢放对齐原片，没有用运动学目标或重跑状态替换实际执行。`video_01/video_audit.json`确认源归档未改，实际视频SHA `2c1d58ea2c62abb463e2dfdebbcb2f3bb821b258cf249b26a94c2a381669084f`，对照SHA `82674e200a71e73a709893a3c8c2f6500049a9555b73cfe7d6583760593e72fc`。

已抽取并查看实际视频2.90/3.20/3.30/3.42/4.00/5.14 s六帧，覆盖纠姿起点、起跳、接地和倒地尾段。图像用于查看姿态，75 ms飞行与毫米间隙来自上述400 Hz几何审计。

![84实际回合六帧，完整保留FAILED尾段](/home/zxc/SuperMimic/logs/ollie_early_balance_landing_01/video_01/montage_6frames.png)
