# Ollie 预蹲接触参考分支对照

## 结果与边界

已建立独立分支 `data/ollie/contact_extended/joint_reference_50hz.npz`，将双脚站板工程先验从视频 0～2 s 扩展到 **0～4.15 s**，恢复段仍为 6～7.716042 s。默认参考没有替换，其 SHA-256 仍为 `646b8fb681da178f994ce47efe67af55c4a92fd93bedd544f124f1f39dc248fb`。

新分支修正了视频 2～4.15 s 中明显的参考足板漂移，几何 QA 通过。但完整 5.14 s 的真实控制没有因此成功：固定偏置基线的首次机器人触地时刻仅从 **1.985 s** 延后到 **2.015 s**；纯 Reference PD 则从 1.025 s 变为 1.035 s。所有对照均未形成合格板腾空窗口，没有完成 Ollie。

因此，“中段接触参考缺失”确实是参考质量问题，但补上这一问题不能替代预蹲动态平衡和起跳控制学习。不能将几何改善描述为已修好摔倒。

## 原视频核对与先验定义

重新抽取并查看原始视频帧，而不是仅看人体恢复图：

- frame 99，视频 4.129125 s：两脚仍在板上。
- frame 100，视频 4.170833 s：两脚仍在板上。
- frame 102，视频 4.254250 s：仍可见两脚踩板。
- frame 106，视频 4.421083 s：板已翘起，头侧脚明显抬起，不能把双脚固定约束继续延伸。

图像留在 `logs/ollie/contact_extended/source_frame099.png` 等文件。4.15 s 位于 frame99 与 frame100 之间，作为保守的双脚几何接触截止先验是合理的。这里的接触来自人工目视，**不是实测接触力**；不能由画面直接推断足板压力、摩擦力或完全无滑动。

新分支继续使用原 s=1.5、GMR common-world 初始化、板 v3 和正常物理模型。s=1.5 仍是用户慢放倍率假设，未由本试验验证。板位姿可信度未提高；准备段 0.35 的工程可信度通过显式阈值 0.3 使用。

## 生成与几何 QA

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
.venv/bin/python scripts/video/build_ollie_reference.py \
  --robot logs/ollie_gmr/common_world/g1_robot_50hz.npz \
  --board logs/ollie/board_reference_v3/board_reference_50hz.npz \
  --out data/ollie/contact_extended/joint_reference_50hz.npz \
  --stance-video-window 0 4.15 --stance-video-window 6 7.716042 \
  --min-confidence 0.3 --ik-iterations 24
```

新参考 SHA-256 为 `0174586f94e68d35eaa63fd02743291e34db5660f2401d9e2241abe0968a7387`。输出为 258×92，50 Hz，物理时间 0～5.14 s。每只脚明确 stance 样本从 125 增加到 **197**，中间释放和腾空阶段仍没有足板相对位姿绑定。

新分支全段 QA：最大受约束足底支持间隙 **0.820 mm**、最大 anchor 平面误差 **0.142 mm**、真实足板碰撞最大穿透 **1.235 mm**、关节限位误差为 0、最大关节速度 12 rad/s。完整 `JointReference.load()` 验证通过。

对视频 2～4.15 s 的实际 72 个参考姿态，比较同样的真实足底几何：

- 支持间隙绝对值均值/最大值，旧版 **14.78 / 48.12 mm**，新版 **0.800 / 0.807 mm**。
- 相对于各自准备段 anchor 的足底平面偏移均值/最大值，旧版 **72.64 / 157.34 mm**，新版 **0.028 / 0.073 mm**。

上述“偏移”是运动学参考相对 anchor 的变化，不能当作真实动力学接触滑移测量，也不能把很小的优化残差称作毫米级视频重建精度。

## 根校准与初始化变化

机器人根平移校准的各轴最大绝对量：旧版约 `[0.12806, 0.42037, 0.11049]` m，新版约 `[0.12806, 0.41796, 0.11051]` m。最大量级变化不大，但校准的**时间分布**改变：旧版视频 2～6 s 之间主要插值；新版在 2～4.15 s 继续使用接触条件下的校准。

板 21 维状态在两个版本中逐点完全一致，没有通过改变板轨迹来获得改善。足底 anchor/yaw 按更长连续段重新估计，因此初始机器人位姿也略变：初始根 xyz 差约 `[-0.188, +0.767, -0.358]` mm，最大关节角差约 **0.02679 rad**。本对照并非“相同初始 qpos 只换后半段目标”，首次触地 0.01～0.03 s 的差别不能被过度解释为接触先验的明确因果改善。

几何比较完整数值留在 `logs/ollie/contact_extended/reference_comparison.json`。

## 总质心时间尺度复核

对新参考重新实际执行相同审计，没有改变时间倍率：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python scripts/video/audit_ollie_timing.py \
  --reference data/ollie/contact_extended/joint_reference_50hz.npz \
  --out-dir logs/ollie/contact_extended/timing_audit
```

在相同视频 4.71～5.26 s 窗口、同一模型全部质量与相同板轨迹下：

- 旧参考展示竖直加速度约 -2.79051 m/s²，s=1.5 时物理加速度约 -6.27865 m/s²，条件重力等效倍率约 **1.87496**，二次拟合 RMSE **9.51 mm**。
- 新参考展示竖直加速度约 -4.66184 m/s²，s=1.5 时物理加速度约 **-10.48914 m/s²**，条件重力等效倍率约 **1.45063**，二次拟合 RMSE **28.89 mm**，最大残差约 **68.66 mm**。

新参考局部平均曲率更接近正常重力，但更不符合单一抛物线。只改变接触校准，条件等效倍率就从 1.875 变到 1.451，说明该反推非常依赖参考优化，不能用于确定原视频实际慢放倍率。这进一步支持 [时间尺度审计](19_Ollie物理时间尺度审计.md) 的保守结论：保留 1.5 基线，联合校准参考、接触、尺度和无支撑 COM 动力学，而不是根据一次条件拟合直接覆盖全链路时间。

## 未训练固定偏置 actor 容器

为复用现有 `evaluate()` API，将 [准备段站稳审计](20_Ollie准备段站稳审计.md) 的 `limit025_gated` 29 维固定偏置放入一个**未训练** actor 检查点：`logs/ollie/contact_extended/untrained_fixed_bias025.pt`。

该检查点 `iteration=0`、`trained=False`，没有梯度更新。decoder 最后一层权重为 0，bias 为 `atanh(bias_rad/0.25)`，因此其输出恒为当前参考关节目标加同一固定偏置。已数值确认输出偏置与源 JSON 一致。它只是兼容评测接口的固定 PD 基线，不能把 actor 文件扩展名或通用网络类误当作训练策略。

检查点 SHA-256 为 `b9421b56be1df073dabd7856d25cde4edce86b5cc24d125da17a1ea27910d8ec`，源偏置为 `logs/ollie_stance_bias/limit025_gated/result.json` 的 `bias_rad`。评测产物的 mode 和 metadata 已明确改标为 `Untrained fixed-bias PD baseline (actor container)`；实际状态和动作数组未修改。

## 完整真实 5.14 s 控制对照

所有实验均从各自参考首帧初始化一次、零初始速度，正常重力，物理步长 0.0025 s，控制周期 0.02 s，真实关节目标限位；只使用 29 个机器人位置驱动器，板无驱动，无根/板姿态回写，无外力，无回合内重置。使用已有 API：

```python
from scripts.local.evaluate_ollie_policy import evaluate

evaluate(None, "data/ollie/contact_extended/joint_reference_50hz.npz",
         "logs/ollie/contact_extended/extended_reference_pd", render=False, device="cpu")
evaluate("logs/ollie/contact_extended/untrained_fixed_bias025.pt",
         "data/ollie/contact_extended/joint_reference_50hz.npz",
         "logs/ollie/contact_extended/extended_fixed_bias025", render=False, device="cpu")
evaluate("logs/ollie/contact_extended/untrained_fixed_bias025.pt",
         "data/ollie/joint_reference_50hz.npz",
         "logs/ollie/contact_extended/original_fixed_bias025", render=False, device="cpu")
```

实际结果：

- **旧参考 Reference PD**：已有连续评测首次机器人触地 **1.025 s**，最大穿透约 **24.95 mm**，FAILED。
- **新参考 Reference PD**：首次触地 **1.035 s**，最大穿透约 **29.29 mm**，FAILED。
- **旧参考＋相同固定偏置**：本轮重新连续执行，首次触地 **1.985 s**，最大穿透约 **16.52 mm**，FAILED。
- **新参考＋相同固定偏置**：首次触地 **2.015 s**，最大穿透约 **25.22 mm**，FAILED。

三次新执行均完成完整 5.14 s 数值连续回合并保存全部真实 qpos/qvel/state/actions 与每子步接触记录，但都没有尾部触地窗口或合格板腾空窗口。这里“完成回合”仅表示没有中断，不表示任务成功。未启动 GL 渲染或 GPU 训练。

新版固定偏置评测首个触地物理时刻 2.015 s 对应视频 3.0225 s，仍在明确预蹲接触窗内，尚未到预期弹板阶段。固定偏置并未解决预蹲中随姿态变化的平衡和接触载荷问题。

## 下一步建议

保留此独立分支作为更完整的准备/预蹲参考候选，不覆盖默认参考或已训练结果。可用新分支单独做课程和控制对照，但应重新验证站稳偏置或学习反馈，因为连续接触段重新估计改变了初始姿态与中段目标。

优先在物理 0～2.76 s 的准备/预蹲课程中优化姿态反馈、COM 横向稳定、髋踝腰协同和足板接触保持。之后才连接踩尾、离地、接板、落地；不能继续靠固定常量贯穿全回合。对 COM 弹道的评价也应同时报告重建非抛物线残差，不能只因平均加速度接近 g 就宣称物理可行。
