# Ollie 连续闭环评测记录

2026-10-23。本记录对应 `scripts/local/evaluate_ollie_policy.py`。输入是经过联合几何校正的 `data/ollie/joint_reference_50hz.npz`，不是把参考动画直接当成执行结果。当前已实际运行参考 PD 基线和训练 2 轮的原型策略；两者均失败，实际视频和接触数据已保存。

## 物理执行约定

原生 CPU MuJoCo 3.16.0，物理步长 0.0025 秒，每 8 个物理步更新一次控制，即 50 Hz。只设置 G1 的 29 个关节绝对位置 PD 目标；使用场景原有 PD 增益、阻尼、执行器力限和碰撞配置。目标按训练环境相同规则裁剪至每关节物理范围内侧 0.005 弧度。滑板和四个轮子均无执行器，没有外力、支撑焊接或中途重置。

回合只在起始处写入参考第 0 帧的机器人 / 板位姿，默认将全部初速度置零；`--initial-velocity reference` 是明确标注的另一初始化条件，当前报告均未使用。之后不再写机器人根或板位姿。跌倒仍持续模拟到片段末端，保留真实失败行为。若 MuJoCo 出现非有限值或时间倒退 / 自动不稳定重置，记录数值失败并停止，不静默重开回合。

参考 PD 使用下一帧的 29 关节角。策略入口严格读取 `supermimic.ollie.actor.v1`，核对四个板轮关节的顺序，按保存的 `learning_config` 构建 `JointResidualPolicy`。第 `step` 次控制的目标参考为 `min(step+1+offset, last_frame)`，当前偏移 `(0,5,10)`；输出是绝对关节 PD 目标，包含网络自己的有界残差，不另叠加一次残差。默认在 CPU 推理，不占用训练用 Torch CUDA 设备。

此入口是使用完整仿真状态的研究原型 / 基线评测，尚不是论文完整 SuperMimic 感知与训练实现。几何参考中的视频时间倍率 1.5 仍未被证实。

## 接触、腾空和成功判据

每个 2.5 毫秒物理子步都重新读取实际几何与 `mj_contactForce`，默认法向力至少 0.1 N 才记作有力接触。四轮通过独立轮碰撞几何 ID 的集合统计，数值范围只能是 0–4；同一个轮子的多个接触点不能变成“四轮落地”。左右脚分别统计对板面 / 板端 / 轴盒的接触与对地接触。

穿透检查覆盖初始化后的第 0 帧和之后每一个物理子步，使用所有实际接触的最小距离，不受 0.1 N 接触统计阈值过滤。一个完整 5.14 秒回合有 257 个控制目标、2056 个物理步和 2057 行接触记录。

滑板中心、翘起板端和轮轴使用其实际旋转盒子的最低支撑高度；四轮使用实际旋转圆柱的解析最低支撑高度。腾空要求全板最低高度大于 5 毫米、没有板与地面的有力接触，并连续至少 40 毫秒。尾端需要在这次腾空开始之前真实触地；40 毫秒要求针对连续腾空窗口，不要求尾端本身持续触地 40 毫秒。尾触地窗口单独记录。

`success=True` 需要全部满足：

- 完成整段参考时长且无数值异常，初始穿透不超过 2 毫米。
- 尾触地在先，随后出现合格的连续全板腾空；四轮最低高度曾达到至少 15 毫米。
- 全程没有机器人几何触地，包括脚部；全程最大接触穿透不超过 10 毫米。
- 最后 300 毫秒内，左右脚各自板上接触占比至少 80%，四个独立轮同时触地占比至少 80%。完整回合此窗口包含 120 个物理子步，不能用初始接触一次代替。
- 最后窗口机器人竖直余弦至少 0.8，板至少 0.95；机器人根高于板根至少 0.55 米。机器人最大线速度 / 角速度分别不超过 0.3 m/s、0.8 rad/s；板分别不超过 0.35 m/s、0.8 rad/s。
- 没有施加外部广义力或刚体外力。

上述阈值是本项目明确采用的工程判据，不声称是原论文的官方成功指标。`ranking_cost` 为独立 CPU 筛选用的工程代价，越低越好；它结合坐标分量位置 RMSE、关节 RMSE、机器人触地占比、超限穿透与未完成惩罚。排名改善不能替代全部成功判据。

## 命令行与训练内调用

实际参考 PD 基线：

```bash
.venv/bin/python scripts/local/evaluate_ollie_policy.py \
  --reference data/ollie/joint_reference_50hz.npz --reference-pd \
  --out-dir logs/ollie_eval/reference_pd_v2
```

真实策略检查点：

```bash
.venv/bin/python scripts/local/evaluate_ollie_policy.py \
  --reference data/ollie/joint_reference_50hz.npz \
  --ckpt logs/ollie_actor_smoke_01/best.pt --device cpu \
  --out-dir logs/ollie_eval/learned_smoke_first
```

添加 `--no-render` 可用于周期 CPU 筛选。输出目录必须为新目录，避免覆盖已经保存的回合。也提供可导入接口：

```python
import importlib.util

spec = importlib.util.spec_from_file_location(
    "ollie_evaluation", "scripts/local/evaluate_ollie_policy.py"
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
result = evaluation.evaluate(
    ckpt="logs/ollie_actor_smoke_01/best.pt",
    reference="data/ollie/joint_reference_50hz.npz",
    output="logs/ollie_eval/another_new_directory",
    render=False,
    device="cpu",
)
```

`ckpt=None` 选择参考 PD。接口不会改训练器模型、优化器或 Torch 线程数；CLI 使用 1 个 Torch CPU 线程。原生 MuJoCo 的全局控制回调在评测期间清空，完成后恢复，避免调用者的其他回调偷偷加入控制。该 API 创建独立原生 CPU 场景，与 GPU 并行训练场景隔离。

每次保存：

- `rollout.npz`：258×92 实际状态，以及实际 `qpos/qvel`、257×29 控制、裁剪前目标、物理 / 视频时间、实际参考索引和布局说明。全部来自同一次连续回合。
- `contacts.csv`：物理子步接触、唯一轮数、左右脚、尾 / 头、穿透、几何离地高度、根速度 / 姿态及外力范数。
- `result.json`：输入哈希、检查点轮数、初始化与控制约定、逐项成功判断、失败原因、跟踪误差和排名代价。
- `rollout.mp4`：默认 800×600、50 fps、H264，标题为 `Reference PD baseline` 或 `Learned full-state prototype`，明确标注 `Outcome - FAILED/SUCCESS`。渲染仅回放实际记录的 qpos，在独立渲染数据中进行，不把位姿回写进物理回合；当前未加参考幽灵。

EGL 仅用于画面渲染，动力学和默认策略推理均在 CPU。258 帧视频容器约 5.16 秒，最后一个真实采样状态为物理 5.14 秒；没有伪造额外末帧状态。

## 本机实际结果

参考输入哈希为 `646b8fb681da178f994ce47efe67af55c4a92fd93bedd544f124f1f39dc248fb`。

参考 PD 基线，产物 `logs/ollie_eval/reference_pd_v2/`：

- 连续完成 5.14 秒，0 中途重置、0 外力、0 板驱动；初始接触穿透近似 0。
- 无尾触地、无合格腾空，最大全板最低高度仅 0.392 毫米。
- 机器人第一次触地约在 1.025 秒，峰值穿透 24.953 毫米。
- 末 300 毫秒左脚板接触占比 63.3%，右脚 0%，四轮同时接触 55.8%；站稳不合格。
- 排名代价 4.981，含完整视频的本次墙钟约 2.27 秒；结果 **FAILED**。

训练 2 轮的真实策略 `logs/ollie_actor_smoke_01/best.pt`，产物 `logs/ollie_eval/learned_smoke_first/`：

- 检查点 schema、网络参数和 `(0,5,10)` 目标接口实际加载成功；完整回合、状态与接触记录检查通过。
- 第一次机器人触地约 1.0275 秒，无合格腾空，峰值穿透 25.968 毫米，最大全板最低高度约 0.645 毫米。
- 末窗口左脚 / 右脚板接触占比 46.7% / 0%，四轮同时接触 64.2%。
- 排名代价 4.985，含完整视频的本次墙钟约 2.44 秒；结果 **FAILED**。两轮 smoke 只验证训练 / 保存 / 推理链路，不能据此声称学会 Ollie。

另用明确标为“未训练、零残差”的临时检查点验证接口，其控制相对参考 PD 的最大差异约 `1.1e-7` 弧度，状态 qpos 最大差异约 `7.4e-6`，来自参考 float64 与网络 float32 的数值差别；它不作为学习效果展示。末步未来索引均被正确裁剪到 `[257,257,257]`。

已检查两种真实回合的状态 / 控制形状、全部有限性、50 Hz 单调时轴、2057 行含初帧接触、独立轮数范围 0–4、外力恒零；视频编码经 ffprobe 确认为 H264 / 800×600 / 258 帧 / 50 fps。已目视抽查基线起始、1–4 秒和最终帧，保留全过程倒地行为。Ruff、格式检查与语法编译通过。
