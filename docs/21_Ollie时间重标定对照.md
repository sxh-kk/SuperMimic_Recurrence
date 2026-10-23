# Ollie 的 s=2.0 时间重标定诊断

2026-10-23。已完成独立 `s=2.0` 分支：重新执行 GMR、按新物理时间重采样并重算速度、重新求解板的弱重力先验、重新做足板接触 IK，再完成真实连续原生 CPU MuJoCo 参考 PD 对照和总质心审计。没有修改默认参考、相机、人体恢复资产、训练或评测源码，也没有启动 GPU 神经训练。

**结果：时间重标定没有解决站稳或 Ollie。** s=1.5 与 s=2.0 两次实际回合都在物理约 1.025 秒首次出现机器人触地，均没有尾端触地或合格全板腾空。s=2.0 的部分重建质心曲率更接近正常重力，但不能据此认为视频真实倍率是 2.0，也不能认证轨迹受力可行。

## 固定条件与独立输出

时间定义仍是 `t_video = s × t_physics`。输出保持 50 Hz；不是把旧文件的 fps 或 metadata 改成另一个数。

固定使用原人体恢复 `body_world_raw.npz`、相机 K、`data/ollie/common_world.json`、假设板 / 图案尺寸、GMR 身高 1.732 m 和部位缩放。接触窗保持旧版视频 **0～2 s、6～7.716042 s**，阈值 0.3、IK 24 次、相同速度上限与平滑配置；没有混入 [文档 22](22_Ollie预蹲接触参考对照.md) 的扩展接触窗。

输入哈希保持：

- 人体恢复：`e1be98c17d0ca6981d2b9a9b7e8afa1781cb0144fe1662322940d4073c192137`。
- 共同相机 / 世界变换：`5e6ff665d12c5deb579666fa145e4028e9b9f78b51d8a9cb5460dc137010446c`。
- 未改默认 s=1.5 联合参考：`646b8fb681da178f994ce47efe67af55c4a92fd93bedd544f124f1f39dc248fb`。

新联合参考为 `data/ollie/retime_s2/joint_reference_50hz.npz`，SHA-256 为 `589e24e4d54ca2d3132524537f58f68bcd1c2e818a81509ed4b756d223db1185`，状态 193×92。物理时间 0～3.84 秒、视频时间 0～7.68 秒。原视频最后采样为 7.716042 秒；50 Hz 物理网格没有向容器末端补一个虚假姿态。

本机输出均在 `logs/ollie/retime_s2/`：`gmr/`、`board/`、`timing_audit/`、`native_reference_pd/`、`native_s15_control/` 和 `comparison.json`。这些派生数据、检查点和视频留在忽略目录，文档是本轮唯一新增的受版本管理文件。

## 实际重跑命令

现有 CLI 参数足够，没有新增参数或修改实现。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
references/GMR/.venv/bin/python scripts/video/retarget_gmr.py \
  --body logs/ollie_gvhmr/ollie/body_world_raw.npz \
  --out-dir logs/ollie/retime_s2/gmr --time-scale 2.0 --output-fps 50 \
  --human-height 1.732 \
  --joints-field joints_camera --rotations-field joint_rotmat_camera \
  --camera-world-transform data/ollie/common_world.json

CUDA_VISIBLE_DEVICES='' .cache/board-cv/bin/python scripts/video/fit_board_reference.py \
  --observations logs/ollie/board_tracking/board_image_observations.json \
  --body logs/ollie_gvhmr/ollie/body_world_raw.npz \
  --common-world data/ollie/common_world.json --video video/ollie.mp4 \
  --extra-landmarks logs/ollie/board_reference/manual_board_landmarks.json \
  --slowdown 2.0 --fps 50 --out-dir logs/ollie/retime_s2/board

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
.venv/bin/python scripts/video/build_ollie_reference.py \
  --robot logs/ollie/retime_s2/gmr/g1_robot_50hz.npz \
  --board logs/ollie/retime_s2/board/board_reference_50hz.npz \
  --out data/ollie/retime_s2/joint_reference_50hz.npz \
  --stance-video-window 0 2 --stance-video-window 6 7.716042 \
  --min-confidence 0.3 --ik-iterations 24
```

GMR 在原 233 帧展示时刻重新做了 CPU IK；原生逐帧 qpos 与旧版完全相同是预期结果，因为人体图像姿态没有改变。之后使用新的物理输入速率 30×2、50 Hz 输出网格计算位置、SLERP 和速度，而非沿用旧速度。在共同视频时刻，GMR qpos 最大差异约 `1.46e-14`；新文件内部关节速度与新姿态的物理时间梯度一致。

板优化在相同 186 个原视频帧上重新运行，新倍率改变弱重力项中的物理时间间隔。源板根位置相对 s=1.5 的最大绝对差分别约 x=0.157、y=6.785、z=5.727 毫米；源旋转矩阵最大元素差约 0.01791。清晰空中 9 帧的图案重投影 RMSE 中位数 1.064、最大 2.969 像素。说明本次确实重新求解了轨迹，同时仍受同一组单目和图案假设约束。

新时间网格和物理速度限制也改变后续 IK 的结果。共同视频采样时刻的最终根位置最大差约 `[2.188,83.134,8.396]` 毫米，最大关节角差 0.09994 弧度。初始根差约 `[-1.271,-0.346,+0.014]` 毫米，最大初始关节角差 0.001741 弧度。因此这是完整重建链路的时间分支，不是“完全相同初始与姿态、只调执行快慢”的严格单因素控制试验。

## 几何参考 QA

新参考每脚明确接触约束 94 个样本；相同视频窗在较短物理时长中，样本数自然从原版 125 减少。最大受约束足底支持间隙 0.820 毫米、最大平面 anchor 误差 0.150 毫米、全段足板碰撞最大穿透 1.104 毫米；关节限位误差 0，最大关节速度 11.900 rad/s，小于配置 12 rad/s。`JointReference.load()`、全部有限性、单位四元数、50 Hz 递增物理时间和双时间轴对齐检查通过。

这些仍只是几何 QA；没有求解平衡、冲量、力矩或摩擦可行性。

## 真实原生 CPU 参考 PD 对照

两次均使用当前相同评测脚本，从各自首帧初始化一次、零初速度；正常重力，8×0.0025 秒物理步 / 0.02 秒控制，只驱动机器人 29 个关节，目标限位内缩 0.005 rad；没有外力、根 / 板姿态回写或中途重置。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/local/evaluate_ollie_policy.py \
  --reference data/ollie/retime_s2/joint_reference_50hz.npz --reference-pd \
  --out-dir logs/ollie/retime_s2/native_reference_pd

CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/local/evaluate_ollie_policy.py \
  --reference data/ollie/joint_reference_50hz.npz --reference-pd \
  --out-dir logs/ollie/retime_s2/native_s15_control
```

s=1.5 控制回合：连续 5.14 秒，258 个实际状态、2057 行含初帧的物理子步接触。首次机器人触地 1.025 秒，最大穿透 24.953 毫米；无尾端触地、无合格全板腾空；末 300 毫秒左右脚板接触占比 63.3% / 0%，四轮同时触地 55.8%，结果 **FAILED**。

s=2.0 新分支：连续 3.84 秒，193 个实际状态、1537 行含初帧的物理子步接触。首次机器人触地同为 1.025 秒，最大穿透 22.274 毫米；无尾端触地、无合格全板腾空；末左右脚板接触占比 0% / 0%，四轮同时触地 100%，结果 **FAILED**。四轮触地率高仅说明板留在地面，不表示机器人站板成功。

两次每子步的外力范数均为零；轮接触按独立轮几何计数，范围 0～4。s=2.0 的排名代价为 4.555，旧版 4.981，但片段总物理时长和跌倒后占比不同，不能将跨时长的代价降低解读为站稳进展。首次触地的物理时刻相同，映射回视频时刻分别约 1.5375 / 2.05 秒，也不能用较大的视频时间声称活得更久。

两次均保存 `rollout.npz`、`contacts.csv`、`result.json` 和明确标记失败的实际 `rollout.mp4`。s=2.0 视频经 ffprobe 核对为 H264、800×600、50 fps、192 帧：当前渲染器对累计浮点末时刻 `3.839999999999929` 做 floor，漏显示了最后一个 3.84 秒采样，最后画面约 3.82 秒。实际 NPZ 的 193 个状态和全部接触记录完整，评测数值未受影响；此渲染边界问题已反馈主任务，本轮没有修改评测器。

## 人板总质心审计

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python scripts/video/audit_ollie_timing.py \
  --reference data/ollie/retime_s2/joint_reference_50hz.npz \
  --out-dir logs/ollie/retime_s2/timing_audit \
  --video-window 4.71 5.26 --support-clearance 0.01
```

使用真实模型全部 G1 与板刚体质量和世界惯性中心计算总 COM，口径与 [文档 19](19_Ollie物理时间尺度审计.md) 相同。相同请求视频窗 4.71～5.26 秒，新网格实际选取 4.72～5.24 秒的 14 点：

- 展示时间竖直二次拟合加速度为 **-2.76775 m/s²**；在 s=2.0 下物理加速度为 **-11.0710 m/s²**，强于正常重力。
- 条件重力等效倍率 **1.88266**；二次拟合 RMSE **9.620 毫米**，最大残差 **17.780 毫米**；固定正常重力、s=2.0 时 RMSE **10.285 毫米**。
- 几何最低碰撞点高于 1 厘米、包含峰值的更宽窗口为视频 4.32～5.40 秒，共 28 点，对应物理 0.54 秒；拟合加速度 **-9.5936 m/s²**，条件倍率 **2.02243**，二次 RMSE **16.253 毫米**。
- 对主窗口边界各扰动 ±0.09 秒得到 49 次拟合，条件倍率最小 / 中位 / 最大约 **1.676 / 1.920 / 2.316**。这不是置信区间。

旧 s=1.5、旧接触窗的同口径主窗结果是物理加速度 -6.27865 m/s²、条件倍率 1.87496、二次 RMSE 9.51 毫米。新分支部分窗口的曲率靠近正常重力，仍不代表整个离地参考是可实现的无外部支撑弹道。板拟合本身已使用该倍率下的弱重力先验，因此总质心审计也不是完全独立的时间验证。

尤其 [文档 22](22_Ollie预蹲接触参考对照.md) 显示：只扩展接触校准、完全不改 s=1.5，条件倍率就可从 1.875 变到 1.451，同时抛物线残差增大。这比本次 s=2.0 分支带来的 1.875→1.883 变化更大，说明接触校准、根平移、尺度、机器人质量分布和窗口选择均影响时间反推。不能把 s=2.0 当成原视频实测结果。

审计原脚本的 JSON `caveats` 有一句历史固定文案写“板先验使用 1.5”；本分支实际板命令为 `--slowdown 2.0`，其日志和 metadata 均已保留，数值审计读取的 `time_scale_assumption` 正确为 2.0。该文案更正在本记录和 `comparison.json` 中明确说明，未暗改原输出数值或源码。

主审计图 `timing_audit.png` 已打开核对；实际视频、状态 / 动作尺寸、物理时轴、包含初帧的接触记录与外力恒零检查均通过。本分支保留作诊断，不替换默认参考或现有检查点；后续仍应优先处理准备 / 预蹲的真实平衡、支撑保持与参考动力学可行性。
