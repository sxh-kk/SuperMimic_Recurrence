# Ollie 视频时序参考与在线训练

## 本轮输入与当前边界

以文档32的新候选 `data/ollie/air_relative_video_timed_v1/joint_reference_50hz.npz` 继续真实物理评测和在线WM+BPTT。参考SHA为 `941f748cf3b13edbcf60dd20f2db41810cee2cbd1e9b4dccba1e3b1d4b5a71f5`。它修正提前腾空和空中脚板错位，保留准备段；仍是带单目尺度、接触时序及动力学不确定性的运动学目标，不能视为已验证可执行轨迹。

迁移的准备段教师使用 `logs/ollie/air_repair/video_timed_v1/teacher_migrated_v2.npz`，SHA `c953f231a5d0d47ce7d850f71ddf01ba64807e5b0ec78f102a50ad86cf964a8c`。139帧物理状态及实际动作保持原值，15个未来目标向量和参考来源已更新。旧迁移副本的顶层hash与metadata不一致，被BC和连续评测入口实际拒绝，未进入训练；旧文件保留，不能再用。

## 新参考下重新拟合初始化

从随机网络完整重做BC/DAgger的 `logs/ollie_bc_video_timed_02` 作为对照保留，其最佳筛选为2000次更新，准备段最小upright0.84259。没有把原网络的未来目标潜变量直接当作已兼容。

随后新增 `--policy-init` 权重初始化接口：严格检查网络、关节、残差范围和未来目标契约，但允许以旧网络作为**重新拟合的起点**；输入来源hash单独保存，所有评测仍用新目标。实际 `logs/ollie_bc_video_timed_03` 从旧03的7000监督更新网络开始，再做2000次有限拟合；实际物理筛选选中新增加1000次更新的版本，而非最后版本。

新推荐初始化 `logs/ollie_bc_video_timed_03/best.pt`：0～2.76s纯网络无触地，最小upright **0.85159**、最大root-board XY **193.851mm**、峰穿透 **7.816mm**。这里报告的是本轮新增监督更新数，旧训练历史通过初始化来源保留。完整5.14s初始评测首次机器人触地 **3.2575s**，仍FAILED，没有合格Ollie。

## 恢复演员并覆盖世界模型的修正

此前同时传 `--resume` 与 `--wm-init` 时，恢复检查点可能把显式指定的预训练WM覆盖为BC中未训练的WM。当前顺序明确为：恢复演员及优化器后，再应用显式WM初始化，并重建WM优化器，避免旧动量污染替换后的模型。

新增 `--actor-lr` 显式覆盖恢复后的演员优化器学习率；不暗改网络配置。`logs/ollie_actor_smoke_09/initialization_precedence_check.json` 实际证明初始模型全部参数与离线WM逐值相同，演员与指定BC逐值相同，WM优化器无旧状态，演员lr为1e-5。该冒烟还实际完成128条物理转移、2次WM更新、1次H4 BPTT和完整原生评测，有限性及保存恢复通过，不作为技巧收益证明。

## 在线有限预算

本轮 `logs/ollie_actor_video_timed_01` 使用：128环境、每轮128步、10轮；每轮300次H8模型更新；前3轮只拟合WM，之后每轮10次H8 BPTT；batch128、演员lr1e-5、教师动作约束权重100、探索噪声0.01rad。每一轮均用独立原生CPU完整回合筛选，结果不会因训练窗口达到某个参考阶段就标为成功。

WM从 `logs/ollie_wm_offline_02/model.pt` 开始；所有阶段初始化先经过全身几何门槛，评测仍只从首帧零速度初始化一次。训练中的阶段重置属于数据采集，不能作为最终连续视频。原一小时LAFAN1训练保持停止，未重启。

本轮已完成并停止：耗时 **291.643 s**，采集 **163840** 条真实物理转移，完成3000次在线WM更新、70次H8 BPTT更新。全部完整原生回合仍FAILED，最佳检查点停留在迭代0，即新参考BC1000次更新的初始化，**没有选择任何在线BPTT更新后的actor作为更好结果**。模型/日志/视频留本机，只有代码和Markdown上传。

初始和前3轮仅训练WM时，首次机器人触地均为3.2575s、原生ranking cost为2.88641。开始actor更新后，第4、5轮均3.13s触地，第6轮2.9275s，第7～10轮约2.785、2.7875、2.7725、2.740s。第10轮cost为3.289，较初始化更差。`best.pt`保留初始化演员，`latest.pt`与`candidate.pt`保留第10轮用于失败分析；不按训练更新次数推断效果。

后续真实动作梯度对照见文档36，发现部分滑板响应的幅度和方向存在问题，因此没有继续扩大相同训练预算。自由飞行augmentation的另一次离线拟合与本轮分开记录，本轮未使用该新数据。

```bash
env -u PYTHONPATH -u PYTHONHOME OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
MUJOCO_GL=egl uv run --locked python scripts/local/train_ollie.py \
  --reference data/ollie/air_relative_video_timed_v1/joint_reference_50hz.npz \
  --out-dir logs/new_ollie_video_timed_actor \
  --resume logs/ollie_bc_video_timed_03/best.pt \
  --wm-init logs/ollie_wm_offline_02/model.pt \
  --residual-limits-json logs/ollie_bc_video_timed_03/residual_limits.json \
  --teacher-anchor logs/ollie/air_repair/video_timed_v1/teacher_migrated_v2.npz \
  --teacher-weight 100 --actor-lr .00001 --num-envs 128 --iterations 10 \
  --collect-steps 128 --model-updates 300 --model-warmup-iterations 3 \
  --policy-updates 10 --model-window 8 --horizon 8 --batch-size 128 \
  --eval-every 1 --replay-chunks 8 --exploration .01
```
