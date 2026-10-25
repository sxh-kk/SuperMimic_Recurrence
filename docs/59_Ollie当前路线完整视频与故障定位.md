# Ollie 当前路线、完整视频与故障定位

2026-10-25。**视频到物理仿真和结果审计的工程链路已执行；完整 Ollie 尚未成功。**本文明确现阶段路线与边界，避免把运动学回放、MPC教师或准备网络叫作最终SuperMimic结果。

## 现在走的路线

1. `video/ollie.mp4` → GVHMR/SMPL-X人体恢复 → GMR重定向到29关节G1；另行追踪板面图案、估计板三维状态，统一世界坐标与物理时间。SMPL-X和所需公开权重已下载并实际使用，没有等待用户补资产。
2. 修正足板关系、真实翘尾站姿和手臂自碰，生成50 Hz联合参考。视频姿态不能直接当动力学可行解：空中接触、COM与落地候选仍有明确失败记录（文档46、49～51、54）。
3. 原生MuJoCo中运行G1＋自由滑板，只作用原29个机器人PD目标，用采样MPC寻找真实可行教师。准备0～2.76 s通过原门槛，唯一完整延续5.14 s仍FAILED（53、56）。目前的直接排错控制器是MPC，不是论文SuperMimic策略。
4. 可行教师 → 行为克隆初始化 → 人板联合世界模型＋BPTT策略优化，是学习路线。世界模型和BPTT原型早期已运行，但实际动作响应留出检查不可靠；最新准备教师的固定BC、单回合真实MPC纠偏监督更新也没通过全部闭环门槛（55、57、58）。没有合格完整教师，未把失败回合作为成功样本继续训练。

原一小时LAFAN1平地训练保持归档停止，与这次Ollie物理问题分开。当前成果不能称为论文级完整复现。

## 最新完整视频与核验

- [原片／真实物理完整对照](/home/zxc/SuperMimic/logs/ollie_sampling_mpc_continuation_01/video_01/comparison_full.mp4)。右侧是保存的连续实际qpos，明确FAILED，不显示运动学目标冒充实际。覆盖物理0～5.14 s，保留失败尾段。
- [实际物理原速](/home/zxc/SuperMimic/logs/ollie_sampling_mpc_continuation_01/video_01/rollout.mp4)。258帧/50 Hz，物理末时刻5.14 s；最后姿态显示一个20 ms采样周期，编码时长5.16 s。
- [已通过的准备段对照](/home/zxc/SuperMimic/logs/ollie_sampling_mpc_prep_01/video_01/comparison_preparation.mp4)。只认证单初态0～2.76 s准备质量，不认证起跳或扰动鲁棒性。

最新完整回合只初始化一次，无相位reset、根/板回写、板驱动、吸附或外力。257个控制、258边界状态、2057接触观测均已用原模型独立连续重放，qpos/qvel/state92误差0、接触全部原字段相同、完整严格v3结果逐字段相同。

严格结果：全板最高最低点仅离地1.282 mm，未形成连续40 ms、全板>5 mm的flight；3.785 s首次机器人几何触地，14.350 mm最大穿透，板曾倒置，最终四轮接地比例0。3.94 s才发生首次尾地接触，晚于机器人触地，不是有效pop。四轮最低点最大22.862 mm不能冒充整板腾空。

原始证据在`logs/ollie_sampling_mpc_continuation_01/`：`actual_rollout.npz`、`result.json`、`contacts.json`、`selected_plans.npz`、`independent_replay_audit_retarget.json`、`video_01/video_audit.json`。状态SHA256为`44467039d304d547af23557136ee96cbaeca347ba7ab0ab701564ea6658a0fcc`，对照视频SHA256为`fa8d3127a2de58c40a90cbfdddc656a60327ca0cb0cb1838aacbd51add1e0030`。完整解码、帧数与六个时点画面人工检查已通过；渲染不运行新的物理回合，状态hash不变。

1.5倍慢放仍是用户估计。原片完整186帧、7.757750 s；对照统一24 fps、187帧、7.791667 s，最后物理姿态对应视频7.71 s，最后输出采样点最多保持40 ms。它是显示尾端保持，不是站稳时间或额外仿真。

## 问题是否来自滑板模型

后续[模型审计61](61_Ollie滑板与足板接触模型审计.md)已确认足板参数被脚的高priority覆盖：原实际μ=0.6、solref=[0.02,1]，不是板XML的0.9/[0.008,1]。这属于明确的配置遗漏，应与控制时序问题一起处理。当前模型的重力、自由飞行、滚动、尾地碰撞与足板接触已单独测试（11），但这不等于实物校准准确；尚不能确定两类问题对最终失败各占多少。

现模型是2.18 kg刚性板＋翘尾/板鼻＋四个被动轮，轮半径27.5 mm，板长约0.8013 m。质量、摩擦、碰撞柔顺性、尺寸和人体尺度含工程估计；无板面弹性、无truck转向，也无实物辨识数据。刚体尾地反弹与真实木板变形可能不同，这些是最终物理精度的实质不确定性。原片是近原地Ollie，本轮不以转向模型缺失解释当前的前脚压板问题。

当前更直接的证据是：准备末真实动量低于参考，前脚保持高负荷，抵消后脚抬头力矩；随后双脚卸载时序没有接上尾地反弹。详细逐接触力矩诊断继续记录，不能因“踩尾失败”就直接调整板参数使结果看似成功。后续应先检验前脚卸载与后脚踩尾的控制顺序，再在可解释的参数范围检验模型敏感性。

力矩采样另作更正：执行力矩应在`mj_step`后、额外诊断`mj_forward`前读取。独立审计原准备腿峰35.59%cap，最新延续仅左髋pitch≥95%累计80 ms，且发生在失败后段；没有证据表明起跳前是腿部普遍达到上限。旧83.96%是post-forward诊断峰值，已在53/55更正，未改历史数据。

## 重导视频

```bash
env -u PYTHONPATH -u PYTHONHOME MUJOCO_GL=egl \
  uv run --locked python scripts/local/export_ollie_actual_comparison.py \
  --evaluation logs/ollie_sampling_mpc_continuation_01 \
  --out-dir logs/ollie_sampling_mpc_continuation_01/video_new
```

入口先从400 Hz观测重算严格结果并与原JSON对照，再渲染保存qpos；只支持完整5.14 s回合，拒绝partial控制与已有输出目录。生成原速和原片对照后自动完整解码、核帧数、检查状态hash不变，并写视频SHA和显示时间说明。独立物理重放由审计完成，导出器本身不声称完成重放。
