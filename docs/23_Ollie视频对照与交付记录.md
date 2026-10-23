# Ollie 视频对照与当前输出

2026-10-23。当前尚无通过完整Ollie物理判据的学习策略。为核对视频恢复、运动学目标和真正物理执行，新增 `scripts/local/export_ollie_comparison.py`，输出三栏H264视频，依次为原视频、明确标记的运动学参考、实际连续物理回合。运动学参考只用于显示，不能用于证明物理成功。

实际首版对照：`logs/ollie_comparison/pilot01_first_v2/comparison.mp4`。右侧使用固定保存的第一轮学习第10轮CPU评测，保留全过程倒地，并标记 `Outcome - FAILED`。中栏底部为 `REFERENCE / kinematic pose visualization`，与实际物理标签分开。第一版临时渲染曾沿用物理标签尾缀，已修正并重新完整输出v2，交付仅引用v2。

## 命令与时间对齐

```bash
env -u PYTHONPATH -u PYTHONHOME uv run --locked python \
  scripts/local/export_ollie_comparison.py \
  --reference data/ollie/joint_reference_50hz.npz \
  --evaluation logs/ollie_eval/pilot01_best_native_01 \
  --out-dir logs/ollie_comparison/pilot01_first_v2
```

脚本核对参考与评测保存的SHA256相同，并使用评测实际导出的MP4。原片保留展示时间；中/右两栏按参考声明的倍率1.5放慢到同一展示时间，不改变物理回合、控制、状态或成功结论。倍率仍为假设。容器输出1920×480、24fps、H264、185帧，展示约7.71s；末端仅有编码帧采样取整，未增加新的物理状态。

真实执行的摄像机只用于展示，跟随实际机器人/板中点；原片相机并不与仿真摄影机完全相同。渲染参考时使用独立MuJoCo数据，渲染记录时读取已保存的实际qpos，均不回写进物理回合。运行信息、源视频/参考hash、物理成功标志和失败原因保存为 `report.json`。

已检查实际H264编码、分辨率、帧数/帧率，并抽查约4.9s的画面：原视频在空中，中栏展示估计目标，右栏真实倒地，标签清楚。Ruff通过。视频、模型及派生数据留在本机忽略目录，只上传代码和Markdown记录。

## 当前边界

首轮学习控制尚未进入有效起跳；第二轮通过站稳热启动、长回合采集及独立CPU筛选改进。另一路PD参数搜索的旧代理曾提示离地，完整几何复核未确认合格全板腾空，且接板失败，详见文档14/28的更正。最终必须分别说明动作相似程度、连续物理成功、训练算法和验收结果，不能由三栏视频中看似正确的中栏代替右栏。

第二轮30轮训练已结束，最新可查看对照为 `logs/ollie_comparison/pilot02_best/comparison.mp4`，右栏使用独立CPU实际筛选的第20轮状态，首次机器人触地2.135s，仍FAILED。原生实际视频为 `logs/ollie_actor_pilot_02/native_0020/rollout.mp4`，由原保存qpos离线渲染，实际物理记录及接触统计没有修改。它比首轮晚摔倒，但未完成起跳/接板，详细训练结果见文档16。
