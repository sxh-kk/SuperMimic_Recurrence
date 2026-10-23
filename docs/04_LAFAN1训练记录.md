# LAFAN1 第一轮真实训练

## 本轮目标与边界

在本机单张 RTX 4090 上训练默认观测、seed=0 的 LAFAN1 基线。使用 40 段真实动作，单轮上限 6 小时；320000 次环境迭代或时间上限先到先停。达到时间上限不等于完成全量论文复现。完整跟踪评测在本轮结束后单独执行。

## 2026-10-23：数据准备与验收

- 来源：[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)，仅下载 `g1/*.csv`。
- 固定数据版本：`ce1572906efe6157840e8474d5a0d7aa87481e74`。
- 40 份 CSV 共 90487276 字节，源帧率 30 Hz；输入位置为米、关节角为弧度、根四元数为 xyzw。
- 12 个 CPU worker 转换到 50 Hz，40/40 成功、0 失败。NPZ 根及刚体四元数为 wxyz。
- 全库检查：40 对名称一一对应，264705 个原始帧、441140 个输出帧，总时长分别为 2.45060185 / 2.45055556 小时；差异来自每段输出时间网格的端点取整。
- 所有数值有限；逐段帧数、帧率、29 个关节及 30 个刚体的名称顺序、首帧 pelvis 位置 / 四元数和关节角均通过。
- 输出四元数最大范数偏差 `5.13e-8`，首帧 pelvis 位置最大误差 `2.93e-8 m`。检查无警告、无错误。
- 本地 raw 约 87 MiB、NPZ 约 755 MiB；数据不上传 GitHub。

集合 SHA256 按相对文件名排序，对每行 `文件名<TAB>文件SHA256<换行>` 的 UTF-8 内容求 SHA256。NPZ 哈希标识本次生成的文件产物，不要求重新转换后的 ZIP 元数据逐字节相同。

```text
CSV: 0e8f485b239551a7997c792fb9531e1893aed2ea09ef30a3cda8c569b4ae133c
NPZ: 9cf3cce08b645d50e744364b49a2464f7b98471a95b45aa4331b02cd34f15765
```

准备与复查命令（项目根目录执行）：

```bash
env -u PYTHONPATH -u PYTHONHOME HF_HUB_DOWNLOAD_TIMEOUT=300 uv run --locked python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lvhaidong/LAFAN1_Retargeting_Dataset", repo_type="dataset",
    revision="ce1572906efe6157840e8474d5a0d7aa87481e74",
    allow_patterns=["g1/*.csv"], local_dir="data/raw/lafan1", max_workers=8,
)
PY

env -u PYTHONPATH -u PYTHONHOME OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --locked python scripts/data/csv_to_npz_batch.py \
  --input-dir data/raw/lafan1/g1 --output-dir data/npz/lafan1 \
  --input-fps 30 --output-fps 50 --workers 12

env -u PYTHONPATH -u PYTHONHOME uv run --locked python scripts/local/check_lafan_data.py
```

## 启动前验证

- `test_motion_convert.py`、`test_env_smoke.py`、`test_analytic_heads.py`：**14 passed，0 skipped，13.02 s**。
- 覆盖真实动作重采样、CPU FK、GPU 环境 / replay、解析奖励 / 观测 / 终止对拍；上游 TorchScript FutureWarning 不影响结果。
- 后台入口通过成功退出、非零退出、无效命令、并发锁拒绝、SIGTERM 转发和保留参数别名检查；`--help` 经完整后台入口正常返回。
- 两个新增 Python 脚本通过 Ruff 检查。

## 正式配置与启动方式

- 运行名：`lafan_paper_s0_20261023`。
- 默认观测：策略本体历史 930 维，目标 680 维；`anchor_obs=False`。
- 4096 环境，策略批量 4096，想象 H=32；WM 批量 2048，每次更新 10 个长度 8 的窗口批次。
- replay 每环境 1000 步，UTD=0.4，warmup=40；seed=0。
- 观测噪声、重置噪声、域随机化均开启；动作噪声标准差 0.1。
- 标准网络和默认编译模式；总环境迭代 320000，最长 6 小时。
- 检查点间隔由预设的 4000 改为 **1000 次环境迭代**，算法批量与学习率不变。每次同时保存 latest 与编号文件，退出成功后再保存 latest。

```bash
env -u PYTHONPATH -u PYTHONHOME uv run --locked python scripts/local/start_train_4090.py \
  --profile lafan --run-name lafan_paper_s0_20261023 -- \
  --seed 0 --schedule.checkpoint-interval 1000
```

后台入口拒绝覆盖已有运行目录；不要用相同运行名重启。进程脱离当前聊天会话，控制台、机器配置、进程号及退出状态写入本机 `logs/lafan_paper_s0_20261023/`。`status.json` 是生命周期记录，并非心跳，实际进度以控制台新迭代和进程存活为准。所有对外实验结论在本 Markdown 文档追加；机器日志和权重不提交。

6 小时从环境 reset 后计时，每 50 次迭代检查一次，允许超过一个日志周期。首次编译计入运行时间；初始化不计入这 6 小时。

## 运行状态

数据与验证完成，等待提交当前启动工具后派发正式训练。实际开始时间、进程、早期更新及显存检查结果在启动后追加。

## 评测与恢复约定

本轮结束后使用稳定的编号检查点或已停止写入的 latest 做全 40 段评测。训练中不并行评测，以免占用同一张 GPU、改变本轮时间预算。当前 benchmark 的 `norm_return` 使用实际执行步数作分母；报告须同时列失败数，并另外核对整段 `frames-1` 分母结果，不能直接将其称为严格论文分数。

```bash
env -u PYTHONPATH -u PYTHONHOME uv run --locked python scripts/eval_benchmark.py \
  --ckpt logs/lafan_paper_s0_20261023/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 --num-envs 40
```

现有检查点不保存 replay 或仿真状态，恢复不等于逐步完全接续。后续若恢复编号 N 的检查点，需要另建运行名、使用相同动作集和观测配置，并将 `--schedule.warmup-iterations` 设为 `N+40`，先重新填充 replay；该恢复路径尚未做真实训练的独立进程验证。
