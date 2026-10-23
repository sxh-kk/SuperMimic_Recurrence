# LAFAN1 第一轮真实训练

## 本轮目标与边界

在本机单张 RTX 4090 上训练默认观测、seed=0 的 LAFAN1 基线，使用 40 段真实动作。最初按 6 小时上限启动；用户随后要求改为**本轮累计约 1 小时，保存检查点后停止**，目标上海时间 2026-10-23 19:25 左右。当前首跑不等于完成全量论文复现。完整跟踪评测在本轮结束后单独执行。

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
- 启动时使用标准网络和默认编译模式；内部总环境迭代 320000、最长 6 小时。后续用户要求的 1 小时截止由外部控制器执行，保留原始配置文件用于追溯。
- 检查点间隔由预设的 4000 改为 **1000 次环境迭代**，算法批量与学习率不变。每次同时保存 latest 与编号文件，退出成功后再保存 latest。

```bash
env -u PYTHONPATH -u PYTHONHOME uv run --locked python scripts/local/start_train_4090.py \
  --profile lafan --run-name lafan_paper_s0_20261023 -- \
  --seed 0 --schedule.checkpoint-interval 1000
```

后台入口拒绝覆盖已有运行目录；不要用相同运行名重启。进程脱离当前聊天会话，控制台、机器配置、进程号及退出状态写入本机 `logs/lafan_paper_s0_20261023/`。`status.json` 是生命周期记录，并非心跳，实际进度以控制台新迭代和进程存活为准。所有对外实验结论在本 Markdown 文档追加；机器日志和权重不提交。

6 小时从环境 reset 后计时，每 50 次迭代检查一次，允许超过一个日志周期。首次编译计入运行时间；初始化不计入这 6 小时。

## 运行状态

- 上海时间 **2026-10-23 18:25:06** 派发。原预计次日 00:25 达到 6 小时上限，现由用户新要求覆盖：累计约 1 小时保存并停止，目标为今天 **19:25 左右**。
- 实际运行代码版本：`cecca1237a4090acf2bfa8795df62817175d9713`，已推送公开 GitHub 仓库。
- worker PID `1933699`，uv PID `1933700`，实际 GPU 训练 Python PID `1933706`；三者同属独立会话 `1933699`。已验证实际训练进程继承并发锁文件。
- 运行时配置文件逐项核对通过，实际加载 **40 clips / 441140 frames / 2.451 h**，动作库报告 GPU 占用 0.57 GB。
- `it150 / updates44`：训练回报 0.448，WM 1 步 / 8 步误差 0.0658 / 0.2478；37 个 TensorBoard 标量标签、111 个已写入样本全部有限，包含 loss 和梯度范数。
- 初始速度约 4.0–4.2 次环境迭代 / 秒。训练回报是已结束短回合的均值，早期波动及想象回报不代表长动作跟踪成绩；尚未做完整评测。

### 首个检查点确认（上海时间约 18:30）

- 训练持续推进至 `it1100 / updates424`，后台进程仍在运行。
- `ckpt_0001000.pt` 与 `ckpt_latest.pt` 均已保存；编号文件大小 **215430607 字节，约 205.45 MiB**。
- 在 CPU 上读取已完成写入的编号文件，确认 `iteration=1000`、`num_updates=384`，三个模型的全部状态张量均为有限值。该检查不等于恢复训练验证。
- 截至 it1100，37 个 TensorBoard 标量标签、814 个已写入样本全部有限。
- it1000 日志：训练回报 0.344，WM 1 步 / 8 步误差 0.0300 / 0.0814；it1100 为 0.347、0.0307 / 0.0721。训练回报未呈持续上升，不据此宣称收敛；后续仍需长段评测。
- 启动阶段以约 5 秒间隔采样，18:25:50–18:29:57 的 50 个样本中，训练进程显存最高 **20472 MiB，约 19.99 GiB**；整卡最高 21438 MiB，包含桌面等占用。系统可用内存最低约 157.12 GiB，磁盘剩余约 146 GiB。
- 上述采样从启动约 44 秒后开始，离散查询可能漏掉短时尖峰，只代表该观察窗口，不代表完整 6 小时峰值。
- 检查点和机器日志保留在本机，训练结束与全 40 段评测结果尚未产生。

## 用户要求改为累计约 1 小时保存停止

用户于上海时间约 18:43 要求“训练 1h 就停下来保存 ckpt”。这里按本轮累计时长处理，不是从该消息起再训练一小时。

- 以原派发时间 18:25:06 加 1 小时作为外部截止边界，即 **2026-10-23 19:25:06**。
- 当前训练已加载内部 6 小时配置，修改磁盘上的 `config.json` 不会改变运行中对象，因此增加独立的 `stop_after_checkpoint.py`，不重启训练、不清空 replay。
- 到点等待首个新编号检查点完成写入；通过后续日志迭代和 CPU 读取核对其完整性、迭代数、运行名及模型有限值，先原子归档为本机 `logs/lafan_paper_s0_20261023/ckpt_1h.pt`，再请求停止该运行的 worker。
- 正常每 1000 次迭代约 4 分钟，可能晚于 19:25 几分钟结束。新检查点等待宽限为 300 秒；若异常超时，保留最近可读的完整编号检查点并停止，实际采用的迭代与回退原因写入 `stop_status.json`。
- 使用绑定进程身份的 Linux pidfd 发送 SIGTERM，worker 会转发至 uv / Python；确认已捕获的三层进程都退出。不会因为同名程序而影响其他任务，也不使用全局进程名批量终止。
- 该停止方式可能产生非零训练退出码；这是用户要求的主动停止。原始生命周期状态和退出码保留，是否成功归档 / 停止以独立的 `stop_status.json` 为准；不把它冒充训练自然跑完。
- 控制器仅在停止时使用 CPU 检查检查点，不执行额外 GPU 评测。未来直接启动新的一小时训练，可在启动参数中设置 `--schedule.max-hours 1`，无需使用这个运行中追加控制器。

验证结果：本机真实运行的 `--dry-run` 通过，确认截止 UTC 为 `11:25:06`、进程身份及 pidfd 后备接口可用，没有发信号或改训练状态。独立测试进程链与小型 CPU 检查点的三组验证全部通过：正常到点归档停止、等待超时回退、非有限模型归档并报告异常后停止。均未向真实训练发送信号。新增脚本 Ruff 和空白检查通过。

```bash
env -u PYTHONPATH -u PYTHONHOME uv run --locked python scripts/local/stop_after_checkpoint.py \
  --run-dir logs/lafan_paper_s0_20261023 --hours 1 --dry-run
```

控制器将在本批代码提交后独立后台派发；派发状态随后追加。本段为截止安排，尚不代表训练已经停止或 `ckpt_1h.pt` 已生成。

## 评测与恢复约定

本轮结束后使用稳定的编号检查点或已停止写入的 latest 做全 40 段评测。训练中不并行评测，以免占用同一张 GPU、改变本轮时间预算。当前 benchmark 的 `norm_return` 使用实际执行步数作分母；报告须同时列失败数，并另外核对整段 `frames-1` 分母结果，不能直接将其称为严格论文分数。

```bash
env -u PYTHONPATH -u PYTHONHOME uv run --locked python scripts/eval_benchmark.py \
  --ckpt logs/lafan_paper_s0_20261023/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 --num-envs 40
```

现有检查点不保存 replay 或仿真状态，恢复不等于逐步完全接续。后续若恢复编号 N 的检查点，需要另建运行名、使用相同动作集和观测配置，并将 `--schedule.warmup-iterations` 设为 `N+40`，先重新填充 replay；该恢复路径尚未做真实训练的独立进程验证。

## 训练后如何查看效果

本轮无需额外提供参考视频。LAFAN1 NPZ 已有完整三维参考动作，可以直接比较目标轨迹与策略实际控制的 G1：

1. 对全部 40 段从首帧进行整段评测，报告失败数、执行步数 / 总动作长度、失败时刻和跟踪回报；不能仅用训练时随机 10 秒片段的回报判断效果。
2. 使用 `scripts/play.py` 查看三维回放，默认半透明机器人为参考动作、实心机器人为策略实际动作。重点看走路、舞蹈、跳跃的姿态偏差、脚滑、漂移和摔倒；较差片段也须保留展示。
3. 必要时使用 `scripts/diag_drift.py` 补充位置和朝向漂移诊断。当前回放入口支持本机窗口或 viser 浏览器，尚未提供直接导出 MP4 的参数；本机图形回放和视频导出尚未验证。

例如，本轮训练结束后查看舞蹈：

```bash
env -u PYTHONPATH -u PYTHONHOME uv run --locked python scripts/play.py \
  --ckpt logs/lafan_paper_s0_20261023/ckpt_latest.pt \
  --clips dance1_subject1 --viewer viser
```

当前 40 段全部参与训练，再评测这 40 段衡量的是训练动作覆盖率与控制跟踪质量；未见动作泛化需另设未参与训练的数据。若后续希望测试用户指定的视频，需要先得到三维人体动作并重定向到 G1，转换成参考动作数据；当前策略不直接读取视频像素。
