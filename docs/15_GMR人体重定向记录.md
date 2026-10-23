# GMR 人体到 G1 重定向记录

## 本轮完成范围

2026-10-23，已经实际跑通 Ollie 视频人体前端输出 → 原版 GMR CPU 逆运动学 → 项目约定的 G1 29 关节顺序 → 50 Hz 机器人参考。未调用 GPU、未创建窗口或 GL 上下文、未改训练环境和 GVHMR 环境。

这一步提供机器人运动学初始化。它没有求解足板接触力，没有让机器人通过关节控制完成 Ollie，也没有训练联合人板 SuperMimic 世界模型或策略。已有 LAFAN 权重不能直接解释为已学会滑板。

实现入口为 `scripts/video/retarget_gmr.py`。原版 GMR 克隆保存在被 Git 忽略的 `references/GMR/`，固定提交 `25f56db79ac411258b377b6d4ccba0c8ff3372ac`，只检出算法、脚本和 G1 资产。项目适配入口不改上游 IK 实现。

官方依据：

- [GMR 仓库](https://github.com/YanjieZe/GMR)。
- [固定版本的 GVHMR 适配入口](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/scripts/gvhmr_to_robot.py)。
- [GVHMR 坐标转换和人体高度经验式](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/utils/smpl.py)。
- [G1 缩放和 IK 目标配置](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/ik_configs/smplx_to_g1.json)。

## 独立运行环境

`references/GMR/.venv` 使用 Python 3.12.13，只安装 CPU IK 所需依赖。GMR 顶层包会自动导入 viewer 和 Torch；本项目入口使用标准 Python 模块加载机制，直接加载未修改的 `motion_retarget.py` 和相对导入的 `params.py`，避免这些不需要的导入。该路径不需要 Torch、SMPL-X 模型重复计算或任何渲染依赖实例化。

可复现安装命令：

```bash
git clone --filter=blob:none --no-checkout https://github.com/YanjieZe/GMR.git references/GMR
git -C references/GMR sparse-checkout init --cone
git -C references/GMR sparse-checkout set general_motion_retargeting scripts assets/unitree_g1
git -C references/GMR checkout --detach 25f56db79ac411258b377b6d4ccba0c8ff3372ac
uv venv references/GMR/.venv --python .venv/bin/python
uv pip install --python references/GMR/.venv/bin/python --no-config \
  --default-index https://pypi.org/simple \
  mink==1.3.0 mujoco==3.13.0 numpy==1.26.4 scipy==1.15.3 \
  qpsolvers==4.13.0 daqp==0.9.1 rich==15.0.0
```

不直接复用 GVHMR 环境的原因是：该环境的 SciPy 1.13.1 不支持现版 GMR 使用的 `Rotation(..., scalar_first=True)`；其 NumPy 1.23.5 也需保留给前端旧依赖。独立环境避免改动这些版本。完整间接依赖可用以下命令读取：

```bash
uv pip freeze --python references/GMR/.venv/bin/python
```

## 输入、坐标与缩放

输入为已实际恢复的 `logs/ollie_gvhmr/ollie/body_world_raw.npz`，SHA-256 为 `e1be98c17d0ca6981d2b9a9b7e8afa1781cb0144fe1662322940d4073c192137`，包含 233 帧、30 Hz 的 22 个 SMPL-X 身体关节位置和全局旋转。22 个关节覆盖 G1 IK 配置的全部人体目标，因而无须再次运行 SMPL-X。

适配入口检查位置与旋转维度、有限值、旋转矩阵正交性、正行列式、递增且均匀的展示时间轴；输入帧率读取 `mocap_framerate=30.0`，避免六位小数 PTS 舍入导致错误的 `30.0003` 帧率。循环从第 0 帧开始，全部 233 帧保留。

两条分支分别保存，不相互覆盖：

1. `raw_global` 使用 `joints_world` 与 `joint_rotmat_world`，按 GMR 官方 helper 做 `Rx(+90°)`，即 `(x,y,z) → (x,-z,y)`，从 GVHMR Y-up 转为 Z-up。这仅改变坐标基，不修复 GVHMR 全局漂移。
2. `common_world` 使用 `joints_camera` 与 `joint_rotmat_camera`，只应用 `data/ollie/common_world.json` 提供的一个固定相机到 Z-up 世界旋转和平移，不再叠加上述旋转。该 JSON 来源是准备阶段重力方向均值、图像板端人工标注和假设板长 0.8013 m，属于单目工程校准。逐帧相机到 GVHMR 世界的变换不够恒定，不能当作独立标定的静态相机外参。

固定变换 JSON 的契约为 `rotation_camera_to_world`：一个正交且行列式为 +1 的 3×3 矩阵；`translation_camera_to_world`：长度 3 的米制向量，可省略为零。变换文件和完整内容、SHA-256 会写入重定向报告。

GMR 会缩放人体根位置，以及相对根的各部位位置；它没有统一的单一全身缩放系数。必须把缩放记录传给后续人板几何校正：

- 原始分支按 GMR 的 `1.66 + 0.1 × beta[0,0]` 经验式得到人体高度 1.6752367 m。这不是实测真人身高；`height_ratio=0.9306871`，根、腿、脊柱比例为 0.8376184，上肢比例为 0.7445497。
- 共同世界分支使用 `--human-height 1.732`，来源是 SMPL 静态高度估计，仍不是已知真人身高；`height_ratio=0.9622222`，根、腿、脊柱比例为 **0.866**，上肢比例为 0.7697778。
- GMR 会连同固定变换的平移一起缩放根位置，板参考不自动跟随缩放。因此，这一步输出的人与米制板几何尚需后续共同校正，不能直接把脚位当作已落在板面。

`offset_to_ground=False`、`ground_offset=0`。没有逐帧最低脚高度归零，没有锁地，没有压平板腾空；原版关节限位保留，沿用原版默认的无速度限位 IK。时间连续性和动力学可行性需另行检查。

## 实际运行命令与输出

原始全局分支：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
references/GMR/.venv/bin/python scripts/video/retarget_gmr.py \
  --body logs/ollie_gvhmr/ollie/body_world_raw.npz \
  --out-dir logs/ollie_gmr/raw_global --time-scale 1.5 --output-fps 50
```

共同固定坐标分支：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
references/GMR/.venv/bin/python scripts/video/retarget_gmr.py \
  --body logs/ollie_gvhmr/ollie/body_world_raw.npz \
  --out-dir logs/ollie_gmr/common_world --time-scale 1.5 --output-fps 50 \
  --human-height 1.732 \
  --joints-field joints_camera --rotations-field joint_rotmat_camera \
  --camera-world-transform data/ollie/common_world.json
```

两条命令均已真实完成，单次 CPU IK 与输出处理约 0.3 秒，不含 Python 导入时间。每个输出目录包含：

- `gmr_native_30hz.npz`：233×36 原始 MuJoCo qpos；根 xyz 米、根四元数 **wxyz**、原模型具名 29 关节弧度；保留展示时间、最近原视频帧索引和 IK 误差。
- `g1_50hz.csv`：258×36，无表头；根 xyz 米、根四元数 **xyzw**、按 `G1_JOINT_NAMES` 排列的 29 关节弧度，符合本项目 CSV 输入契约。
- `g1_robot_50hz.npz`：258×71 `robot_state`，以及项目顺序 qpos、两条时间轴、关节名和报告。71 维顺序为根 xyz、根 wxyz、29 关节位置、根世界线速度、根机体系角速度、29 关节速度。
- `report.json`：输入 hash、上游提交、变换来源、完整部位缩放、输出尺寸、关节限位误差和局限。

此处 `time_scale=1.5` 仍是用户估计的慢放倍率假设，定义为 `t_video = 1.5 × t_physics`。原始最后展示采样时刻为 7.716042 s，视频容器时长不等于最后有效姿态时刻；50 Hz 物理网格保留 258 帧，末帧 5.14 s。速度基于物理时间重算，不把 30 Hz 展示帧误当 30 Hz 真实运动速度。归一化前端的 233 帧和末尾展示 hold 保存在原始输出里，最终 50 Hz 输出不延伸到原视频最后姿态之后。

## 已验证结果与后续边界

已在 CPU 核验两分支：所有数组有限、根四元数单位范数、CSV/NPZ 四元数转换一致、29 关节名称和顺序一致、258 帧时间轴严格为 `arange(258)/50`。用本项目实际 G1 模型核对关节限位，最大误差不超过浮点舍入量级。入口 `--help` 与 Ruff 检查通过。

原始全局分支根 z 约为 -0.479～-0.179 m，反映未做地面对齐，不能直接放入地面场景。共同世界分支根坐标范围约为 x[-0.359,-0.0059]、y[-0.968,0.0259]、z[0.744,1.197] m；其中仍包含单目漂移及原版缩放后的位移。共同分支最大关节速度约 **27.8 rad/s**，说明逐帧 IK 的时间连续性尚不足，关节位置合法并不意味着能在真实力矩/速度范围内跟踪。

后续由统一人板参考阶段进行：准备期静态根与板对齐、脚底到板面的接触 IK、腾空阶段保留相对姿态和离板事件、着陆阶段连续接触约束、物理时间上的速度/加速度平滑和模型限位复核。平滑不能把腾空和落板事件平均掉。完成后还须运行真实接触动力学控制、记录尾部触地和轮子离地、腾空和着陆，并与运动学参考视频区分。

本脚本输出的是机器人参考初始化 NPZ，尚不是 `JointReference` 联合人板文件，也尚未包含 `MotionLoader` 的全部 body FK 字段。后续将板状态、接触标签、阶段和可信度加入联合参考；如需机器人单独的原训练入口，可在主环境调用既有 `convert_csv_to_npz(..., input_fps=50, output_fps=50)` 生成兼容数据，不能把这一步转换当作滑板策略训练完成。
