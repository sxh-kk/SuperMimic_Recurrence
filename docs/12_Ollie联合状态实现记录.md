# Ollie 人板联合状态与参考格式

日期：2026-10-23。本次实现独立的人板状态和参考数据基础组件，服务于 29 DoF G1、被动板和四个无驱动轮的物理控制。原平地 `StateIO`、71 维状态、世界模型、策略和训练入口均未改动。新增组件不代表已训练联合世界模型，也不代表已有 Ollie 控制器。

## 已实现接口

- `src/supermimic/ollie/state.py`：`JointStateLayout`、按 MuJoCo 名称 / 地址映射的 `JointStateIndices`，以及可微的 `JointStateSpec`。
- `src/supermimic/ollie/reference.py`：`JointReference` 的 NPZ 读写、版本 / 数值 / 时间轴校验和 `resample_reference()`。
- `tests/test_ollie_state.py`：仅使用 CPU 的地址映射、积分、误差、不变性及参考数据测试。

### 状态排列与坐标

联合状态按实体拼接，不是把整个场景的全部 qpos 后面直接接全部 qvel：

```text
robot_state[71] | board_state[13 + 2K]

每个实体：
position_world[3] | quaternion_wxyz[4] | joint_position[K]
| linear_velocity_world[3] | angular_velocity_body[3] | joint_velocity[K]
```

机器人 K=29，关节次序保持 `G1_JOINT_NAMES`。板的 K 和关节名称显式记录：纯自由板总状态 84；四轮板 92；包含七个内部单轴关节的另一板模型名义状态为 98。四元数均为 wxyz，使用米、弧度、秒；两实体位置共同减去相同的环境原点。根角速度使用 MuJoCo 的 body frame，不能把它直接当世界角速度。

当前与物理组件约定机器人自由关节为 `floating_base_joint`，板自由关节为 `board_free`；四轮顺序为：

```text
wheel_front_left_joint
wheel_front_right_joint
wheel_rear_left_joint
wheel_rear_right_joint
```

`JointStateIndices.from_model(model, board_joint_names=...)` 从模型读取地址；`pack(qpos, qvel, origins=None)` 与 `unpack(state, origins=None, qpos_template=None, qvel_template=None)` 完成双向转换。额外场景自由度不属于这两个实体时，unpack 应提供模板以保留它们；不提供模板时其余量初始化为零。

板关节状态属于模型状态，并不意味着它们被驱动。机器人 action 仍应为 29 关节的控制命令。状态中的关节顺序与模型 actuator 顺序是不同契约；写入控制前应使用物理组件提供的 actuator 地址映射，不能把 qpos 地址直接用作 ctrl 下标，不能假定旧策略输出与新控制接口天然同序。

### 可微积分、特征和误差

`JointStateSpec(layout, cfg=None, device="cpu", dtype=torch.float32)` 支持任意前导批量维度：

- `canonical(state)`：机器人原有 71 个特征，附加板相对机器人位置、去除共同世界 yaw 后的板旋转、板速度和内部关节量。四轮板共 94 个特征；同时旋转 / 平移整个人板系统时特征不变。
- `integrate(state, delta)`：机器人部分复用原 70 维更新，板部分另建速度增量、显式位置 / 旋转积分和小位置修正。四轮板总 delta 为 90 维。零 delta 表示按现有广义速度外推；这只是运动学先验，不是无力自由飞行或接触求解器。
- `error_vector()`、`normalized_sq_error()`：旋转使用 SO(3) 误差，接受等价的正负四元数；其余量按显式尺度归一化。

板的归一化和输出尺度是工程初值，不是论文原参数或经训练验证的最优值。积分不投影 equality 耦合约束；以后引入 truck 耦合时，必须增加相应约束检查 / 投影。当前四轮模型没有 truck，不应把此接口视为已解决倾斜转向动力学。

## 参考 NPZ 契约

格式标识为 `supermimic.ollie.joint_reference.v1`，状态标识为 `supermimic.ollie.entity_state.v1`，不兼容旧的单机器人 LAFAN1 NPZ。读取使用 `allow_pickle=False`。

必备数组为 `time_video[T]`、`time_physics[T]`、`robot_state[T,71]`、`board_state[T,13+2K]`、`phase[T]`、`contacts[T,C]`、`confidence[T]`；另保存机器人 / 板关节名称、阶段 / 接触通道名称、四元数与单位约定和 JSON 元数据。

阶段默认是 prepare / crouch / pop / flight / landing / recovery，也可显式自定义名称。阶段 -1 表示未知。接触值 -1 / 0 / 1 分别表示未知 / 不接触 / 接触；默认通道包括双脚—板、双脚—地、板尾—地与四轮—地，但可显式提供其他通道。置信度范围为 [0,1]，来源和含义须写入 provenance，不能将估计值标为接触力真值。

metadata 至少包含 `time_scale`、`video_time_origin_s`、`physics_fps` 和非空 `provenance`。时间关系固定为：

```text
t_video = video_time_origin_s + time_scale × t_physics
```

物理参考从 t=0 开始，均匀采样、严格递增。使用原视频子段时，以其真实首帧 PTS 作为 video_time_origin_s；不要丢失原片时间映射。provenance 应记录实际视频 SHA256、所用模型 / 提交、估计或优化方法和置信度定义；测试中使用明确标注的 synthetic_unit_test，不将合成数组冒充已恢复参考。

`resample_reference(reference, fps=50)` 线性插值位置和内部关节，用 Slerp 插值旋转，并在**物理时间**重新求线速度、body frame 角速度和关节速度；不会保留错误的视频时间速度。阶段和接触采用左侧保持，不生成分数接触标签。轮角应预先连续化；此函数不对所有机器人关节盲目 unwrap，也不做穿板、接触或动力学修复。

最后一个时间是有姿态支持的采样点，不是容器末帧展示终点。当前视频在 s=1.5 假设下，最后 PTS 对应约 5.144028 s；50 Hz 网格截止 5.14 s，共 258 个姿态点。不补造 5.171833 s 容器展示终点的运动；如需保持末帧，应在播放层单独记录。

结构验证包括形状、有限值、四元数范数、名字顺序、标签范围及时间映射。它不证明三维恢复准确、足板接触真实或 Ollie 动态可行。

## CPU 验证

本次新增测试覆盖：板先于机器人的模型地址排列、两个实体的原点往返、与 `mujoco.mj_integratePos` 的积分对拍、84 / 92 / 98 三种维度、共同 yaw / xy 平移不变性、可微梯度、四元数正负等价、NPZ 往返、s=1.5 下重新求物理速度、未知标签保持以及错误参考拒绝。

```bash
env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest -q tests/test_ollie_state.py
```

当前结果：**9 passed**。Ruff 检查通过。原平地 StateSpec 和 GAE 两个 CPU 基线测试另有 **2 passed**。

与物理组件的实际合并模型进行 CPU 联调：nq=47、nv=45、nu=29，92 维状态 pack/unpack 往返通过，94 维特征和 90 维积分输出均有限。没有运行 GPU 联合世界模型或 Ollie 学习训练。

## 后续训练接入与物理控制诊断

原 `DynamicsMLP`、`WorldModel`、解析头、参考窗口和 imagination 并非只更换 StateSpec 就能使用：原动力学输出固定 70，解析头对 qpos/qvel 切片固定 36/35，历史固定 93×10，目标固定 68×10，终止头只读高度与倾斜。下一阶段应新增联合任务入口，保留原平地入口，并依次完成：

1. 从真实人板接触环境记录 92 维状态与实际执行动作；回放窗口和终止末行均保存板。明确新增 episode 参数和动态接触量，不能将动态板状态塞进恒定 xi。
2. 新建 94 维状态特征、29 维动作和扩展上下文输入、90 维输出的联合动力学；分踩尾、离地和落地验证多步误差，保留真实仿真对照。
3. 机器人 FK 只接 robot_state 的 71 维；另建板几何 / 接触代理、联合观测 / 目标和阶段终止。按实际 rollout 数据校验解析奖励，不能凭 FK 声称得到真实冲击力。
4. 新建联合策略、critic、目标历史和 BPTT 训练入口。旧权重只能在顺序、尺度、维度核对后做部分初始化；当前组件不提供直接兼容声明。

在模型式训练完成前，可以先在同一被动物理场景中，以少量预蹲 / 踩尾 / 收腿 / 落地时序参数优化 29 关节 PD 目标，检查动作是否可行并构造课程起点。该结果必须标为**物理控制参数优化基准**，不是已训练的 SuperMimic，也不是把参考关节逐帧强写进仿真；板仍只能由真实接触驱动。没有足够可行状态转移数据前，不能用运动学动画替代世界模型训练和无辅助物理评测。
