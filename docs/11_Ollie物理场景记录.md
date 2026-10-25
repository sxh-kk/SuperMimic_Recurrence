# Ollie 被动板物理场景：首轮实现与验证

日期：2026-10-23。用于 `video/ollie.mp4` 所示原地 / 极低速 Ollie 的独立物理分支。没有改动原平地环境、71 维 StateIO、训练 loop 或已有权重。

## 当前可用内容

- `assets/ollie/passive_skateboard.xml`：本项目新写的简化滑板，刚性板中部、翘起板头 / 板尾、固定轴架、4 个被动轮铰接、自由浮动根。
- `src/supermimic/ollie/physics.py`：原生 CPU MuJoCo 构造、29 DoF G1 组合、命名索引、诊断初始化、物理接触统计和模型导出。
- `scripts/local/check_ollie_physics.py`：CPU 落板、自由飞行、初速滚动、尾端碰撞、G1 足板接触和导出重载检查，可选 EGL 单帧预览。

这一轮完成的是**可检验的物理场景**，没有训练出 Ollie 控制器。尾端碰撞检查使用人为设定的初始角速度，不是机器人自主踩尾；初速滚动是检查初始动能下的被动滚动，不代表板有电机。

## 与 HUSKY 的关系和许可

官方 [HUSKY 仓库](https://github.com/TeleHuman/humanoid_skateboarding) 已固定克隆到忽略目录 `references/humanoid_skateboarding`，HEAD：`d93833e80deff7f927c0b80ef9c435d8b5c488fe`，未安装其独立依赖或运行其 GPU 训练。

核对了其[板模型](https://github.com/TeleHuman/humanoid_skateboarding/blob/d93833e80deff7f927c0b80ef9c435d8b5c488fe/src/mjlab_husky/asset_zoo/robots/skateboard/xmls/skateboard.xml)和 23 DoF 机器人配置。HUSKY 具有前后 truck、内部横滚—转向耦合、4 轮，服务推地 / 转向 / 阶段转换；本轮没有复制整套环境或策略。其 README / LICENSE 明确为 CC BY-NC 4.0，而 pyproject 元数据写 Apache 2.0，许可描述不一致，不能把上游代码整体改标成本项目 MIT。

本项目滑板 XML 为新写的参数化简化模型，不是 HUSKY 模型的忠实复刻。G1 直接读取本机已锁定 mjlab 的原有机器人资产、碰撞和 29 个位置执行器，原资产许可仍适用；没有将第三方 G1 网格复制到公开源目录。生成的含网格 ZIP 留在忽略的 `logs/` 下。

## 简化取舍与物理假设

板总质量为 2.18 kg；平板中央长 0.56 m、宽 0.21 m、厚 0.012 m，两端分别约 0.124 m、翘起 12°。轮半径 0.0275 m，前后轴距 0.42 m。这些是**工程初值**，还没有依据视频尺度、实物测量或接触实验完成标定。

所有板段与轴架属于同一个刚体；两端不是可转动尾铰。轮子各有一个绕板局部 Y 轴的自由铰接，无阻尼、无限位、无执行器。板根为自由关节，可整体俯仰、横滚、腾空；没有脚板焊接、磁吸、运动学轨迹牵引或对板施加的主动推力。

暂不加入 truck 转向和 HUSKY 的内部板倾斜机构：本片段是原地 / 极低速起跳，先隔离踩尾、带板和落地问题，减少未知接触与惯量参数。**这不适用于复现倾斜转弯、carving 或一般滑板运动**。需要转向时再引入并校准 truck，而不是用固定轮轴声称完整滑板动力学。

使用真实 MuJoCo 碰撞：脚—板面、尾端—地、轮—地相互独立。板头尾为多个简单凸碰撞体，避免将整块带翘曲板强行当一个凸包后把凹形支撑区域填平。摩擦、软接触参数和 0.0025 s 物理步长均只是首轮数值配置；冲击峰值、弹板高度与可迁移性仍需检验。

## 命名与状态接口

2026-10-25 建模审计补记：板XML参数不等于实际足板pair参数。G1脚priority1覆盖板priority0，原足板实际μ=0.6、solref=[0.02,1]；尾地仍为μ=0.9、solref=[0.008,1]。`make_scene_spec(foot_deck_friction=...)`现可显式指定仅足板摩擦，默认None保留旧物理，未将0.9或1.2晋升为真实材料值。实际参数读取、有限nose向上力边界与尾撞组件检查见[模型审计61](61_Ollie滑板与足板接触模型审计.md)。

G1 的名字保持原样：根 body 为 `pelvis`，freejoint 为 `floating_base_joint`，关节采用原 `G1_JOINT_NAMES`。滑板为：

- 根 body：`board`；自由关节：`board_free`。
- 轮关节顺序：`wheel_front_left_joint`、`wheel_front_right_joint`、`wheel_rear_left_joint`、`wheel_rear_right_joint`。
- 局部坐标：X 为板长 / 前进，Y 为轮轴，Z 朝上。根四元数 wxyz；根线速度 world、角速度按 MuJoCo freejoint 的 body frame。

纯板 `nq=11, nv=10, nu=0`。组合 29 DoF G1 后 `nq=47, nv=45, nu=29`；机器人原 71 维加板的 21 维，名义联合状态为 92 维。接入联合状态模块时仍按关节名字 / 地址映射。

`SceneIndex.robot_actuators` 明确按 `G1_JOINT_NAMES` 对应控制索引。mjlab 创建的执行器顺序不同于 CSV 关节顺序，不能把 29 列目标直接赋给原始 `data.ctrl[:]`。

执行器为 mjlab 原有位置 PD，`ctrl` 接受绝对目标角（rad），不是力矩或归一化动作。当前模型实测的 Kp / Kd / 力矩限制如下；没有调整原执行器增益来制造跳跃：

| 关节组 | Kp | Kd | 力矩上限 Nm |
| --- | --- | --- | --- |
| hip pitch / yaw、waist yaw | 40.179 | 2.558 | 88 |
| hip roll、knee | 99.098 | 6.309 | 139 |
| ankle pitch / roll、waist roll / pitch | 28.501 | 1.814 | 50 |
| shoulder、elbow、wrist roll | 14.251 | 0.907 | 25 |
| wrist pitch / yaw | 16.778 | 1.068 | 5 |

```python
from supermimic.ollie.physics import compile_scene, contact_summary

model, data, index = compile_scene(include_robot=True)
# root 坐标和轮关节按 index 获取；这里没有创建 CUDA 仿真。
joint_target = data.qpos[index.robot_joint_qpos].copy()
data.ctrl[index.robot_actuators] = joint_target
print(contact_summary(model, data))
```

`reset_scene()` 将机器人初始化为本机 mjlab 的屈膝诊断姿态，朝向与板长垂直，并将足底置于中央板面；没有稳定站立成功保证，更不是视频重定向后的起始姿态。后续控制器需要自己设定正确的站距、脚的板上位置和参考初始化。

当前两足 site 的板局部 X 分别为 −0.11851 / +0.11851 m，Y 约 −0.00120 m，左右间距约 0.23701 m。这个窄站距仅用于初始化和接触检查；真实踩尾前应依据视频或接触 IK 扩大站距，使后脚达到板尾。导出模型的 `init_state` keyframe 已替换为这个组合场景诊断姿态，重新加载后的 keyframe 也核验有足板接触。

## 实际验证

执行：

```bash
env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' uv run --locked \
  python scripts/local/check_ollie_physics.py --out-dir logs/ollie_physics --render
```

物理推进为 CPU MuJoCo；`--render` 只创建小型 EGL 图形预览，不启动 CUDA 学习或训练。

- 0.30 m 板根高度自由落下 2 s，最终根高度 0.073997 m，接近轮几何确定的 0.074 m；轮—地 8 个接触点（不是 8 个轮子），稳态最大穿透约 0.003 mm。着地瞬间最大穿透约 5.09 mm，仍需后续冲击标定。
- 板离地 0.8 m、向上初速 2 m/s，0.15 s 无接触自由飞行高度与解析重力轨迹差约 1.84 mm，符合本轮离散积分误差量级。
- 初速 0.6 m/s 且匹配轮速，0.75 s 前进 0.44694 m，末速 0.59187 m/s；无电机、无外力推进。
- 带初始俯仰角速度的尾部碰撞产生 38 个尾—地接触样本；瞬间最大穿透约 7.14 mm。它只证实尾部碰撞实际发生，不证明机器人能弹板或完成 Ollie。
- 组合模型准确为 47 / 45 / 29，4 个轮关节均无执行器；初始化有 10 个足—板接触点，0.2 s 位置 PD 诊断状态有限，最大穿透约 1.34 mm。短诊断不作为站立稳定性评测。
- 原生 MJCF 导出并重新加载通过；ZIP 显式打包所有 G1 网格，解压到临时目录后独立重载通过；EGL 800×600 场景预览通过。

输出：`logs/ollie_physics/checks.json`、`g1_passive_board.xml`、`g1_passive_board.zip`、`g1_passive_board.png`。MJCF 使用本机绝对网格路径，ZIP 适合保存配套模型资产；两者都留在本机忽略目录。

## 下一步物理验收

收到视频人板联合参考后，检查脚落点、板尾触地与起跳事件；对 G1 刚性足底是否能产生正确的冲量与前脚带板进行实际试验。对 0.0025 / 更细物理步长及接触参数做收敛检查，并观察能量、穿透、尾端撞地、轮接地和接板冲击。固定参考姿态直接回放属于运动学结果，必须与机器人关节驱动、被动板通过真实接触起跳的结果分开报告。
