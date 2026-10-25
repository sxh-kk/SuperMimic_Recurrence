# GitHub 滑板 Demo：建模与人板接触借鉴

2026-10-25。用户要求借鉴GitHub滑板demo的建模与处理方式，**控制算法保持SuperMimic，不使用PPO替代**。本轮查了实际配置、模型文件与运行控制路径，不把README里的滑板图像直接当作Ollie成功证据。GitHub连接器搜索超时，随后通过网页搜索、官方仓库及固定版本源码完成核验。

## 推荐参考与用途

| 项目 | 值得借鉴 | 对当前Ollie的边界 |
| --- | --- | --- |
| [HUSKY](https://github.com/TeleHuman/humanoid_skateboarding) | MuJoCo人形滑板、板侧倾与前后桥转向耦合、被动轮、接触分类 | 公开示例以推行/转向为主，板面平直；不能直接提供拖鼻起跳解 |
| [HAIC/HOIC](https://github.com/Cybercal/HOIC-baseline) | 独立自由板、四被动轮、box鞋碰撞及人板联合状态接口 | board6是上板/滑行/下板，无板腾空；不是Ollie专家 |
| [dancher00/skateboard](https://github.com/dancher00/skateboard) | CAD分部质量惯量、真实倾斜kingpin轴与轮子层级 | 宽短板，尺寸不符本片；bushings/材料未给完整标定 |
| [ollie-optimization](https://github.com/mechmotum/ollie-optimization) | 翘尾几何、尾触地角、分质量惯量和分阶段接触力学 | 二维点质量/直接足力控制，不能把它的力注入G1或替换SuperMimic |
| [DHAL](https://github.com/UMich-CURLY/DHAL) | 轮/板桥与被动弹簧阻尼结构例子 | 实际默认板绑在一只脚上且有额外推进力；排除该处理方式 |

## HUSKY：不止一个“平板加四轮”

固定版本`d93833e80deff7f927c0b80ef9c435d8b5c488fe`，已核对远端HEAD相同。关键文件为[原始板XML](https://github.com/TeleHuman/humanoid_skateboarding/blob/d93833e80deff7f927c0b80ef9c435d8b5c488fe/src/mjlab_husky/asset_zoo/robots/skateboard/xmls/skateboard.xml)、[实体配置](https://github.com/TeleHuman/humanoid_skateboarding/blob/d93833e80deff7f927c0b80ef9c435d8b5c488fe/src/mjlab_husky/asset_zoo/robots/skateboard/g1_skater_constants.py)和[实际导出场景](https://github.com/TeleHuman/humanoid_skateboarding/blob/d93833e80deff7f927c0b80ef9c435d8b5c488fe/test_scene/mjlab_scene.xml)。

- 一个自由板根，另有板面侧倾joint、前后桥转向joint和四轮hinge；板面平box为`.8×.2×.02 m`，轮半径`.03 m`、宽`.04 m`、轴距`.4 m`。没有独立翘鼻/尾碰撞面。
- 板面绕局部−X侧倾，桥joint沿±Z转向，两个joint equality使用`polycoef=[0,−.577,0,0,0]`联系侧倾和转向。前后实际反向转向还来自其相反轴向，不能只抄一个系数忘记坐标系。该关系是转向近似，不是尾部弹性。
- 配置将脚和板碰撞priority均设1，脚μ=1，轮condim=6、friction=`[1,.01,.001]`。原始资产XML和配置处理后的demo不是相同的碰撞参数；必须看最终编译模型，而非只搜到原始XML就照抄。
- 导出场景有7个板actuator：侧倾和两桥以零目标PD实现回中弹簧阻尼，四轮gain为0。demo主循环只更新机器人`ctrl[:-7]`，没有逐帧板位姿覆盖或板root wrench；这些零目标回中项不是主动轮推进。但它也不是本项目“板actuator数量为0”的同一实现，借鉴时应将bushing写成明确的被动弹簧阻尼，并核算能量。

### 编译后质量的重要发现

新增`scripts/local/audit_open_skateboard_models.py`实际编译HUSKY资产和demo。两者板子树质量均为**6.95 kg**，不是看到root `<inertial mass=1.5>` 就能称整板1.5 kg：root1.5、两桥合计1、四轮合计0.8，另有`board_tilt_body`的两个box按默认density推得3.65 kg。这个值是该快照编译结果，不是对其实物质量的判断，更不是本片板的标定。

同一只读审计得到当前板2.18 kg、HOIC独立板子树2.802 kg；记录每个body的质量/COM/惯量、joint阻尼/armature/摩擦、几何材料和actuator。HOIC完整机器人缺部分mesh，故只在保留原default/材料的情况下编译其未修改板子树，不假称跑通完整策略。结果保存于本机`logs/open_skateboard_model_comparison_01/result.json`，没有改变canonical板。

首次隔离编译漏带groundplane材质引用的procedural texture而报错；补齐原texture元数据后编译通过，没有进行动力学或改参数。脚本和所有原文件SHA可复核，第三方源码/资产仍留在ignored references。

## 怎样用于我们当前的模型

1. **保持自由板和可分离鞋板接触。** 当前模型这一点正确；不引入绑脚、weld、吸附、直接加跳跃速度或给板注力。
2. **把材料按接触角色区分。** 足板、尾地、轮地分别检查实际生效的摩擦与柔顺性。已发现并修复的priority覆盖问题继续用显式足板pair处理；μ=.9仍是工程假设，不能因为别的demo用1或2就直接增大到“会跳”为止。
3. **优先核对Ollie需要的鞋—鼻/尾几何。** 当前12°翘面与7胶囊脚、HOIC的平板/box鞋、Ollie力学示例20°翘面明显不同。下一项应对照原片/可用CAD检查真实翘面、过渡曲面和鞋边接触见证；不能把圆滑capsule、尖锐box角或游戏绑脚看作同一个接触模型。当前尺寸/12°尚未实物标定，不在本轮盲改成20°。
4. **借鉴质量惯量的完整编译检查。** 本轮已落地参数审计入口，避免漏算子body、默认density和关节armature。板参考图像相似不代表惯量正确。
5. **转向扩展采用可解释的被动板桥。** HUSKY的侧倾—转向近似和CAD倾斜kingpin可作两种候选；本片是近原地Ollie，当前固定轴的简化不能独立解释为何前脚没有卸载，也不先用新增转向自由度掩盖这个控制缺口。后续若加入3个板关节，联合state92、参考格式和WM维度都须同步扩展，旧权重不能直接复用。
6. **人板处理进入SuperMimic的真实数据。** 本轮增加的400 Hz接触测量已能核对实际上表面支持、尾地事件、nose力、9几何净空及COM；用于联合WM辅助监督与真实动作响应检查，再验证BPTT梯度。它不改变控制算法，也不把接触力直接写回仿真。具体结果和跨运行数值限制见[68](68_GPU真实接触观测与SuperMimic数据核验.md)。

当前结论是：可借鉴的机械结构和处理方法确实存在；现成滑行demo不能替代原视频Ollie的接触验证。本轮没有替换canonical几何/质量、开始新训练或宣称完成Ollie。后续模型变更须固定profile、先做被动动力学/鞋鼻接触检查，再重新采集对应SuperMimic训练数据。

以下为其它官方实现的逐行核验。

## DHAL：可参考被动轮/板桥，不能迁入其绑脚和辅助推进

核验官方 [UMich-CURLY/DHAL](https://github.com/UMich-CURLY/DHAL)，固定 commit `62a08b292c0d4349026f1e52992a45a956a9d9a1`。没有安装 Isaac Gym、执行策略或改变当前 SuperMimic 控制算法。

**关键区别是实际入口，而不是仓库里恰好存在的独立板文件。**[go1_config.py:326–328](https://github.com/UMich-CURLY/DHAL/blob/62a08b292c0d4349026f1e52992a45a956a9d9a1/legged_gym/legged_gym/envs/go1/go1_config.py#L326-L328) 实际加载组合 `go1.urdf`。其中 [1030–1071 行](https://github.com/UMich-CURLY/DHAL/blob/62a08b292c0d4349026f1e52992a45a956a9d9a1/legged_gym/resources/robots/go1_skate/urdf/go1.urdf#L1030-L1071) 用 X/Y/Z 三个 revolute 串接 `2_FL_foot → skateboard_joint_link_x → skateboard_joint_link_y → skateboard_deck`，三者 origin 都为零、damping 均 .5。因此板可绕脚转动，但锚点平移永久跟随前左脚；**不是独立自由板，不能实现要求鞋板分离的 Ollie**。不是 MuJoCo weld，但其平移绑定同样不能照搬。

每个物理子步 [legged_robot.py:142–147](https://github.com/UMich-CURLY/DHAL/blob/62a08b292c0d4349026f1e52992a45a956a9d9a1/legged_gym/legged_gym/envs/base/legged_robot.py#L142-L147) 都调用 `_against_rolling_friction()`。[965–971 行](https://github.com/UMich-CURLY/DHAL/blob/62a08b292c0d4349026f1e52992a45a956a9d9a1/legged_gym/legged_gym/envs/base/legged_robot.py#L965-L971) 的实际公式是局部 X 速度 >.3 m/s 时施加 +10..25 N、<−.3 时施加 −10..25 N，并 `apply_rigid_body_force_tensors(...LOCAL_SPACE)`。**该力与速度同号，属于持续辅助推进，函数名不能当作“被动滚阻”证据。**不把它迁入本项目，也不拿该 demo 作为无板外力成功的证明。

实际组合模型的可查参数：

- [deck:878–896](https://github.com/UMich-CURLY/DHAL/blob/62a08b292c0d4349026f1e52992a45a956a9d9a1/legged_gym/resources/robots/go1_skate/urdf/go1.urdf#L878-L896)：visual `.75×.23×.02 m`，collision `.75×.25×.02 m`，平 box，无翘尾；质量 1 kg；COM origin 为零，而 collision origin 是 `[-.2,−.07,−.04]`，惯量对角均 `.01 kg m²`，不能直接称实测分布。
- [truck:1074–1119](https://github.com/UMich-CURLY/DHAL/blob/62a08b292c0d4349026f1e52992a45a956a9d9a1/legged_gym/resources/robots/go1_skate/urdf/go1.urdf#L1074-L1119)：每个质量 .5 kg，X 轴 roll，范围 ±.05 rad。它们是绕板 X 的 roll，并没有独立建出真实倾斜 kingpin 的 roll/yaw 几何耦合。
- [config:318–349](https://github.com/UMich-CURLY/DHAL/blob/62a08b292c0d4349026f1e52992a45a956a9d9a1/legged_gym/legged_gym/envs/go1/go1_config.py#L318-L349)：truck 是 underactuated、零目标被动 PD，k=100、d=10；wheel 和三个板锚关节不主动驱动。这可作为**被动弹簧阻尼结构的实现例子**，不能把数值当本机板的标定。
- [wheel:1123–1244](https://github.com/UMich-CURLY/DHAL/blob/62a08b292c0d4349026f1e52992a45a956a9d9a1/legged_gym/resources/robots/go1_skate/urdf/go1.urdf#L1123-L1244)：圆柱半径 .03 m、宽 .044 m，每轮 .2 kg、Y 轴转动。原始部分惯量矩阵有零特征值，运行端又 `recomputeInertia=True`（legged_robot.py:1103），因此不能把 URDF 原值当运行时最终惯量。
- [材料随机化:499–512](https://github.com/UMich-CURLY/DHAL/blob/62a08b292c0d4349026f1e52992a45a956a9d9a1/legged_gym/legged_gym/envs/base/legged_robot.py#L499-L512) 给全部 shapes 同一环境 friction；[config:385–387](https://github.com/UMich-CURLY/DHAL/blob/62a08b292c0d4349026f1e52992a45a956a9d9a1/legged_gym/legged_gym/envs/go1/go1_config.py#L385-L387) 范围 .6..2。四个硬编码 wheel shape 的 rolling_friction 被置零。没有可直接迁入的足板分材料标定。
- [脚接触分类:719–738](https://github.com/UMich-CURLY/DHAL/blob/62a08b292c0d4349026f1e52992a45a956a9d9a1/legged_gym/legged_gym/envs/base/legged_robot.py#L719-L738) 主要是足点到 marker 的 3 cm 距离阈值，不是本项目原生 pair/向上支持力认证，不能照搬成功标签。

仓库另有 `skateboard.urdf`（deck `.7×.25×.02`、独立树根、4 轮+2 roll）；**实际 go1 配置未加载它**。完整 tree 只有 `rsl_rl/LICENSE`，未看到覆盖 robot asset 的顶层许可证，不将子库许可证泛化到整个模型。

## 可下载的独立 CAD 板：dancher00/skateboard

官方 [dancher00/skateboard](https://github.com/dancher00/skateboard)，commit `bedcc7c70905fb3a86b42192da0e3de5ad32d95a`。提供 SolidWorks `.SLDASM/.SLDPRT`、7 个 STL、URDF；[quadruped-skateboard-mounting](https://github.com/dancher00/quadruped-skateboard-mounting) README 也直接链接该来源。这是机械模型候选，**不是已验证的 G1 Ollie demo**。

本轮实际下载 URDF 和 deck STL，不只核对目录：

- [robot.urdf:7–45](https://github.com/dancher00/skateboard/blob/bedcc7c70905fb3a86b42192da0e3de5ad32d95a/robot/urdf/robot.urdf#L7-L45)：deck/base 质量 `1.8204384946 kg`，惯量 `Ixx=.0083867461,Iyy=.0436475937,Izz=.0519774548 kg m²`（交叉项也保留）；所有 link 显式质量和 `2.1261741810 kg`。deck STL 未设 scale，读取顶点包络为 `.575×.25×.07142354 m`，**Z 包络包含底部结构，不能把它当板厚或翘尾高度**；作者整体 CAD 图看起来也是宽短板，不能默认与原视频 .8013 m 板相同。
- [trj0:87–104](https://github.com/dancher00/skateboard/blob/bedcc7c70905fb3a86b42192da0e3de5ad32d95a/robot/urdf/robot.urdf#L87-L104)、[trj1:251–268](https://github.com/dancher00/skateboard/blob/bedcc7c70905fb3a86b42192da0e3de5ad32d95a/robot/urdf/robot.urdf#L251-L268)：轴中心 X=±`.2484512384 m`；安装旋转 Y=`.3804442639 rad`（约21.8°），局部轴 `[0,0,−1]`，关节范围 ±`.2278 rad`。真实倾斜轴比单纯绕板 X 的转轴更值得借鉴，且前后镜像必须连同局部旋转一起转换。
- 四轮 continuous hinge，局部轴 `[0,−1,0]`；URDF 无根到机器人连接、无 actuator/外力控制脚本，也未写轮地/足板摩擦。
- [README:1–3](https://github.com/dancher00/skateboard/blob/bedcc7c70905fb3a86b42192da0e3de5ad32d95a/README.md#L1-L3) 明确提示 bushings 需要额外 springStiffness 或 Gazebo torsional-spring plugin，**原 URDF 没有给出完整的已标定弹簧阻尼**；truck limit 的 effort/velocity 还填 0，导入时不能机械照搬。
- `robot/package.xml:11` 声明 BSD，顶层未附具体 BSD 条款文件；mounting 仓库顶层为 Apache-2.0，但不能据此抹去来源模型的声明。资产保持 ignored，不迁公开仓。

**可借鉴：**独立板树、四轮转动、倾斜 kingpin 轴、分 link 质量惯量及 CAD→碰撞近似的检查流程。**不能借鉴为既成事实：**翘尾/鞋边碰撞、有效足板摩擦、真实 bushings 参数、当前原视频板的质量尺寸。本轮未在 MuJoCo 编译或落体测试该资产。

## 真正围绕 Ollie 的力学源码：mechmotum/ollie-optimization

官方 [mechmotum/ollie-optimization](https://github.com/mechmotum/ollie-optimization)，commit `dd60b1984a236dd682c9677345c55a168957651f`。这是论文“Maximizing Ollie Height by Optimizing Control Strategy and Skateboard Geometry Using Direct Collocation”的公开结果源码；[README/LICENSE](https://github.com/mechmotum/ollie-optimization/blob/dd60b1984a236dd682c9677345c55a168957651f/README.rst#L1-L10) 明确代码 MIT、论文文字/图 CC BY 4.0。

`thesis-files/MultiPhaseOllieOptimization.py` 已实际下载：

- [130–140 行](https://github.com/mechmotum/ollie-optimization/blob/dd60b1984a236dd682c9677345c55a168957651f/thesis-files/MultiPhaseOllieOptimization.py#L130-L140) 的一个示例参数是轴距 `.444 m`、尾长 `.13 m`、平段 `.57 m`、翘角20°、轮半径 `.0245 m`、足板 μ=.8、人体80 kg。**这是该研究参数，不是我们实物/视频标定值**。
- [42–128 行](https://github.com/mechmotum/ollie-optimization/blob/dd60b1984a236dd682c9677345c55a168957651f/thesis-files/MultiPhaseOllieOptimization.py#L42-L128) 分材料计算 deck/轮/轴质量、COM 和平行轴惯量；[283–302 行](https://github.com/mechmotum/ollie-optimization/blob/dd60b1984a236dd682c9677345c55a168957651f/thesis-files/MultiPhaseOllieOptimization.py#L283-L302) 明确区分板 COM、后轮、尾、前脚、nose；[404 行](https://github.com/mechmotum/ollie-optimization/blob/dd60b1984a236dd682c9677345c55a168957651f/thesis-files/MultiPhaseOllieOptimization.py#L404) 从轮到尾相对几何推尾触地角。
- [187–204、450–485 行](https://github.com/mechmotum/ollie-optimization/blob/dd60b1984a236dd682c9677345c55a168957651f/thesis-files/MultiPhaseOllieOptimization.py#L450-L485) 表明人体只有 XY 点质量状态，控制变量包括直接足部法向力/切向力及足点加速度。**不是全身 G1、不是 29 个电机关节目标下涌现的 Ollie，也没有可直接加载的 MuJoCo机器人 checkpoint/URDF。**不能把优化的接触力直接注入当前板。

值得借的是**质量惯量来源、尾触地几何与分阶段接触一致性诊断**，不是替换算法。保持 SuperMimic 的 WM+BPTT；该理想化解最多提供建模/必要条件交叉检查，不能晋升成合法 G1 teacher。

另查 [MIME Lab / Elias Mikkola 项目页](https://mimelab.github.io/projects/skateboard)：确为 MuJoCo/MJPC 物理滑板动画，公开页面重点推行/转向，摘要明确 Ollie 只有有限成功；该页只链接论文而没有 GitHub 模型入口。本轮没有取得其可固定 SHA 的模型，因此不把它列为可立即迁移资产。

## 游戏 demo 的排除证据

不是把所有“physics skateboard”搜索结果都算作人板真实接触成功：

1. [Shahzaib52/Skateboard-Physics-Controller-Unity](https://github.com/Shahzaib52/Skateboard-Physics-Controller-Unity)，commit `dce5d5d4b3c1ade4a12fcf6ea9532a5f5144a5f3`。唯一 controller 的 [42–47 行](https://github.com/Shahzaib52/Skateboard-Physics-Controller-Unity/blob/dce5d5d4b3c1ade4a12fcf6ea9532a5f5144a5f3/Assets/Skateboard%20Physics/Scripts/SkateboardController.cs#L42-L47) 直接 AddForce 推刚体，[64–87 行](https://github.com/Shahzaib52/Skateboard-Physics-Controller-Unity/blob/dce5d5d4b3c1ade4a12fcf6ea9532a5f5144a5f3/Assets/Skateboard%20Physics/Scripts/SkateboardController.cs#L64-L87) 依据 raycast normal 直接写 skateboard.localRotation；未实现全身脚施力 Ollie。排除作为无板控制的参考。
2. [StarKnightt/skate-park](https://github.com/StarKnightt/skate-park)，commit `ca3a7153d21e118842c5452e10e4c681dfd0f8f8`。`src/player/physics.ts:368` 自称 kinematic controller；[856–865 行](https://github.com/StarKnightt/skate-park/blob/ca3a7153d21e118842c5452e10e4c681dfd0f8f8/src/player/physics.ts#L856-L865) 的 pop 直接 `vel.addScaledVector(dir,popV)` 赋予跳跃速度；[skater.ts:120](https://github.com/StarKnightt/skate-park/blob/ca3a7153d21e118842c5452e10e4c681dfd0f8f8/src/player/skater.ts#L120) 明示 procedural IK feet glued to deck。虽然有脱板后的刚体翻滚实现，骑行/Ollie 路径仍不能作为本项目无 reset/无板冲量/可分离接触的验证。

## 本节证据留存

所有来源只下载到 ignored `/home/zxc/SuperMimic/references/skateboard_model_research_20261025/`，commit、关键文件 SHA256、CAD 关节/质量摘要在 `manifest.json`。主要哈希：

- DHAL 实际 `go1.urdf`：`a7b61a001bb2b9351822a9188d1927748f63590c9765223b06d402f8b0a45a80`。
- DHAL 独立 `skateboard.urdf`：`a95a81ac3db2fa27fe7da5ed59d00118fbe7ff370ae393d31082706b01115013`。
- DHAL `legged_robot.py`：`a5bf09bed0afc5b59ae8cdbd4a1124fe504c093c857f5776014857011825218a`。
- CAD `robot.urdf`：`e96100d472cb0ff10e58aea93c489ed2bba8272c09d9c21452d178bfe3bd7f7f`；deck STL：`de4d3ee76220866d74a69de12cabf1382f7ed7a36b763a6b6b01b1a41c61d096`。
- Ollie 力学脚本：`3086826d8fa9ce29cb99fd5fd4da864e0e358d3145732c38b9d92ffed521c463`。

本节仅支持从机械结构/几何证据中借鉴，不启动训练、不给板施额外力、不把板绑定脚、不更换 SuperMimic，也不把第三方 demo 的图像好看或“physics”名称当成已通过当前严格 Ollie 验收。
