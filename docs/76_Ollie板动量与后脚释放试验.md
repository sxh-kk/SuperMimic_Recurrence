# 76：原模型板动量与后脚释放试验

本轮是为 SuperMimic 补充真实人板动力学探索数据的有界试验。它没有训练世界模型或 actor，也不是 PPO。探索等级不代表复现成功；实际回合只使用原来的 `whole_board_geometry_top_support_v3` 判断。

## 已知证据与唯一问题

旧 60 的实际记录显示，3.2075 s 首次尾触地时，板 COM 竖直速度为 +0.84678 m/s，而机器人为 −0.72441 m/s，系统为 −0.62798 m/s。板在 3.2125 s 达到 +0.94604 m/s，3.24 s 已变为 −0.06268 m/s。系统 COM 下降不能直接证明板没有起跳动量。

因此本轮保留原尾地模型，检验原 29 个 PD 目标能否在已有板上升动量期间卸载双脚，维持整板腾空，然后 catch 和恢复站姿。不添加板驱动、根轨迹、板轨迹、外力、约束、弹性尾地参数或仿真相位重置。

## 固定物理与热前缀

- 原默认足板摩擦有效值 μ=0.6，`foot_deck_friction=None`，没有 42 个显式足板 pair；板、重力、惯量、质量、执行器和限位不变。
- 源：`logs/ollie_pop_events_preflight_02/actual_rollout.npz`，SHA256 `43cfef43b9cf5f966df2bb0a8b7e31a2446ee040f6fef5d3cab10baf221c027d`。
- 唯一主场景从原初态真实执行 156 个原控制，到 3.12 s。逐边界 qpos、qvel、state92 与全部 1249 个原 ContactObserver 记录必须逐值一致，不冷置末态。
- 在该热状态的独立 scratch 上继续旧 60 的 14 个控制到 3.40 s。14 个边界与 112 个新 Observer 记录必须逐值一致，主场景指纹不变；这是候选评估之前的硬门槛。
- 候选仅增加只读传感器、两足观察 site。所有物理数值数组及 option 逐值比较，原 site 按名称核对；新 site 不参与碰撞、无质量。manifest 同时记录板 XML、生成原场景 XML、网格资产、代码和输入 SHA。

## 冻结控制参数与先验

剩余 101 个 50 Hz 控制覆盖 3.12–5.14 s，每个控制 8 个 2.5 ms 子步。固定 256 候选、64 轮、16 精英、16 CPU 线程、seed=76；只在前 15 个腿腰关节维度加入残差，σ=0.25 rad，下限 0.025 rad。时间结点为：

`[0,.02,.04,.08,.12,.18,.26,.38,.54,.76,1.04,1.42,2.0] s`。

所有目标限制在原机械目标范围内缩 5 mrad。基础 nominal 从最后真实控制保持至 +0.06 s，再 quintic 收敛到原初始站姿的 `q+tau_static/Kp`，于 +1.42 s 到达。旧 60 先验精确保留后续 14 个原控制至 3.40 s，再从 +0.28 到 +1.42 s quintic 回到同一 standing target；它只是目标动作先验。

每轮固定包含当前 mean、nominal、旧 60 先验和每个已发现等级的 best。初始 mean 来自旧 60 先验。特殊先验必须通过显式参数传入 NPZ 与 result.json 及双 SHA，默认不会读取新文件。经 root 审查的右脚静态抬高 80 mm 先验，仅取右腿 6 个目标关节：

`[-.8237093295116416,-.1861654278824522,.15128122072333175,1.2839078880734835,-.524720301687585,-.06175753741504575]`。

该先验的其他 23 个目标沿旧 60 加 standing 路径；右腿保留 drive 至 +0.02 s，quintic 于 +0.08 s 到 tuck，保持到 +0.30 s，再 quintic 于 +1.42 s 回 standing。不使用 IK 保存的整份 qpos，不复制根或板状态。静态几何通过仅说明该姿态可达，不是动力学教师。

## 全轮相同的等级与成本

每轮所有完整有限候选使用相同 `(grade, -cost)` 排序，先比较等级，再比较固定连续成本。保存每个等级的 best；最后只执行最高等级内成本最低的一条已评估完整分支，不能执行未评估的 CEM mean。

- Grade 0：尚无满足下条的真实尾地接触。
- Grade 1：真实单点尾地 Fn≥0.1 N，并且同一时刻板 COM vz>0；截至该时刻机器人未触地、穿透≤10 mm、板未倒置。
- Grade 2：满足 grade 1 的尾地事件先于整板飞行窗口。所有 9 个碰撞几何最低点连续 16 子步 >5 mm，地面净正 Fn 合计各几何均<0.1 N，且截至窗口末的整板最低点峰值≥15 mm；从回合开始至窗口末没有机器人触地、穿透>10 mm 或板倒置。窗口后的摔倒不会抹掉探索等级，但实际最终结果仍然 FAILED。

尾地使用原生 CONTACT `maxforce` 选中的一个真实接触的 Fn，避免多个亚阈值接触的合计制造“真实 tail”。它按完整力范数选点，可能漏掉个别 Fn 更大的点，属于保守标签。候选机器人触地按原生接触计数判定，比原 Observer 的 0.1 N 门槛更保守。飞行的正 Z 合力合计门槛同样更保守，不能把相互抵消的足载当作释放。

连续成本均先冻结，不根据结果改权重。以同一个 +0.02–+0.40 s 软尾地 anchor 联合计算：尾几何离地间隙（300，尺度 68 mm）、尾地 5 N 不足（200）、板 COM vz 距 +0.5 m/s 不足（200）、前/后脚 20 ms 法向载荷（100/200，尺度 40 N）、随后 40 ms 整板间隙距 8 mm 不足（600，尺度 40 mm）与 15 mm 峰值不足（200）。允许尾前继续短 drive；不要求 3.12 s 瞬时卸载，不要求系统 COM +0.6 m/s 或鼻端净向上力。

恢复项包括 catch 支撑代理（200）、末 300 ms 双脚和轮子支撑代理（600）、姿态/相对高度/线角速度（300），机器人触地比例（2000）、最大穿透/真实限位/倒板严重度（500）、板 XY 漂移（10）和目标变化（0.1）。双脚净 Fz 仅用于 catch/final soft proxy，不能替代原 Observer 的板上表面支持分类。软 anchor/flight 按同步成本贪心选定后再算 catch，并非联合最小化全部 catch 项。

## 法向载荷与时间阶段

两足新增 BOX site 精确覆盖原 7 个旋转 capsule 的局部联合 AABB，再各向外扩 15 mm；只添加原生 TOUCH 传感器。它汇总该 body 的接触法向力，包含足板、足地和可能的自身接触，不是净 Fz，因而不会因相反方向而抵消。但 site 内点/射线规则可能遗漏区域外接触，因此 TOUCH 只作指导。

首分支与 winner 均在原模型 scratch 上逐点读取 `mj_contactForce`，独立比较 TOUCH 与全足 Fn、选中 tail 的 Fn，并保留最大误差。TOUCH 不吻合时必须如实标记，不能称为 Fn exact。最终实际执行另外保存每脚全部接触 Fn 合计、足板 Fn 合计和板作用脚的世界 Z 净力。`actual_release_windows.json` 用真实 Fn 计算两脚同时连续 20 ms 的最大载荷及尾事件邻域窗口，不用净零力证明卸载。

原生批量 rollout 的 pose/COMv 是每个积分前状态，contact sensor 为同控制当步求解力，FULLPHYSICS 是积分后状态。独立 branch proof 按此阶段逐子步验证。额外实际 Fn 采样在 `mj_step` 之后、额外 `mj_forward` 之前，明确记录执行区间 start/end；它不是 post-forward 标签积分。原严格 Observer 始终在额外 `mj_forward` 后，保持旧记录的阶段语义。[MuJoCo CONTACT 与 TOUCH 文档](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-contact) 规定普通/maxforce FORCE 在接触坐标系，netforce 的合成接触使用世界坐标系。

## 预算、保存与验收

默认 `inspect` 仅校验与冻结输入、代码、资产和参数，不积分。`benchmark` 只做同一固定首批候选与独立 proof，不更新分布、不执行主场景后缀；由 root 审查后再授权唯一正式 `run`，不网格、不追加配置。

总预算 300 s；完整搜索结束后保留 10 s proof 和 5 s actual，winner proof 前与实际执行前再次检查 deadline。不足时只保存已评估 best 和失败原因，不启动 actual。实际完整执行 5.14 s，即使中途跌倒也不提前截去尾段；主实际后缀内的数值失败和 KeyboardInterrupt 保留完整已完成控制、partial 控制进度、raw terminal qpos/qvel/time 与已取得接触/执行力记录。搜索阶段中断保存已评估 best、主场景指纹和时间；任意文件写出中途被再次打断不保证所有归档完整。

最终只调用未修改的原严格 v3；接触判定 0.1 N，上表面真实支持 >1 N，整板 9 几何，完整 400 Hz 记录与末 300 ms 站稳要求不变。任何探索候选的尾事件/腾空/后续摔倒均如实保存，不能将探索 grade 2 晋升为成功或教师。

## 当前状态

默认 inspect 已完成，输出 `logs/ollie_momentum_release_inspect_01/frozen_manifest.json`；静态证明见同目录 `static_checks.json`。Ruff 与编译检查通过；原生/观察模型全部物理数组及原 site 比较通过。6 个 CPU 合成案例等级为 `[2,0,1,2,1,1]`，分别覆盖合格窗口、无 tail、窗口前触地、窗口后摔倒、仅 15 子步飞行、tail 晚于飞行，全部连续成本有限。

独立接触夹具只调用 `mj_forward`，不积分：4 个平面/盒接触中 maxforce 输出为 `[found=4, Fn=3.07, Ft1=0, Ft2=0]`，与 `mj_contactForce` 选中接触的 local force 误差为 0。这确认本机 CONTACT maxforce 的首个 force 分量是 Fn，与 netforce 的世界 Z 分量不同。先验形状 `(101,29)`、结点形状 `(13,15)` 与 14 个旧控制逐值一致检查通过。上述合成数组、冷置 scratch 和夹具均只用于数学/接口验证，不是物理轨迹或主实际状态初始化。

以上作者静态阶段积分次数为 0、GPU 初始化次数为 0。随后 root 执行唯一固定配置 `logs/ollie_momentum_release_benchmark_01`：156 个主前缀与额外 14 个 scratch 原控制的 qpos/qvel/state 和全部 Observer 均 exact；808 子步 branch proof 的 state 误差为 0，force 误差约 2.84e−14 N，全足 Fn/TOUCH 误差为 0；64 轮投影 180.224 s，满足 300 s 门槛。这是基准评估，不是实际 Ollie 成功或训练。

独立 review 的无积分故障注入发现：在一个边界的 `actions.append` 完成、其余 snapshot 尚未提交时发生 KeyboardInterrupt，旧版可能留下 `len(state)==len(action)` 的不一致归档。旧源码 SHA `6b68d8d710ce04e076d573fc373725c6f80b4ce5c40647db55b594237e4919dd`、inspect_01 和 benchmark_01 原样保留。只修复实际归档事务：先构建全部 snapshot；完整 append 后才推进 committed_control_count；中断时按该计数统一裁回上一个完整边界，同时保留已发生的真实 partial、raw terminal、接触和执行 Fn。未改任何物理、成本、目标先验、随机种子或配置。

事务修复后的源码 SHA 为 `3800f75a0be59c29f6eea356ea41ab69999799544088cab4df6abcee1e99395d`。新冻结记录见 `logs/ollie_momentum_release_inspect_02`，其 `transaction_revision.json` 确认 config、sources、依赖、canonical assets、物理和选择声明与 inspect_01 逐值一致；Ruff 与编译检查通过。该修复没有重跑任何前述物理结果。

独立审查者使用 fake time、不调用真实 `mj_step`，对 native-return、snapshot pack、action append、全部 append 后 commit 前四处中断进行故障注入，全部回滚到 action156/state157，partial 保留真实模拟进度1或8个子步、raw terminal 保留，结论明确FAILED。审核为 `logs/ollie_momentum_release_review_01/final_review.json`。root另外独立核对7种合成等级边界，包括一个轴盒低于5mm、仅15个样本飞行和不完整时间，全部符合预期；这些都不是实际腾空结果。

正式运行前只使用 Ruff 统一格式，前后 `ast.dump` 完全相同，证明逻辑未变。最终源码 SHA 为 `bea4d2137d4313497b68e6720b13297cf192af8528f258e663664c72a2a92cc5`，格式记录 `logs/ollie_momentum_release_format_01/result.json`；inspect_03冻结所有输入，Ruff check/format-check/语法检查通过。2026-10-25 17:46北京时间启动唯一固定正式运行 `logs/ollie_momentum_release_01`，完整真实结果及独立重放随后补充。

## 正式完整执行：真实低跳通过，接板仍失败

唯一64轮搜索已全部完成，含实际执行与保存耗时 **75.8273 s**，在300 s预算内。首次在搜索第16轮输出时已有grade2；最终执行的是零基第53轮candidate137、固定成本5902.66890的已评估分支。未追加seed、轮数或修改代价；也未使用未经评估的均值。257个原29维目标从唯一初态连续执行到5.14 s，258个边界、2057条原400 Hz记录，`partial_control=null`，实际与预测FULLPHYSICS最大误差0。

**原严格v3仍为FAILED，但首次通过尾触地后的真实整板离地条件：**

- tail-ground：3.2025–3.2400 s，连续窗40 ms。
- 所有9个真实几何均>5 mm且无有效板地接触：3.2550–3.3175 s，连续窗 **65 ms**。
- 全板最低点峰值 **23.4328 mm**，满足原15 mm门槛；板全程未倒置。
- 首次机器人接地为 **3.3850 s**，在上述飞行窗结束之后；全段最大穿透7.51854 mm，初始穿透约0。
- 末300 ms：左足上表面支持100%、右足0%、四轮同时接地100%。右足未接回板，机器人随后倾倒，末段未站稳。
- 原v3仅`no_robot_geom_touches_ground`与`final_300ms_standing_stable_on_board`失败；不把其余规则通过称为完整Ollie或原片幅度已复现。

这次只是实际29PD物理探索。**没有训练SuperMimic actor、没有新的策略checkpoint，`verified_teacher=false`。**新μ=.6轨迹也不直接混入72/73的μ=.9数据/权重。板12°、原接触材料、重力、质量/惯量、原29PD/cap均未改变，不能把改善归因于增大翘角或反弹；多个腿腰目标同时改变，也不能把结果单独归因于某一关节或唯一释放时序。

## 实际释放、独立重放与能确认的范围

真实执行力显示：3.2025 s首次尾撞时，板COM Vz为+.97205 m/s、机器人COM为−.62973 m/s；后脚Fn约438.29 N，尾地Fn总和555.60 N。双脚并非在首次尾撞前就全部卸载。随后 **3.2475–3.2675 s** 的连续20 ms中，两足全部真实接触Fn逐样本均为0；之后形成上面的整板飞行。这支持允许“尾撞后快速释放”的接触模式，不能声称实现了2021论文完全相同的双脚提前离板时序或人类完整Ollie。

另一个执行者从原初态独立原生重放全部257目标：qpos/qvel/state92边界最大误差均0；时间与归档规则网格仅有5.55e−14 s累计舍入差；2057条原Observer字段及完整v3结果逐值相同。808条后缀执行阶段的每脚全Fn、足板Fn、净Fz、尾地Fn和起止时间逐字段误差全0。首156目标/157边界与源60逐值相同，原53初态也相同。全部电机目标在原目标范围内，实际执行力矩无cap越界；无外力、无中途reset/状态回写、neq=0、板电机0。

独立证据为 `independent_replay_01/result.json` 与 `independent_force_audit.json`。胜者808子步的另一路branch proof：FULLPHYSICS误差0，合力约2.84e−14 N、力矩2.27e−13 N·m，9几何净空误差3.12e−17 m；TOUCH与真实全足Fn、maxforce与真实tail Fn均误差0。本轮数据上的吻合不扩大为所有异常接触下TOUCH必覆盖的保证。

实际轨迹SHA256：`c8754cad0ec9f77dc816790c9919d918f2c1e1fd9dfb5460b13ad2581ba0f0a2`。它是通过真实起跳子条件的**失败完整回合**，不晋升为成功教师或学习策略。

## 全长视频与接板的具体缺口

[原片／完整真实物理对照](/home/zxc/SuperMimic/logs/ollie_momentum_release_01/video_01/comparison_full.mp4) 已导出，右栏明确FAILED并保留后续跌倒。单场景实际渲染258帧/50 Hz、原片对照187帧/24 Hz，均完整解码通过；只渲染保存的qpos，没有新积分或替换空中状态，轨迹SHA不变。慢放1.5倍仍为用户估计，未成为实测参数。元数据与视频SHA见 `video_01/video_audit.json`，root检查了起跳、落地前和失败后的图像。

独立几何诊断进一步定位了“卸载”与“可接回”的区别。3.30 s左脚site在板局部 `[.162,.303,.060] m`，而板半宽只有.105 m，有限脚板最近间隙约177.3 mm；右脚site约`[-.475,.116,.093] m`，已经越过尾部，有限间隙61.9 mm。机器人COM此时向下约1.375 m/s。首次接地具体为right_foot5_collision，不能把它当成成功踩回板。

纯FK回查源60：左脚site局部Y在frame150/152/154/156分别约.0613/.1016/.1522/.2014 m；仅奖励卸载会允许把脚踢向板侧。初始站姿site局部XY为left`[.205021,-.024923]`、right`[-.320000,.022453]`，site本身不是足底点。下一项应从更早 **frame150=3.00 s** 的真实热前缀处理竖直抬脚及可回收XY走廊，保持已获得的跳跃动量，再尝试接板；不从3.30 s强行假设能在几十毫秒内收回横向偏离20 cm的脚。该后续控制尚未实施，不增加虚假的成功、训练或运行中状态。

## 复核入口

保留原目录，先用新目录只读inspect；改为`--mode benchmark`才评估首批，`--mode run`才执行上述固定64轮及实际后缀。以下是同一冻结配置的入口，不代表又追加运行：

```bash
env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked python scripts/local/optimize_ollie_momentum_release.py \
  --mode inspect \
  --extra-prior logs/ollie_rear_release_prior_01/rear_tuck_prior.npz \
  --extra-prior-sha256 0cce4b0ef02036ba578ff9785e198ab9f7e13d1eca8f5bf55a23be7123c8171f \
  --extra-prior-result logs/ollie_rear_release_prior_01/result.json \
  --extra-prior-result-sha256 aabf09b1cffe7ee621c785960e3a7832dce2439450ac7331d4b5b6dbcff487ff \
  --out-dir logs/new_momentum_release_inspect
```
