# Ollie真实触地优先与整段几何筛选

2026-10-25，实验前固定方案。84综合cost选择3.405 s脚先触地且穿透10.7569 mm的分支，虽然原82安全几何低跳直到3.7125 s才手触地，说明加权平均允许以更早触地换其他项改善。本轮修改选择规则，不能把修改后的探索等级当作成功判定。

## 固定实现

入口`../scripts/local/optimize_ollie_survival_priority_landing.py`复用84的P145=2.90 s/H112、原82完整112目标nominal及三个prior、全29 PD目标、256候选×64轮、最高等级最多16精英、原σ与600 s预算。只固定新seed85和以下选择规则；83/84/top/76源码、原模型、初态、摩擦、力矩、严格v3不变。

原真实起跳等级2只有在**整个原前缀、候选400 Hz几何与最后5.14 s几何**均穿透≤10 mm且板up>0时才保留，否则降1。位置几何不依赖暖启动求解力：最后姿态只在新的scratch MjData中调用`mj_fwdPosition`补上最后一帧碰撞距离，绝不步进、写入主回合或宣称末帧接触力认证。候选sensor覆盖2.90至5.1375 s的间隔起点，补末帧后覆盖全部几何采样。

在同等级内用`2*lost_steps + raw/(50000+raw)`排序。无已观测真实人体触地时lost_steps=0；否则为`1+round((5.14-first_ground)/.0025)`。真实触地仍按原canonical Fn≥0.1 N，不使用found、几何间隙或小于阈值的接触。第二项单调且在[0,1)内，因此多保持一个2.5 ms步的未触地状态必定优先于任意次级cost改善；同一首次触地时刻才比较原83完整双足top/同窗姿态等cost。raw各项仍逐候选归档为指标，可查实际交换关系。

执行阶段最后一个力样本几何在5.1375 s；上述排名**不认证5.14 s末帧无触地**，完整实际执行和独立postforward严格v3照常检查全部2057点及末300 ms。更晚摔倒不是成功；最终仍须真实双足顶面承重并稳定站板。

原82零残差的整段穿透9.70352 mm且未翻板，仍有可评估等级2候选。84预检direction prior虽延迟触地至3.82 s，但穿透15.2805 mm，本轮会降级，不作为合格优解。无新prior调参，不改变视频时间倍率。

## 预检与完整验证

先做无积分静态检查，再一次完整batch预算和新优先winner接触/原物理校验，通过后固定64轮、唯一原初态完整执行。原83的first-top proof仍保留，85另核优先winner同stage真实top和ground，避免只核次级原cost的赢家。

```bash
env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --locked python scripts/local/optimize_ollie_survival_priority_landing.py \
  --mode benchmark --out-dir logs/ollie_survival_priority_benchmark_01
```

正式使用`--mode run --out-dir logs/ollie_survival_priority_landing_01`。完整失败尾段同样保存；这仍是SuperMimic物理教师数据探索，actor更新0。

预检`logs/ollie_survival_priority_benchmark_01/benchmark.json`：一完整batch rollout .8413 s、cost 2.4605 s，64轮投影211.32 s，低于固定600 s预算；未更新分布或执行主后缀。优先winner slot0为原82零残差：first ground3.7125 s，整段穿透9.70352 mm，最小板up .84501。新旧top proof、原模型896执行步FULLPHYSICS误差0、真实ground不一致0；接触法向最大误差2.84e−14 N。说明修复了84的选择退化，但此预检本身没有带来新动作或成功落地。

## 完整结果：更晚触地，仍未站稳

唯一正式64轮完成**168.034 s**，winner round59/candidate70，优先cost1056.574874。该cost已换为本轮分层排序，不能与84原加权cost直接比较。主回合原初态连续257控制至5.14 s，258边界/2057 Observer/896后缀执行力完整保留，actual_prediction误差0、partial=null。入口冻结SHA `2b7999f0cb3796287c48e6a14d11c16309faa79b0732d3bd8990777407100d1e`，83/84/top/协议源码及原物理未变；actual归档SHA `fc0a113a3003f2ff669510cbfe3ad4d6f0ff31f05ace9738724707d91167a9d6`。

原严格`whole_board_geometry_top_support_v3`仍为**FAILED**：

- 真实全9几何整板飞行3.2550～3.3125 s，共**60 ms**，最低点间隙峰**21.4812 mm**；主尾地窗口3.2225～3.2325 s，共12.5 ms。
- 首次有效机器人触地为**3.8225 s，头部与左手**（`head_collision`、`left_hand_collision`）。比82的3.7125 s及84的3.405 s推迟，但仍未完成安全落地；此时upright−.22456、root−board高度.15228 m，已倒下。
- 全回合最大穿透**9.0677 mm**、板未倒置，保持原10 mm几何门槛；实际软关节越界最大.02904 rad另记。原严格失败仅剩身体触地与末300 ms未稳定站板两项。
- 末300 ms左/右真实top支持100%/45%，四轮接地45.83%，最小身体upright−.29718。倒地后脚仍与板顶接触不能认证站稳。

整段几何筛选与首次触地优先确实避免了84那种穿透超限且更早倒地的原winner；结果只说明该固定回合更晚摔倒，不能称已完成Ollie或策略能力提升。**本轮actor更新0、SuperMimic训练0，无新策略检查点。**

## 双阶段顶面与完整重放核验

`independent_top_stage_audit_01/result.json`由新ignored审计脚本`logs/ollie_top_landing_audit_tools/verify_winner_stages85.py`固定重放唯一257目标；同时核验85 caller、84/83引用、top模块与协议的冻结hash，原83/84审计入口不改。脚本SHA `1a02f59dc181b9ec2b81f52057646a8e0e933244823ee783f1f864df6899af50`。原模型与2453维只读观测模型的全程物理、258边界、2057原Observer逐值一致；执行求解与额外forward两个阶段各2056样本，top、逐接触Fn/位置/法向/count、真实ground判断误差全部0。单组top最多14接触/容量32、ground最多8/容量64，无overflow。

实际安全双足同时top>1 N的最长窗口是**3.5375～3.5650 s，共30 ms**，另有10 ms及更短片段。搜索原winner的保存指标本身也是.03 s，未达到连续40或50 ms。合并upright≥.9和root−board高度≥.6 m后窗口仍为**0 ms**。执行与postforward结果一致；执行首触地几何/区间起点3.8225 s对应返回data.time3.825 s，postforward则在data.time3.8225 s判定，未混用两个阶段。

通用`independent_replay_01/result.json`和v3 `independent_force_audit.json`独立确认qpos/qvel/state92、全部2057 Observer、896后缀执行力及所有strict字段逐物理值一致，真实力矩超限0；后缀腿部最高仅为上限59.9547%，不能把倒地单独归因于腿电机饱和。理想时间轴与原生累计时间残差5.06e−14 s单列。82前145、old60公共前145、原53初态及模型来源hash通过。首次执行阶段头部地接触Fn4250.815 N，属于碰撞冲击，不是站立支持。v3审计脚本SHA `3d1ae496bdaee27a2cedf2a41f02c46f102e21ce7203bf00311dfefab641d24a`，报告SHA `fcd57cc597eaf3472ec755a12121bbd8abdcdea8e6bc4f3b298d5d4dc169e5cd`，未新增搜索或训练。

## 接板方向与时序

`independent_top_stage_audit_01/top_direction_diagnosis.json`仅分析已存82诊断、84/85逐接触观测，没有新积分或动作拟合。板坐标为+X朝nose、+Y沿轴左侧，center半宽.105 m。

85在3.24～3.32 s左前脚body原点Y约+.118～+.129 m，靠板+Y侧，且没有真实top力；body原点只作方向描述，不把它当整只鞋的包络证据。右后脚3.30 s已有约34 N真实top，却落在极后尾X−.388 m/Y+.032 m；3.34～3.50 s主要仍是尾端单脚支持。左脚直到**3.5375 s**才与center顶面同时承重，接板明显晚于3.295 s首40 ms飞行结束。

3.5375 s两足top约109.1/16.6 N，upright已降至.86930、高度.42356 m；3.54 s虽有150.8/104.9 N双top，upright仅.86502、高度.42400 m。此时COM沿世界重力线在板顶平面的投影Y为**−.25732 m**，所有真实top接触Y只在**+.01734～+.07603 m**，投影已向−Y越过接触包围盒274.66 mm，3.60 s进一步扩大到324.86 mm。这个静态投影/包围盒是失衡方向诊断，不能单独证明动态不可恢复；实际持续倒下和随后头/手触地另有完整物理记录。

对照84，3.30 s左脚实际接触在+Y边缘约+.1037 m，normal·deckUp仅.162，属于侧边；后续右脚又出现底面接触。82既有独立诊断也显示COM投影3.24/3.34 s为Y−.1284/−.1764 m，左脚先擦侧边、右脚短时尾端top，未同时接稳。85已经从“始终错过双top”推进到真实短时双top，但主要缺口是**脚在+Y一侧接板偏晚、身体重心持续向−Y倒，支撑出现时躯干高度和姿态已失去余量**。后续纠偏必须覆盖起跳前及起跳时的横向状态，不能仅追求更晚触地或倒地后top比例。

## 完整视频

[原片/真实物理完整FAILED对照](/home/zxc/SuperMimic/logs/ollie_survival_priority_landing_01/video_01/comparison_full.mp4)与[实际物理原速视频](/home/zxc/SuperMimic/logs/ollie_survival_priority_landing_01/video_01/rollout.mp4)均保留5.14 s倒地尾段。`video_01/video_audit.json`记录258实际帧/187对照帧及源归档未改；实际视频SHA `4b421c664e5fdfc649927e862d9ebcf50ba0fe4df54d58d5e6d7781d7b01a26a`，对照SHA `9ffb118aac2ff3dacdc7e6a6a423119e13c8d4cb99df42871dfbab3f55eaec92`。已抽取并查看实际2.90/3.28/3.54/3.70/3.84/5.14 s六帧，能看到短时接板后身体翻倒；图像不替代400 Hz接触测量。

![85实际回合六帧，完整保留FAILED尾段](/home/zxc/SuperMimic/logs/ollie_survival_priority_landing_01/video_01/montage_6frames.png)
