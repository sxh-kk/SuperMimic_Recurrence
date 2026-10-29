# Ollie低维相位动作原生搜索

2026-10-29，正式搜索前固定。95已经在板接地后30 ms出现真实前脚顶面承重51.953 N，早落地双top+身体up≥.9/根板高度≥.45 m最长15 ms，但未站稳。97保留接板事件精英后，实际早落地同类窗口增至22.5 ms；其完整原v3仍FAILED，不能直接成为SuperMimic教师。

99把原17×29=493个独立目标残差缩为24个协同角度及1个末段混合系数。新入口`../scripts/local/optimize_ollie_phase_basis.py`默认inspect；只编辑该入口和本文，不改冻结76/95成本、模型、传感器或旧结果。SuperMimic仍为下游学习；本试验不训练actor/WM，不用学习模型替代真实接触。

## 固定协同与相位

每个相位有8个幅度参数，单位rad，原G1关节顺序保持。下表是**PD目标**的代数方向，不是脚朝向的硬约束；真实响应仍由原29PD、力矩上限和接触产生。

| 组 | 原目标增量 |
| --- | --- |
| 1 | 左右髋pitch各+δ，左右踝pitch各−δ |
| 2 | 左右膝各+δ，对应髋pitch/踝pitch各−δ/2；负δ为伸膝方向 |
| 3 | 左髋roll+δ、左踝roll−δ |
| 4 | 右髋roll+δ、右踝roll−δ |
| 5 | 左踝pitch+δ |
| 6 | 右踝pitch+δ |
| 7 | 腰pitch+δ |
| 8 | 两肩pitch各+δ |

三相位按20 ms控制区间中点取quintic权重：3.24～3.36 s catch从0升1；3.36～3.62 s catch退、absorb升；3.62～4.10 s absorb退、stand升；其后stand保持1。权重非负、和≤1，最早3.24 s之前的全部源动作逐值不变，避免91早期负髋pitch响应破坏pop。髋踝补偿只表示目标角度协同，不能保证动态脚底姿态保持。

原95晚段目标已经偏离站姿，例如5.0 s右髋pitch约.933 rad、腰yaw/roll/pitch约.498/.205/.465 rad。仅24维各±.3 rad的盒约束不能保证覆盖末段站起，因此在首次物理预检前增加第25维`z∈[0,1]`：先生成24维加性协同目标，再按`z×quintic(3.60→4.10)`向原初始站姿29PD目标混合。站姿目标严格由原reference第0帧加`FF0/kp`后按原targetlimits裁剪得到；没有写入qpos、root或board。z=0且24角度均0时逐值还原源动作，z在3.60 s之前完全无影响。

协同参数化、97目标nominal、末段z和接板后排序同时变化，本轮不能据此作维度降低的隔离因果结论。phase/basis/z是固定搜索空间，不能从其中失败推断完整29PD动作空间不可行。

参数数组为25维：前24按`phase3×group8`展平，最后1维无量纲。角度均值初始0，σ=.06 rad、floor=.01 rad、边界±.30 rad；z初始均值.5、σ=.2、floor=.04、边界[0,1]。参数采样先按各自盒约束裁剪，最后生成的原29目标统一按原targetlimits裁剪；每候选保存目标裁剪数量及最大幅度。固定slot0为本轮mean、slot1为精确源动作、slot2为只有z=1的末段站姿目标种子，后者仍须整段真实物理评估。其余保存每级best参数再插入，全部参数与每轮RNG结果留档以便重建。

## 目标与精英选择

真实物理事件、完整几何门槛、逐接触顶面支持和蹲姿catch沿用冻结95；传感器仍2709维。原v3是唯一成功证书，空中双top不被额外加入原v3必需条件，.45 m中间catch高度也不替代末段.55 m站立判据。

99新增一层明确的排序包装，不编辑95：对已经达到tier0/1的候选，使用

`4*tier + 2*lost_steps/2057 + 1e−4*final/(1+final) + 1e−5*quality/(1+quality)`。

无机会窗时qualityBound=1。一次2.5 ms更晚真实身体触地带来的改善约.0009723，严格大于全部secondary上界.00011，鞋底软质量不能再换掉更晚触地。tier2/3继续逐值使用95原分数；原95 score、全部分项及物理metrics另存，不能将新分数当旧分数。tier0仍只是预测站立proxy，最终必须完整原canonical v3成功。

每批通过已有显式同批缓存选择器，先最高物理event grade、再最低progress tier、再按本轮新score取最多8个精英；只有1个合格catch时不混入其余未catch目标填满配额。全局winner始终按同一`(最高grade,最低新score)`保存，并且只能执行已完整评估的目标序列，不能执行未评估mean。

## 固定预算与物理证明

固定P130/H127，32轮×128候选、8精英、seed99、16线程，600 s动态预算。首次完整batch后运行原生proof并估算剩余32轮成本；benchmark只跑这一批及proof，没有CEM分布更新，也没有实际后缀。正式搜索仍需root另行明确启动；每轮与最终proof前检查15 s余量，实际执行前检查5 s余量。预算不足只保存已评估best与失败记录，不启动实际后缀。

来源选择必须显式path+SHA。最初数组inspect使用95，未做动力学；root在看到97真实进展后指定97：`logs/ollie_progress_landing_01/actual_rollout.npz`，SHA `21b6ef7c0f6fbc0b1b1a39e3dba7dc575d19298c8d183463b9abc7424f81cc64`。运行时还须指定其独立完整物理审计SHA，程序核验审计通过且对应同一actual；不自动查找“最新”轨迹，也不会自动退回95。

主回合从原唯一初态按源动作连续重放130控制到2.60 s，核验131边界/1041条Observer逐值一致。候选只从该热状态做原NativeSamplingMPC rollout，保留FULLPHYSICS与qacc_warmstart；源保存末态不置入主回合。首batch的零残差slot1必须与源完整127后缀的FULLPHYSICS、全部2709传感器及127边界逐值一致；所有候选至3.24 s的前256个子步也必须一致，证实相位残差没有提前改pop。

只读top/air传感器构建时逐数组核对原物理、原site和全部option；主模型无pair/eq/mocap/board驱动。首个选定分支与最终winner分别做原模型逐子步state、pose/COM、wrench、9几何、真实normalFn和top/ground同阶段证明。最终只在同一主回合执行一个已评估后缀，保存257目标/258边界/2057条postforward Observer/1016条真实执行力，原严格v3继续覆盖整个失败尾段。

`execute_winner`复用76已审计的事务式partial归档：控制中断时regular数组回滚至最后完整边界，raw terminal及真实已执行子步保留，不能截掉跌倒尾段。所有结果保持`verified_teacher=false`；成功也须独立完整固定目标重放，才可讨论下游数据资格。actor更新0。

## 零积分检查与预检记录

初版25维数组检查`logs/ollie_phase_basis_inspect_01`通过：8组方向rank8，目标生成器与直接矩阵式误差8.88e−16，零参数exact95，3.24前不变，z=1末段exact原站姿目标、3.60前不变。该检查在新生存优先排序加入前完成，保留原目录；不能当作最终源码hash或物理预检。修订后inspect/唯一benchmark的实际结果随后追加；本阶段没有正式搜索，也没有声称Ollie成功。

纯数学排序检查`logs/ollie_phase_basis_static_01/result.json`通过，0积分：更晚一物理步即使承担最大secondary仍严格胜出、tier2/3原95score逐值保留、无效grade保持inf。该产物也记录检查时源码hash，后续只增加同阶段contact证明和依赖哈希清单时保留旧记录。

## 97源的最终inspect与唯一benchmark

97独立完整双阶段审计已PASS，SHA `102ec417c316af89b66004181e1cc6036034e71d686607c79b4e5657909fa8e8`，对应实际SHA与上文21b6…一致。99显式使用该来源，最终`logs/ollie_phase_basis_inspect_02`通过，0积分；25维生成器与直接矩阵表达式差9.02e−16，零参数exact97、3.24前不变、z在3.60前不变、z=1终末exact原站姿目标全部通过。

随后仅运行一次固定配置`logs/ollie_phase_basis_benchmark_01`：128候选完整真实后缀rollout .477397 s，cost 2.451608 s，固定32轮投影93.7282 s，低于600 s预算。没有更新CEM分布，也没有执行实际主后缀；主场景保持原连续前缀2.60 s。该投影是首批测量估算，不是正式搜索已经完成的耗时或结果。

source97零残差slot1的完整127后缀FULLPHYSICS、全部2709传感器及127个控制边界均逐值exact0；所有128候选3.24 s前256个物理子步也逐值一致。原native1016子步proof state误差0，pose/COM误差1.33e−15，force≤2.84e−14 N、moment≤5.68e−14 Nm、count/depth/ground count均0，9几何包络差≤5.55e−17 m。真实normalFn/TOUCH差0，同阶段左右top与真实robot-ground、board-ground差0；模型、足板μ=.6及全部执行上限保持原样。

首批30个physical grade2、98个grade1；进展tier1共11个，但只有其中4个同时处于最高物理grade，精英为`[1,0,89,2]`。选择器没有混入较差tier或为填满8个配额混入不合格几何。best仍是精确源97的slot1，99新cost=4.6126427483，原95 raw score=5.2291027793，早落地双top+蹲姿22.5 ms、firstfloor3.5675 s；这只是源动作回归，不是新改善。z=.5及z=1目标种子也保留22.5 ms窗口，但本批没有推迟首次触地，仍须把站起与更早的全身吸收协调起来。每候选targetclip已记录，首批范围0～49个值、最大裁剪.2008374 rad；角度参数盒约束与实际29目标限位是不同层次，不能把有界参数当作无裁剪目标保证。

冻结源码SHA `6f0d8b75f3b5d0bdc5612a6a6f0d2ebf57cb00a60398e680207c866bdae94131`，Ruff check/format通过。benchmark SHA `5d861320f5fb4e9bd0fdb9810555d7d2300b43c29a153f1342ba20ef896b4feb`；完整manifest SHA `6c646cb4dac31973a4a11fd58561c64dae82917e30fc5c1b4d32639ca3125d98`。原95/97、模型及所有旧结果未改写。当前停在预检完成，未正式搜索、未训练、未Git提交。

本次实际预检命令：

```bash
env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/local/optimize_ollie_phase_basis.py \
  --mode benchmark \
  --nominal-source logs/ollie_progress_landing_01/actual_rollout.npz \
  --nominal-sha256 21b6ef7c0f6fbc0b1b1a39e3dba7dc575d19298c8d183463b9abc7424f81cc64 \
  --nominal-audit logs/ollie_progress_landing_01/independent_air_stage_audit_01/result.json \
  --nominal-audit-sha256 102ec417c316af89b66004181e1cc6036034e71d686607c79b4e5657909fa8e8 \
  --out-dir logs/ollie_phase_basis_benchmark_01
```

输出目录必须全新；以上目录保留证据，不得覆盖。正式模式及新目录由root在审阅本轮后决定。

## 唯一正式32轮结果：触地推迟，仍未站稳

root审阅源码和预检后明确授权，使用同一冻结SHA、同一verified97来源/审计、seed99与全部既定参数，在新目录`logs/ollie_phase_basis_01`完成唯一正式32×128搜索及一个完整实际回合，总耗时96.4894 s，低于600 s预算。没有中途改相位、权重、σ、限位或模型，没有追加搜索。

winner为round30/candidate76，physical event grade2、progress tier1，新cost4.55527910305。目标参数25维保存在`winner_evaluated_branch.npz`及完整`round_parameters.npz`，末段z=.3422867881；z处于[0,1]仅代表目标混合强度，不能当作起身程度或实际姿态指标。

从原唯一初态连续执行257目标，258边界/2057条postforward Observer/1016条后缀真实执行力完整落盘，终点5.14 s、无partial，实际与已评估winner的FULLPHYSICS最大差0。实际归档SHA `a884dedd37963cc03f3744ac74d3e4545b3bdbe254d256acb4cc64faf683119f`；result SHA `0250a9c7f5d2355514d33eb037667421b891b537149f4b9988ea4b71a1a11f6c`；正式manifest SHA `4d144ea65791b8636aee18172d38daa6a1c2ad5d1f03d9c9044f053e5ae5c1df`。

原严格v3仍为**FAILED**，失败项为机器人触地和末300 ms未在板上稳定站立。全板9几何真实飞行3.2650～3.3275 s、65 ms，最高最低包络16.5102 mm；仅轮几何包络峰值25.3511 mm，不能把较大的轮代理数值当作全板高度。最大穿透9.98316 mm，接近但未超过原10 mm门槛；实际关节最大限位偏差.0239845 rad继续原样记录。

首次身体触地3.7150 s，比97的3.5675 s推迟147.5 ms，但末300 ms左右真实顶面承重均0，左脚触地98.33%、四轮同时触地100%，站立proxy也为0。晚触地没有解决站稳，本轮不能据此宣称稳定控制或完整Ollie成功。`eligible_reference=false`、`verified_teacher=false`、actor更新0。

### canonical真实顶面承重与姿态窗口

只读分析完整保存的原canonical400 Hz postforward记录，不做新积分：真实板首次接地3.3375 s；右脚3.3150 s已有top42.9801 N，位于空中；左脚首次top为3.3725 s、63.7431 N，位于板接地后35 ms，处在既定60 ms早落地窗口内。当时身体up=.994629、根板高度.521546 m；不能把这次蹲姿承重计为≥.55 m站姿。

| 真实窗口 | raw双top | 双top+up≥.9/高≥.45 m | 同姿态/高≥.55 m | 同姿态/高≥.60 m |
| --- | ---: | ---: | ---: | ---: |
| 首次合格飞行开始至真实板接地前 | 0 ms | 0 ms | 0 ms | 0 ms |
| 板接地后60 ms以内 | 25 ms | 25 ms | 0 ms | 0 ms |
| 飞行后至首次身体触地前，全段最长 | 40 ms | 40 ms | 0 ms | 0 ms |

40 ms真实双top窗为3.3725～3.4100 s，共16个样本；早落地60 ms内保留其中3.3725～3.3950 s的25 ms。原95执行阶段cost metrics也得到25 ms早双top+蹲姿，仍须区分执行力/区间起点几何与postforward重算力；上表来自后者，未将其积分称执行冲量。此处早窗口未额外施加下降速度筛选，已在产物字段名明确；既定95成本的原下降规则没有改动。所有空中双top仍0，属于落地后接板进展。

保存记录分析结果`logs/ollie_phase_basis_01/saved_canonical_phase_summary.json`，SHA `fc1175e28229c35e988535a00979e4d17e2c6e6e8ee9cffdbf4c9d2768ee8de3`；分析工具位于ignored logs，0新积分，不改变原结果。上述原记录结论仍待root组织的独立完整重放交叉复核，不能以本脚本自己的执行exact代替独立审计。

### winner原生证明与后续边界

最终1016子步原生分支proof state差0，force≤1.42e−14 N、moment≤5.68e−14 Nm，pose/COM≤1.33e−15、9几何≤5.55e−17 m；真实normalFn/TOUCH差0，左右top、真实robot-ground与board-ground同阶段差0，主热场景保持不变。`winner_branch_proof.json` SHA `83e9b75979271879e3557c32f1f68a67557d7ec035785e40eeebb1a773dbd150`。

本轮证明了该固定低维目标空间可以在保留接板事件时推迟触地，并未证明其覆盖稳定落地解，更不能从失败推断全29PD空间不可行。初始CP−COP偏离及鞋底朝向问题仍是未解决机制；25维参数化和更长触地时间都不等于重心已经可支撑。本轮正式已结束，代码/配置保持冻结，不再自动搜索；后续迭代由root根据完整失败和独立审计判断。

### root验证与完整视频

新增`tests/test_ollie_phase_basis.py`四项CPU检查通过，覆盖恢复后更晚一个2.5 ms物理步优于全部次要代价、tier优先/未恢复95代价逐值保留、任意有界参数不改3.24 s前目标、末段站姿blend的因果性和终值。Ruff通过，冻结99源码未改。

通用独立完整原生重放全部257控制，qpos/qvel边界误差0，2057条Observer字段exact，真实全板峰16.5102 mm与原v3失败一致；产物`independent_replay_01/result.json`。[实际完整视频](/home/zxc/SuperMimic/logs/ollie_phase_basis_01/video_01/rollout.mp4)及[原片完整对照](/home/zxc/SuperMimic/logs/ollie_phase_basis_01/video_01/comparison_full.mp4)已导出并验证解码，258实际帧/187对照帧；root已查看六帧蒙太奇，保留完整倒地尾段和FAILED标签。视频只渲染保存的实际qpos，未额外积分或加入运动学替身。

独立双阶段专项审计也已PASS：`independent_air_stage_audit_01/result.json` SHA `8d7a5b5ec7ec86538b7c2b43b39f7728cf8ce09e90427abb1fa3014119de1b9b`。全部32×128 RNG/参数、每轮CEM更新/精英、八组方向/三相位/z/目标裁剪/winner逐值重建；两模型257目标的边界/2057 Observer/1016执行力，以及exec/post两阶段的raw top接触和板地Fn全部0差，无overflow。raw双top40 ms、早落地.45 m姿态窗25 ms、.55 m窗0及原严格FAILED均独立重现。

审计还定位了一处纯序列化差别：同一冷进程首次`model_sources()`的scene XML SHA为`7d9d…`，后续为`59ed…`，仅因mesh显式`content_type="model/stl"`省略，.STL路径与内容相同；两份编译模型全部481数组和29项option逐位一致。证据`logs/ollie_phase_basis_audit_tools/scene_xml_audit.json`。未修改冻结99或静默忽略hash；后续102另建显式零积分证明/记录入口来处理此差别，原物理、实际回合与判据保持。
