# Ollie低维接板连续迭代

2026-10-29，02轮任何inspect/benchmark/正式运行前登记。99首轮在完整原生物理下把首次身体触地从源97的3.5675 s推迟到3.7150 s，canonical真实双顶面支撑最长40 ms，早落地双top+up≥.9/根板高度≥.45 m为25 ms；空中双top和高≥.55 m支撑姿态仍0，末300 ms未站稳，原严格v3保持FAILED。不能把触地时间推迟当成稳定落地。

root明确授权仅一次02轮：先等首轮99独立完整双阶段审计PASS及其固定SHA，再做新源inspect、一个首批benchmark；预算/原动作回归/proof都通过后直接完成唯一正式32轮。没有03轮自动授权，不更改99源码、任何Config字段、物理模型或判据，不提交Git。

## 输入与固定配置

源为`logs/ollie_phase_basis_01/actual_rollout.npz`，SHA `a884dedd37963cc03f3744ac74d3e4545b3bdbe254d256acb4cc64faf683119f`。读取的nominal只包含其127×29 PD后缀；源完整qpos/qvel只用于逐值审查共同前缀和已评估实际动作，不把保存状态设入执行场景。源仍是失败的探索数据，`verified_teacher=false`。

入口继续使用`../scripts/local/optimize_ollie_phase_basis.py`，源码SHA `6f0d8b75f3b5d0bdc5612a6a6f0d2ebf57cb00a60398e680207c866bdae94131`。配置与首轮逐值一致：P130/H127、128候选×32轮、8精英、seed99、16线程、600 s动态预算、proof/actual余量15/5 s。24角度参数σ=.06 rad/floor=.01 rad/边界±.30 rad；第25维z均值.5/σ=.2/floor=.04/边界[0,1]；catch/absorb/stand相位和3.60～4.10 s末段目标混合完全沿99，不调时间或权重。

每个新回合仍由原唯一初态连续执行130控制到2.60 s，所有参数3.24 s前无作用；不是从首轮5.14 s倒地末态续积分。变化的是下一次固定搜索的目标nominal，不能把跨回合的搜索迭代误写成同一个物理回合连续运行。

排序继续沿99：最高原物理event grade；同级由95真实接板progress tier和99新score排序；tier0/1中首次真实身体触地严格优先，其余末段/鞋底质量为有界secondary；精英同批仅取最高physicalgrade中的最低tier，不填充较差tier。tier2/3原95分数保持，全部原始metrics留存。tier0站立proxy不能替代最终完整原canonical v3。

源更换后，相同seed的采样仍围绕不同nominal、经历不同接触响应；这是暖启动目标序列上的顺序优化，没有同总预算的从零对照，不能当作维度降低或任何单个控制参数的隔离因果证明。末段z是在新的目标曲线上再次向原站姿目标混合，不能把两轮z数值相加称为实际关节位移或恢复比例。

## 门槛、产物与边界

固定新目录计划为`logs/ollie_phase_basis_02_inspect_01`、`logs/ollie_phase_basis_02_benchmark_01`、`logs/ollie_phase_basis_02`。运行时显式传入源actual路径/SHA、独立首轮99审计路径/SHA，不使用程序默认95源、不自动选择最新目录。独立审计尚未通过时，不启动这些物理步骤；其路径/SHA和结果随后追加。

inspect为0积分数组检查；benchmark仅一个128完整分支batch及原生proof，不更新分布、不执行实际后缀。零残差slot1须完整复现首轮99的127后缀FULLPHYSICS、2709传感器和边界；所有候选3.24 s前256物理子步逐值一致。传感器构建的原物理数组/site/options、原29PD/力矩/限位、默认足板μ=.6/被动板保持。

预算门槛通过后，只执行一个已完整评估的best后缀；原生state、真实normalFn、top/robot-ground/board-ground均同阶段证明，actual执行必须exact。完整257目标/258边界/2057条Observer/1016执行力及失败尾段全部归档，原严格v3无放松。真实顶面支撑分别报告空中、接地后60 ms以及首次身体触地前的窗口，并区分.45/.55/.60 m姿态门槛；末300 ms standing单独报告。

本文为02运行前冻结登记，后续只追加真实检查/结果，不追改以上实验定义。actor/WM训练更新0；正式02结束后即停，不追加搜索。

## 运行前暂缓：scene XML冷进程序列化哈希

在02任何inspect、benchmark或正式物理步骤启动前，独立首轮99审计发现：同一冷进程首次`model_sources()`的scene XML哈希为`7d9d2c644abefa064d1e8099df857794b5304e63c82f048bdee974a0cd7925db`，第二次及以后为`59ed71b4de62ab1f83f956db5eabf4b7be7c5bc6d4c53bd0a91606ddfe9f1453`；board及mesh字节哈希相同。首轮99 manifest保存后者，因此冻结99入口在新冷进程可能会被来源门槛正确拒绝。

root要求先停在门槛前，独立审计继续比较XML差分和全部编译物理数组。当前尚不能仅凭资产字节相同就宣布物理模型相同；也不通过预调用prime、忽略scene哈希或改冻结99代码绕过。02的三个计划目录均未启动。若审计确认需要稳定的compiled-physical指纹与明确新102入口，将由root另行授权后追加新的预注册定义；原99结果和源码哈希保持不变。

## 审计结论与明确的新launcher

独立审计随后完成：`logs/ollie_phase_basis_audit_tools/scene_xml_audit.json`证实差异仅为mesh的冗余`content_type="model/stl"`在后续`to_xml()`输出中省略，481个编译模型数组和29个option逐值相同，board/mesh文件相同，0积分。首轮99完整独立审计也已PASS：`logs/ollie_phase_basis_01/independent_air_stage_audit_01/result.json`，SHA `8d7a5b5ec7ec86538b7c2b43b39f7728cf8ce09e90427abb1fa3014119de1b9b`，对应实际SHA与本轮a884…来源一致。

root明确授权新增`../scripts/local/run_ollie_phase_basis_continuation.py`，不编辑冻结99。该launcher在调用原driver.main之前，显式执行并记录0积分serialization warmup：生成首/后两个spec的原始XML及编译模型，XML仅允许在mesh元素上删除该**完全相同的字面属性**，任何名称、数值、其他元素或其他content_type变化均拒绝；board/全部mesh字节须与source manifest一致；全部模型数组和数值options须连dtype/shape/原始bytes都相同，另记录稳定的compiled fingerprint。没有创建MjData或应用源状态。

证明通过后，正常调用未修改的`model_sources()`并要求其完整返回值与源manifest逐值一致，再把原命令行原样交给冻结99的`main()`。不patch函数/返回值，不跳过driver的来源审计、模型匹配、全后缀baseline、预算或actual exact检查。warmup报告存于新鲜旁侧`<out-dir>.serialization_preflight.json`，包含launcher/driver/source SHA、首/后XML SHA及编译数组逐项hash；不提前建立out-dir，保持原driver的新目录门槛。

这与此前禁止的静默prime有明确区别：修复入口、唯一允许差异、编译物理等价和warmup本身均公开记录且是root授权的运行前步骤。若任一证明失败，只留失败sidecar，不调用原driver，也不启动真实物理步骤。

0积分合成检查`logs/ollie_phase_basis_continuation_static_01/result.json`通过：指定mesh属性省略被接受，改name或其他元素属性被拒，编译后timestep改变被拒。实际新源inspect、benchmark与正式结果随后追加；未改25维参数或任何Config字段。

## 新源inspect与唯一benchmark通过

新launcher冻结SHA `a23fad68515898015c5831da7ad069ece0eefbfd5b1d228e5df2e9152ae8a4ce`，Ruff check/format通过；原99代码仍6f0d8b…未改。使用上文actual99/audit的显式路径与固定SHA，`logs/ollie_phase_basis_02_inspect_01`通过：0积分，目标生成器与直接矩阵式差8.05e−16，零残差exact首轮99，3.24前不变、z在3.60前不变、z=1末段exact原站姿目标均通过。

首/后XML的35个冗余mesh属性删除、481数组/29options逐bytes相等均记录。稳定compiled fingerprint SHA `488559149f25001accb7bd6c07f2cf7d01301f956a47f01295297fb98400c132`；inspect与benchmark两个独立冷进程的旁侧proof JSON均为SHA `d9ff99effcdaf27f52de808ca23b0e15f47bfcd38eda5f03f9a60ac0e739dba4`。该文件记录显式warmup，不取代来源审计或正常model_sources检查。

唯一`logs/ollie_phase_basis_02_benchmark_01`通过：128完整分支rollout .480880 s，cost2.519202 s，固定32轮投影96.0026 s<600 s；没有分布更新或actual后缀。零残差slot1完整127后缀FULLPHYSICS/2709 sensors/127边界全部exact0，所有候选3.24前256子步exact。1016子步原生proof state0、真实normalFn/TOUCH0、左右top/robot-ground/board-ground同stage0，wrench误差≤1.42e−14 N。首批best仍精确首轮99，所取同最高物理grade/最低tier精英为`[1,106,109,117]`。

benchmark JSON SHA `3e3627c515de008a40bb8eaac5ad73435c6de37d2625fbc8b18a15fb3de519bd`，manifest SHA `1c459e6b674db37c8211bbc2c954221e11773d39b9efe05967f44b28049e9f6b`。根据root已授权的条件，门槛全部通过后直接启动唯一正式`logs/ollie_phase_basis_02`，配置不变；这是预登记02轮，没有追加03。

## 唯一02正式结果：触地再晚17.5 ms，承重窗口未同步变好

固定32×128完整结束，总耗时89.5410 s。winner为round29/candidate12，physical event grade2、progress tier1，新cost4.54847320808；本轮z=.1432057804，其余24角度参数及每轮完整128×25参数均原样保存。没有修改已注册代码、Config、seed或中间阈值，没有追加03。

主回合从唯一初态连续执行257目标至5.14 s，258边界/2057条Observer/1016条真实执行力完整，无partial；实际FULLPHYSICS与已评估winner逐值差0。首次身体触地3.7325 s，较首轮99的3.7150 s再晚17.5 ms；原严格v3仍为**FAILED**，失败项继续是机器人触地和末300 ms未站稳。全板9几何真实飞行仍为3.2650～3.3275 s、65 ms，峰值16.5315 mm，最大穿透9.9420 mm；末300 ms左右真实top均0，左脚触地100%、四轮同时触地100%。不把轮子稳定落地或更晚身体触地当作人板站立稳定。

真实板接地3.3375 s，右脚3.3150 s空中top41.8006 N；左脚首次top3.3750 s、48.2653 N，位于板接地后37.5 ms，仍处在既定60 ms早落地窗口内。当时身体up=.994638、根板高度.518794 m，仍不是≥.55 m站姿。

| 原canonical postforward实际指标 | 首轮99 | 本轮02 |
| --- | ---: | ---: |
| 首次身体触地 | 3.7150 s | 3.7325 s |
| 空中真实双top | 0 ms | 0 ms |
| 早落地60 ms内双top+up≥.9/高≥.45 m | 25 ms | 22.5 ms |
| 身体触地前最长双top+up≥.9/高≥.45 m | 40 ms | 37.5 ms |
| 高≥.55/.60 m的同类双支撑窗口 | 0 ms | 0 ms |
| 末300 ms左右top比例 | 0/0 | 0/0 |

本轮37.5 ms窗为3.3750～3.4100 s，共15个样本；早落地窗口保留其中3.3750～3.3950 s的22.5 ms。25/40降至22.5/37.5 ms仍在原tier1进展定义内，所以冻结的生存优先排序选择了稍晚触地的候选。这正是所注册的取舍，不代表接板质量或稳定性同步改善；不能事后把较短窗口隐藏或改权重再选。本表由完整原canonical保存记录计算，0新积分；早落地报告未额外施加下降筛选，原cost下降规则保持未改，执行阶段winner metrics也报告22.5 ms早双top。

### 来源、证明与归档

- 实际`logs/ollie_phase_basis_02/actual_rollout.npz` SHA `816e3556ed00f311bf1a12dbb4ed7f3326561607061feda134cbb0674c623596`。
- `result.json` SHA `1919bfa98fa574f629e8877ba0b932387c77e09d93eae54718c8847ed462a756`；正式manifest SHA `ed0e6887ae3c24b253a5f7afcdc882cd3e0037e142f50df932ccafa63b959819`。
- 旁侧`logs/ollie_phase_basis_02.serialization_preflight.json` SHA仍为`d9ff99effcdaf27f52de808ca23b0e15f47bfcd38eda5f03f9a60ac0e739dba4`，首/后XML差别、481数组/29options及稳定compiled fingerprint与inspect/benchmark相同。
- 保存记录分析`saved_canonical_phase_summary.json` SHA `51a44700140d94fea7d0993ca31a8d4191b6d0efaefa8b879fdbf5793122f24b`；工具为ignored logs下原有`ollie_phase_basis_tools/summarize_saved_actual.py`，没有新物理分支。
- 最终`winner_branch_proof.json` SHA `0ce7c0de935cc2aac8692c03b1fbecfaf699a93fb8cb70400615f2b5b68156cc`。1016子步state0、normalFn/TOUCH0、top/robot-ground/board-ground同stage0；force≤1.42e−14 N、moment≤1.14e−13 Nm、pose/COM≤1.33e−15、9几何≤5.55e−17 m。

代码仍是冻结99的6f0d8b…与新launcher a23fad…，原99文档和结果未改写。本轮自己的actual exact与证明不能代替独立完整重放，后续由root组织独立审计/视频；`verified_teacher=false`、`eligible_reference=false`、actor/WM训练更新0。02正式已结束，停止搜索，下一轮需另行授权，不自动启动03。

## root完整重放、视频与源码检查

通用独立原生重放已完成全部257目标：qpos/qvel边界最大差0，2057条原Observer字段exact，完整v3失败及全板峰16.5315 mm重现。产物`logs/ollie_phase_basis_02/independent_replay_01/result.json`；该通用检查与99首轮的专项双模型/双stage审计范围分开，不把它写成02也额外完成了后者。

[02实际完整视频](/home/zxc/SuperMimic/logs/ollie_phase_basis_02/video_01/rollout.mp4)与[原片完整对照](/home/zxc/SuperMimic/logs/ollie_phase_basis_02/video_01/comparison_full.mp4)均已验证解码，258实际帧/187对照帧。root已查看六帧蒙太奇，保留倒地尾段与FAILED标签；视频只渲染保存状态，没有额外动力学积分。

新增`tests/test_ollie_serialization_preflight.py`四项CPU检查通过：仅允许冗余STL mesh字面属性省略；其他XML/name变化、编译数值变化、dtype变化和timestep变化全部拒绝。Ruff通过，99冻结代码与物理资产未修改。单次续轮仅有17.5 ms首次触地收益且承重窗稍退，后续转向101诊断支持的独立髋/膝有限动作干预，不把再跑同basis当作已解决接板。
