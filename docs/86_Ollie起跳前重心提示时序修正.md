# Ollie起跳前重心提示时序修正

2026-10-25，实验前固定。85修复选择退化并实际完成60 ms/21.4812 mm低跳，首次身体触地延迟至3.8225 s，穿透9.06774 mm合规；实际接板双top仅30 ms且根板高度约.424 m、up约.865，仍倒向板−Y，FAILED。

## 发现与固定修改

只读COM核验发现2.90 s横向相对COM位置/速度−.0366 m/−.0385 m/s，CP软值−.0462 m仍在.065 m提示区间；3.00 s原82/84软值已分别−.0897/−.0988 m。83源码虽声明“只有top奖励延迟”，实际CP项复用同一个`weighted()`，也要等首40 ms飞行完成才启用，错过起跳前纠偏。证据`logs/ollie_lateral_momentum_comparison_01/result.json`只读取既有实际轨迹，没有新积分。

新入口`../scripts/local/optimize_ollie_prelaunch_capture_landing.py`保持原83/84/85/top/76冻结文件。用相同世界水平/板heading坐标CP公式、目标和600权重，将CP加权窗换成**原postflight ramp/window与优化起点之后前160 ms的并集**，仍只计算一个加权均值。前窗对应[2.90,3.06) s，**3.06 s之后至首40 ms飞行完成之间仍没有CP权重**，不是整个起跳阶段都在约束重心。同事件等级内仍优先首次身体触地时间，CP只是次级软提示；不把启发式LIP值当成接触可行性证明，top奖励仍只在真实40 ms飞行后启用。新旧CP/成本逐候选分别归档，不修改旧记录。

同时把85已完整实际执行并独立验证的112个后缀PD目标作为新增首prior，SHA `fc0a113a3003f2ff669510cbfe3ad4d6f0ff31f05ace9738724707d91167a9d6`。原82仍为nominal与唯一原初态连续145目标前缀，原三个prior保留；仅读85 action作目标，不把其qpos/qvel设进主回合，不追认85为合格教师。这是调度修复加真实候选热启动，不能声称隔离单因素对照。

其余P145/H112、全29PD、256×64、最高等级16精英、σ .10/floor .025、600 s预算、原模型/摩擦/力矩与whole-geometry gate/首次真实触地优先/严格v3均不变；固定seed86。先静态review，再一次完整batch预算和同stage真实top/ground/分支校验，通过后完整64轮、原初态唯一主回合至5.14 s，独立审计与完整视频。

```bash
env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --locked python scripts/local/optimize_ollie_prelaunch_capture_landing.py \
  --mode benchmark --out-dir logs/ollie_prelaunch_capture_benchmark_01
```

正式目录`logs/ollie_prelaunch_capture_landing_01`。这仍是SuperMimic物理教师探索，未引入PPO，actor更新0。

预检通过：一完整batch rollout .8045 s、cost 2.6767 s，64轮投影222.80 s，低于600 s预算；无分布更新或主后缀执行。winner slot2为保存的actual85目标prior，原生896步FULLPHYSICS误差0、真实ground误判0、新winner双足top误差0，接触法向最大误差1.42e−14 N；first ground仍3.8225 s、9.06774 mm穿透、30 ms双top，证明真实热启动保留原动作，没有凭修改成本虚增物理改进。

## 静态核验与完整实际结果

`logs/ollie_prelaunch_capture_review_01/result.json`在0原生积分、0 GPU初始化下通过：旧post-only公式等价误差1.82e−12，前窗恰为64个400 Hz样本、并集权重不超过1；无飞行时仍保留前窗而关闭post窗。原82 nominal零残差逐值一致，新增首prior仅取actual85动作，前145目标共用、全部目标限位合格，主状态fingerprint未变。这是算式与接口核验，不替代动力学结果。

唯一正式64轮完成**167.153 s**，winner round63/candidate194、优先cost1036.542635。从原唯一初态连续执行257个29维目标至5.14 s，完整258边界/2057 Observer/896后缀执行力保留，actual_prediction误差0、partial=null。入口冻结SHA `6d5477cf65703576a9a2d9097433f21558bb1c16533910943463e210b10b7a5a`，actual归档SHA `b7866c9421eaaefd727e0774756a000e18013b65dcc8061e75e35f3ca674fc8f`。原12°板、默认足板μ=.6、29PD、重力/惯性/力矩上限不变，没有外力、板驱动、阶段reset或根/板回写。

原严格`whole_board_geometry_top_support_v3`仍为**FAILED**：

- 真实全9几何整板飞行3.2525～3.3075 s，共**57.5 ms**，最低点间隙峰**18.8896 mm**；尾地3.2325～3.2350 s，共5 ms。
- 首次有效身体触地为**3.8475 s右手**（`right_hand_collision`），比85再晚25 ms，但仍未落地站稳。此时身体upright−.18663、root−board高度.15888 m，已倒下。
- 全回合最大穿透**9.9221 mm**、板未倒置，通过原10 mm几何门槛；最小身体upright−.28531，实际软关节越界最大.02904 rad另记。
- 末300 ms左/右脚真实top>1 N支持100%/0%，四轮接地29.17%。原严格失败仍为身体触地和最终稳定站板两项。

与85相比，触地仅推迟25 ms，飞行时长和间隙反而减少；同时更改CP窗、加入actual85先验并更换seed，不能将变化隔离归因于CP修正。这个固定结果没有证明全起跳重心可行或完整Ollie成功。**本轮actor更新0、SuperMimic训练0，无新策略检查点。**

## 独立双阶段接触核验

`independent_top_stage_audit_01/result.json`从原初态将相同257目标在原模型和2453维只读观测模型固定重放。新审计脚本`logs/ollie_top_landing_audit_tools/verify_winner_stages86.py`核验86 caller、85/84/83引用、top/协议及warm_source哈希，原85审计文件不改；脚本SHA `c25f7338d6165614cd86fc1e3323cbfe4ceb2887c51b110f27ca588acfa31f97`。完整物理状态、258边界和全部2057原Observer逐值一致，执行求解/额外forward两阶段各2056样本，逐接触Fn/位置/法向/count及真实ground误差0，top合计仅2.84e−14 N浮点求和差。单组top最多14接触/容量32、ground最多10/容量64，无overflow。通用`independent_replay_01/result.json`另确认qpos/qvel边界、9几何包络与原接触字段一致，实际执行力矩超限0。

两阶段都测得：首40 ms飞行完成后、真实触地之前，双足同时top>1 N的最长窗口为**3.5400～3.5675 s，共30 ms**，另有2.5～5 ms零散片段；加入upright≥.9与root−board高度≥.6 m后最长仍**0 ms**。未达到同一个50 ms姿态承重窗。执行首触地几何/区间起点为3.8475 s，对应返回data.time3.850 s；postforward则在data.time3.8475 s判定，同阶段各自与原Observer一致，没有移位比较或将postforward力当执行冲量。

v4 `independent_force_audit.json`又独立核验896条后缀执行力的全部字段，最大误差0；qpos/qvel/state92、2057 Observer及strict全部字段逐值一致，warm85/82 prior与source链全部匹配。真实目标和执行力矩超限0，后缀腿部最高54.7117%上限、腿≥95%时长0；腕达到100%，不能泛称所有关节都未饱和。原生累计边界相对理想时间残差5.06e−14 s单列。报告SHA `218ae58e3e269426851a9eb67ba553da6a5d74e533bf6baf8d091cad95adeac8`，独立脚本SHA `1d5f09db57f64875f62cf3f521638a29e313fb5e321ed05e244c2a0e9b02052b`。

`top_direction_diagnosis.json`只读86/85/84已存sensor与82既有诊断，新增积分0。86在3.54 s两脚真实top已达110.16/94.97 N，但身体upright仅.86874、高度.42459 m；COM重力线投影Y−.24457 m，而实际top接触Y范围+.01001～+.08135 m，已向−Y超接触包围盒254.59 mm。脚在+Y一侧偏晚接板、身体继续向−Y倒的主要关系仍在，未因短前窗提示消失。重力投影与包围盒仅用于描述失衡方向，不能单独认证动态不可恢复；倒下及右手触地由完整物理回合确认。

## 完整FAILED视频

[原片/真实物理完整对照](/home/zxc/SuperMimic/logs/ollie_prelaunch_capture_landing_01/video_01/comparison_full.mp4)及[实际物理原速视频](/home/zxc/SuperMimic/logs/ollie_prelaunch_capture_landing_01/video_01/rollout.mp4)保留5.14 s失败终态。`video_01/video_audit.json`记录258实际帧/187对照帧、源归档未改；实际视频SHA `b7bc0d54af1efd989fbda2fb3e2c55388b3419ff960e684675d8680377e4c2f7`，对照SHA `5afe53bdbc6fc17db8432ebc21db7d60cb284c7564b4960ebf352cc44ece21cf`。已抽取并查看实际145/163/177/185/193/257帧，覆盖起跳、短时接板及倒地尾段，毫米间隙与接触时长仍以400 Hz审计为准。

![86实际回合六帧，完整保留FAILED尾段](/home/zxc/SuperMimic/logs/ollie_prelaunch_capture_landing_01/video_01/review_montage.jpg)
