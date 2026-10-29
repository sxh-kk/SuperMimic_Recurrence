# Ollie保留接板事件精英与落地起身

2026-10-29，运行前固定。95首次真实左脚在板落地后30 ms承重，独立原canonical top=51.953 N，早落地双top+蹲姿窗15 ms，未站稳。其64轮的每8轮快照几乎都只有1个tier2候选，其余为tier3；旧`highest_grade_elites`仅保留物理grade2，仍会用较差未接板候选填满16精英。这使稀少的接板进展在分布平均时被抹去。

## 修改与边界

新`ollie_progress_elites.py`提供成对闭包`bind_progress_elites(cost)`：成本每批捕获显式`progress_tier`、grade与score，selector核验同批、显式schema、tier评分区间并单次消费，先最高物理event grade，再最低接板tier，只取1～16个精英，不以低进展填满配额。它不改76协议，不靠全局patch，也不只从浮点分数猜tier；26项独立测试覆盖单一接板不与15个未接板平均、原grade优先、评分溢出、跨批缓存和非法输入。

新`optimize_ollie_progress_preserving_landing.py`复用冻结95的成本/几何/传感器与原严格v3，将零残差nominal换为完整实际95的127个PD后缀，首130控制仍是共同唯一原初态连续前缀；输入完整source与独立双阶段审计先验哈希都固定。source95还只是FAILED探索，不是教师；其qpos/qvel不会设进执行回合。

此外保留精确95与原82种子，增加实际95的纯29PD站起blend：3.38 s开始、时长.16/.28/.40 s；另有3.38 s起保持当时目标、3.60～4.10 s再渐变原初始站姿的种子。首3.38 s前动作逐值不变，原增益/力矩/目标限位不动，候选在首batch完整物理评估，不能凭目标曲线宣称站稳。选择器、nominal与这些种子同时改变，不作隔离单因素因果结论。

P130/H127、256×64、17结点、σ=.10/floor=.025、seed97、1200 s预算；新Config显式声明stride4、4个tier与每tier内部cost<3。原物理模型/足板μ=.6/被动轮、29PD、完整v3、末300 ms站立高度.55 m及无身体触地均保持，actor更新0。预检`logs/ollie_progress_landing_benchmark_01`，正式`logs/ollie_progress_landing_01`；结果随后追加，不预先认定成功。

## 唯一正式结果

64轮234.459 s，胜出第9轮/154号。真实早落地双top+upright≥.9/高度≥.45 m由95的15 ms增至**22.5 ms**（3.3725～3.3925 s），进入tier1。末轮只保留2个同进展精英，未用较差tier填满16名额。原严格v3仍**FAILED**：全板离地65 ms，最低几何点峰16.57485 mm、最大穿透9.83072 mm，3.5675 s左脚触地，末300 ms双足top均0。没有空中双top，也没有.55/.60 m姿态承重窗。

板实际3.3375 s重新接地，左脚3.3725 s首次top，晚35 ms。首双支持左/右56.56/418.87 N，左真实接触法向·板up=.5564，右为1，side/under均0；左鞋底·板up仍−.341、右+.058。真实顶面支持的出现没有同时得到平脚与均匀承重。

首双支持窗板heading中COP约[−.334,+.0196] m，COM投影[+.0639,−.1142] m，LIP软CP[+.1438,−.1777] m，CP−COP=[+.4780,−.1973] m，末窗X差增至.5664 m。CP仅为软诊断，不是完整接触稳定判据；其与实际接触分布共同说明身体虽直立，侧向位置和动量仍未回到承重点附近。不能仅凭触地变晚宣称落地稳定。

## 独立审计与视频

两模型各执行全部257目标/2056子步，258边界qpos/qvel/state92、2057原Observer、1016后缀执行力全部0差；exec/post各2056点逐接触Fn/位置/法向、top、robot-ground/board-ground均0差，无overflow。source95 nominal、5个95/站起/保持种子与82种子逐值重建一致；64轮每批精英ID独立重算通过。目标和执行力矩超限0；腿部最高81.8707%上限，全回合实际q软越界最大.025165 rad。通用重放也exact。

- 实际归档：`logs/ollie_progress_landing_01/actual_rollout.npz`，SHA `21b6ef7c0f6fbc0b1b1a39e3dba7dc575d19298c8d183463b9abc7424f81cc64`。
- 独立审计：`independent_air_stage_audit_01/result.json`，SHA `102ec417c316af89b66004181e1cc6036034e71d686607c79b4e5657909fa8e8`。
- [实际完整视频](/home/zxc/SuperMimic/logs/ollie_progress_landing_01/video_01/rollout.mp4)及[原片完整对照](/home/zxc/SuperMimic/logs/ollie_progress_landing_01/video_01/comparison_full.mp4)：258实际帧/187对照帧，已查看六帧蒙太奇，倒地尾段与FAILED标签保留。

目标生成2项、显式进展选择器26项测试及Ruff通过。它是较早真实接板的探索来源，**不是合格教师或SuperMimic策略**；下一轮99用低维协调目标和恢复后的首次真实触地优先，原物理和严格v3不动。
