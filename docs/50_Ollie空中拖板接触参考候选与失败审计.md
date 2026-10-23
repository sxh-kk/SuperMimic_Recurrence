# Ollie 空中拖板接触参考候选与失败审计

本轮实现了保留整体COM节点、固定板轨迹和准备前缀的空中接触重定向，但正式v3仍 **FAILED**。安全回退后几何、真实关节范围和12rad/s通过，前脚拖板与落地接触却没有达成，不能送去训练或称为可执行参考。

源视频支持前鞋侧/尖沿nose拖板和后脚较早回板，见49文档。本次局部IK失败不证明原片无解，也没有把请求的接触标签当成已满足接触。

## 正式输入与输出

- 新脚本：`scripts/video/repair_ollie_air_contacts.py`。
- 输入46最终v3：`data/ollie/ballistic_tail_upright_candidate_v3/joint_reference_50hz.npz`，SHA256 `1611f169de8e5df7442aadccabd309f2ecbc33b9acd1f530ca9f774c68b997d8`。
- 正式失败候选：`data/ollie/air_contacts_v3/joint_reference_50hz.npz`，SHA256 `39cb87079fc94d0b40e8df65f54c37264770a65a14b3600271eb89101aeced54`。
- 诊断：`logs/ollie/air_contacts_v3/`，状态 `FAILED_KINEMATIC_OR_TEMPORAL_CHECKS`，`qa.passed_geometric_thresholds=False`。

这是运动学候选，没有真实仿真回合、外力、weld、物理成功或已测接触力。原视频、旧参考、教师均未覆盖。

## 显式目标与不变量

保护原准备phys≤2.76s全部92维state逐值不变；相邻2.78s位姿也保留，以避免重算中心差分速度时偷偷改变保护前缀。首个可改节点为2.80s，过渡不能伪称源片已经测到接触。

候选仅改根、腿、必要腰；板全部21维state和14个手臂关节q/qd逐值保留。源contact标签全部原样保留：请求的工程接触写入独立 `requested_contact_targets.npz`，实际几何见证另存，绝不因目标存在就补 `contacts=1`。

目标分阶段设置，全部属于需检验的工程先验：

- 左/前脚2.80..3.28s从center向nose滑动；之后向landing锚点过渡，期望3.56s前回板。
- 右/后脚原有tail接触至2.90s保留；之后释放。3.14..3.32s是明确不标接触的接近轨迹，3.32s起尝试catch。
- 用真实有限center/tail/nose盒上面或上边缘与7个真实足胶囊建立距离目标，脚方向自由，不强行把脚Z轴对齐板法向，不把平脚掌焊到nose。
- 4.0s位姿回原恢复姿态，4.02s起全部state回源；末段3.6..3.98增加原姿态正则，同时以双向可达盒约束返回4.0s时的关节速度。没有无记录延迟视频落地时刻。

46的25个主COM节点phys3.10..3.58s为硬目标。用全系统COM Jacobian消去机器人根平移自由度，每次关节/根旋转更新后再按机器人质量分数精确补偿根平移。板质量2.18kg、机器人33.341142kg，补偿不是施加外力。

只保留节点位置不等于全部动态条件：最终节点COM误差1.67e−15m；2.5ms插值COM相对输入最大变化0.0737mm，Jacobian得到的COM速度与真实节点位置中心差分仍最大相差0.13476m/s。速度由修改后的位姿重算，没有为了让COM速度好看而补写根速度。

## 版本与正确性修复

首轮v1曾出现板-腿连杆33.8mm穿透和返回原姿态时40rad/s速度，明确失败。v2补上末端双向12rad/s约束，但其回溯只看接触误差，遗漏了局部最小二乘中已有的碰撞和原姿态项。

根代理复核指出这一实现不一致后，本轮只进行一次v3正确性修复，权重、板轨迹、COM目标和速度限制全部不变：回溯 merit 现在包含与局部LS相同的原姿态平方误差、接触平方误差、时间区间碰撞违反量平方误差。

数值核对LS零步残差cost与merit最大差1.82e−12。共779次迭代，接受728次；56个求解帧总cost下降，33帧峰值穿透下降。所有接受步骤cost均下降。原peak门槛保持 `peak≤max(old_peak,0.5mm)+1e−7m`，因此允许在0.5mm裕量内小幅增加；实测最大增加0.438mm，不能描述成严格峰值单调不增。

几个原始局部解的迭代例子，均尚未经过安全回退：

- 3.10s：前脚最近有限板gap203.286→47.091mm，cost1861.60→219.46，仍没有接触。
- 3.20s：峰值穿透21.593→19.460mm，cost920.39→785.27，仍严重不合格。
- 3.32s：左右足gap61.718/32.140→4.392/0.092mm，局部接触明显改善，但不能代表整段通过。
- 3.50s：左右足最终仍154.611/96.258mm；3.56s仍63.862/43.410mm，落地接触未达成。
- 全部原始局部解按2.5ms插值后，自碰3.481mm、板碰33.018mm、地面3.094mm，明确不通过。

## 安全回退后的正式v3

脚本沿原轨迹到局部修正的广义位姿差进行全轨迹回退，逐档检查完整2057姿态，且每档重新投影到相同COM节点。只保留3.125%修正后才满足几何/速度检查；这不是放宽门槛或假称目标已实现。

正式失败候选结果：

- 自碰最大1.312586mm、机器人-板1.640700mm、地面0.107736mm，全类别无>2mm穿透。
- 真实关节范围违规0，邻帧速度最大11.911480rad/s。
- 准备全state、2.78s位姿、板全state、手臂q/qd、源contact标签、4.02s后全state均逐值保持。
- 前脚拖板193个致密样本：≤2mm有限上/边面见证比例0，最大gap187.500mm。
- 前脚landing287样本：比例0，最大gap213.758mm。
- 后脚catch272样本：比例3.676%，最大gap296.498mm。
- 旧右脚tail窗口41样本比例100%，说明没有把已存在的准备接触当成新空中接触成功。

组件审计仍显示长时间机器人与板分离；板无支持时非弹道残差仍存在。整体COM节点可满足不能补救没有真实力路径的分离组件。这版只用于失败诊断，不是合格新教师或BC输入。

## 标签安全更正

原v2临时归档曾把94个“请求的工程目标”写成contact=1；虽然整体QA是失败，也不允许把这种标签交给下游冒充满足。原归档保留但撤回使用。

标签安全副本：`data/ollie/air_contacts_failed_v2_safe/joint_reference_50hz.npz`，SHA256 `f4b3a84e77c6b1852695e2d89e4a297f434a8ab557a28033121d67b0e95458f0`。它与原v2的92维state逐值完全相同，只恢复源contact标签和明确失败元数据。证明在 `logs/ollie/air_contacts_failed_v2_safe/label_safety_provenance.json`。正式v3与脚本已经保留源标签，不再有此问题。

## COM固定与足部可达的静态对照

另在7个时刻进行相同目标/权重的简化静态局部解，比较固定COM与允许COM移动。该诊断不生成新轨迹，不含时序速度或碰撞约束，每个局部解最多100次函数评估；所有解都没有达到求解器终止收敛标准，不能当全局不可达证明或合法落地解。

早段3.10/3.20/3.30s，即使固定COM，指定接触见证点距离也可逼近0.8mm目标，但完整足部/身体仍出现约5..15mm穿透。释放COM没有自动消除穿透。因此早段不能简单归因于COM太高，有限足部形状、碰撞和连续路径同样限制当前方法。

晚段差异更大：

- 3.50s固定COM的局部解，前/后脚真实最近板gap约133.7/93.2mm；允许COM移动约204.8mm的局部解能逼近指定触点目标，但整体仍穿透13.5mm。
- 3.56s固定COM仍约85.5/60.4mm；允许COM移动约127.0mm可逼近触点目标，但仍穿透8.7mm。

这提示旧整体抛物线与落地可达目标存在实质冲突，尤其晚段；但COM-free触点残差小的解同样不合法，不能拿它直接作为新landing参考。应先从已几何验证的站姿和目标板姿态构造有来源的landing COM，再联合处理完整有限几何、时序和力学。该下一候选本批未执行。

## 可复核输出与重跑

`logs/ollie/air_contacts_v3/`：

- `result.json`：严格失败状态、不变量、完整几何和组件必要条件。
- `iteration_trace.json`、`iteration_trace_summary.json`：每次接受/拒绝、各项cost、峰值穿透和真实有限gap。
- `raw_local_ik_qpos.npz`：未通过的局部原始位姿，不能用于训练。
- `surface_witnesses.json`、`requested_contact_targets.npz`：实际几何与请求目标分开。
- `actual_finite_gaps_400hz.npz`：完整2057×左右脚×3有限盒的有符号距离、最近点对、盒子上半部见证、脚胶囊ID和参考hash；同名最小gap CSV便于检查。
- `after_samples.npz`、`com_scope.json`：真实位姿/速度导出的COM和接触图，没有伪造力。

静态简化对照在 `logs/ollie/air_contacts_failed_v2_safe/static_com_reach_ablation.json`。源视频二维证据在49文档对应目录。

重跑需要新目录，不能覆盖当前已绑定候选：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/video/repair_ollie_air_contacts.py \
  --source data/ollie/ballistic_tail_upright_candidate_v3/joint_reference_50hz.npz \
  --out-dir data/ollie/air_contacts_new \
  --diagnostics logs/ollie/air_contacts_new
```

已实际完成v3完整CPU流程、2057姿态严格审计和7时刻静态对照；Ruff格式与检查通过。没有进行神经训练或把失败候选运行成成功视频。
