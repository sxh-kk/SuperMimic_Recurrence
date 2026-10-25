# Ollie可回收腾空候选完整验证

2026-10-25。81已恢复真实70 ms整板腾空且侧移减小，完整回合仍在3.705 s右手触地、未站稳。其64轮产生114次`recoverable_qualified_flight=1`候选记录，意味着某个真实16子步飞行窗两只整足包络都留在预声明走廊并有deck投影重叠。**该代理仍不保证接板支持或最终站稳**。81成本全局最优原winner61/224的可回收代理为0；本次对一个单独候选做明确分开的完整物理验证，不改写81原结果。

## 不再搜索，只重建已评估目标

脚本`../scripts/local/reconstruct_ollie_evaluated_candidate.py`读取81原冻结manifest、prior、64轮完整等级/成本/elite_ids，按seed81重放NumPy RNG和目标数组更新。每轮current mean、nominal、固定prior、各级历史best参数重插入以及最高等级精英规则均逐项验证。**无mj_step、rollout、成本重评、训练或新候选生成**；仅编译原模型获取原真实目标限位。

必须逐值重建原winner真实action[150:]和保存parameters，并重建最终grade0/1/2 best的action/parameters/metadata，才能接受结果。114次记录中固定选择`(cost, round, candidate)`字典序最小的合格候选：**round63/candidate130，原已评估cost6165.3556911123405**。相同候选出现多次也只是记录数，不称114个独立成功技巧。

`logs/ollie_hop_recoverable_reconstruction_01/selection_result.json`全部通过，原winner61/224、所有级别best、全部64轮elite与special labels精确匹配。候选107×29目标NPZ SHA `23b3f5cfcaffa5b4fb410e442f42982dc5eb891442e973fea4dea8570a91969f`；源81原actual SHA `42424ce7f2e2909dc5216b3a72ad7b0ad61d30719ccfdfe1a5e3d5d7fa6b6465`，search rounds SHA `ea3f5edbdeca35c9ecddec7848df44af5f60bdd31c452f43711fad00171f836a`。原重建工具在ignored目录保留，公开工具仅整理格式并要求输出新目录。

## 单条原生验证协议

`../scripts/local/verify_ollie_recoverable_candidate.py`要求候选NPZ和selection结果的显式hash，校验所有源搜索文件、原81代码/模型/配置/依赖及旧60/reference/feedforward来源，保持原12°、默认μ=.6和原29PD。唯一初态连续执行旧60前150个控制到3.00 s，并对余下20个旧控制做scratch精确回归。

随后只试算这条**固定**107控制；全部代价项、指标及总cost必须重现原63/130搜索行，且raw2与可回收代理均通过，再做fullphysics/COM/姿态/原生执行力及双足真实normal proof。通过后才在同一个真实热状态执行这条完整后缀。没有分布更新、重新采样、目标重拟合、状态回写或换初态。最终仍保存全257控制/258状态/2057 Observer/856执行力，并按原完整严格v3验收；失败和尾段必须保留。

## 实际结果

公开重建工具已另用新目录`logs/ollie_hop_recoverable_reconstruction_public_01/`重跑，selected、action、parameters及候选metadata与原工具逐值相同。driver经独立只读审查无阻塞，运行前SHA `59173c07ed991b95412e95f196532bf9b3a8d6d84eaff4ba0cf01e7ccad62b95`。

固定验证命令：

```bash
env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --locked python scripts/local/verify_ollie_recoverable_candidate.py \
  --candidate logs/ollie_hop_recoverable_reconstruction_01/selected_evaluated_candidate.npz \
  --candidate-sha256 23b3f5cfcaffa5b4fb410e442f42982dc5eb891442e973fea4dea8570a91969f \
  --selection-result logs/ollie_hop_recoverable_reconstruction_01/selection_result.json \
  --selection-result-sha256 6eb1f72a2acb40045c0dd2b21b3d044f2aad5a8428c68ae20078aa6e6ce038de \
  --out-dir logs/ollie_recoverable_verified_candidate_01
```

单条验证已完成，wall3.928 s，无新采样或分布更新。原63/130总cost误差0，全部terms/metrics最大误差1.42e−14。856子步fullphysics差0、COM/姿态≤1.78e−15、执行力≤2.85e−14 N、力矩≤1.14e−13 Nm；真实尾Fn与足TOUCH差0，主热状态不被scratch改写。实际执行预测差0，partial=null。

严格v3结果仍**FAILED**，只剩“无人体触地”和“末300 ms稳定站板”两项未通过。真实尾地窗3.2325～3.2350 s共5 ms；真实连续整板飞行**3.2500～3.3350 s，共87.5 ms**，9几何全板最低点峰**25.8031 mm**。相对76的65 ms/23.4328 mm，该固定候选保留并略延长低跳，也满足原搜索16样本可回收代理；不能将其等同原视频Ollie的幅度、完整接板或站稳。

首次身体触地为**左手 @3.7125 s**（真实Fn572.81 N），末300 ms左足真实top>1 N支持100%、右足44.17%，四轮100%，机器人已倒地。最大穿透9.70352 mm、板未倒置，完整回合实际关节最大软越界.02904 rad仍包含旧前缀，选中后缀自身约.00295 rad；没有删除这些误差。

`independent_replay_01/result.json`和`independent_force_audit.json`独立从同一原始初态重放保存目标，确认qpos/qvel/state92、2057 Observer、856后缀执行力、全部严格v3字段误差0，原53初态与旧60前150不变；默认μ=.6/原29PD/neq0/无外力。执行力矩超限0，后缀腿最大61.96%上限，腿≥95%用时0，腕仍达到100%。实际archive SHA `5cf90dd936724fe920e15578b054724811c0368129fea253dfc6b6514e5c29f3`；独立执行力审计SHA `0900b838b1e84a30415f19b0489be08f1bff8a5e6fd5c471455680cb1cbc36af`。

完整[原片/真实物理对照](/home/zxc/SuperMimic/logs/ollie_recoverable_verified_candidate_01/video_01/comparison_full.mp4)及[实际原速](/home/zxc/SuperMimic/logs/ollie_recoverable_verified_candidate_01/video_01/rollout.mp4)已导出。实际258帧50 Hz、对照187帧24 Hz，H264完整解码核验；原片1.5倍率仍是用户估计。录像来自已保存真实qpos，显示到5.14 s、保留FAILED倒地尾段，导出不积分且轨迹hash未变。已抽查起跳、接近板面及倒地帧，`video_01/review_montage.jpg`只作QA，不替代全视频。

这条轨迹仍不是完整教师，SuperMimic actor本轮更新为0；物理可回收飞行代理的进展与落稳失败分别保留。

## 400 Hz真实几何与承重时序：尚未双脚接稳

独立从原始初态再次重放所有257目标，2057条postforward几何/Observer逐值一致，真实全七胶囊AABB与有限deck距离核验见`independent_execution_01/hot400_corridor_and_catch.json`（SHA `af3d20498c7570ac656d7f507234d9e454e892b3e3ea4972a23edc7b63708bba`）。**只有3.2500～3.2875的16个400 Hz样本满足飞行与双足完整包络走廊，按原计时为40 ms**，prior tail与15 mm peak也通过。3.2900 s右足Ymax=.150905 m，超过.15 m边界.905 mm，左仍在内；不能说整个87.5 ms飞行双脚都留在走廊内。

几何接近没有自动变成有效承重。`postforward_support_windows.json`（SHA `72cdc8bdeddeda6fadba22c77f66e9ade8698645cdf1353e5c9f429b1cb80cb8`）按原Observer真实top>1 N门槛发现：**3.15～3.70 s两脚同时有效top支持窗口为空**。右脚支持3.3125～3.3400共30 ms、3.3475～3.3650共20 ms，另有3.3775单点；左脚直到3.4850才首次有效支持，至3.5425共60 ms，二者错开。3.28/3.30 s左足虽有2.65/10.01 N实际足板normal，top支持仍为0，不能把侧/底接触、负gap或足Fn叫接板成功。

因此当前瓶颈是**双足接板承重的时序与全身姿态协调，尚未接稳**，不是已经双脚站稳以后才失去平衡。后续目标需要直接区分真实上表面支持与足板几何接近；本轮不伪造新已验证接板或学习策略。
