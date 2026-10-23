# Ollie 配对动作响应世界模型对照

## 结论与本次范围

已给 `scripts/local/fit_ollie_wm_dataset.py` 增加显式配对数据入口、整 snapshot group 留出、真实正负动作响应 loss，以及只恢复 WM 权重的新优化器初始化。完成同条件 **500+500 updates** 对照，合计1000次正式 GPU 更新，没有追加 actor 训练或修改共享训练核心。

配对 loss 明显修正训练组的虚假动作响应，但**没有在2.9 s留出组学到可信的滑板竖直响应**。不能因为训练 loss 下降或普通状态预测改善，就称动作梯度已经正确、恢复 board actor loss 或宣称完成 Ollie。

两份权重均冻结留档：

- 普通状态 loss 对照：`logs/ollie_wm_paired_control_01/model.pt`，SHA-256 `21fd12a245c885fd03c14e448ac6dc6af263640b49970455659aa98e1e3efa13`。
- 同数据加入配对 loss：`logs/ollie_wm_paired_effect_01/model.pt`，SHA-256 `69fdc1859167f8c106b851e4582f02fbeb7144ab36b024966a3ef2127dc568b2`。

完整数值见两个目录的 `diagnostics.json/manifest.json`，跨实验一致性核验见 `logs/ollie_wm_paired_comparison.json`。两者均与原 `--wm-init` 权重接口兼容；“可以加载”不代表适合用于接板或起跳优化。

## 为什么要单独约束动作响应

[38中的真实干预](38_Ollie真实动作干预配对数据.md) 从同一个完整 MuJoCo 状态复制正负分支，仅改变一个关节的真实位置目标。该数据允许检验 WM 是否预测了动作造成的差异，而不仅是拟合状态随时间的平均推进。

本批原始正负配对的 H1 board-VZ 响应差分布差异很大：

- 训练组为0.5、1.5、2.5、2.7 s，共116对。绝对响应中位数约 `3.1e-6 m/s`，q90约 `2.84e-5 m/s`，最大 `2.56e-4 m/s`。
- 留出组为2.9 s，共29对。绝对响应中位数 `3.88e-4 m/s`，q90约 `1.79e-2 m/s`，最大 `3.32e-2 m/s`。

这是接触状态变化带来的实际分布差异。不能把训练组近乎贴地的微小板竖直响应学好，就认为能推断释放支撑后的响应。上述中位数在独立 NumPy 描述中采用偶数样本两中间值平均；训练诊断使用 `torch.median`，116对时报告较低中间值约 `2.97e-6`，不是数据改变。

原有90维平均归一化 state loss 对这样的小响应不敏感，因此加入单独的配对标定项；它是工程诊断与预训练扩展，不是原论文新增算法的复现声明。

## 数据读取与严格组划分

`--paired-source` 接受38的输出目录，或其中的 `paired_branches.npz`。读取同目录的 `transitions.npz/manifest.json`，核验两文件的记录 hash，并逐项验证：

- 92维 state、29维 actual action、H8形状、规范 G1/四轮名字、50 Hz控制间隔和根四元数。
- 每条标准 transition 与配对文件中的原始 state、next_state、float64 actual ctrl完全对应；标准 `action` 与真实 ctrl的float32表示相同，不读取 expert label。
- baseline和正负分支的起点完全一致，snapshot fingerprint和group一致；除指定关节以外的28个目标与baseline一致。
- 同一group的所有正负变体和baseline拥有同一split，且与collector manifest中的group/fingerprint/split关联一致；同一fingerprint不能出现在两侧。
- 仅完整、质量有效且baseline有效的正负对进入响应loss。首步正负实际动作必须有差别；按真实裁剪后的动作推演，不假设请求的±0.05一定完整施加。

38的 group4，即 **2.9 s整组59回合**始终留出。配对训练116对、配对留出29对。普通 state loss中同样按这个固定group split使用236/59个配对回合，绝不重新做 episode 随机划分。

原始18/19/DAgger/air源继续按完整回合及重复内容组留出。相同H8窗口不得跨集合；如果配对训练分支在其他源的留出H8内容中再次出现，入口明确拒绝，避免删去普通窗口后又通过 paired loss泄漏。

本次五个实际动力学源分别是：

- `logs/ollie_pd_pilot_18/transitions.npz`：3630条、33回合。
- `logs/ollie_pd_pilot_19/transitions.npz`：3630条、33回合。
- `logs/ollie_bc_prefix_03/dagger_actual_transitions.npz`：690条、5回合，仅使用实际 `action`。
- `logs/ollie_air_augmentation_04/transitions.npz`：1600条、64回合，训练专用合成初态，不是视频观测或成功示范。
- 38配对分支：2360条、295回合。

合计 **11910条、430回合**。最终343 train / 87 heldout episodes，去重后 **6954 train / 1866 heldout H8窗口**。各源文件hash保存在manifest。旧03 DAgger文件缺失的schema/关节名/间隔仍经显式 `--infer-dagger-schema` 和受审 producer推断，并记录producer hash；没有偷偷补写源文件或把它伪装成自带schema的数据。

## 配对 loss 的精确定义

每次从训练组均匀采样group/joint正负对。两条模型轨迹从完全相同的 `initial_state` 开始，分别执行实际保存的正负H8动作序列；不在中间teacher force到真实状态，也不将已经分叉的后来状态称为同state单步对照。

响应向量 `R` 为正分支减负分支，共24个SI量：机器人和板各自的世界位置3、世界相对旋转log3、世界线速度3、世界角速度3。原生body-frame根角速度先用各自根四元数转到world，再做差；旋转使用 `log(q_positive * inverse(q_negative))`，不直接减wxyz四元数。

真实响应先在保存的 **float64原生状态**上求差，再转float32训练target，避免先舍入两个几乎相等状态后丢失微小差分。模型仍使用float32。

本次目标为：

```text
L = L_state + paired_weight * mean(((R_model - R_native) / scale_h)^2)
```

`L_state` 保持原H8、被动轮权重0.05；配对项取 **H1和H8** 平均，比较真实有限幅动作响应，没有除以名义±幅度冒充无穷小Jacobian。正负目标受裁剪的差异已在各自实际action中体现。

固定工程尺度在运行前确定，没有用留出组调参：

- robot/board位置各轴：0.001 m。
- robot/board旋转log各轴：0.005 rad。
- robot世界线速度各轴：0.05 m/s；世界角速度各轴0.1 rad/s。
- board世界XY线速度：0.01 m/s；世界角速度各轴0.05 rad/s。
- **H1 board-VZ：0.0001 m/s；H8 board-VZ：0.01 m/s**。H8接触切换与累积响应较强，不能也套用极小的H1尺度。

`--paired-horizons` 可显式选择1～8的有序不重复步数；若额外选择H2～H7，其board-VZ使用H8的0.01尺度。本次只用了1和8。

`--paired-loss-weight` 默认0，不传 `--paired-source` 和 `--resume-wm` 时维持原训练行为。本次对照即使weight0，也仍将配对的普通H8 state windows作为同一第五源；训练日志中weight0的 `paired_response_scaled_mse=0` 表示该更新未计算配对项，并非模型实际响应误差为零。真实响应精度必须看独立的 `initial_paired/final_paired` 诊断。

## 初始化、相同批次与预算

两次均从 [33的offline02](33_Ollie真实数据世界模型预训练.md) 恢复相同WM权重，SHA-256 `95d42ddbaefc5865b806419f439d788b34f86e3db71a43c568204468b1fff72b`。`--resume-wm`严格核对结构、关节与时间契约，只加载 `model`，使用新Adam，不继续旧优化器动量。

同seed0、batch128、学习率2e-4、H8、五源均匀采样、原梯度范数上限1.0。控制组weight0更新500次；实验组weight1更新500次，实际耗时约 **38.0/69.7 s**。

配对采样使用独立Torch generator，不扰动普通source-balanced批次的RNG。已验证两次初始诊断完全相同，manifest除paired weight外完全相同，普通训练batch索引SHA-256也相同：

`1c4c8d920e7ee6e01fc7273e9feba0b10344faac266a0c1156c252df27f817c3`。

因此两次之间没有混入不同普通样本采样、不同数据源或重新从零初始化等混杂因素。本次只做既定一次对照，没有依据留出结果追加调参或扩大预算。

## 响应评测：改善局限于训练组

“零响应”基准指模型完全忽略action，因而正负分支预测差恒为0，**不是把下一state预测为0**。它的误差就是原生真实响应的RMS，必须与模型响应误差一起报告。

训练组 H1 board-VZ：

- 普通状态loss对照：响应RMSE **0.0033923 m/s**。
- 配对loss：**0.00002378 m/s**，零响应基准 **0.00003091 m/s**。
- 配对模型的response gain投影约 **1.018**；只对真实绝对响应>1e-6 m/s的87对统计，符号准确率 **71.3%**，对照为57.5%。

训练组 H8 board-VZ：

- 对照响应RMSE **0.013171 m/s**，配对模型 **0.005294 m/s**，零响应 **0.005673 m/s**。
- 配对模型gain仅 **0.103**，仍明显低估较大响应；符号准确率69.2%是训练组结果，不能称泛化。

留出组2.9 s H1 board-VZ：

- 对照响应RMSE **0.010845 m/s**，配对模型 **0.010897 m/s**，零响应 **0.010911 m/s**。
- 配对模型gain仅 **0.00136**，符号准确率 **50.0%（28对）**。误差接近零响应基准说明仍未学会该接触阶段的实际竖直响应。

留出组 H8 board-VZ：

- 对照响应RMSE **0.029185 m/s**，配对模型 **0.027561 m/s**，零响应 **0.026860 m/s**。
- 配对模型仍**劣于忽略动作基准**，gain **-0.0228**，符号准确率 **48.3%（29对）**。不能将它相对普通state-loss对照的少量下降当作可信board梯度。

这里gain定义为 `sum(predicted_response * native_response) / sum(native_response²)`；不是相关系数，也不是逐样本除以很小真实响应。符号统计只计绝对真实差>1e-6 m/s的样本，避免把接近浮点/微小力效应的正负判别放大。

普通五源混合holdout的加权H8窗口NSE由共同初值 **0.5261** 变为对照 **0.4596**、配对模型 **0.5001**。这不是动作因果指标，而且混合了新增air/paired源，不能直接与33旧三源的总体数字比较。对照的state-loss变好而train H1板响应明显变坏，正说明两项必须分开评估。

## 验证、复现与后续限制

已完成：

- reader固定236/59配对回合、116/29正负对、group4整组留出。
- 四类故障注入均被拒绝：文件hash损坏、单variant跨split、snapshot fingerprint与manifest不一致、标准action与实际ctrl不一致。
- 两次CPU冒烟更新模型有限，真实配对项可反向传播。
- 不传新选项时，与提交前33脚本按相同CPU两次更新对比，全部WM权重最大差 **0**。
- 两次正式模型参数有限、初始诊断/数据划分/普通batch索引严格一致；Ruff、编译与diff检查通过。

CPU reader与护栏结果在 `logs/ollie_wm_paired_smoke_01/reader_validation.json`。

对照复现示例：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python scripts/local/fit_ollie_wm_dataset.py \
  --datasets logs/ollie_pd_pilot_18/transitions.npz \
    logs/ollie_pd_pilot_19/transitions.npz \
    logs/ollie_bc_prefix_03/dagger_actual_transitions.npz \
    logs/ollie_air_augmentation_04/transitions.npz \
  --paired-source logs/ollie_action_branches_01 \
  --resume-wm logs/ollie_wm_offline_02/model.pt \
  --infer-dagger-schema --updates 500 --batch-size 128 \
  --paired-loss-weight 0 --paired-horizons 1 8 \
  --paired-board-vz-scale-h1 0.0001 --paired-board-vz-scale-h8 0.01 \
  --seed 0 --device cuda:0 --out-dir logs/new_wm_paired_control
```

实验组完全相同，只将 `--paired-loss-weight` 改为1并使用另一个新输出目录。输出模型附manifest hash、源权重hash和optimizer；原参考、原transitions、38数据及共享 `learning.py/train_ollie.py` 未改动。

已把两个冻结权重交给负责 [36原生梯度审计](36_Ollie世界模型动作梯度审计.md) 的agent，要求用同一实际snapshot进一步对比autograd、模型secant与原生小幅动作secant。**本记录完成时该追加审计尚待回报**，上面的有限幅配对结果不能替代它。

当前证据支持“训练组的动作响应可通过配对项校准”，不支持“已具备可靠的board策略梯度”。下一步更适合补充支撑释放/接触切换附近的真实干预覆盖，并保持整个新snapshot组留出；不能把现有2.9 s留出拿去训练后仍当独立测试，更不应仅靠扩大同分布训练预算解决这个分布缺口。
