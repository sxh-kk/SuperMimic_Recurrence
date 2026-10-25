# 87｜Ollie起跳前真实动作响应先验

本实验在执行前固定以下协议。目标是用少量原生物理动作干预，构造一个可核验的起跳前横向动量目标先验；不是训练策略、世界模型或替代SuperMimic算法，也不是已验证教师。原12°板、默认足板μ=.6、原29PD/力矩/关节限位、重力及严格v3不变。

## 首次执行前固定的预算与配置

- 从原始唯一初态，真实重放82实际回合的前145个目标到2.90 s；不复制热端点，不重置相位。82源SHA `5cf90dd936724fe920e15578b054724811c0368129fea253dfc6b6514e5c29f3`。
- baseline是85已执行的 `action[145:]`；85源SHA `fc0a113a3003f2ff669510cbfe3ad4d6f0ff31f05ace9738724707d91167a9d6`。前145个目标须与82逐值相同。
- 所有分支从同一个真实热 `MjData` 深复制，保留FULLPHYSICS与qacc_warmstart；主场景始终停在2.90 s。无外力、根/板/状态写回、相位reset。
- 仅干预前8个20 ms控制，即160 ms。正弦幅形固定为 `sin(pi*t/.16)`，按控制区间中点 `t=.01,.03,…,.15 s` 采样；目标为原实际目标加 `wave*delta`，随后按原targetclip裁剪。中点采样使首末ZOH目标偏移较小，不声称PD目标连续。
- 固定59个短分支：1 baseline，加29关节各±.03 rad幅度。任何裁剪或非比例扰动都记录；该列可列无效，不能称对称中央导数。有效列用实际目标offset对wave的投影幅度差作分母。
- 只进行一次有界 `scipy.optimize.lsq_linear`。末3.06 s目标以板heading横向CP到−.03 m为主要修正，其余保持baseline：相对COM世界VZ、root−board世界高度、robot upright、board upright、board世界Z、左右真实foot site板系XY。尺度依次 `.04 m/.25 m s⁻¹/.05 m/.1/.1/.02 m/.05 m`（最后四个XY分量共用.05 m）。
- 最小化归一化末态线性残差平方和，加 `.1*||delta/.15||²`；腿12维幅度界±.15 rad，腰臂17维±.30 rad，另与原targetlimits对应的正弦幅度可行区间取交集。没有改PD或力矩。
- 线性求解后仅验证一次8控制修正，再验证一次完整112控制后缀：总共**61个分支**。不做line-search、网格、随机搜索、第二次线性迭代或训练；墙钟上限120 s，超时保留已有原始结果，不重试优化。
- 新先验只包含修正前8个目标＋剩104个原85目标，形状112×29；不应用保存的qpos/qvel。完整候选由同一热场景scratch真实执行，补上已核验原前缀后按原严格v3验收；丢掉原flight、提前floor、穿透超标或未站稳均明确失败。

## 时刻与证据边界

短分支baseline每个20 ms端点的qpos/qvel/state92必须与85存档逐值相同；原145前缀还与82存档及所有canonical Observer行逐值比较。分支不能靠冷设同一state92替代同一求解器历史。

末态特征在真实积分结束后，以独立scratch `mj_fwdPosition`、`mj_jacSubtreeCom @ actual_qvel`计算。CP使用83相同的世界水平位置、相对COM水平速度和板heading公式；它是LIP软启发式。保存机器人和板各自真实COM世界速度，不能把根速度混称COM速度。

执行力在每次`mj_step`之后、额外`mj_forward`之前归档，属于本次执行solve和积分前几何；canonical Observer在额外`mj_forward`后归档。末态位置/Jacobian特征是3.06 s的实际位置速度，不把前一子步传感器缓存当作末态COM或末端力认证。

源码入口：`../scripts/local/solve_ollie_prelaunch_balance_prior.py`。首次输出固定为 `logs/ollie_prelaunch_balance_prior_01/`；本节冻结时尚未执行。后续结果只追加实测，不修改上述样本、目标、权重、幅度或筛选。

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/local/solve_ollie_prelaunch_balance_prior.py \
  --out-dir logs/ollie_prelaunch_balance_prior_01
```

## 唯一实际执行：61分支、1次线性求解

上述固定运行已完成。共59个原生响应分支、1个修正8步验证、1个完整112步验证，**没有追加迭代、line-search、网格、随机搜索或训练**。程序报告至结果构造的墙钟耗时3.6966 s，随后完成原始归档，低于120 s预算。执行源码SHA：

`7e6a279a7f1cb733d1f1193dcb91b0e48be5e998328721043294cd68bc28f839`

canonical资产与85冻结manifest逐项相同。真实145前缀的qpos/qvel/state92及所有Observer行逐值相同；短baseline到3.06 s的9个20 ms端点qpos/qvel/state92误差均0，与85原评估branch前64个FULLPHYSICS子步误差0，与85保存执行力误差0。证据为`baseline85_exact_proof.json`，没有用冷设state92替代热历史。

29个关节中26列满足未裁剪、正弦profile比例一致；`left_ankle_pitch_joint`、`left_ankle_roll_joint`、`right_ankle_roll_joint`有目标裁剪，保存了全部正负分支，但**没有将这3列用作中央响应矩阵**。每列实际有效幅度和profile误差在`response_clipping.json`，完整59×64实际FULLPHYSICS和动作在`native_responses.npz`。

仅一次`lsq_linear`收敛，内部11次求解器迭代属于这一次有界最小二乘，不是11次物理试验。其预测与真实修正为：

| 3.06 s指标 | 原85 baseline | 线性预测 | 一次真实修正 |
| --- | ---: | ---: | ---: |
| heading横向CP | −.151526 m | −.076955 m | −.090967 m |

真实改善60.559 mm，但没有达到−.03 m目标。有效有限幅响应中，CP对幅度的较大割线为：right hip roll `−.137670 m/rad`、left hip roll `+.137523 m/rad`、waist pitch `−.083451 m/rad`。这些是±.03 rad、160 ms真实动作分支的条件有限割线，不是已认证光滑Jacobian；响应正负均来自同一个热状态。

解中多个腿幅度达到±.15 rad，left shoulder roll达到−.30 rad，意味着对有限幅响应外推了5～10倍。没有因此扩大机械或力矩限位，也没有为了缩小预测误差再求解或重试。完整数值见`one_linear_solve.json`和`cp_response_directions.json`。

## CP改善的来源与保持量误差

这次主要改变了被动板与机器人之间的相对运动，不能称为已消除机器人的世界横向动量：

| 3.06 s真实量 | baseline | 修正后 |
| --- | ---: | ---: |
| robot COM世界Vy | −.142291 m/s | −.134452 m/s |
| board COM世界Vy | +.187487 m/s | −.007225 m/s |
| robot−board COM世界Vy | −.329778 m/s | −.127226 m/s |
| robot−board COM世界Vx | +.092260 m/s | +.342033 m/s |
| robot−board COM世界Vz | +.167206 m/s | +.086324 m/s |
| root−board世界高度 | .765531 m | .773416 m |
| robot upright | .930852 | .920768 |
| board upright | .999345 | .999332 |
| board世界Z | .081543 m | .081617 m |

左foot site板系X/Y分别偏移`+.021747/+.012265 m`，右脚分别`−.002852/+.000044 m`。所谓“保持”是已预注册尺度下的软最小二乘项，没有被硬等式强制满足。CP相对量的改善主要来自板COM速度变化；板没有执行器，这仍是原足板接触产生的真实动力学，而非直接给板施力。

本次没有从以上相关性断言某一个关节或某一保持量是丢失起跳的唯一原因。单个末态的有限响应近似和10项软保持，尚不足以保持整段launch接触历史、各关节速度及后续原85控制的适配性。

## 完整续接验收：FAILED，不可晋升

修正前8个目标后，其余104个目标逐值沿用原85，仍从2.90 s同一热scratch连续执行至5.14 s。主场景本身保持2.90 s，末指纹逐值不变；所有61个分支均只控制原29PD目标。

完整严格v3为**FAILED**：

- 没有合格整板飞行窗口；全板最低包络峰值仅`3.452758 mm`，未达到5 mm连续净空及15 mm峰要求，**没有保留原85低跳**。
- 尾部仍有真实地面接触`3.2100～3.2225 s`，但不能拿尾触地替代腾空。
- 首次人体触地`3.4025 s`，早于原85的`3.8225 s`。
- 最大实际穿透`20.338655 mm`，超过原10 mm门槛；板未倒置，但末300 ms没有稳定站板。
- 动态实际关节最大软越界`.034254 rad`仍照原样记录；目标在原targetclip内，不等于动态实际关节没有软约束误差。

没有更改严格验收、补接触、删除失败尾段、施加外力或重新设定根/板状态。`verified_teacher=false`、`eligible_reference=false`，**此文件只是一份失败的动作先验和实际动作响应数据，不是可用教师或新推荐参考**。CP局部方向改善与完整动作失败分别保留，不据此宣称原模型或SuperMimic不可行。

归档中`prior_targets.npz`只含112×29动作目标、29名字、delta和资格标记，**没有qpos/qvel**；`actual_rollout.npz`另外保存完整257动作、258状态及2057条Observer对应的实际物理回合，供独立审计或失败视频使用。使用者不能拿实际证据中的状态覆盖仿真。

主要输出SHA256：

```text
4f24a2d8eae3d9fa7fb37b88869f44a718d002334915917469255190a0907d4b  prior_targets.npz
66fe3b76e53e7c117143225bca8e026693a29c853600c56ad2293cef4bb04724  actual_rollout.npz
5bfcf0e87cf6bd265dc3069eb8603b8d20cec4bb0e7977ef0bc7d023ce721770  native_responses.npz
edbc20cb5eebc63156a23ace61da31db02bb92fe6d6fecc6f810550ad9db66d4  frozen_manifest.json
0f1a0298d7f268016c942f721c8f9b567e0ee27ed08c7fd5b6a4d77f9677b797  result.json
```

Ruff检查、格式检查及Python语法检查通过。该响应试验没有训练；后续root另行完成以下独立重放和失败视频导出，没有追加优化。

## 独立重放、数学核验与完整失败视频

`independent_replay_01/result.json`从唯一原初态重放全部257目标，258个qpos/qvel边界、2057条canonical Observer与原严格v3全部逐字段相同。`independent_response_audit_01/result.json`另以已存59组FULLPHYSICS纯FK/Jacobian重算全部末态特征、有效幅差与26列Jacobian，误差0；目标尺度、原限位幅度交集、正则、预测目标和唯一求解目标值一致，KKT残差2.11e−15。审计没有重新积分或求解。其SHA `d99f99c012b3d9ee88a07b958ff52076e573e3707e9c005086e18ac8afe0df45`。

已导出[完整原片/实际物理FAILED对照](/home/zxc/SuperMimic/logs/ollie_prelaunch_balance_prior_01/video_01/comparison_full.mp4)，保留所有倒地尾段；[实际原速视频](/home/zxc/SuperMimic/logs/ollie_prelaunch_balance_prior_01/video_01/rollout.mp4)的六帧已查看。此失败候选不替换86低跳基线，也不用于actor训练。

![87纠偏候选完整失败抽帧](/home/zxc/SuperMimic/logs/ollie_prelaunch_balance_prior_01/video_01/review_montage.jpg)
