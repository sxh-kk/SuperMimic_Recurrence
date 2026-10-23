# Ollie 准备段行为克隆与真实闭环初始化

## 当前结果

已将文档29的实际准备/下蹲控制接入 `JointResidualPolicy`，并在原生MuJoCo中独立闭环运行。推荐实验初始化为 `logs/ollie_bc_prefix_03/best.pt`：累计7000次监督更新，包含行为克隆与DAgger数据聚合，**BPTT更新数为0，世界模型尚未训练**。它是准备段初始化，不能称已学会Ollie。

推荐检查点在0～2.76s的独立纯网络闭环中，没有机器人触地，最小直立余弦0.85603、最大穿透6.665mm、root-board XY最大196.516mm，末300ms双脚板面向上支持和四轮同时接地占比均100%。它延续了原前缀的窄余量和动态末状态，不认证扰动鲁棒性或静止平衡。

另将该网络回合保存的138个实际动作在新建模型中重放，1105个含初帧物理采样的qpos/qvel与原记录一致。逐2.5ms重算足板目标偏差：高度最大5.751mm、平面滑移26.596mm、根高RMSE19.644mm/最大68.779mm，均在原有限准备窗对应阈值内。审计位于 `logs/audit/bc_prefix_03_best/`，包括全几何包络、真实支持力和 `prefix_tracking.json`；仍保留实际软关节角偏差，未把这项说成严格零越界。

同一检查点连续执行完整5.14s时，首次机器人触地为3.255s，最大全板离地6.877mm、峰穿透12.398mm，完整结果仍为 **FAILED**。不能把监督拟合误差小或准备段不摔倒写成完整Ollie成功。完整实际视频在 `logs/ollie_bc_prefix_03/native_full/rollout.mp4`，标签为行为克隆初始化。

## 输入与透明动作范围

新增入口 `scripts/local/warm_start_ollie_policy.py`，输入必须是已通过所声明有限窗门槛的真实 `teacher.npz`；失败候选会在训练开始前被拒绝。实际验证了该拒绝分支。

输入教师为 `logs/ollie_quasistatic/corrected_static_final_v2/feedback_0.45/teacher.npz`，SHA-256 `becf8edd26596b6ae858418e3a5f20f1f8bea5ab42c89c862d29d1347dd1d738`；参考SHA为 `f6150d1217a4cf4952c3c03e601df4abe20b649ce03abb6f6b4c208d894f8dbc`。脚本检查规范29关节/4轮名字、138×92实际状态、138×29实际动作、未来目标及动作前时间，对齐规则是当前step+1+[0,5,10]。

原策略残差范围不能覆盖教师五个关节的实际目标。因此显式采用逐关节规则 `max(原范围, ceil(1.10*教师实际峰值/0.05)*0.05)`，把原范围、实际峰值、新范围、来源hash及规则全部写入输出 `residual_limits.json`。主要变化包括左髋pitch .75→.95、左右踝pitch .7→.75/.9、右踝roll .35→.45、腰pitch 1.1→1.15，以及部分肩关节范围。没有静默截断教师，也没有改变原机械目标限位、扭矩、PD增益或物理参数。

**真实关节角仍受MuJoCo软限位的数值行为影响。** 推荐网络准备段最大实际角度越界约0.02956rad；全段失败回合最大约0.09924rad。目标合法不能当作实际角度严格合法。文档29保留原教师逐关节偏差；本轮没有通过修改软限位或验收容差掩盖该问题。

## 实际训练和筛选

先对138个实际状态/目标做3000次监督拟合。纯拟合只提供训练集误差，不声称独立泛化：初轮末误差RMSE约0.000737rad、最大约0.007526rad，但原生闭环最小直立余弦仅0.78571。误差虽小，累积接触偏差仍会改变控制效果。

随后做5轮DAgger，每轮真实连续执行138个动作，专家混合系数依次为1、.75、.5、.25、0，并加入0.002rad目标噪声；每轮再做1000次监督更新。所有回合只在起点初始化一次，实际控制是经过真实目标限位的29维混合动作，没有向板或机器人根注力。专家标签是同一实际状态下的前馈/反馈控制查询，**它与真实执行动作分开保存**。

共产生690条实际转移，保留可表示且姿态可恢复的专家标签，共累积826条监督样本。超出策略残差范围的标签明确不用于拟合，未被悄悄裁成可表示目标。数据归档为 `dagger_actual_transitions.npz`，明确 `verified_teacher=False`，不能将混合控制失败数据当成功示范；世界模型若使用该数据，只能读取 `action`，不能误用反事实 `expert_label`。

每次保存后都重新做纯网络的原生闭环，选择代价在原评测代价上增加低于.85的直立余弦和超过.20m根板XY距离惩罚。仅按位置RMSE选择会偏好明显倾斜的候选，已纠正该筛选方式。推荐为DAgger第3轮后的7000更新检查点；最后8000更新版本最小直立余弦0.83467，较差，保留但未选为推荐。整个03轮约35.77s。

03轮初版DAgger输出尚未自带schema/名字/控制步长，动力学接入必须核对生成脚本并在manifest公开推断依据；当前入口已为未来输出补齐这些字段，不覆盖03归档。02轮在首次DAgger保存时因局部变量遮蔽模型导致异常，已修复并以新03目录完整重跑；02不是完成结果。

## WM+BPTT衔接验证

`scripts/local/train_ollie.py` 新增可选 `--teacher-anchor` / `--teacher-weight`。训练时仍由真实物理收集转移、拟合世界模型、冻结世界模型做BPTT；另外在独立抽样的教师实际状态上惩罚动作偏差，帮助保留初始化。教师必须匹配参考hash、关节顺序与未来目标接口；范围不覆盖时明确拒绝。

检查点分别记录监督初始化次数和BPTT次数，恢复训练时保留两者及参考hash。行为克隆检查点包含未训练WM，不能将它误标为训练好的动力学模型。评测标签已区分行为克隆初始化、未训练PD初始化和已进行BPTT更新的原型。

实际 `logs/ollie_actor_smoke_08` 完成8环境×16步=128条物理转移、2次WM更新和1次H4 BPTT更新，梯度/参数有限，teacher anchor loss约4.73e-6，接口和保存恢复通过。该极短更新的完整物理结果反而退化，未采用。它只验证接入，不能据此声称训练收益。

## 复现入口

```bash
env -u PYTHONPATH -u PYTHONHOME MUJOCO_GL=egl uv run --locked python \
  scripts/local/warm_start_ollie_policy.py \
  --reference data/ollie/self_collision_corrected/joint_reference_50hz.npz \
  --teacher logs/ollie_quasistatic/corrected_static_final_v2/feedback_0.45/teacher.npz \
  --base-residual-limits data/ollie/residual_limits.json \
  --out-dir logs/ollie_bc_new \
  --updates 3000 --eval-every 500 --dagger-rounds 5 --dagger-updates 1000
```

已完成真实监督更新、混合控制物理回合、纯网络独立闭环、全段失败视频、检查点读取与BPTT接入冒烟；Ruff和diff检查通过。原空中参考仍有明显人板错位，后续先修正该目标再训练完整技巧，不盲目增加旧目标上的训练预算。
