# Ollie相关工作与起跳接板时序修正方案

2026-10-26。用户要求进一步参考相关工作处理滑板起跳的方式。本轮重新核验论文正文、图示、固定版本源码和数据说明，并对85/86已保存物理回合做只读时序核验；没有改物理、运行新轨迹优化或训练策略。SuperMimic仍是联合世界模型与BPTT学习路线。

## 直接结论

最有用的启发是**准备状态、尾撞卸载、空中调平、接板和落稳必须分别建模，并共同调整时序**。当前真实低跳成立，但主要接板引导启用过晚；已经出现的30 ms双足支持发生在板重新触地以后，不能称为空中接板成功。其次应独立核验尾地碰撞的耗散，而不是直接套用别人的回弹或摩擦参数。

## 相关工作实际做了什么

### The Blind Skateboarder：先获得可执行脚位，再开始参考动作

[作者PDF p2 §II(c)、p4 Fig4](https://joaoamcarvalho.github.io/assets/pdf/TBS_IROS_Workshop_2026.pdf#page=2)在技巧前后安排idle，准备期间获取所需脚位，触发后才推进参考。可借鉴“准备与技巧时钟分开”；论文未披露自动事件触发阈值或飞行中重新定相算法。截至本轮作者页面仍未找到可直接下载的Ollie模型、参考和策略。其训练算法不迁入本项目。

同组[Blind Dexterity §V-C](https://arxiv.org/html/2608.29487v1#S5.SS3)指出主动找板的完成时间可变，与固定时间对齐的模仿不易直接兼容；该实验是上板，不能当成Ollie成功证据。

### Shield等2022：接触序列与可靠初值比盲目扩大搜索更关键

[Contact-Implicit Direct Collocation with a Discontinuous Velocity State，§VI](https://www.andrew.cmu.edu/user/amj1/papers/RAL2022_Contact-implicit.pdf)分别建模尾地部分弹性、脚板高摩擦和轮地接触。全身模型在提示一次尾触地或提供已有解后能生成Ollie；空初值及20组随机扰动初值均未成功。可借其接触提示和速度突变处理，不能照搬简化模型的足力、点轮或外力驱动点质量对照。

### Heinen等2024 / mechmotum：人体起跳、尾撞和调平要协调

[论文§2.1–2.6、§3.1、§3.3](https://doi.org/10.1007/s12283-023-00448-y)采用刚板、分相冲击和可变阶段时长。其优化解先通过反向预蹲获得人体动量，再处理尾撞与调平；某个轮距优化解在撞后上升期间无需持续控制，峰前才施小向下力。因此持续拖鼻或前脚对板净向上力不是所有Ollie的必要条件。

[公开源码固定版本](https://github.com/mechmotum/ollie-optimization/tree/dd60b1984a236dd682c9677345c55a168957651f)可借鉴阶段、接触与时间变量；其脚被限制在板面参数曲线上，并直接优化足力/滑动加速度，不能当G1可分离鞋板接触的现成解。尾地恢复系数`.8`也是模型假设。

### Nakashima与Chida2021：不能把脚一直压在尾上

[论文正文、Figs.5/6/10](https://www.jstage.jst.go.jp/article/mej/8/5/8_21-00230/_article)通过规定脚运动和弹簧阻尼接触分析卸载与板转动，提示尾撞附近足部运动时序的重要性。它不是29关节闭环控制器；其力和速度数值不迁为本机验收门槛。

### 其它结果的边界

- [Mikkola2023原文p54](https://aaltodoc.aalto.fi/bitstreams/a4dcb6e7-d352-4594-a09b-8ccc2489d762/download#page=54)的定时truck高度/正弦目标与越障方案仅偶发像Ollie，不能提供稳健专家。提示“抬高板轨迹”本身不足以解决动作。
- [Burgess2023 §II](https://arxiv.org/html/2310.11599v1#S2)的阶段约束有参考价值，但用理想电机操纵与板相连的刚杆点质量；不等价于脚可离板的G1。
- [Open4DHOI §4.3](https://arxiv.org/html/2512.00960v1#S4.SS3)的物体坐标接触点配对可借来标注脚底/脚背/鞋边与nose/tail的关系；没有核验到可用的Ollie闭环控制发布。
- [I’M HOI补充材料Table5](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zhao_IM_HOI_Inertia-aware_CVPR_2024_supplemental.pdf#page=2)确有Ollie名目，但现成G1重定向的滑板条目被维护者标为`object_floating`，不能直接作教师。见[固定版本数据说明](https://huggingface.co/datasets/leggedrobotics/hoi-retarget/blob/0125b55e3e3d26eaf2c49f1f2d368342200a44c2/README.md#L193-L199)。

本轮来源、下载URL和SHA保留在`references/skateboard_research_20261026/ollie_literature_manifest.json`及`primary/source_manifest.json`；未把第三方资产或源码迁入公开仓库。

## 对照本机：接板目标实际落在什么时候

新增只读入口[空中接板时序核验](../scripts/local/audit_ollie_air_catch_timing.py)，原始结果`logs/ollie_air_catch_timing_audit_01/result.json`，SHA `b502bb1e5aa08ba957b60ec711ac9107286497fc3509038d3aea151679e92b00`。

| 实测或原有调度 | 85 | 86 |
| --- | ---: | ---: |
| 原严格>5 mm合格飞行窗 | 3.2550～3.3125 s，60 ms | 3.2525～3.3075 s，57.5 ms |
| 之后首次真实板/轮触地 | 3.3175 s | 3.3175 s |
| 搜索记录的首40 ms结束时刻 | 3.2950 s | 3.2925 s |
| 原top引导40 ms ramp升满 | 3.3350 s | 3.3325 s |
| 原脚板间隙80 ms ramp升满 | 3.3750 s | 3.3725 s |
| 离地后首次左前脚top>1 N | 3.5375 s | 3.5400 s |
| 离地后首次右后脚top>1 N | 3.2975 s | 3.2825 s |
| 首次板重新触地前双top样本数 | 0 | 0 |

**86的左前脚支持发生在真实板重新触地后222.5 ms。**因此85/86记录的30 ms双足top是较晚的地面阶段支持，没有证明空中catch。右后脚在飞行末段确有短暂支持，不能写成“全程完全没接触到板”。严格净空窗口结束和真实板触地是两个分别核验的事件，上表没有将二者混为一谈。

原[83成本225–255行](../scripts/local/optimize_ollie_top_support_landing.py#L225)先等待16子步真实飞行，再用40 ms增加top引导权重，并从首40 ms飞行结束时刻起寻找50 ms同时支持/姿态窗；其`safe`只排除人体触地等，并不要求板仍腾空。因此“短腾空＋较晚承重”仍可降低该目标。几何接近项另有80 ms ramp。这是**软目标时序与本机低跳长度不匹配**，不是严格40 ms腾空验收错误；不应通过降低验收门槛来掩盖它。

另从真实50 Hz qpos/qvel做FK与COM Jacobian：两轨迹的机器人COM大幅上升转下降都落在3.10～3.12 s，板合格离地约3.25 s。86在3.16 s机器人COM VZ已为−.2477 m/s，而板COM VZ为+.5844 m/s。该相位差应与横向漂移一同调整；真人身体也可先于板达到峰值，**不能仅凭此时间差证明它是唯一失败原因**。

本核验无`mj_step`、无新控制分支或训练。COM来自真实保存状态的只读FK/Jacobian；支持力来自已独立重放验证的400 Hz canonical Observer，未使用冷FK的重算力。搜索调度的执行阶段与postforward接触阶段分别标明，未跨阶段比较力。

## 对板模型得到的启发

应单独核验**尾地冲击的耗散及足部压制**。当前板/地面设置`solref=[.008,1]`；其中1是阻尼比，不能解读为恢复系数1。[MuJoCo官方参数说明](https://mujoco.readthedocs.io/en/latest/modeling.html#reference)将其解释为临界阻尼，并说明接触参数还受组合规则影响。论文的Newton恢复系数`.6/.8`不能直接填入此字段，更不能由XML单值推断真实有效回弹。

下一项应记录尾撞前后板COM速度、角速度/动能、尾地冲量及同时存在的脚下压力，判断能量是被持续压尾还是接触耗散消耗。刚板已有相关工作能起跳，目前没有证据要求先引入木板弹性或主动板驱动；尺寸/鞋形与材料仍是未实物标定的模型假设。

## 下一批修改收敛为三项

1. **准备与技巧推进分开。** 先获得合法板坐标脚位、真实支持和可接受的起跳前侧向动量，再推进pop参考。对准备、尾撞、卸载、上升、调平、catch采用可调整持续时间；进一步事件门控是本项目待验证设计，不冒称TBS已公布完整实现。仅暂停/推进参考，不能暂停重力或重置物理状态。
2. **把空中调平与落地承重拆开。** 保留原40 ms腾空/15 mm净空及末300 ms稳定标准；在离地至峰值、下降至首次轮触地的实际窗口内，提前规划脚板接近、相对法向速度和板俯仰变化。允许需要时的前脚调平，不强迫持续nose净向上力；真正catch必须另外报告是否早于首次板触地。不要把TBS机器人pop-shove-it约.3 s的飞行或固定抬脚保持时长直接套到60 ms低跳。
3. **联合修正与重新验证。** 从起跳准备段开始同时调整机器人动量、踩尾和回板；87已说明单独改善相对CP会破坏后续旧控制。全程仍只输出原29PD目标，保持被动自由板。先完成原生完整物理验证及对应阶段真实数据采集，再用于SuperMimic联合WM/BPTT；论文的直接足力、固定脚约束和可视化位姿不能充当我们的控制器或教师。

上述是本轮文献与实测共同支持的实施方向，尚未实施新策略或宣称成功。88离线世界模型对照仍未通过动作响应，原BPTT限制保持。完整Ollie继续判FAILED。

复核命令：

```bash
env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --locked python scripts/local/audit_ollie_air_catch_timing.py \
  --out-dir logs/ollie_air_catch_timing_audit_new
```
