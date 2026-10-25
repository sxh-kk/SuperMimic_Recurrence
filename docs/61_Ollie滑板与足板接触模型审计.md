# Ollie 滑板与足板接触模型审计

2026-10-25。用户提出“是不是对滑板的建模有问题”后，暂停追加动作搜索，独立检查实际生效的模型。**已确认足板摩擦与接触响应存在参数覆盖问题；当前模型也尚未完成实物校准。不能再把所有失败简单归为训练或控制。**另一方面，已有动作的卸载、踩尾、上升与拖板并未同步，单改摩擦没有使整板腾空，不能把当前失败也全部归到一个参数。

## 实际生效的接触参数

板的XML声明摩擦0.9、`solref=[0.008,1]`，但脚来自G1原资产：摩擦0.6、`solref=[0.02,1]`、priority=1；板priority=0。

MuJoCo动态接触在两个geom优先级不同时，采用较高优先级的摩擦和对应接触参数。显式contact pair则直接指定接触参数，绕过geom间的合成。[官方参数合成规则](https://mujoco.readthedocs.io/en/latest/modeling.html#contact-parameters)

这不只是根据XML推测。对60的170个保存目标完整原生重放，逐子步读取实际`data.contact`，有效足板接触全部为：

```text
sliding friction = [0.6, 0.6]
condim = 3
solref = [0.02, 1]
solimp = [0.9, 0.95, 0.001, 0.5, 2]
```

因此，“板写了0.9”不能称足板实际用了0.9。接触柔顺性也由脚的参数接管；0.008与0.02在这里是接触求解参数中的时间尺度，不是直接的材料恢复系数。此前模型配置遗漏了这一跨资产合成关系。

完整原模型重放qpos/qvel最大误差均0。输入实际轨迹SHA256为`43cfef43b9cf5f966df2bb0a8b7e31a2446ee040f6fef5d3cab10baf221c027d`，实际参数与逐接触记录见`logs/ollie_contact_model_audit_04/{result.json,native_foot_deck_contacts.json}`。

## 代码修正：让足板接触成为显式模型选择

`make_scene_spec`新增可选`foot_deck_friction`：只对两脚各7个胶囊与center/tail/nose三个板面建立42个显式pair。保留脚原有condim、接触柔顺性、margin/gap、扭转/滚动摩擦，显式指定足板滑动摩擦；质量、惯量、电机和脚地参数不变。

```python
# 历史模型：priority合成后，足板实际μ=0.6。
spec = make_scene_spec()

# 显式足板材料假设，脚地仍保持μ=0.6；这不是实物校准值。
spec = make_scene_spec(foot_deck_friction=0.9)
```

默认值仍为`None`，保留历史物理与复现记录，原XML未修改。选择0.9或1.2只用于敏感性诊断，尚未晋升为最终模型。旧教师和控制动作不能在新模型中假定依然有效；后续必须记录参数版本并重新验证。

实际验证包括：默认模型170动作全重放误差0；显式μ=0.6的42pair模型从原始初态及2.60 s真实热状态两种入口，都与原轨迹qpos/qvel逐值一致。三组显式参数的实际足板接触值均与指定值相等；另用独立几何接地probe确认脚地仍为0.6。probe只检查参数，不是机器人执行或新的可行初态。9项联合状态测试、Ruff及语法检查通过。

## 同一控制下的摩擦敏感性

未重新训练或搜索，只执行60已经保存的同一组控制。每个显式pair模型均做两种诊断：从同一原始初态执行全部0～3.40 s；以及从原模型真实连续到2.60 s的热状态开始，单独改变模型参数后展开0.80 s反事实分支。后者有模型切换，**不能作为全程使用新模型的完整动作成功证据**。

- **μ=0.6**：全程和热分支均精确复现原结果；尾触地70 ms，无机器人接地，最大穿透7.519 mm；全板最大间隙0.002674 mm，无flight。
- **μ=0.9**：全程控制首次机器人接地3.1775 s、最大穿透13.643 mm，没有tail事件；热分支仍出现尾触地，但3.40 s脚已触地，全板最大间隙0.01419 mm，无flight。
- **μ=1.2**：全程控制首次机器人接地3.1675 s、最大穿透16.344 mm，没有tail事件；热分支3.39 s机器人接地，全板最大间隙0.000722 mm，无flight。

这些结果说明控制对摩擦敏感，且**单把摩擦调大并不会让现有动作成功**。它们没有比较“各模型分别重新优化后的最佳可能效果”，也不能反推出真实滑板应采用哪个摩擦系数。

## 接触形状与向上带板

对倾斜平面顶面的单个接触，设板面上倾角为α、法向力为N。板所受力的向上分量，在圆形库仑锥这一较宽上界下满足：

```text
F_board_z / N <= -cos(α) + μ sin(α)
```

以60板头最大上倾30.2199°、nose自身再翘12°为例，顶面α约42.2199°、μ=0.6，右侧仅为−0.3374；单纯压住该顶面并沿面摩擦，仍不能给板向上的合力。纯顶面向上力需要α>59.036°。这只是指定平面接触的条件，**不能扩展成整块有限滑板或任意Ollie的不可能证明**。

实际60没有前脚—nose接触。它在先上升、后下落踩尾时，前脚已经离鼻端很远；提高一个根本未闭合的接触的摩擦不会自动产生带板力。nose的有限后边缘可以提供不同法向，局部几何审计另行检查；正的摩擦锥上界只表示存在可用力方向，不等于全机器人可达、实际滑动方向正确或已实现该力。

独立入口`scripts/local/audit_ollie_nose_contact.py`已实际运行并保存`logs/ollie_nose_contact_audit_01/`：完整170目标重放qpos/qvel/state92和1361条Observer原字段均逐值相同。它另用当前libmujoco的`mju_decodePyramid`核验dim3四条锥射线，按真实接触法向、两切轴和μ计算精确pyramidal上界，而非用较宽圆锥代替。

2.60～3.40 s的有效真实接触仅left-center、right-tail和right-center；其最佳`F_board_z/N`上界仍分别约−0.98561、−0.76250和−0.56493，真实板受竖直力也均向下。**当前动作确实没有采用可向上带板的接触模式。**

另一方面，在2.90 s实际姿态中，前脚第6胶囊距nose有限后端—顶面边缘仅35.835 mm。独立局部probe将整只脚平移约`[+35.213,+0.104,−6.911] mm`，保留原7胶囊形状、姿态、材料优先级与nose几何；用50微米闭合量生成真实native contact，其精确pyramidal上界为 **+0.396170 N/N**。这提供了“有限边缘允许向上力”的局部几何反例，原XML输出为`geometry_only_probe.xml`。该probe没有积分步骤，刻意不记录或解读其无支撑静态求解力；未检查全身IK、自碰、腿部可达性或实际滑动/受力实现，不能称已经做出拖板。

## 尾部碰撞与步长组件检查

原模型总质量2.18 kg，其中板/轴架刚体1.86 kg、四轮共0.32 kg；系统COM在板系Z约−9.20 mm。各体惯量和质量单位已核查，暂未发现明显的轴向或数量级错误，但这些数值仍不是原视频实物的测量结果。板长的名义tip尺度约0.8013 m，完整旋转盒碰撞包络X长约0.803785 m，两者并非同一个几何量；碰撞检查使用后者的原生包络。

尾地真实接触为μ=0.9、condim3、`solref=[0.008,1]`、`solimp=[0.95,0.99,0.001,0.5,2]`；轮地为condim6、相同normal solref/solimp。它们与被脚优先级覆盖的足板接触不同。

组件测试仅用自由被动板，无G1或电机。固定人工初态：板pitchY=−0.6 rad、tail底高40 mm，初始角速度分别0和−3 rad/s；每种以2.5 ms、1.25 ms步长运行0.35 s，共4个短案例。人工初始能量只用于检查碰撞响应，不作为机器人自主Ollie。各case首次连续tail力窗口没有其它active接触。

- 初始ωY=−3 rad/s、2.5 ms步长：首次尾碰撞约0.035 s，选定尾部材料点入射法向速度约−1.30235 m/s；0.120 s释放时该点约+0.001275 m/s，首窗机械能减少约0.54336 J，tail法向冲量约1.64365 N·s。
- 相同步长减半：入射约−1.30235 m/s，释放约+0.001343 m/s，能量减少约0.53194 J，冲量约1.61276 N·s。
- 零初始角速度、仅重力下落两步长：入射约−0.8829 m/s，释放约+0.00424/+0.00488 m/s，首窗能量减少约0.35752/0.35701 J。

当前数值尾碰撞在这些人工案例中明显耗散，同一尾部点释放速度很小；这值得在最终技巧模型中专门校准。**释放/入射速度比不是木板材料恢复系数**：它属于这个多自由度刚体的指定材料点、指定事件与重力/摩擦环境；没有实物对照，也没有从这4个case证明Ollie不能完成。步长减半仅提供初步敏感性检查，不是完整收敛证明。

组件原始资料保存在`logs/ollie_model_audit_01/{component_model.json,component_impacts.json,component_timeline_*.json}`，原XML hash为`b0275651d4ca6e8455e0a8cdcf65204078017007c8c3416635b2460050bfc17b`且未变化。可复跑的标准入口随本批记录一起保存。

标准入口`scripts/local/audit_ollie_board_impact.py`已在新目录`logs/ollie_model_audit_01/component_repro_01/`重跑，模型/结果/4条timeline共6份JSON与原组件输出逐值完全相同，证据为`component_reproduction_check.json`。未调整初态、接触、步长或测量逻辑来获得更好结果。

## 模型精度仍未认证

现板为刚体板面、刚性翘尾/板鼻、固定轴架和四个被动轮。没有板弯曲、truck弹性/转向或鞋底/脚趾结构；质量、惯量、尺寸与材料接触参数为工程假设，未测量原视频中的实物。组件能滚动、能碰撞、能腾空，不等于已忠实还原原板的撞击和带板行为。

下一步的模型验收应先检查真实生效的足板/轮地/尾地pair参数、有限足鼻接触及碰撞步长响应，保留不确定参数范围，再在明确固定的物理模型中联合优化卸载、上冲、尾撞、释放和前脚带板。不是继续追踪失败54参考，也不是调参数到视频看起来成功后反称模型正确。

## 重跑审计

```bash
env -u PYTHONPATH -u PYTHONHOME OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked python scripts/local/audit_ollie_contact_model.py \
  --out-dir logs/ollie_contact_model_audit_new

env -u PYTHONPATH -u PYTHONHOME OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked python scripts/local/audit_ollie_nose_contact.py \
  --out-dir logs/ollie_nose_contact_audit_new

env -u PYTHONPATH -u PYTHONHOME OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked python scripts/local/audit_ollie_board_impact.py \
  --out-dir logs/ollie_board_impact_audit_new
```

入口绑定60实际轨迹SHA；新输出目录必须不存在，原输入文件hash在结束后再次核验。没有修改原物理资产、旧日志、教师、模型权重或旧控制。所有敏感性结果均标注`does_not_certify_full_ollie=True`。
