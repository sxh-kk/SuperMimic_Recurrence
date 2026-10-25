# Ollie真实双足承重与全身落地修正

2026-10-25，用户要求继续修改。本轮针对82未实现的双足同时承重，不把脚靠近板面当接板成功。仍为原12°/2.18 kg被动板、有效足板μ=.6和原29 PD增益/力矩上限，无外力、reset或根板state回写；保持SuperMimic学习路线，本轮是原生真实教师轨迹探索。

## 直接诊断

`logs/ollie_landing_diagnosis_01/diagnosis.json`读取82已核验400 Hz状态/真实执行力：COM沿世界重力投影的板Y在3.16/3.20/3.24/3.34 s分别为−.0926/−.1080/−.1281/−.1764 m，板半宽.105 m；3.24躯干upright仍.963，重心却已越板边。左足3.26～3.30的有力接触均在center侧边，normal·surfaceUp约−.056～+.004，达不到原top门槛>.5。右足3.3125～3.34短暂top在tail X约−.399～−.390/Y+.04～+.06，而COM投影Y约−.162～−.176，横向错开20～24 cm，身体向板−Y倒。

因此起点前移到**P158=3.16 s**，原COM世界Vz−.206 m/s，尚有约72.5 ms至尾撞；P162下落已−.798 m/s，等首40 ms飞行结束后才改动作只剩47.5 ms到板落地。唯一3.34静态catch IK未过（site误差18.54/9.10 mm、穿透8.813 mm），已拒绝，没有追加高度网格或使用失败pose作prior。

## 原生真实顶面支持

新增`../scripts/local/ollie_top_contact_sensors.py`：原277维观测再加左右脚×center/tail/nose六个CONTACT NONE，每组32槽、每槽`found/Fn/Ft1/Ft2/pos3/normal3`共10维，总2197维。物理数组和选项逐项保持，overflow/非有限样本显式invalid，不能静默截断。

逐接触Fn≥.1 N、deck局部contact z≥0、指向脚的normal·surfaceUp>.5，才累加`Fn*max(normal_worldZ,0)`；每脚sum>1 N是原Observer支持标签。普通总Fn、net Fz、侧边/底面不替代top。执行求解阶段和额外forward后阶段分别对齐，不能混时刻或事后shift。

`logs/ollie_top_contact_sensors_audit_01/result.json`核验82原185控制/1480子步，状态/Observer逐值一致。执行与postforward各217样本，逐slot Fn/位置/法向/count、top支持误差均0，实测单组最多4接触；合成侧/底/阈值/严格>1/单样本overflow过滤通过。模块SHA `39cf134944a92b1ec9980ab5a197e6028f31c33e5ad7397c290ca32770ed3ee7`。

预检中又定位旧`robot_ground` FOUND代理在3.400 s提前报接触，而canonical实际受力触地为3.7125 s。**首次正式搜索之前修正**：增加robot XBODY×ground的CONTACT NONE64槽，每槽found/Fn/Ft1/Ft2共4维，增加256，总**2453维**；只按真实Fn≥.1 N计身体触地，overflow并入invalid。`logs/ollie_top_contact_sensors_audit_02/result.json`对原82固定轨迹延至3.72 s，execution/postforward各225样本，地接触bool/逐Fn/足top均误差0，最大ground count3。首次触地的几何时间两stage同为3.7125（execution data.time3.715，postforward data.time3.7125），未跨stage混比。模块新SHA `48499f8c5333abc11f7991f9713207507c84fb8b2c2682d1e86943047ad92c66`，原39cf源码保留归档。

## 首次正式动力学前固定配置

入口`../scripts/local/optimize_ollie_top_support_landing.py`。原唯一初态连续重放82前158目标到3.16 s，优化剩99目标到5.14 s。nominal逐值为82实际99目标，零残差保留真实低跳；从原15腿腰残差扩为**全部原29机器人目标**，腿/腰/臂从第一步即可纠姿，仅承重奖励延迟到首个真实40 ms飞行窗完成后。

256候选×64轮，最多16个最高真实事件等级精英，13节点`[0,.02,.04,.08,.12,.18,.26,.38,.54,.76,1.04,1.42,1.96] s`，σ=.10/floor.025 rad，seed83，16原生候选线程。因新增2197维接触观测，**正式搜索之前明确固定预算600 s**。先完整batch、top同stage/原生物理/预算预检，再唯一正式完整回合。

固定prior有exact82、原站姿+.12→+.60 s渐入，以及静态COM Jacobian指导的方向候选：腰pitch偏−.45 rad、双肩pitch各+.60 rad，前+.12 s渐入、保持+.50、1.0 s回零，原限位截断。原正腰pitch的dCOM板Y/dq约−.0797 m/rad，两肩各约+.009/+.012 m/rad；只解释候选方向，不能预称动态可行。

每脚真实top40 N为软目标，首40 ms飞行完成后40 ms渐入、后400 ms平均；另在后300 ms找**同一个50 ms窗**：逐样本先取双足最小支持、合并upright≥.9及root−board世界Z≥.60的缺口，再对20样本取最大缺口，最后在候选窗间取最小。不能用两个脚各自最佳时刻或滚动平均的短冲击冒充持续双支撑；窗末前必须没有身体接地、穿透>10 mm或板倒置。

同时约束真实侧/底接触、躯干高度/倾斜、实际关节范围、全身地接触和最终稳定。capture point只作飞行后的LIP软引导，使用**世界水平**COM−boardCOM与相对水平速度，再转板heading；不用完整board-local COM.XY把板俯仰带来的高度投影当横向失衡。足XY与有限见证点gap仍只作辅助，不能认证top受力。固定权重/时间见脚本TASK和frozen_manifest，轮间不切目标。最终成功仍是原严格v3。

## 前缀视图与证据

为复用未改76协议，只读视图精确截取82真实前170控制/171状态及前1361 Observer；全部qpos/qvel/state/action与原档逐值相同。`time_physics`仅是原协议要求的理想20 ms lookup，原生累计边界时间另存`time_native`，Observer保持原时间。源SHA `5cf90dd936724fe920e15578b054724811c0368129fea253dfc6b6514e5c29f3`。不把保存末态设入主仿真；始终由原初态逐控制重放，额外12旧控制只在scratch回归。

prior工厂安装2197观测，另存扩展模型物理proof。首完整batch胜出在原无新观测模型逐子步重放，与新增sensor同执行阶段canonical top比较；通过后公共协议另核fullphysics/COM/力/包络和真实normal。主执行/最终Observer仍原模型，257控制/258状态/2057 Observer，本轮后缀792执行力。

## 运行

先benchmark，再将mode改run并使用全新输出目录。所有文档为MD；数据和视频仍为本机ignored产物。

```bash
env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --locked python scripts/local/optimize_ollie_top_support_landing.py \
  --mode benchmark --out-dir logs/ollie_top_landing_benchmark_01
```

## 预检与首次正式运行

无新增积分的`logs/ollie_top_landing_static_01/result.json`确认：adapter源state/action/native时间/1361 contacts逐值一致，nominal是82后99实际目标，全部prior99×29、限位和主状态不变通过。五个合成案例区分同窗20样本、异步左右脚峰、单点overflow、无飞行、同窗19样本，连续50 ms项分别0/1800/7200/7200/1800，双top最长.050/0/.025/0/.0475 s；overflow仅该条candidate invalid。后来新增真实ground通道的阈值/overflow语义另见audit02。无air不能报告“after-air支持”。

预检01使用原2197维/FOUND地接触代理，256完整候选rollout3.289/cost2.746 s，投影64轮386.21 s；原生792步fullphysics差0、执行力≤2.85e−14 N，top同stage误差0。随后发现并在正式前修正上述FOUND代理。其原运行优化器与模块源码分别保留在预检01和audit01，未改写旧记录。

预检02使用2453维和真实Fn≥.1 N身体触地：rollout2.760/cost2.612 s，投影64轮343.80 s；190/256候选保留真实飞行等级2。首批胜出193，真实top同stage误差0、canonical身体触地判断错误0、fullphysics0；原COM/姿态≤1.56e−15、执行力≤2.85e−14 N、尾Fn/TOUCH差0。exact82 nominal的首次执行阶段有效触地指标正确重现3.7125 s。源158前缀及额外12个原目标回归均逐值一致。所有预检未更新分布、未执行主后缀。

首次正式配置入口SHA `cd564a077f23b6efdfdb5675cda58da85a0fe0c2efcf076cccd406dd9c79a81f`、top模块SHA `48499f8c5333abc11f7991f9713207507c84fb8b2c2682d1e86943047ad92c66`，输出`logs/ollie_top_landing_01/`。额外保留达到同一个50 ms真实双top+姿态代理的已评估完整分支，避免只保留全局cost winner而丢失有用接板数据；该归档不改变选择/均值/σ，也不认证完整v3或教师。搜索、实际回合及独立检查的结果随后按原指标记录。

## 首次正式结果：仍未接稳

唯一正式64轮完成149.721 s，winner62/96、raw2、cost46575.390237。实际257控制到5.14 s、258状态/2057 Observer/792后缀执行力，actual_prediction误差0、partial=null；原严格v3仍**FAILED**，未训练策略或晋升教师。真实整板飞行3.2550～3.3550共102.5 ms、全9几何最低点峰23.4938 mm，尾地3.2350～3.2375共5 ms；板未倒置、最大穿透9.78753 mm。

首次有效身体触地是**左脚4/5几何，3.3975 s**，其中left_foot4 Fn247.30 N；相对82的3.7125 s更早，不能把代价下降称落地改善。3.38 s身体upright仍.9554，但root−board世界高度仅.4906 m、robot COM Vz−2.119 m/s、重力线板Y−.128 m已越边。末300 ms两脚真实top支持95.83%/98.33%、四轮100%，然而5.14 s身体upright仅.426、root−board高度.0426 m，仍是倒地状态，不等于站立。

独立`independent_force_audit.json`确认qpos/qvel/state92、2057 Observer、792执行力、全部strict字段逐值一致，82前158/old60前150与原53初态保持。源actual SHA `13bc652ffc920b2b3bb44f91aed3db3eb4fa0b19c447887ac168d3bf89ef277c`，独立force审计SHA `0b9a96af6c781deccd201958a1fb52d8b2edd2eb805dff1b77e1b0fbd640402f`。力矩超限0，后缀腿最大58.01%上限，腿≥95%时间0；腕仍达到100%。原完整关节软越界.02904 rad保留。

`independent_top_stage_audit_01/result.json`另将全部257目标双模型固定重放，保存完整2453列执行/后forward各2056样本：物理状态/Observer/逐接触Fn/位置/法向/top/ground/count全部误差0，无overflow。**首40 ms飞行完成后、真实身体触地前，双足top>1 N同时支持最长仍0 ms，合并upright≥.9及高度≥.6同样0**。没有shadow合格分支。`static_02/result.json`补核8例，真实ground .099不触地/.1触地、旧FOUND=3不误判、overflow仅使对应候选invalid通过。

完整[原片/真实物理视频](/home/zxc/SuperMimic/logs/ollie_top_landing_01/video_01/comparison_full.mp4)已导出，258实际帧/187对照帧完整解码，hash未改；已抽查起跳、左脚接地及倒地尾段，保留FAILED。下一项[84](84_Ollie提前纠正重心与全身落地.md)将同一真实top/全29目标的纠偏前移到2.90 s，未因这轮失败放宽物理或成功标准。
