# GPU 真实接触观测与 SuperMimic 数据核验

2026-10-25。算法保持 **SuperMimic：真实物理数据 → 联合世界模型 → 模型内 BPTT 更新策略**。此前提出的PPO分支已取消，从未创建实现文件或开始训练。本批只增加通用实际观测，不引入其他控制器、奖励或critic。

## 新增内容与时序

`JointEnvConfig.contact_observations=False`为默认；关闭时保持原8次step后1次forward，不增加传感器。开启后，每个2.5 ms执行`mjw.step → mjw.forward → record`，每个20 ms控制产生`contact_observations[N,8,49]`。`step`返回原92维state的接口保持不变。

49个列名和schema定义于`src/supermimic/ollie/gpu_contact_observer.py`。包括两脚实际足板世界Z合力、前脚作用于nose的世界Z合力、两脚真实上表面支持、尾/鼻/四轮/脚/机器人地面接触、接触数量、穿透、9几何净空、机器人/板姿态速度、关节越界、COM竖直速度及overflow。只新增两只读SUBTREELINVEL传感器以取得真实COM速度，不改质量、惯量、电机、约束、控制或外加力。

接触分类阈值固定**0.1 N**，与原MPC ContactObserver一致；最终上表面承力是否超过1 N仍是另一项判据。两脚足板Fz为板对脚（站立正）；`front_nose_board_fz_n`为脚对nose（上提正），符号不可混用。净Fz使用所有已求解接触，不截为正数，也不把接近nose当成真实带板。

这些是**积分后新姿态的额外forward观测力**，不是前一步积分执行的驱动力。可监督接触和时序，不能把它们累加成精确执行冲量来硬约束MΔv。执行力矩仍须在step后、额外forward前另采。

## 独立核验

入口：`scripts/local/audit_ollie_gpu_contact_observer.py`。使用4个环境、原显式μ=0.9准备初态及固定前5个控制，覆盖0.10 s、每个环境初态加40子步。先保存8fe329a旧环境基线，再比较新路径；没有新动作搜索或训练。

逐子步用`mjw.get_data_into`将**GPU已经计算的contact、efc_force和位姿**取回独立native数据，调用原ContactObserver和独立公式，不重新CPU求力。世界/接触约束地址重排误差为0，物理数组和option逐值保持。

同一实际GPU状态的观测计算通过预先固定门槛：

- 接触布尔、计数和分类逐值相同；未观察到外力或overflow。
- 支持力/净Fz最大误差约3.91e−5 N，9几何误差约5.13e−9 m，COM速度约1.93e−8 m/s。
- 正常CUDA graph step的每个控制末slot另有20个直接读回样本，力误差≤3.20e−5 N，分类相同。

**整体`audit_pass`仍为False，未隐藏失败：**事先要求默认关闭路径与旧GPU运行逐值0，以及不同运行graph/manual的支持力差不超过固定门槛。这两项未过：默认off相对旧基线state92最大差2.092e−5，graph/manual约2.603e−5（92维含不同物理量，不能当米）；跨运行右脚支持差约0.0979 N超过门槛。旧8fe329a自身按同样初态/固定动作重复也有3.307e−5的state92差，说明后端本来存在同量级重复漂移，但该对照不能把失败门槛偷偷改为通过。同状态直接读回一致与跨运行逐值一致分别记录。

完整证据为本机`logs/ollie_gpu_contact_observer_audit_01/observations_02/audit.json`。首次报告有NumPy bool JSON编码错误，原样重复固定核验恢复输出，正式02另增加旧代码自身重复诊断；没有变动作或阈值。

局限：0.10 s站板没有真实尾撞、带板、腾空或接板；4个重复世界也没有覆盖任意初态。此核验不能认证新模型真实材料、完整Ollie或合格教师。原完整65仍FAILED，不开始新训练。

## 用于 SuperMimic 的后续接法

每个真实50 Hz transition可附8×49个高频标签及模型profile/backend hash。旧μ=0.6数据与新μ=0.9数据须隔离或显式条件化，不能继续当作相同动力学。当前联合state92、动力学delta90的主接口保持。

先训练/核验世界模型：使用尾地力、两足支持、前脚作用于nose的力、几何净空、COM速度作为辅助监督，并用同一热状态下的真实正负动作分支检查板响应的增益与方向；接触阶段整组留出，报告零响应基线。新增辅助head不意味着其动作梯度已经可靠，预测力也不能写回仿真。

只有留出卸载/尾撞/nose阶段的真实板响应和模型动作梯度通过检查后，才恢复相关板目标的BPTT。当前测量准备段本身不提供这些事件数据；不因测量公式通过就直接扩大想象视界或换成PPO。

## 冻结版本

- `gpu_env.py`：`ed4ae4a465d93608806ac31e4c2eb5134d7149b2a3082eeccafb5d4d758c01d8`。
- `gpu_contact_observer.py`：`3f1d779ff742f6ab8b94d67d59c954ee8513df5964bb625dcc79fc3fb6992c07`。
- 独立audit脚本：`6a9181868085423493e2928f751300e6fc63ac513d7d56c33fac5135b7dfca25`。

Ruff、语法及diff检查通过。依赖MuJoCo-Warp固定3.16.0的接触力解码接口，升级依赖需重验。
