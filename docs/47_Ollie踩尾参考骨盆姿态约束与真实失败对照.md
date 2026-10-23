# Ollie 踩尾参考骨盆姿态约束与真实失败对照

新增显式骨盆 upright≥0.92 的运动学分支，已通过2.5ms致密姿态和碰撞检查。但根代理运行的三组连续真实准备试验仍全部失败：单独修正骨盆姿态没有救回当前静力前馈与足踝反馈控制，后续应转向实际状态反馈控制，不再反复调整同类根姿态参数。

## 正式输入与输出

- 修改入口：`scripts/video/repair_ollie_tail_contact.py`，新增 `--minimum-pelvis-upright` 显式选项。默认 `None`，维持44文档中v3算法行为。
- v4输入：`data/ollie/tail_contact_v3/joint_reference_50hz.npz`，SHA256 `9ef91bcbac47ab919d80525b1330baed9e520c36cb02089bc7793d064189c653`。
- v4输出：`data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz`，SHA256 `38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7`。
- 诊断：`logs/ollie/tail_contact_repair/upright_v4/`。

右脚仍在真实12°翘尾上表面，site X≈−0.32m，明确工程接触窗口0..2.90s；左脚中心板 X≈+0.205m，窗口0..2.76s。修改范围仍只有根、腿、腰。板全部21维state、两时轴、9个PnP节点以及14个手臂关节的位置和速度逐值不变。phys≥3.16s全92维state与v3/941f逐值相同。

原片只能支持后脚位于尾部的定性证据；−0.32m锚点、接触窗口和upright门槛都是显式工程先验，没有伪称视频测得的3D接触或力。

## 实现与检查

upright定义为骨盆局部Z轴在世界Z方向的分量，等价于 `1−2(qx²+qy²)`。约束在求解接触姿态阶段生效，2.90s之后随原来的修正量平滑释放。使用骨盆世界旋转Jacobian的准确导数，同时增加躯干方向软目标，由腰关节补偿骨盆变化；没有修改物理控制器或验收标准。

回溯目标同时考虑脚的位置和upright不足；碰撞上限独立检查。首个临时版本虽足部几何通过，却只达到0.915726，严格判定不满足0.92，未作为正式候选。将upright项纳入回溯目标后才得到当前v4。

正式v4结果：

- 准备0..2.76s每2.5ms采样，最低 pelvis upright **0.9223388794**，达到显式0.92要求；同一最低值也适用于0..2.90s接触求解范围。
- 全片2057姿态：自碰最大1.312586mm，机器人-板0.178052mm，地面0.107736mm，全部无>2mm穿透。
- 左脚1105个采样全部存在真实中心盒上表面近点，最大距离1.438637mm；右脚1161采样全部存在真实尾盒上表面近点，最大距离1.815976mm。足部XY锚点最大误差分别1.206451/1.248782mm。
- 真关节限位违规0，邻帧关节速度最大12rad/s。
- 相对v3根平移最大变化56.011mm；相对941f为107.871mm。躯干世界朝向最大变化0.206911rad，约11.86°，因此“保留躯干”是软目标，不能称逐值不变。
- 全片根速度最大3.168327m/s、加速度66.526219m/s²；准备段分别1.825726m/s、58.627084m/s²。v3准备段为1.605605m/s、56.230161m/s²，运动学姿态通过没有消除动态跟踪难度。

骨盆Jacobian另外用原生 `mj_integratePos` 对全部45个广义坐标作±1e−6有限差分核查，最大绝对差9.99e−11。记录在 `upright_jacobian_check.json`。该检查验证数学导数，不代表策略训练或物理稳定。

## 默认行为兼容证明

新增选项后，按默认未启用upright约束的完整流程重新运行，得到 `data/ollie/tail_contact_v3_default_compat/joint_reference_50hz.npz`。它与已冻结v3的两时轴、robot_state、board_state、contacts、phase、confidence七组数组逐值完全相同，92维state最大差0。

兼容重跑文件SHA256 `65f8cecdcc30afc3e3cb746996f04544e16e33a63f3fa4389a3b55d2abc9afc0` 与旧v3不同，只因新增选项、QA和诊断元数据。没有覆盖旧v3。机器证明：`logs/ollie/tail_contact_repair/upright_v4/default_v3_compatibility_proof.json`。

## 真实物理对照：仍然失败

读取根代理实际输出 `logs/ollie_quasistatic/tail_upright_prep_01/`；各trial均绑定上述v4 hash。每个连续试验只初始化一次、零中途reset、无板执行器、无施加外力；139个静力QP均可解，但QP力没有注入仿真。

- 动态feedback scale0.45：身体首次触地2.595s，min upright−0.01618，最大板上XY偏移0.69637m，FAILED。
- 动态scale0.25：身体首次触地2.6125s，min upright−0.02972，最大板上XY偏移0.57581m，FAILED。
- 动态scale0.65：2.76s内未身体触地，但min upright0.12870、XY偏移0.34772m、足滑移82.222mm，末300ms右脚上表面支持比例仅60%，FAILED。
- 三组实际PD目标受真实关节目标边界裁剪的次数分别45、11、43。该限制没有放宽，报告使用实际执行的控制，不能把未裁剪目标当作成功跟踪。

单独1s恒初姿试验的姿态门槛可通过，min upright0.99189；它没有产生有效的完整动态教师，`verified_teacher=False`。恒深蹲试验则min upright0.83081，仍未达到原0.85物理门槛。局部初姿可站、QP可解、几何可达三件事都不能替代完整准备动作的实际稳定性。

因此本批只完成必要的参考几何修复。三个动态试验均未生成可进入BC的合格新准备教师，不能复用旧教师的hash，也不能宣称Ollie成功。后续实际状态WBC和空中系统COM条件由各自记录维护。

## 重跑正式v4

必须给新输出目录，不覆盖已经绑定的v3/v4。

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/video/repair_ollie_tail_contact.py \
  --source data/ollie/tail_contact_v3/joint_reference_50hz.npz \
  --minimum-pelvis-upright .92 \
  --out-dir data/ollie/tail_contact_upright_new \
  --diagnostics logs/ollie/tail_contact_repair/upright_new
```

默认安全余量0.005、upright权重300、躯干方向软权重15均随参数保存在元数据。最终是否满足要求由致密实测最小值判断，未达标不会靠元数据宣称通过。本批已实际完成正式v4和默认兼容全流程，Ruff格式与检查通过。
