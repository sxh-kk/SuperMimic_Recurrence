# Ollie 翘尾接触前馈与实际准备段

## 支持真实倾斜接触面

原准静态前馈把足部四个假定支撑点裁到中心平板±0.275m，再投影到平板上表面。它不能用于文档44中位于X=−0.32m翘尾上的后脚。

`ContactStatics(..., surface_patches=True)` 新增显式模式：对每个足部假定点，计算真实center/tail/nose三个碰撞盒的有限上表面矩形最近点，选取最近表面；相对机器人/板Jacobian乘以该盒子的世界旋转矩阵，力在各自表面的切向X/Y及向上法向Z表示。摩擦锥、非负法向力、全45维平衡和原29关节力矩上限仍用于求解。地面轮接触力保持世界坐标表达。

解出的力仅转换为真实机器人PD目标，绝不写入板或根的外力。点位仍是近似足底矩形的支撑假设，求解成功不是接触确实发生或动态跟踪成功的证据。每帧另保存12个 `contact_surface_names`，可结合该帧qpos恢复各力坐标系。

`track_ollie_quasistatic.py --surface-patches` 显式开启并在结果中记录；连续衔接和DAgger从教师metadata读取同一布尔配置，避免以后在尾脚教师上误用平板标签。旧教师缺少该字段时仍走原平板路径。

## 实际验证

`logs/ollie_quasistatic/surface_solver_validation_01/check.json`：重新计算旧参考的139个准静态解，所有29维tau与原落盘值最大差严格 **0**；默认路径没有改变。初始实际参考上新模式平衡残差5.68e-14，求解没有修改调用者的qpos/qvel/ctrl/外力数组。

文档44的v2新尾脚参考实测139个解全部可行；每帧左4点选择center、右4点选择tail。代表帧平衡残差约1e-14～6e-14。v2与正式v3全部状态、时间及标签数组逐值相同，证明见文档44；本节物理文件保留真实输入v2 hash，不改写为v3。新参考与旧941f准备教师不兼容。

实际原生连续准备 `logs/ollie_quasistatic/tail_surface_prep_01`：从首帧零速度初始化一次，只执行关节目标，共138个控制/1104子步；三组均未通过教师门槛。

- scale0.45：身体2.6250s触地，最小upright −0.06464，峰穿透5.545mm，足平面偏移最大142.69mm。
- scale0.25：身体2.6225s触地，最小upright −0.07289，峰穿透7.269mm，足偏移140.48mm。
- scale0.65：2.76s内身体未触地，但最小upright0.16202、root-board XY最大358.41mm、足偏移62.26mm，仍失败。

初始固定姿态1s能保持upright0.99189、足偏移4.84mm；最深固定姿态1s无身体触地，但最小upright0.76458，不能作为合格动态教师。两者均明确不标记`verified_teacher`。

另做 `--absolute-upright` 闭环对照，`logs/ollie_quasistatic/tail_surface_prep_abs_01`：scale0/0.45/0.65/1分别在2.6800/2.2450/2.2250/2.2400s身体触地，全部失败，未继续增加同配置预算。绝对upright反馈是控制目标变化，没有重新置姿。

## 当前结论

真实尾面力坐标和接触假设已经接入，旧模式逐值回归通过；新的尾脚几何参考尚未转成合格连续准备控制。本批没有生成新已验证教师、启动尾脚BC/actor训练或认证起跳。

参考准备pelvis upright约0.824，源参考本身约0.826，都低于准备教师门槛0.85。下一候选需把姿态可达性、动态平衡、空中质心运动一致性一起处理，不能放宽实际验收来把失败标为成功。

```bash
env -u PYTHONPATH -u PYTHONHOME OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
uv run --locked python scripts/local/track_ollie_quasistatic.py \
  --reference data/ollie/tail_contact_v3/joint_reference_50hz.npz \
  --out-dir logs/new_tail_surface_prep --seconds 2.76 \
  --feedback-scales .45 .25 .65 --surface-patches
```

上述命令使用正式v3重跑，预计结果对应相同数值数组，但本记录中的已执行结果以原v2文件及落盘hash为准。
