# 同事真机接入约定

## 1. cam10 视频

入口配置 input.source 接受文件路径、摄像头编号字符串（例如 "0"）或 RTSP URL。mode 为 replay 或 live。图像须先去畸变为针孔投影，尺寸严格等于 camera.image_size；程序不隐式旋转、缩放或对鱼眼直接求几何。

当前 MoGe 适配器要求居中主点、等焦距像素：cx=(W-1)/2，cy=(H-1)/2，fx=fy。若真实相机不满足，先在上游重投影到这种虚拟针孔相机，并将相应 K 及光学坐标变换填入配置。不要把原始鱼眼 K 填进来。确认标定后设置 camera.intrinsics_verified=true。

实时 OpenCV 的时间戳是主机接收到图像时的 Unix 秒，不是硬件曝光时刻。硬件同步相机应替换 inputs.FrameSource，使迭代器输出 (frame_id, exposure_timestamp_s, BGR_uint8)，并把机械臂状态转换到相同时间域。默认 .5 秒帧龄限制仅限制控制建议，不阻止视频复核。处理慢时自动丢旧帧，frames.jsonl 记录实际帧号和耗时。

默认捕获到断流后结束；max_observations 可限制长度，stop_file 出现则停止处理。生产控制端应通过自身压力/接触状态机触发停止，而不是用视频帧号代替接触传感器。

## 2. 同步机械臂状态

input.joint_states 指向 JSONL，一行一条状态。实时可由控制端持续写入；建议维护至少覆盖推理延迟的滚动历史，不只保存最后一条。

```json
{"timestamp_s": 1795176000.123, "clock": "unix_seconds", "position_unit": "rad", "positions": {"shoulder_y": 0.0, "upper_arm_p": 0.0, "elbow_p": 0.0, "lower_arm_y": 0.0, "wrist_p": 0.0, "flange_y": 0.0}}
```

离线 clock 为 replay_seconds，时间等于 timestamp_origin_s + source_frame / 源视频FPS。默认匹配差最大 .1 秒；缺失、JSON 半写入或时间域不匹配均返回无同步状态。输入文件应原子替换或按完整 JSONL 行提交。

EVT 链为 base_link → shoulder_Link → upper_arm_Link → elbow_Link → lower_arm_Link → wrist_Link → flange_Link。关节名字必须完全匹配。禁止直接按六个数字的顺序猜测其他机型的关节映射。

## 3. 相机、吸盘与尺度接口

所有 T_A_B 都是 4×4 刚体变换：将 B 系坐标变到 A 系。平移单位米，旋转右手系。

- robot.cam10_urdf_link：cam10 实际装配 link，留待现场确认，不能猜 cam6/cam7 的对应关系。
- robot.T_flange_cam10_optical：flange → 实际输入图像的去畸变光学坐标系，包含安装变换和光学轴/重投影修正。不能直接把 CAD 相机 link 当光学系。
- robot.T_flange_suction_contact：flange → 吸盘接触中心。+z 指向待吸物体，+x 沿希望对齐的吸盘边。CAD suction_cup_Link 原点不一定是接触中心。
- robot.depth_scale_to_m：模型三维坐标乘此尺度才成为米，须用实物验证；仅凭单目预测不能保证间距。
- robot.standoff_m：正数，预吸附中心到目标面的法向间距。
- robot.suction_size_m：[沿 +x 的接触宽度、沿 +y 的接触长度]。
- robot.edge_margin_m：矩形边界与吸盘接触外形的最小内缩量。
- robot.calibration_verified：以上接口经现场核对后置 true，默认 false。

robot/evt/source.urdf 是原始 CAD 导出，robot.urdf 仅把网格路径改为 package://cam10_evt/meshes/，几何与关节值未改动。URDF 版本由 SHA256 校验，换模型必须同步更新配置。该 CAD 导出的速度/力矩限额为零，控制端必须使用自己的有效动态限额。可用 colcon 构建 robot/evt 描述包，但本仓库不声称已做 ROS2/真机验证。

## 4. 几何与找平输出

frames.jsonl 中 tracks[].current 是当前 RGB SAM + 深度观测。normal_camera 是单位向量；corners_camera_model_units、center_camera_model_units 和 d_model_units 属于模型尺度。quad_px 是矩形投影近似，不是已标定真实角点。current=null 表示仅预测/过期，不输出可用于找平的新法向量。

机器人建议先将点、法向量转换到 base 系。目标吸盘 +z=-n_base；在四条目标边中选择足够容纳吸盘、旋转幅度最小的平行方向作为 +x，得到完整旋转。目标中心为四角中心+n_base×standoff。输出：

- T_base_suction_target / T_base_flange_target。
- rotation_delta_base_axis_angle_rad（左乘在 base 系中的旋转增量）。
- normal_sine_residual，以及 normal_facing_dot（应接近 +1）。
- edge_sine_residual、chosen_edge_index、预测边距。
- 带关节限位的数值 IK 结果与位置/姿态误差。
- collision_checked=false、robot_motion_ready=false。

这里使用闭式旋转解将平行残差消到零，再进行受限 IK；不需要用迭代调整正弦值来求旋转。IK 成功仅表示几何可达，不包含碰撞、轨迹、负载或物体尺寸真实性验证。

## 5. 参考节点与返回流程

融合矩形与 SAM 边界采用双向平均轮廓距离，除以图像对角线归一化。每个轨迹保留误差最小参考，不根据单帧面积最大或最新来选参考。遮挡或过期帧不能更新参考。此指标衡量边界一致性，不是抓取正确率；SAM 若含多个箱面，仍可能失配。

参考目录 reference.json 保存 geometry、joint_state、clock、urdf_sha256、config_sha256、原图哈希。运行目录保存 URDF 与配置快照。若关节状态缺失，can_return_to_reference=false。坐标快照绑定当时图像，不能在移动以后直接当作当前相机系平面。

reference-plan 输出三个阶段的草案：
1. return_reference：记录的关节位置，由控制端规划无碰撞返回轨迹。
2. verify_reference：用新状态确认到位、底座未变、目标在新画面仍有效。
3. align_and_standoff：使用参考姿态下的几何和配置生成找平/旋转/间距建议。

控制端必须反馈每阶段完成后再推进。建议接入急停、姿态/速度限制、接触压力互锁及视觉失效暂停。此工程不发送关节命令，也不自动开启吸附。底座移动或箱子移动之后应重新观测建立参考，不能仅凭机械臂关节值重用旧节点。

一次 run 对应一次接近/吸附任务。目标丢失不会自动换箱子；控制端在释放或取消后结束本轮并开始新轮。长期运行时非目标过期轨迹会释放图像缓存，参考节点仍保存到磁盘。frames.jsonl 的 model_timing 提供 MoGe 总耗时、SAM CUDA 事件耗时与 PyTorch 分配/保留显存；这些不是整机 GPU 利用率，也不等于所有并发进程的显存占用。
