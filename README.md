# Recongnition and Recover

唯一主线：**cam10 视频 → MoGe2 深度平面 → 原始 RGB 的 SAM3 实例 → 融合平面与相机系法向量 → EVT 运动学对齐建议 → 箱面四边形补全和跟踪 → 吸盘边缘平行与预吸附间距。**

支持文件回放、摄像头和 RTSP 实时输入。模型常驻 GPU；实时采集仅保留最新帧，不排队处理旧画面。预览视频的帧率与实际推理速度分开记录。

## 入口

在仓库根目录执行，使用安装了模型依赖的 Python：

```bash
python -m pip install -e .
cam10-align check --config configs/mainline/replay.json
cam10-align run --config configs/mainline/replay.json --out outputs/my-run
# 实时输入：先填写 live.json 中的设备/RTSP、内参、关节状态来源
cam10-align run --config configs/mainline/live.json --out outputs/live-run
python -m unittest discover -s tests -v
```

也可以用 `python -m cam10_alignment.cli`，不依赖安装命令行入口。输出目录必须是新的。

**配置中的内参是旧 256×256 去畸变样本的示例，不可直接当成真机标定。** 当前 cam10 link、吸盘接触坐标系、深度尺度、吸盘尺寸和预吸附距离均保留为显式接口。未填写时，感知正常运行，机械臂建议返回缺失原因。

## 最优参考节点

每个跟踪实例在有效融合帧中，比较四边形边界与 SAM 实例轮廓的**双向平均轮廓距离**，取截至当前误差最小的一帧，而非最早或最后一帧。得分相同时保留更早参考帧。无法一次在线预知未来的全局最优帧；整段结束后保存的就是该跟踪实例的最终最优参考帧。

每次参考帧改善都会保存原图、相机系平面、四角、帧时间、同步关节状态和 URDF/配置指纹。运行目录保存当时 URDF 与完整配置副本。没有同步状态的参考帧可以复核，但不能用于返回机械臂节点。

目标实例选择与参考帧选择是两件事：目前以吸盘前方 ROI 的覆盖率、候选歧义和连续观测确认目标；确认后不静默切换另一个实例。参考帧最优也不意味着目标箱体一定选对。

```bash
cam10-align reference-plan \
  --bundle outputs/my-run/references/track-0000/frame-00000240 \
  --out outputs/my-run/controller-handoff.json
```

控制端交接分三步：回到参考关节位姿 → 用新反馈确认到位、底座未移动且目标仍存在 → 找平、边缘旋转和预吸附间距。这里生成的是计划草案，**没有下发机械臂的网络/ROS 执行器**。同事需要在控制端接入轨迹规划、碰撞检测、速度约束与接触保护，不能直接把历史平面当成当前观测。

## 输出

- `review.webm` / `index.html`：原图与融合叠加视频。绿色是当前深度支持；淡橙色是补全/历史预测；灰色是过期。
- `frames.jsonl`：每个实际处理帧的时间戳、候选、当前有效法向量、参考帧、机械臂建议和耗时。
- `references/track-*/frame-*/`：不可覆盖的参考节点记录、原图。
- `robot.urdf`、`config.json`：运行时运动学和配置快照。
- `summary.json`：最终目标与各轨迹最佳参考节点索引。
- `model_cache/`：本轮原图和 SAM 掩膜，可用于复核。

法向量使用去畸变 cam10 光学系：x 向右、y 向下、z 向前；平面法向量朝相机一侧。吸盘接触坐标系约定 +z 朝目标，+x 沿吸盘指定边。找平同时约束法向量平行和面向目标，避免仅优化正弦值时出现 180° 朝向歧义。

箱面补全是三维平面内矩形假设，未知隐藏边界不是测量真值。预测帧不提供新的有效法向量；超时不继续向控制端提供历史结果。

## 当前目录

| 目录 | 内容 |
|---|---|
| cam10_alignment/ | 唯一运行包：视频、模型、融合、跟踪、参考记录、运动学 |
| configs/mainline/ | 回放与实时配置模板 |
| robot/evt/ | 本机 EVT URDF、原始副本、网格、来源指纹及可选 ROS2 描述包 |
| cpp/ | SAM3 TensorRT 常驻推理桥接 |
| tests/ | 主线几何、参考节点、运动学与接口验证 |
| docs/ | 部署、接口与验证记录 |
| models/、third_party/ | 模型与固定版本依赖，不进 Git |
| data/、outputs/ | 数据及所有历史结果，保留原路径、不进 Git |
| archive/20261120-pre-mainline/ | 已移出的旧功能与清单；不打包、不进 Git |

详细接入：[接口约定](docs/INTERFACES.md) · [部署运行](docs/DEPLOYMENT.md) · [验证范围](docs/VALIDATION.md)。
