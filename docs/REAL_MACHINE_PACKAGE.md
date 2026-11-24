# Thor 真机视频流测试包

## 包的区别

- `cam10-suction-alignment-0.2.0.tar.gz`：代码包，不含权重、引擎、上游源码和预编译程序。单独复制到空环境不能推理。
- `cam10-thor-with-models-20261124.tar.gz`：完整应用包，包含主线代码、接口文档、EVT URDF/meshes、MoGe2 权重、三个 SAM3 TensorRT 引擎、tokenizer、固定版本 MoGe/SAM 部署源码、Thor C++ 程序、样例视频、配置及校验工具。

完整应用包不包含操作系统、JetPack、CUDA/TensorRT 动态库、Python 环境或依赖 wheel。SAM3 提供的是已编译的引擎，不包含可供任意硬件重新导出的原始 SAM checkpoint。换 GPU/TensorRT 版本可能需要重新生成引擎。

包针对现有 Thor aarch64 环境。采集到的环境记录在 `deployment/thor/environment.json`，C++ 动态库清单在 `deployment/thor/binary-libraries.txt`；包含 CUDA 13、TensorRT libnvinfer.so.10 和 C++ OpenCV 4.15。记录供排查，不是建议在其他平台直接安装的依赖锁定表。

## 1. 解压和检查

在新的目录解压，避免覆盖已有工作目录。将压缩包和同名 `.sha256` 文件放在一起：

```bash
sha256sum -c cam10-thor-with-models-20261124.tar.gz.sha256
mkdir cam10-deployment
 tar -xzf cam10-thor-with-models-20261124.tar.gz -C cam10-deployment
cd cam10-deployment
python3 tools/check_deployment.py
```

校验应在编辑配置之前执行。MANIFEST.json 记录全部文件大小和 SHA256。

## 2. 使用现有 Thor 环境

在目前已验证的 Thor 上可复用已有环境：

```bash
source /home/nvidia/segment_depth_estimation/.venv/bin/activate
python -m pip install --no-deps -e .
python -m pip install --no-deps -e third_party/MoGe
python tools/check_deployment.py --runtime
python -m cam10_alignment.cli check --config configs/mainline/sample.json
python -m cam10_alignment.cli run --config configs/mainline/sample.json --out outputs/sample-check
```

`--no-deps` 仅适用于依赖已经齐全的现有 Thor；新机器需先准备匹配硬件的 PyTorch/CUDA/TensorRT，以及 MoGe 的 Python 依赖，参见 DEPLOYMENT.md 和上游 pyproject.toml。依赖检查通过不等于 TRT 引擎兼容，最后的三帧样例推理才验证完整加载链路。

预编译程序在 `build/sam3_export`。若 C++ 动态库不匹配，在新目录编译后修改配置中的 `sam_executable`：

```bash
cmake -S cpp -B build-local -DCMAKE_BUILD_TYPE=Release
cmake --build build-local -j4
```

## 3. 接入实时视频

复制 `configs/mainline/live.json` 为同目录下的 `live-machine.json`，填写：

- `input.source`：摄像头编号字符串（如 `"0"`）或 RTSP 地址；`input.mode` 保持 `live`。
- `camera.image_size`、`camera.K_pixels`：对应实际输入的已去畸变图像；示例中的 256×256 内参未经真机标定，不能直接用于控制。当前深度适配器要求居中的主点和 fx=fy，其他输入需先按接口约定校正。
- 若需要参考姿态及对齐提案，接入 `input.joint_states`，填写外参、吸盘接触系、尺度及预吸附距离等 robot 配置，详见 INTERFACES.md。仅测试视觉时可保留这些接口为空。

```bash
python -m cam10_alignment.cli run --config configs/mainline/live-machine.json --out outputs/live-test-001
```

每次使用新的输出目录。输出包括预览视频、逐帧 JSON、状态日志及参考帧快照。参考帧按融合平面与实例边缘吻合程度更新，快照保存图像、法向量、四边形、同步关节状态（有输入时）、URDF 和配置指纹。未接关节状态时不能恢复参考机械臂姿态。

这是实时视频输入接口，并不保证实时推理帧率；已有 Thor 回放实测约 1.58 次观测/秒，现场帧率需重新测量，live 模式取最新帧。

## 4. 控制端交接

此包输出相机系几何和对齐位姿提案，不包含机械臂通信驱动、碰撞检测或执行程序。相机/吸盘外参、深度米制尺度和时间同步未完成时，仅输出视觉结果。`robot_motion_ready` 保持 false。

控制端按 INTERFACES.md 接入关节状态和位姿提案；回到参考姿态后重新观察目标，再执行找平与边缘对齐。URDF 是机器人结构描述，不能代替实时关节状态。法向量/四边形预测也不能视为已验证的箱体真值。
