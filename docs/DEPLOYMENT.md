# 部署与运行

从仓库根目录运行。核心需要 Python>=3.10、NumPy、SciPy、OpenCV；MoGe2 需要匹配 GPU 的 PyTorch。Thor 请保留 JetPack 对应的 CUDA/PyTorch/TensorRT 环境，不要用通用 pip 包替换系统 TensorRT。

```bash
python -m pip install -e .
python -m pip install -e third_party/MoGe
python -m pip install tokenizers
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
python -m cam10_alignment.cli check --config configs/mainline/replay.json
```

固定上游版本见 configs/sources.lock.json。models/moge-2-vitl/model.pt 是 MoGe2 权重；models/sam3 下放 decoder.engine、text_encoder.engine、vision_encoder.engine。引擎必须与部署机 TensorRT/GPU 兼容。C++ 桥接程序 build/sam3_export 与 sam3-deployment/demo 一起编译，读取原始 RGB、文本 box，输出同尺寸掩膜。

正式运行前填写配置的视频源、尺寸/K、同步关节状态。live.json 用最新帧采集；replay.json 用固定 stride 回放。绝对路径可直接填入，其他路径相对 project_root，默认仓库根目录。

```bash
python -m cam10_alignment.cli run --config configs/mainline/replay.json --out outputs/first-run
python -m http.server 18788 --bind 127.0.0.1 --directory outputs/first-run
```

访问 http://127.0.0.1:18788/。失败记录在 status.json，SAM 日志在 sam.log。参考节点返回计划命令见 README。每次运行使用新输出目录，避免覆盖参考记录。

给同事打包仅需 cam10_alignment、cpp、configs、robot、tests、docs、README.md、pyproject.toml。权重、TensorRT 引擎和 third_party 另外按固定版本部署，原始数据/outputs/archive 不需要复制。执行 tools/package_mainline.py 可生成不含旧功能和大模型的代码包。

实时接口尚需现场完成：硬件曝光时间同步、cam10 optical 外参、吸盘接触系与尺寸、尺度验证、预吸附距离，以及控制端轨迹与压力状态机。本仓库的 UDP/TCP/ROS 控制发送端有意留空，交接格式是可审查 JSON。
