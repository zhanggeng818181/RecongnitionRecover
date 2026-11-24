# GitHub 协作

远程仓库：git@github.com:zhanggeng818181/RecongnitionRecover.git

```bash
git clone git@github.com:zhanggeng818181/RecongnitionRecover.git
cd RecongnitionRecover
git switch -c codex/my-change
# 修改后运行测试
python -m pytest tests/test_mainline.py -q
git add <修改的代码文件>
git commit -m "描述修改"
git push -u origin codex/my-change
```

GitHub 保存主线代码、配置模板、EVT 描述、接口文档和测试。模型、引擎、视频数据、推理输出、编译产物和部署压缩包留在本地，通过完整部署包单独交付。克隆代码仓库不等于获得模型，真机接入见 REAL_MACHINE_PACKAGE.md。

原初始化提交中包含验证缓存和旧代码包；后续提交停止跟踪，未重写历史。`.gitignore` 不会清除历史版本。

GitHub Actions 在 push / pull request 时运行 CPU 单元测试，不加载大模型、不代表 Thor GPU 推理或真机动作验证。现场配置和访问凭据不要提交；在本机独立管理。
