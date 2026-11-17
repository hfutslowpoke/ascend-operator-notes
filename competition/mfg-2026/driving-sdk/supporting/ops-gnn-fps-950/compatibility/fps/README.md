# FPS 兼容性差异

此目录保存随包旧 oracle 和原始 torch_cluster CPU 两个对拍面。它们不在 pyproject.toml 的默认 testpaths 内；正式门禁会显式执行并保存逐例结果。

## 随包旧 oracle

- supplied_attachment/test_fps.py
- SHA256 47eb2ce17d4679d6cf3eb5f7bf8cd0440afd2c4ef6f32e02e7c4ce68203e46ab
- supplied_attachment/golden.py
- SHA256 ce498dce169b98e5850b96edb23dc5250f4db36bc5db5e7874a686730723e281

文件保持冻结 v7 包内原字节。已知结果为 44 passed / 17 failed，不能改成 skip。差异来自逐维 FP16 舍入、mod-256 tie、float32 count 和 torch.manual_seed 随机首点。

## raw torch_cluster

- raw_upstream/test_raw_torch_cluster_task_matrix.py
- SHA256 f2b7a51ab252603be9cb5519cd16c6a809cd5c8afa460a26dc499a7231123dc0

该脚本使用原输入直接调用 torch_cluster 1.6.3 CPU，不改 ratio。2026-09-21 旧候选的已知结果为 58 passed / 3 failed：

- FP16，N=10000，ratio=0.3：当前合同3000，raw upstream3001；
- FP16，ptr=[0,150,200]，float32 ratio=[0.2,0.8]：当前合同72，raw upstream70；
- FP32，同一 ptr/ratio：当前合同72，raw upstream70。

三项都是旧版 float64 计数造成的 count-only 差异。2026-09-23 的正式验收 Issue #2 要求按 CPU dtype/float32 语义修复。旧结果保留；新版须在同一冻结版本上重跑完整 61 项，不能沿用旧结果宣称通过。

## 66 项验收套件

正式 Issue #2 给出 `pytest test/fps -vv` 的 63/66 摘要、三个失败用例与 golden 54/54 交叉对拍，但未附完整测试和 golden 源文件。本仓自测不能冒充验收方的 66 项；重交前须在同一冻结版本上完成可取得的全量功能、性能和真实 CPU 对拍，并保留无法运行官方原套件的边界。
