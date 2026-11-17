# FPS 自测与复现说明

本目录的默认测试入口遵循任务书的真实 `torch_cluster` CPU 标杆及验收 Issue #2：标量 ratio 先存入 `src.dtype`，Tensor ratio 保留原 dtype；随后用 `degree.float() * ratio` 的 PyTorch dtype promotion 计算并 ceil。选点序列使用真实 PyTorch CPU 表达式 `(x - x[idx]).pow_(2).sum(1)`、minimum 和 first-index argmax 逐步验证。本套件不是验收方未提供源码的 66 项套件。

## 固定环境

- Ascend 950
- CANN /usr/local/Ascend/cann-9.1.0/set_env.sh
- Python /usr/local/python3.12.13/bin/python3
- PyTorch 2.9.1、匹配的 torch_npu
- torch_cluster 1.6.3 + pt29cpu，仅用于 raw upstream 兼容性对拍

不要向全局 Python 环境安装依赖。使用候选目录的隔离 PYTHONPATH 和实际构建 SO：

    source /usr/local/Ascend/cann-9.1.0/set_env.sh
    export PYTHONPATH="$PWD/python"
    export LD_LIBRARY_PATH="$PWD/python/ops_gnn:${LD_LIBRARY_PATH:-}"
    export FPS_EXPECTED_ROOT="$PWD"
    export NPU_DEVICE_ID=0

所有构建和 NPU 测试必须持有 /tmp/ascend_npu0_codex.lock。正式门禁使用 scripts/run_fps_resubmit_gate.sh，不要并行启动另一个 NPU 进程。

## 当前合同专项

    /usr/local/python3.12.13/bin/python3 -m pytest \
      test/fps/arch35/test_fps.py \
      test/fps/arch35/test_fps_count_contract.py \
      -vv --tb=long

test_fps.py 保留随包 61 个输入场景，并按 Issue #2 的类别摘要补 1D 输入 2 项、ratio=None 1 项、随机首点单点 batch 2 项，组成自建 66 场景。每个 NPU 输出先用独立 count oracle 校验长度，再用真实 CPU 运算逐轮检查首点、距离更新、minimum 与 argmax；没有 ratio_override。test_fps_count_contract.py 另有 24 个 CPU 参数化用例，直接检查当前 `fps.py` 的 eager 元数据函数，并脚本化相同函数；公共 TorchScript 分支使用只返回正确长度的 CPU 测试算子执行。该测试算子不验证 NPU 选点。关键边界包括：

- `src=float16`、Python float `0.3`、degree 10000 得 3001；`src=float32` 得 3000；
- float32 Tensor `[0.2,0.8]`、degree `[150,50]` 得 `[30,40]`，ptr/batch 两条路径合计 70；
- Tensor ratio 分别为 float16/float32/float64 时保留存储 dtype，并按 PyTorch promotion 计算；
- Python float `0.500000005` 存为 float32 `0.5` 后，degree 2 得 1。

random_start=True 使用 libc srand/rand。torch.manual_seed 不控制 torch_cluster 1.6.3 CPU 的首点。逐 batch 对拍固定 torch.set_num_threads(1)。

## TorchScript

    /usr/local/python3.12.13/bin/python3 \
      test/fps/arch35/check_torchscript.py

该脚本覆盖 ratio 为 float/Tensor、ptr 为 Tensor/List 的四种 overload，并比较 scripted 与 eager NPU 输出。

## 完整仓回归

当前专项通过后，按 pyproject.toml 的默认 testpaths 执行完整仓库测试：

    /usr/local/python3.12.13/bin/python3 -m pytest \
      test -vv --tb=long --junitxml=/tmp/fps-results/repo.xml

这一步必须包含 test_import.py 及 FPS 以外的 gather_coo、gather_csr、graclus_cluster、radius、random_walk、scatter、segment_max_csr、sparse、spmm_max 测试。报告必须保留 passed、failed、skipped、error、timeout 的逐项状态，不能只写摘要。

## 兼容性对拍

两套历史/上游对拍位于 candidate/compatibility，因此不会被默认 testpaths 静默混入当前合同回归。正式门禁仍会显式运行并保存 JUnit：

    /usr/local/python3.12.13/bin/python3 -m pytest \
      compatibility/fps/supplied_attachment/test_fps.py \
      -vv --tb=long

    /usr/local/python3.12.13/bin/python3 -m pytest \
      compatibility/fps/raw_upstream/test_raw_torch_cluster_task_matrix.py \
      -vv --tb=long

冻结 v7 的历史结果分别为 44 passed / 17 failed 和 58 passed / 3 failed；原始失败证据不改动。前者包含旧附件的逐维 FP16、mod-256 tie、计数和 torch.manual_seed 合同差异；后者 3 项均为本次修复的计数差异。修复后的结果必须由新源码、新构建产物和新日志重新证明。

## 性能

专项和完整仓回归完成后执行：

    /usr/local/python3.12.13/bin/python3 \
      test/fps/arch35/benchmark_fps.py \
      --warmup 20 --iter 100 --device npu \
      --json-output /tmp/fps-results/performance.json \
      --csv-output /tmp/fps-results/performance.csv

任务书 21 个形状乘两种 dtype，共 42 项。每项 performance_ratio = official_reference_ms / measured_ms，必须逐项大于等于 0.45；任一 fail、missing、skip、timeout 或 unknown 均不允许提交。

## 证据边界

验收方 [Issue #2](https://gitcode.com/gcw_i0yFZkCV/ops-gnn-fps-950/issues/2) 报告官方 CPU 扩展交叉对拍 54/54、专项功能 63/66；失败是 `test_large[fp16]` 与 `test_per_batch_ratio_ptr[fp16/fp32]` 的三个计数用例。Issue 给出了场景摘要和日志片段，但未提供验收方 66 项测试源码、完整原始日志或被测 checkout SHA。任务书随附 `test_cases/test_fps.py` 可读，但其手写 `fps_single` 对 fp16 标量 `0.3` 的 10000 点边界算出 3000；Issue #2 已用编译版官方 CPU 扩展核验为 3001，因此本轮按真实 CPU 标杆修复。本目录的自建 66 场景和新增 CPU 边界用例是复现准备，即使数量相同也不能作为官方 66/66 结果。正式复审、完整回归、42 项性能和本地测试分别报告。
