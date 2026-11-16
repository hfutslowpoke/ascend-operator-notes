# 本机 Ascend 910B3 复测记录（2026-11-16）

状态：本机纯 Triton 复测完成。未编译 Ascend C 自定义算子，未运行 profiler，未提交比赛平台。结果只适用于本机环境，不是作者环境的严格复现。

## 1. 环境

| 项 | 值 |
|---|---|
| 设备 | Ascend 910B3，NPU index 2 |
| `npu-smi` | 26.1.1，健康 OK，HBM 65536 MB |
| CANN | 9.1.0（`/usr/local/Ascend/cann-9.1.0/opp/version.info`） |
| Python | 3.10.12（隔离 venv） |
| PyTorch | 2.10.0+cpu |
| torch_npu | 2.10.0 |
| Triton-Ascend | 3.2.0 |
| venv | `/workspace/acw/var/private/venvs/triton-ascend-310` |
| 串行锁 | 所有 NPU 运行均通过 `bin/npu-lock` |

原始环境输出：[`our_work/evidence/local-910b-20261116/environment.txt`](../../our_work/evidence/local-910b-20261116/environment.txt)。

### 1.1 本地兼容性 workaround

首次运行 `triton-ascend 3.2.0` 时，驱动在编译 `npu_utils.cpp` 时失败：

```text
error: ‘RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE’ is not a member of ‘rtLimitType_t’
```

本机 CANN 9.1.0 的 `rtLimitType_t` 只有 `RT_LIMIT_TYPE_SIMT_STACK_SIZE`、`RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE` 等，没有 `WARP_STACK_SIZE`。为了让本轮纯 Triton 复测继续，我在**隔离 venv** 的 `triton/backends/ascend/npu_utils.cpp` 中移除了未使用的 `LimitTypeMap` 条目 `WARP_STACK_SIZE`：

```diff
- {"WARP_STACK_SIZE", rtLimitType_t::RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE},
+ // WARP_STACK_SIZE omitted: CANN 9.1.0 rtLimitType_t lacks this enum; local validation workaround.
```

影响与边界：

- 没有修改 CANN、驱动、系统 Python 或第三方仓库源码；
- 本轮 kernel 路径不调用 `set_device_limit("WARP_STACK_SIZE")`；若调用该接口会失败；
- 这不是兼容性修复，也不代表目标部署应这样改；换 CANN/TA 版本应重新评估。

## 2. 复测结果

| ID | 状态 | 本机参考 | 本机候选 | 本机加速比 | 作者数字 | 结论 |
|---|---|---|---:|---:|---:|---|
| `KS2026-ASC910B-GROUPEDTOPK` | PASS | 0.593500 ms | 0.222255 ms | **2.670×** | 0.701666 / 0.283144 / 2.478× | 方向一致 |
| `KS2026-ASC910B-SINKHORN` | PASS | 3.616230 ms | 0.273420 ms | **13.226×** | 3.032301 / 0.329475 / 9.203× | 方向一致；误差更大但仍过容差 |
| `KS2026-ASC910B-FUSEDMOE` | 编译失败 | — | — | — | 6.942960 / 0.606584 / 11.446× | 当前 TA/CANN UB 溢出，未能复现 |
| `XAYAH-ASC910B-SOFTMAX-NEG` | PASS（负例） | 0.096430 ms | 0.334385 ms | **0.288×** | 0.035 / 0.140 / 0.25× | 负例稳定 |
| `XAYAH-ASC910B-MATMUL-NEG` | PASS（负例） | 0.218790 ms | 0.902870 ms | **0.242×** | 0.163 / 0.699 / 0.23× | 负例稳定 |

机器可读摘要：[`our_work/evidence/local-910b-validation-20261116.json`](../../our_work/evidence/local-910b-validation-20261116.json)。

### 2.1 GroupedTopk

命令（经 `bin/npu-lock`）：

```bash
cd .../platforms/ascend910b/task01_grouped_topk
python benchmark.py --num-warps 1 --warmup 200 --repeat 500
```

原始输出：

```text
correctness: PASS (num_warps=1)
reference: 0.593500 ms
optimized: 0.222255 ms
speedup:   2.670x
```

解释：

- 正确性通过；作者 2.478×、本机 2.670×，方向一致；
- 本机 reference 比作者 0.701666 ms 更快，candidate 也比作者 0.283144 ms 更快；两个加速比不能当作同一硬件上的严格复现；
- 设备是 910B3，作者是 910B1；CANN/TA 版本也不同。

日志：[`grouped_topk_official.log`](../../our_work/evidence/local-910b-20261116/grouped_topk_official.log)。

### 2.2 Sinkhorn

命令：

```bash
cd .../platforms/ascend910b/task08_hc_split_sinkhorn
python benchmark.py --num-warps 1 --warmup 200 --repeat 500
```

原始输出：

```text
output[0] max_abs_diff=0.00000000e+00
output[1] max_abs_diff=0.00000000e+00
output[2] max_abs_diff=6.76512718e-06
PASS accuracy
reference_median_us=3616.230
optimized_median_us=273.420
speedup=13.226x
```

解释：

- 正确性通过；本机 13.226×，作者 9.203×，方向一致；
- 本机最大绝对误差 `6.765e-06`，作者 `1.192e-07`；两者都在 `atol=rtol=1e-2` 内，但本机误差明显更大；
- 该 benchmark 是逐 sample 同步、中位数；`--warmup 200 --repeat 500` 生效。

日志：[`sinkhorn_official.log`](../../our_work/evidence/local-910b-20261116/sinkhorn_official.log)。

### 2.3 FusedMoE：当前环境编译失败

命令：

```bash
cd .../platforms/ascend910b/task02_fused_moe
python benchmark.py --num-warps 4 --warmup 200 --repeat 500
```

结果：

```text
[ConvertLinalgRToBinary] encounters error:
[NOTE] Ub overflow detected; automatically set enable-code-motion to false and retrying compilation.
[NOTE] Ub overflow detected; automatically set enable-auto-multi-buffer to false and retrying compilation.
error: ub overflow, requires 2359296 bits while 1572864 bits available!
```

尝试过的配置：

1. 原始 `BLOCK_M=32`、`num_warps=4`：失败；
2. 原始 `BLOCK_M=32`、`num_warps=1`：失败；
3. 本地副本 `BLOCK_M=16`、`num_warps=4`：仍失败，required bits 不变；
4. 本地副本 `BLOCK_M=32`、`num_warps=4`、`num_stages=1`：仍失败。

结论：

- 这不是正确性或性能结果，而是**当前 TA 3.2.0/CANN 9.1.0 的编译兼容性失败**；
- 作者 11.446× 是在 910B1/CANN 9.0.0/TA 3.2.1 上的数字；不能把它外推到本机；
- 需要 TA 3.2.1 或调整 kernel 的 `tl.dot`/buffer 结构后才能继续验证。

日志：[`fused_moe_compile_fail.log`](../../our_work/evidence/local-910b-20261116/fused_moe_compile_fail.log)。

### 2.4 52Xayah Softmax：负例复现

作者脚本（批量 50 次后统一同步）：

```text
Triton 极速版耗时: 0.155 ms
PyTorch 耗时: 0.035 ms
性能对比: 0.23x
```

本机逐 sample 同步（warmup 200/repeat 500，中位数）：

```text
softmax PASS shape=4096x1024 fp32
triton_median_ms=0.334385
torch_median_ms=0.096430
speedup=0.288x
```

解释：

- 两种协议都显示 Triton 慢于 PyTorch；
- 逐 sample 的绝对时延包含同步/launch，因此 0.334 ms 高于作者批量计时的 0.155 ms；
- 负例结论稳定：朴素一行一 program、固定 1024、无 mask 的 Triton Softmax 不适合作为高性能实现。

日志：[`xayah_softmax.log`](../../our_work/evidence/local-910b-20261116/xayah_softmax.log)、[`xayah_persample.log`](../../our_work/evidence/local-910b-20261116/xayah_persample.log)。

### 2.5 52Xayah Matmul：负例复现

作者脚本（批量 20 次后统一同步）：

```text
Triton 耗时: 0.675 ms
PyTorch 耗时: 0.164 ms
性能对比: 0.24x
```

本机逐 sample 同步（warmup 20/repeat 50，中位数）：

```text
matmul PASS shape=1024x16384x1024 fp16
triton_median_ms=0.902870
torch_median_ms=0.218790
speedup=0.242x
```

解释：

- 负例方向稳定；
- 本地绝对时延高于作者，主要来自逐 sample 同步、不同 SoC/编译器和 env；
- 仍然支持“规则大 GEMM 优先厂商库或做 Cube 友好 tile/流水”的判断。

日志：[`xayah_matmul.log`](../../our_work/evidence/local-910b-20261116/xayah_matmul.log)、[`xayah_persample.log`](../../our_work/evidence/local-910b-20261116/xayah_persample.log)。

## 3. 对本批知识库的影响

### 3.1 已被本机支持的判断

- **GroupedTopk 的单调性 + 分母抵消**：正确性通过，2.670×，方向与作者一致。可以把它列为“优先复用/低风险”的代数简化。
- **Sinkhorn 的 4×4 标量化 + 单 kernel**：正确性通过，13.226×，方向与作者一致；但本机误差更大，复用时必须重新验证 eps/顺序。
- **52Xayah 的负例**：Softmax/Matmul 都稳定慢于 PyTorch；不要用“手写 Triton”作为默认优化方向。

### 3.2 被本机削弱的判断

- **FusedMoE 的 11.446× 不是可移植结论**：同一份代码在本机 TA 3.2.0/CANN 9.1.0 上编译失败。它可以作为“小 E/T 下稠密全专家可能值得”的假设，但不能作为当前环境的性能证据。
- **作者数字不能直接横比**：GroupedTopk/Sinkhorn 的绝对值与作者不同，设备/版本/参考实现都不同；只能说“方向一致”，不能说“精确复现”。

### 3.3 仍未验证

- Hansen 的 Fire/Spline/Tree：Ascend C 自定义算子，未在本机编译；需要按 CANN 9.1.0 重新构建并处理可能的 API/UB 差异。
- Mamba-2：自定义 OPP + 复杂 MIX kernel，未在本机构建；作者结果仍是 910B3/CANN 8.2.RC1。
- vLLM-HUST Linear+SwiGLU：需要 vLLM/vLLM-Ascend 依赖链，未在本机搭建。
- 910C：没有任何本机数据。

## 4. 复现边界

- 本机是 910B3，不是作者的 910B1/910B2/910B4；不能把本机数字当作作者环境的复现。
- 仅使用纯 Triton 任务；Ascend C 自定义算子、OPP 构建、vLLM 端到端未验证。
- 没有 msprof/PMU；时延包含 forward/launch/同步，不能拆出纯 kernel 时间。
- `triton-ascend` 的 `npu_utils.cpp` workaround 只用于隔离 venv，不是可部署修复。
- 未提交比赛平台，未验证规则合规。

## 5. 命令与日志索引

- 环境：[`environment.txt`](../../our_work/evidence/local-910b-20261116/environment.txt)
- 机器摘要：[`local-910b-validation-20261116.json`](../../our_work/evidence/local-910b-validation-20261116.json)
- 原始日志：
  - [`grouped_topk_official.log`](../../our_work/evidence/local-910b-20261116/grouped_topk_official.log)
  - [`sinkhorn_official.log`](../../our_work/evidence/local-910b-20261116/sinkhorn_official.log)
  - [`fused_moe_compile_fail.log`](../../our_work/evidence/local-910b-20261116/fused_moe_compile_fail.log)
  - [`xayah_softmax.log`](../../our_work/evidence/local-910b-20261116/xayah_softmax.log)
  - [`xayah_matmul.log`](../../our_work/evidence/local-910b-20261116/xayah_matmul.log)
  - [`xayah_persample.log`](../../our_work/evidence/local-910b-20261116/xayah_persample.log)
