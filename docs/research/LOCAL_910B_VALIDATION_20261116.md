# 本机 Ascend 910B3 复测记录（2026-11-16）

状态：本机纯 Triton 复测 + Be4ver 的两个 Ascend C 算子复测完成。Parity-LRX 未在本机构建，未运行 profiler，未提交比赛平台。结果只适用于本机环境，不是作者环境的严格复现。

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

### 1.2 Ascend C 构建环境

Be4ver 的两个算子使用仓库自带 CMake，目标 SoC 固定为 `ascend910b3`；本机用 CANN 9.1.0 重新配置和构建，没有修改第三方源码：

```text
cmake 3.22.1
gcc 11.4.0
CANN 9.1.0 (/usr/local/Ascend/cann-9.1.0)
SOC ascend910b3
```

构建命令与结果见 [`environment_and_build.txt`](../../our_work/evidence/local-910b-20261116/environment_and_build.txt)。所有设备运行均经 `bin/npu-lock`。

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

### 2.6 Be4ver PACO 批量 Jacobi SVD：Ascend C 构建并复测

源码副本：`repository-research-cache/local-validation-20261116/be4ver_paco`（固定 commit `b027965...`，未修改源码）。构建：

```bash
cmake -S paco-jacobi-svd -B build-local -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0
cmake --build build-local -j 8
```

运行（32×32、batch=512，与作者 `benchmark_32x32_batch512_run1` 同 shape）：

```bash
python3 ../scripts/gen_jacobi_svd_data.py --rows 32 --cols 32 --batch 512 \
  --seed 2024 --out-dir ../input
bin/npu-lock bash -lc 'source /usr/local/Ascend/cann-9.1.0/set_env.sh; \
  cd .../be4ver_paco/build-local && ./jacobi_svd_main 0 32 32 512'
python3 ../scripts/verify_jacobi_svd.py --input-dir ../input --output-dir ../output
```

原始输出：

```text
==== 批量单边块状雅可比 SVD（AIV 旋转 + AIC 算 Gram）====
矩阵: 32 x 32 -> pad 64 x 32, 批大小: 512, AIV 核数: 40, 最大 sweep: 30
耗时: 6.174 ms (batch=512, 32x32)
总结: 512 个矩阵中 0 个 FAIL
  最大奇异值绝对误差 : 3.242e-05  (矩阵 92)
  最差 B 行正交性    : 9.345e-07  (矩阵 407)
  最差 Vt 行正交性   : 1.025e-06  (矩阵 76)
  最差重构相对误差   : 1.650e-06  (矩阵 150)
结论: 全部通过 ✅
```

判断：

- 当前仓库的通用路径在 CANN 9.1.0 上可编译、可运行、512/512 通过；
- 本机 6.174 ms，作者记录 7.061 ms，相差约 12.6%；作者未固定 CANN 版本，不能称为严格同环境复现；
- 这不是 `README` 里 64×64 tournament 原型的 4.26×；tournament 只是 `docs/research-64x64/tournament.patch` 的研究补丁，未进入当前主路径。

日志：[`paco_32x32_b512_run.log`](../../our_work/evidence/local-910b-20261116/paco_32x32_b512_run.log)、[`paco_32x32_b512_verify.log`](../../our_work/evidence/local-910b-20261116/paco_32x32_b512_verify.log)。

### 2.7 Be4ver QDWH+SDC 融合：128/512/1024 方阵

源码副本：`repository-research-cache/local-validation-20261116/be4ver_qdwh`。构建：

```bash
cmake -S qdwh_sdc_fused_jacobi -B build-local -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0
cmake --build build-local --target fused_main -j 8
```

运行（与作者方阵基准同配置：`blockDim=20`、`threshold=64`、`maxIter=4`、`cond=1e2`）：

```bash
python3 gen_data.py --shapes 128x128 --cond 1e2 --seed 42 --out-dir input
bin/npu-lock bash -lc 'source /usr/local/Ascend/cann-9.1.0/set_env.sh; \
  cd .../be4ver_qdwh/build-local && ./fused_main 0 20 64 4 1.0 1e-6'
python3 verify.py --threshold 64
python3 verify_sigma.py
```

| 方阵 | 本机 kernel | 本机 host wait | 作者 kernel | 本机/作者 | leaves | max_leaf | QDWH/SDC/sigma |
|---:|---:|---:|---:|---:|---:|---:|---|
| 128×128 | **4.288 ms** | 4.442 ms | 4.392 ms | 1.024×（本机更快） | 3 | 53 | PASS |
| 512×512 | **20.698 ms** | 21.313 ms | 20.540 ms | 0.992× | 12 | 59 | PASS |
| 1024×1024 | **48.475 ms** | 48.647 ms | 46.839 ms | 0.966× | 23 | 63 | PASS |

精度摘要：

```text
128:  u_err=7.119e-07  eig_err=5.819e-08  trace_err=4.837e-09  frob_err=7.417e-09  sigma_max_abs=9.538e-07
512:  u_err=8.422e-07  eig_err=1.839e-07  trace_err=5.654e-08  frob_err=6.471e-08  sigma_max_abs=1.004e-06
1024: u_err=8.967e-07  eig_err=1.942e-07  trace_err=2.341e-08  frob_err=1.268e-08  sigma_max_abs=1.543e-06
```

判断：

- 三个尺寸都通过 QDWH 极分解、SDC 叶块特征值/迹/Frobenius、sigma 校验；
- 本机与作者在 128/512 基本一致，1024 慢约 3.5%；差异可归因于设备/CANN 版本和单次运行噪声，不能当作算法退化；
- 这证明“单 host main、一次 kernel launch、device 内递归、H 不落盘”的融合路径在 910B3 + CANN 9.1.0 上可执行。

日志：[`qdwh_128x128_run.log`](../../our_work/evidence/local-910b-20261116/qdwh_128x128_run.log)、[`qdwh_512x512_run.log`](../../our_work/evidence/local-910b-20261116/qdwh_512x512_run.log)、[`qdwh_1024x1024_run.log`](../../our_work/evidence/local-910b-20261116/qdwh_1024x1024_run.log)，以及同目录的 `*_verify.log`。

#### 2.7.1 1000×800 seed=42：已知非确定性未在本机触发

作者 README 明确记录：`1000×800`、`seed=42`、`cond=1e2` 在 910B3/CANN 8.2.RC1 上曾出现奇异值误差 `1.5e-5~3.1e-4` 且逐次运行不同；同 shape 换 seed 1/2/3 又稳定。作者归因于 `QdwhPmat` 的 atomic 累加 + 特征值间隙放大。

本机用同一 shape/seed/cond 连跑 3 次：

| 次数 | kernel_event | host_wait | sigma.bin SHA256 |
|---:|---:|---:|---|
| 1 | 36.812 ms | 36.967 ms | `32f444bb...e666c87` |
| 2 | 38.104 ms | 38.273 ms | `32f444bb...e666c87` |
| 3 | 36.953 ms | 37.157 ms | `32f444bb...e666c87` |

三次 `sigma.bin` 完全一致；QDWH `u_err=1.080e-06`、SDC `eig_err=1.457e-07`、sigma max abs `1.801e-06`，全部 PASS。

判断：

- 不能据此否定作者的缺陷：它可能依赖 CANN 版本、atomic 调度、输入生成的末位或特征值间隙；
- 也不能据此说“已经修复”：本机只是 3 次、一个输入，没有做 20 次/多 seed/多进程；
- 正确结论是：**该非确定性是环境相关、概率触发的已知风险**；复用时必须用目标环境重复运行验证，而不是假设它一定出现或一定不出现。

日志：[`qdwh_1000x800_run1.log`](../../our_work/evidence/local-910b-20261116/qdwh_1000x800_run1.log)、[`run2.log`](../../our_work/evidence/local-910b-20261116/qdwh_1000x800_run2.log)、[`run3.log`](../../our_work/evidence/local-910b-20261116/qdwh_1000x800_run3.log)、[`verify.log`](../../our_work/evidence/local-910b-20261116/qdwh_1000x800_verify.log)。

### 2.8 第二批本地复测的边界

- 只复测了 Be4ver 的当前主路径；64×64 tournament 是研究补丁，未合入，未做本地设备复测。
- 作者未在仓库中固定 CANN 版本；本机用 CANN 9.1.0，不能声称完全同环境。
- PACO 没有同环境 PyTorch/厂商库 baseline，6.174 ms 只能与作者 7.061 ms 比较，不能换算成相对 baseline 的加速比。
- QDWH 的 1000×800 `seed=42` 非确定性缺陷来自作者 README，本机没有复现该特定输入；复用时必须把它当作已知风险。
- Parity-LRX 未在本机构建/运行；其性能数字仍是 `O+A`，不是 `V`。

## 3. 对本批知识库的影响

### 3.1 已被本机支持的判断

- **GroupedTopk 的单调性 + 分母抵消**：正确性通过，2.670×，方向与作者一致。可以把它列为“优先复用/低风险”的代数简化。
- **Sinkhorn 的 4×4 标量化 + 单 kernel**：正确性通过，13.226×，方向与作者一致；但本机误差更大，复用时必须重新验证 eps/顺序。
- **52Xayah 的负例**：Softmax/Matmul 都稳定慢于 PyTorch；不要用“手写 Triton”作为默认优化方向。
- **Be4ver 的批量矩阵交错 + 合并归约**：PACO 在 32×32×512 上 512/512 通过，6.174 ms 与作者 7.061 ms 同量级；说明把 K 个矩阵按行交错、把 `Mul/WholeReduceSum` 合并成一条长指令的路径可执行。
- **Be4ver 的单 kernel QDWH+SDC 融合**：128/512/1024 都通过 QDWH/SDC/sigma 校验，4.288/20.698/48.475 ms 与作者 4.392/20.540/46.839 ms 接近；单 launch、device 内递归、H 不落盘的融合路径可执行。
- **1000×800 seed=42 在本机 3 次运行中稳定**：sigma.bin SHA256 三次相同，QDWH/SDC/sigma 全 PASS；作者记录的 atomic 非确定性没有触发。这说明该风险是环境/调度相关的，不是必然现象。

### 3.2 被本机削弱的判断

- **FusedMoE 的 11.446× 不是可移植结论**：同一份代码在本机 TA 3.2.0/CANN 9.1.0 上编译失败。它可以作为“小 E/T 下稠密全专家可能值得”的假设，但不能作为当前环境的性能证据。
- **Be4ver 的 4.26× 不是当前代码成绩**：4.26× 来自 64×64 tournament 研究补丁（`docs/research-64x64/tournament.patch`），没有合入当前主路径；本机验证的是通用路径。
- **QDWH/SDC 的 atomic 非确定性是真实风险**：作者记录 1000×800、seed=42 时奇异值误差 1.5e-5~3.1e-4 且逐次不同；本机没有复现该输入，但复用时不能忽略。
- **作者数字不能直接横比**：GroupedTopk/Sinkhorn/Be4ver 的绝对值与作者不同，设备/版本/参考实现都不同；只能说“方向一致/同量级”，不能说“精确复现”。

### 3.3 仍未验证

- Parity-LRX 的 packed tensor product、mesh electrostatics、capacity chunk：仓库体量大，依赖 Ascend C 构建和完整 MACE 依赖，未在本机构建/运行；其 3.90×/4.76×/1.90× 仍是作者证据。
- Hansen 的 Fire/Spline/Tree：Ascend C 自定义算子，未在本机编译；需要按 CANN 9.1.0 重新构建并处理可能的 API/UB 差异。
- Mamba-2：自定义 OPP + 复杂 MIX kernel，未在本机构建；作者结果仍是 910B3/CANN 8.2.RC1。
- vLLM-HUST Linear+SwiGLU：需要 vLLM/vLLM-Ascend 依赖链，未在本机搭建。
- 910C：没有任何本机数据。

## 4. 复现边界

- 本机是 910B3，不是作者的 910B1/910B2/910B4；不能把本机数字当作作者环境的复现。
- 纯 Triton 任务和 Be4ver 的两个 Ascend C 算子已运行；Hansen Ascend C、Mamba-2 自定义 OPP、Parity 原生库、vLLM 端到端未验证。
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
  - [`environment_and_build.txt`](../../our_work/evidence/local-910b-20261116/environment_and_build.txt)
  - [`paco_32x32_b512_run.log`](../../our_work/evidence/local-910b-20261116/paco_32x32_b512_run.log) / [`paco_32x32_b512_verify.log`](../../our_work/evidence/local-910b-20261116/paco_32x32_b512_verify.log)
  - [`qdwh_128x128_run.log`](../../our_work/evidence/local-910b-20261116/qdwh_128x128_run.log) / [`qdwh_128x128_verify.log`](../../our_work/evidence/local-910b-20261116/qdwh_128x128_verify.log)
  - [`qdwh_512x512_run.log`](../../our_work/evidence/local-910b-20261116/qdwh_512x512_run.log) / [`qdwh_512x512_verify.log`](../../our_work/evidence/local-910b-20261116/qdwh_512x512_verify.log)
  - [`qdwh_1024x1024_run.log`](../../our_work/evidence/local-910b-20261116/qdwh_1024x1024_run.log) / [`qdwh_1024x1024_verify.log`](../../our_work/evidence/local-910b-20261116/qdwh_1024x1024_verify.log)
  - [`qdwh_1000x800_run1.log`](../../our_work/evidence/local-910b-20261116/qdwh_1000x800_run1.log) / [`run2.log`](../../our_work/evidence/local-910b-20261116/qdwh_1000x800_run2.log) / [`run3.log`](../../our_work/evidence/local-910b-20261116/qdwh_1000x800_run3.log) / [`verify.log`](../../our_work/evidence/local-910b-20261116/qdwh_1000x800_verify.log)
