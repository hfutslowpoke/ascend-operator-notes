# 52Xayah/triton-ascend-kernels：Softmax/Matmul 负例

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/52Xayah/triton-ascend-kernels` |
| 作者 | Xayah（曾诚） |
| 场景 | 教学/实验型 Triton-Ascend 算子实践，非比赛提交 |
| 访问日期 | 2026-11-16 |
| 默认分支 | `main` |
| 固定 commit | `6785cbbaf470dbc126102dcdedb1c9e028d7dd63` |
| 主要 DSL | Triton-Ascend |
| 算子族 | Vector Add、Softmax、Matmul |
| 许可证 | 未声明 |
| 本地缓存 | `repository-research-cache/52Xayah_triton-ascend-kernels` |
| 与既有归档关系 | 新来源；无重复 |
| 本报告范围 | 作为负例看 Softmax/Matmul，不作为高性能正面证据 |

### 1.1 环境

| 项 | 值 |
|---|---|
| 硬件 | Huawei Ascend A2 |
| 平台 | HiDevLab |
| Triton-Ascend | 3.2.2 |
| PyTorch | 2.7.1 |
| CANN | 9.2.0 |

作者明确说明：性能数据是本次实验环境的结果，不代表所有 shape/硬件（`README.md:98`）。

## 2. Softmax：一行一个 program，固定 1024，无 mask

### 2.1 源码

`src/softmax.py:8-41`：

```python
@triton.jit
def softmax_fast_kernel(..., BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row = tl.load(input_ptr + row_idx * row_stride + col_offsets)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(output_ptr + row_idx * row_stride + col_offsets, softmax_output)
```

调用：

```python
grid = (n_rows,)
softmax_fast_kernel[grid](..., BLOCK_SIZE=1024)
```

这是一个“每行一个 program、整行 1024 列一次 load”的实现。它做了三件正确的事：

- 减行最大值，避免 `exp` 溢出；
- 把 max/exp/sum/div 融合在一个 kernel；
- 行内归约，不需要跨 program 同步。

但它也有三个明显限制：

- `BLOCK_SIZE=1024` 写死，注释直接说“假设 N 刚好是 2 的幂”；
- 没有 mask，`N != 1024` 会越界或读错；
- 没有 autotune，没有按行数/列数选 tile，也没有处理 `N` 很大时一行放不进 UB 的情况。

### 2.2 结果

作者报告（`README.md:75-83`）：

| 实现 | 延迟 |
|---|---:|
| Initial Triton | 0.180 ms |
| Optimized Triton | 0.140 ms |
| PyTorch Native | 0.035 ms |

Triton 优化版比初始版快约 22.2%，但仍是 PyTorch 的 4 倍慢（0.140 vs 0.035 ms）。作者没有给出 shape，但从代码看是 `M=4096,N=1024,FP32`。

### 2.3 为什么是负例

1. **厂商库已经高度优化**：`torch.softmax` 在 Ascend 上走 ACLNN/内置 kernel，行归约、向量化、tile 都调过；一个朴素 Triton kernel 很难在标准 shape 上超过它。
2. **一行一个 program 的并行度**：`M=4096` 时有 4096 个 program，看起来够，但每行只有 1024 个元素，单 program 的工作量小，launch/调度占比高。
3. **没有 mask/tile**：只支持 `N=1024` 的规则 shape；一旦 `N` 不是 2 的幂或超过 UB，就不能用。
4. **计时很粗**：`time.time()` 包住 50 次调用后统一同步（`src/softmax.py:58-64`），没有逐 sample 同步、没有中位数、没有环境隔离；这只能看方向，不能当精确数字。

### 2.4 可复用/不可复用

- 可复用：减 max、行内融合、按行并行，是 Softmax 的基本正确结构。
- 不可复用：固定 `BLOCK_SIZE=1024`、无 mask、无 tile、无 autotune 的实现不能直接用于一般 shape。
- 如果要改进：按 `N` 选 `BLOCK_SIZE`，加 mask，`N` 超过 UB 时做两遍/分块归约，或者直接调用厂商 `softmax`。

## 3. Matmul：固定 tile + 1D grid，仍慢 4.3 倍

### 3.1 源码

`src/matmul.py:8-89`：

- `@triton.autotune` 四组配置：`(128,128,32,warps=8)`、`(64,128,32,4)`、`(128,64,32,4)`、`(64,64,32,4)`；
- 1D grid：`pid_m = pid // num_pid_n`、`pid_n = pid % num_pid_n`；
- K 循环 `tl.dot` 累加 FP32，最后 `accumulator.to(tl.float16)`；
- A/B/C 都是 FP16，输出 FP16。

这是标准的 Triton matmul 教学实现，数学上正确，但缺少：

- L2 cache swizzle / grouped ordering；
- split-K（K=16384 很大，单 program 串行 K 循环）；
- 更细的 `BLOCK_M/BLOCK_N/BLOCK_K` 搜索；
- 针对 Ascend Cube 的布局/流水（例如 L1/L0 双缓冲、Fixpipe 配置）。

### 3.2 结果

作者报告（`README.md:85-92`）：

| 实现 | 延迟 |
|---|---:|
| Triton Matmul | 0.699 ms |
| PyTorch Matmul | 0.163 ms |

shape 是 `M=1024,K=16384,N=1024`，FP16。Triton 比 PyTorch 慢约 4.3 倍。

### 3.3 为什么是负例

1. **K 很大**：16384 的 K 循环在单个 program 内串行，缺少 split-K/更好的流水，Cube 利用率低。
2. **1D grid 的 L2 局部性**：`pid_m = pid // num_pid_n` 的行优先映射在小矩阵上可以，但 1024×1024 输出时，相邻 program 的 B 块复用不一定最优。
3. **固定 tile 不是 Ascend 最优**：四组配置只是常见 GPU 组合，没有针对 910B/A2 的 UB/L1/L0 预算和 Cube 形状调优。
4. **厂商库更强**：PyTorch 在 Ascend 上走高度优化的 matmul，教学 kernel 很难在规则大 shape 上赢。

### 3.4 可复用/不可复用

- 可复用：`tl.dot` 的 FP32 累加、K 循环、mask 边界，是正确基线。
- 不可复用：把 0.699 ms 当作“Triton matmul 在 Ascend 上的代表性能”；它只是一个未针对 Ascend 深度调优的教学实现。
- 如果要改进：先接 Cube 友好的 tile/流水，做 split-K，按 910B 的 L1/L0/UB 预算搜索；否则直接用厂商 matmul。

## 4. 证据级别与边界

| 项 | 级别 |
|---|---|
| 源码/结构 | `S+L` |
| 性能数字 | `A`（作者粗略计时） |
| 官方评测器/多轮统计 | 无 |
| 本机复现 | `N` |
| 适用性 | 仅 `N=1024` 规则 Softmax、固定 matmul shape；不能外推 |

负例的价值：

- **校准预期**：手写 Triton 不会自动超过厂商库；标准 shape 上厂商库往往是强基线。
- **提醒计时口径**：`time.time()` + 批量同步的结果不能和逐 sample 同步/中位数直接比。
- **提醒边界**：无 mask、固定 tile 的 kernel 只能用于规则 shape；一旦换 shape 就要重写/回退。

复用建议：

- 把这份代码当作“最小正确实现/负例”，不要当性能模板；
- 如果目标是 910B/910C 的固定 Shape Case，优先判断该 op 是否已经被厂商库高度优化；若是，融合/减少 kernel 数/减少中间张量比“重写单 op”更可能有收益；
- 若必须手写，先做 tile/流水/布局，再做 autotune；不要只改 `BLOCK_SIZE`。
