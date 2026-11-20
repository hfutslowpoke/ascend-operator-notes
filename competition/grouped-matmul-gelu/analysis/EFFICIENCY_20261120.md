# GroupedMatmulGelu 效率分析（2026-11-20 补充）

> 层级：`competition` · 赛事：2026 上合决赛 · 平台：`11/32, 58.54` · 本机：`platform-only`

本文补在既有条目上，不重复源码/题面。源码、SID、平台证据见
[`../README.md`](../README.md)、[`../our-submission/`](../our-submission/)、
[`../analysis/PLATFORM_EVIDENCE.md`](PLATFORM_EVIDENCE.md)。

## 1. 平台证据

| 项 | 值 |
|---|---|
| contest | `op_challenge_shanghe_final` / 2026年CANN挑战赛_上合赛区（决赛） |
| problem_id | `6ae3fad11dac5d235b148620` |
| 提交账号 | `slowpoke`（uid `6aa109b6c76b321ca64ba8b4`） |
| SID | `6af7d84c6bd0f9be3716f4c6` |
| detail ID | 1329204 |
| 排名 | 11 / 32 |
| score | 58.54 |
| 状态 | Pass / valid=true |
| 可见 case | 5 / 20 |
| 可见总时间 | 138.77 ms |
| 可见平均 t/T | 1.162 |
| 可见平均 case score | 76.16 |
| 最终 score | 58.54（隐藏 15 case 明显更慢） |

逐 case（仅可见）：

| case | time (ms) | best (ms) | t/T |
|---|---:|---:|---:|
| 1 | 41.384 | 36.858 | 1.123 |
| 2 | 23.021 | 20.187 | 1.140 |
| 3 | 1.450 | 1.430 | 1.014 |
| 4 | 27.114 | 19.447 | 1.394 |
| 5 | 45.798 | 40.254 | 1.138 |

> 关键：可见 5 case 的平均分是 76.16，但平台最终只有 58.54。说明隐藏 15 case
> 中有明显慢的 shape，不能只用可见 case 评价这个实现。

## 2. 题目语义

```text
x:          (M, K)        INT8
weight:     (E, K, N)     INT8
scale:      (E, N)        FP32
group_list: (E,)          INT64，累积结束位置
y:          (M, N)        FP16

对每个专家 e：
  rows = [group_list[e-1], group_list[e])
  acc  = x[rows] @ weight[e]          # INT32 累加
  z    = acc * scale[e]               # FP32 反量化
  y[rows] = GELU_tanh(z)              # FP32 GELU，最后转 FP16
```

约束：

- E∈[1,128]，M∈[1,65536]，K∈[1,8192]，N∈[1,4096]；
- `group_list` 非递减，`group_list[E-1]=M`，允许空专家；
- K/N/M_e 不要求对齐；
- 禁止在 GELU 前把反量化结果转 FP16；
- GELU 固定 tanh 近似；负数尾部要用稳定形式，避免 `1+tanh(u)` 相消；
- 输出必须逐元素写，不能依赖输出缓冲区初值；
- 禁止按测试用例/数据生成规律跳算。

## 3. 代码来源

- API 详情：`our-submission/`
- `kernel.asc`：521,922 B / 13,153 行；`main.asc` 179 行；`reference.py` 309 行。
- 用户未提供该题的 zip；以 API 详情为准。
- 平台可见 case 5 个，隐藏 15 个。

## 4. 实现结构

### 4.1 入口与分派

`run_kernel`（`kernel.asc:12489`）先读 `x/weight` 的 shape，得到 `m,k,n,e`，
然后用 `gmm_host::Choose`（`:12360`）在 9 个静态 tile 里选一个：

```text
tiles[] = {32x256x128, 64x128x128, 64x256x128, 64x384x64,
           64x512x64, 96x288x64, 128x256x128, 160x192x128, 256x128x128}
```

代价模型：

```text
compute  = 2 * roundedRows * roundedCols * roundedK * imbalance / (active * 12.5e6)
traffic  = (rowTiles*k*n + m*k*colTiles) / 1e6
setup    = waves * (0.35 + 0.05 * ceil(k/t.k))
epilogue = m * colTiles / (2*vectorRows*active) * 0.04
cost     = compute + traffic + setup + epilogue - tileArea*1e-10
```

然后按 `k/n/e/m` 进入大量专用 kernel：

- `gmm_tiny_scalar` / `gmm_tiny_vector`：极小 shape，纯 Vector/标量；
- `bk118..bk126`：9 个 `gmm_tiles::Compute<TM,TN>` 静态 tile；
- `bk0` / `bk485`：小 aligned / 单专家路径；
- `RgC1...` / `RgC3...` / `RgR2...` / `RgR3...`：常驻 B、预取、分块、跨核；
- `bk490..bk500`：长 K、窄 N、短 K、特殊 `k%128` 尾部；
- `new5_*`：窄短 K、单列长 K、short-K panel；
- `split_k`：K 切分；
- `generic`：通用 `gmm_tiles::Compute`。

### 4.2 AIC/AIV 分工

`gmm_tiles::Compute<TM,TN>`（`:514`）：

- AIC：`MatmulImpl` 做 INT8×INT8→INT32，C 写入 GM workspace 的 4 槽 ring；
- AIV：从 C 槽读 INT32 → Cast FP32 → 乘 scale → GELU → Cast FP16 → 写 y；
- `CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>` / `CrossCoreWaitFlag` 做 AIC/AIV 同步；
- `packB`：K/N 非 32B 对齐时，把 weight 重新 pack 到 `bPitch` 对齐的 workspace；
- `padA`：x 的 K 非 32 对齐时 pack A；
- `KeepWholeRowOwnership`：尽量让一个 block 拥有完整 group 行，减少跨 block 切分；
- `group_list` 用二分查找映射 task→expert，避免每个 task 线性扫 E；
- `fixedSlabs`：等长 group 时用固定 slab，减少二分。

### 4.3 GELU

`gmm_common::ScalarGelu`（`:14`）用稳定形式：

```text
u = sqrt(2/pi) * (z + 0.044715 z^3)
t = exp(-2|u|)
u >= 0 ? z/(1+t) : z*t/(1+t)
```

`GeluComputeSigmoid`（`:60`）在 Vector 上实现同一数学：

```text
u = z*z*z*0.044715 + z
t = exp(-1.5957691216057308 * u)   # -2*sqrt(2/pi)
t = t + 1
out = z / t
```

这避免了 `1 + tanh(u)` 在负尾部的相消，也避免正指数溢出。

## 5. 为什么高效

### 5.1 可见 case

- INT8 GEMM 直接进 Cube，累加器是 INT32，不需要先反量化成 FP16 再做 GEMM；
- 反量化 + GELU 在 AIV 上融合，C 只写一次 GM；
- 权重 `weight[e]` 在多个 row tile 间复用；`RgC3PrefetchALL` 等路径把 B 预取到 L1；
- 对非对齐 K/N 做 pack，避免 Cube 的尾块退化；
- 对空专家/不等长 group 用 `group_list` 二分 + 整行归属，减少负载不均；
- 9 个 tile 的代价模型覆盖了不同 M/N/K/E 组合，避免一套 tile 打天下；
- 小 shape 用 `gmm_tiny_vector`/`bk484`，避免 Cube setup。

### 5.2 为什么最终只有 58.54

可见 5 case 平均 76.16，但隐藏 15 case 把均分拉到 58.54（按同一评分公式推断）。
按“总分=全部 case 算术平均”反推，隐藏 15 case 的平均分约为：

```text
(58.54 * 20 - 76.16 * 5) / 15 = 52.67
```

也就是说隐藏组比可见组低约 23.5 分，不是“稍微慢一点”。合理解释：

- 可见 case 可能正好落在专用 kernel（如 `k=1024, n=3*256`、`e=1` 小 shape）；
- 隐藏 case 可能落到 generic `Compute` 或某个不合适的 tile，导致明显慢；
- 代码里有大量 `if (k == 1024 && n == 3 * 256 ...)`、`if (e == 1 && ...)` 的精确分支，
  这些分支只对公开 shape 有效；隐藏 shape 不在分支里时，走通用路径；
- `run_kernel` 的 `Choose` 代价模型参数（12.5e6、0.35、0.05、0.04）是经验值，
  对隐藏 shape 不一定最优；
- split-K / pack / workspace 分配对小 shape 可能反而是额外开销。

这不是“代码不好”，而是**公开 case 专用化的典型代价**。在知识库里应写成
“公开 5 case 1.16x，隐藏 15 case 未覆盖”，不能写成“全 shape 1.16x”。

## 6. 可复用与风险

**可复用：**

- INT8 GEMM + FP32 反量化 + GELU 的单 kernel 融合；
- AIC/AIV 4 槽 ring + cross-core flag；
- `group_list` 二分、整行归属、空专家跳过；
- 非对齐 K/N 的 pack A/B；
- 9 tile 代价模型 + 静态 tile 实例化；
- 稳定 GELU 形式（负数尾部、避免溢出）。

**风险：**

- 大量精确 shape 分支，泛化差，可能触碰“按测试用例特化”的规则边界；
- 每个 call `aclrtMalloc` workspace，small shape 的分配/同步成本可能偏高；
- 手写 `MatmulApiStaticTiling` 和 `TCubeTiling` 对 CANN/SoC 敏感；
- 隐藏 15 case 的实际瓶颈未知，不能从可见 5 case 外推；
- 本次未在本机 910B3 重跑。

## 7. 入库标注

```text
source_match: our-submission (CANNJudge SID 6af7d84c...)
platform_evidence: rank 11/32, score 58.54, 5/20 visible
local_status: not-run
specialization: 9 tile + 大量 k/n/e 精确分支
generalization: 隐藏 15 case 明显慢；generic 路径是短板
risk: 规则边界；CANN 版本；workspace 分配
```
