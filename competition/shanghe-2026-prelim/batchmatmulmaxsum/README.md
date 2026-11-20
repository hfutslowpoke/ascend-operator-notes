# BatchMatmulMaxSum（2026 上合初赛）

> 层级：`competition` · 赛事：2026 CAPP 上合赛区初赛 · 平台：`24/661, 64.44` · 本机：`not-run`

## 0. 来源登记

| 项 | 值 |
|---|---|
| 官方题面 | [batchmatmulmaxsum.md](../official/descriptions/batchmatmulmaxsum.md) |
| 官方 problem JSON | [problem-6a9aa054bf41025d6014f3ef.json](../official/problem-6a9aa054bf41025d6014f3ef.json) |
| 平台提交 | `sh_pre_batchmatmulmaxsum` |
| 正式源码 | [`source/`](source/) |
| 证据 | [`evidence/`](evidence/)（detail-summary / cases.csv / ranking-summary / sid / source.sha256） |
| 备注 | 旧候选：[zip-278917-runtime-error](candidates/zip-278917-runtime-error/)（Runtime Error，未采用） |
| 本机状态 | `not-run`（本次未在本机 910B3 重跑） |

## 1. 平台证据

| 项 | 值 |
|---|---|
| contest | `op_challenge_shanghe_prelim` / 2026年CANN挑战赛_上合赛区（初赛） |
| problem_id | `6a9aa054bf41025d6014f3ef` |
| 提交账号 | `slowpoke`（uid `6aa109b6c76b321ca64ba8b4`） |
| SID | `6ad314c2193e507d8e5e08a4` |
| detail ID | 884931 |
| 排名 | 24 / 661 |
| score | 64.44 |
| 状态 | Pass / valid=true |
| 可见 case | 15 / 15 |
| 可见总时间 | 396.75 ms |
| 可见平均 t/T | 1.281 |
| 可见平均 case score | 64.44（与平台 score 完全一致） |

逐 case：

| case | time (ms) | best (ms) | t/T |
|---|---:|---:|---:|
| 1 | 1.33 | 1.18 | 1.127 |
| 2 | 1.92 | 1.47 | 1.306 |
| 3 | 2.42 | 1.72 | 1.407 |
| 4 | 3.18 | 2.11 | 1.507 |
| 5 | 4.99 | 3.42 | 1.459 |
| 6 | 9.71 | 6.26 | 1.551 |
| 7 | 6.60 | 6.26 | 1.054 |
| 8 | 34.18 | 24.97 | 1.369 |
| 9 | 56.50 | 46.46 | 1.216 |
| 10 | 74.72 | 63.08 | 1.185 |
| 11 | 77.37 | 65.53 | 1.181 |
| 12 | 94.15 | 80.07 | 1.176 |
| 13 | 9.92 | 8.15 | 1.217 |
| 14 | 9.78 | 8.22 | 1.190 |
| 15 | 9.98 | 7.84 | 1.273 |

## 2. 题目语义

```text
A[b,m,n] = sum_k x1[b,m,k] * x2[b,k,n]      # FP32 累加
R[b,m]   = max_n A[b,m,n]                  # 初值必须 -inf
y[b]     = sum_m R[b,m]
```

- x1/x2：FP16/BF16，逻辑 shape 固定 `(B,M,K)`、`(B,K,N)`；
- `transposeX1/transposeX2` 只改变 storage shape，不做显式 transpose；
- 四种 layout 组合都要支持；
- B≤64，M/N≤8192，K∈[32,8192] 且 8 对齐；
- 输出 FP32 `(B,)`；
- 精度：FP32 相对/绝对 <1e-4；FP16/BF16 <1e-3。

## 3. 代码来源

- API 详情：``
- `kernel.asc`：224,418 B，API 详情中的**正式提交版本**（detail ID 884931）。
- 用户 zip `batchmatmulmaxsum_submission_278917.zip`：
  - detail ID 278917，状态 **Runtime Error**，2026-09-12；
  - `kernel.asc` 102,218 B，纯 Vector 实现，注释明确写 `No Nd2Nz / Mmad / Fixpipe`；
  - 与正式版本 diff 约 4,828 行。
- 结论：zip 是早期失败版本；本分析以 API 的 Cube+Vector 混合版本为准。

## 4. 实现结构

`run_kernel`（`kernel.asc:2827`）按 shape/layout 做多层分路：

1. **极小**：`M*N<=1 && K<=64` → `short_h/short_b`；`K<=256` → `direct_h/direct_b`；
2. **小矩形**：`M<=4 && N<=8 && K<=64` → `flat_*`；`M<=16 && N<=16 && K<=128` → `trans32_*/aligned_*`；
3. **小 M/N**：`M<=32 && N<=128` → `transsmall_* / native_*`；
4. **瘦长/超宽**：`(M>=256&&N<=64)||(N>=256&&M<=64)` → `tall_* / wide_*`，必要时 L1 常驻；
5. **转置尾块**：`transposeX1 && transposeX2` 且 M/K 非 128 对齐 → `pad_* / zzbounded_*`；
6. **大对齐**：`M,N>=128` 且 128 对齐 → `pipe_* / zzeadense_* / zzrawgroup_*`；
7. **split-K**：M/N 小、K 大、核空闲 → `split_* / parsplit_* / rawsplit_*`；
8. **通用**：`cube_*` + `FusedBmm`。

关键构件：

- `BmmsCubeBmm`（`:8`）：`MatmulImpl` + 手写 `TCubeTiling`，Cube 算 BMM；
- `BmmsCubeBmmSplit`（`:89`）：K 维切分，多核 split-K；
- `BmmsTransBulk`（`:166`）：转置 layout 的批量处理；
- `BmmsCubeCachedN`（`:248`）：N 较大时缓存 B 的 tile；
- `BmmsPairedSmall`（`:335`）：小 shape 的 AIC/AIV 配对；
- `BatchedMaxRows`（`:372`）：AIV 逐行 max，再按 batch 求和；
- `SplitBatchReduce`（`:434`）/ `ReduceBmm`（`:451`）：split-K 的部分结果归约；
- `PipelinedCube`（`:594`）+ `PipelinedVector`（`:757`）：AIC 产 C、AIV 流式 max/sum；
- `run_kernel` 里对 `transposeX1/X2` 做 8 种 route 的模板分发。

## 5. 为什么高效

### 5.1 融合而不是三次 launch

PyTorch 参考是 `bmm → amax → sum` 三次 kernel。这里一次 launch 内完成：

```text
Cube BMM -> C workspace -> AIV Max(N) -> Sum(M) -> y
```

C 仍在 GM，但 AIC/AIV 通过 `CrossCoreSetFlag` 流水，避免三次全局同步和中间 tensor 生命周期开销。

### 5.2 按 case 选不同算法，而不是一套 tiling

- 小 case 用 scalar/direct/flat，避免 Cube setup；
- 中等 M/N 用 Cube + 小 tile；
- 大 M/N 用 128/256 tile + 多核；
- M/N 小、K 大用 split-K，把空闲核用起来；
- `transposeX1/X2` 直接改变寻址，不做显式 transpose；
- 非 128 对齐的转置尾块先 pad 到对齐，再走主路径。

### 5.3 归约细节

- max 的 padding 用 `-3.402823466e38F`，避免 0 被误当最大值；
- 全部 FP32 累加；
- `DataCopyPad` 处理 N/K 尾块；
- `BatchedMaxRows` 对 `N>64` 用两路 `Max` 再 `ReduceSum`，减少长行归约延迟；
- split-K 时用 partial buffer + 第二次 reduce，保证 `max` 不能提前跨 K 合并（必须先完整 BMM 再 max）。

### 5.4 与第一名的差距

| 项 | 本提交 | 第一名 `aleX_5y42` |
|---|---:|---:|
| score | 64.44 | 93.31 |
| 可见总时间 | 396.75 ms | 334.30 ms |
| 15 case 平均 t/T | 1.281 | 1.0 |

差距约 19% 总时间，主要在中大 case（case 8–12）。第一名可能用了更好的 tile/核数、
更激进的 split-K 或更少的 C 往返；本实现已经覆盖了小/中/大/转置/split-K 的主要族，
但每个族的参数还不是最优。

## 6. 可复用与风险

**可复用：**

- 三算子融合的 AIC/AIV 流水；
- 按 shape 族分路，而不是一套通用 tiling；
- split-K 解决“M/N 小、K 大、核空闲”；
- `-inf` padding、FP32 累加、`DataCopyPad` 尾块；
- 四种 layout 只改寻址，不做显式 transpose。

**风险：**

- `run_kernel` 分支极多，隐藏 shape 可能落到不合适的 generic 路径；
- 手写 `TCubeTiling` 对 CANN 版本/SoC 敏感，换平台要重新验证；
- 早期纯 Vector 版本（zip 278917）直接 Runtime Error，说明 bmm 规模上去后
  纯 Vector 不可行；不要把它当“轻量替代”。
- 本次未在本机 910B3 重跑，不能声称本机通过。

## 7. 入库标注

```text
source_match: our-submission (CANNJudge SID 6ad314c2...)
platform_evidence: rank 24/661, score 64.44, 15/15 visible
local_status: not-run
specialization: 大量 shape 分支（M/N/K/layout）
generalization: 未覆盖隐藏 shape 的极端组合
risk: 手写 CubeTiling 跨版本；分支多
```
