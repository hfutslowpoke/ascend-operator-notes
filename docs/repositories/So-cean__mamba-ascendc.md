# So-cean/mamba-ascendc：Mamba-2 SSD ChunkMix 深读

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/So-cean/mamba-ascendc` |
| 作者 | socean |
| 场景 | 生产型 Mamba-2 / SSD 算子，非比赛提交 |
| 访问日期 | 2026-11-16 |
| 默认分支 | `main` |
| 固定 commit | `445a5c001dc3cc09fcec649824f2330634d753f9` |
| 主要 DSL | Ascend C（Cube/MIX + Vector），另有 Triton-Ascend 对照 |
| 算子族 | SSM/选择性扫描、chunked scan、门控、状态传递 |
| 许可证 | Apache-2.0（`LICENSE`），另有 `LICENSES/MulanPSL-2.0.txt` |
| 本地缓存 | `repository-research-cache/So-cean_mamba-ascendc` |
| 与既有归档关系 | 新来源；算子族与 GroupedMatmulGelu 不同 |
| 本报告范围 | 深读 `Mamba2SsdChunkMix`（前向 chunk 内对角项 + chunk 状态） |

### 1.1 环境

| 平台 | 组件 | 版本 |
|---|---|---|
| Ascend 910B3 | CANN | 8.2.RC1 |
| Ascend 910B3 | Python | 3.11 |
| Ascend 910B3 | PyTorch / torch_npu | 2.6.0 / 2.6.0 |
| Ascend 910B3 | Triton-Ascend | 3.2.0 |
| Ascend 950PR | CANN | 9.0.0 |
| Ascend 950PR | PyTorch / torch_npu | 2.7.1 / 2.7.1.post4 |
| Ascend 950PR | Triton-Ascend | 3.2.1 |

本报告的性能数字全部是 910B3；950PR 只作为移植风险提示，不混用。

## 2. 语义与数据流

### 2.1 输入输出

`Mamba2SsdChunkMix` 的接口（`mamba_ascendc_ops/mamba2_ssd_chunk_mix/design.md:5-18`）：

```text
x_cube:       FP16 [B,H,K,T,P]
d_a_cumsum:  FP32 [B,H,K,T]
b_cube:       FP16 [B,K,G,N,T]
c_cube:       FP16 [B,K,G,T,N]
-> y_diag:       FP32 [B,H,K,T,P]
   chunk_states: FP32 [B,H,K,N,P] 或 [B,H,K,P,N]
```

支持 `(T,N,P)=(64,64,64)`、`(64,128,64)`、`(128,128,64)`，且 `H % G == 0`。

### 2.2 每个 chunk 的计算

```text
CB = C @ B^T
W  = tril(CB * exp(dA_i - dA_j))
y_diag = W @ xdt
chunk_states = B^T @ (xdt * exp(dA_last - dA))
```

其中 `xdt` 是 `x` 乘 `dt` 后的加权输入，`dA` 是 `dt*A` 的 chunk 内累加。`W` 是下三角因果矩阵；`y_diag` 是 chunk 内对角输出；`chunk_states` 是传给下一个 chunk 的状态。

这个分解把原始的长度 `L` 串行扫描变成：

- chunk 内：三个矩阵乘（`CB`、`W@xdt`、`B^T@weightedX`）；
- chunk 间：只传递 `chunk_states`，串行依赖长度从 `L` 降到 `L/chunk`。

代价是 chunk 内多出 `O(L*chunk)` 的三角工作；chunk 越大，额外工作越多，但并行度/矩阵乘效率越高。

## 3. 固定源码位置

| 内容 | 路径/行 |
|---|---|
| 设计文档：接口、公式、并行、TilingKey、UB | `mamba_ascendc_ops/mamba2_ssd_chunk_mix/design.md:1-69` |
| kernel 入口、tiling key 分派 | `.../op_kernel/mamba2_ssd_chunk_mix.cpp:1293-1339` |
| AIV 缓冲初始化、因果 mask 一次构建 | 同上 `:37-78` |
| `B[T,N] -> B^T[N,T]` 分块转置 | 同上 `:80-96` |
| `weightedX` / `xdt` 准备 | 同上 `:117-167` |
| `CB` 的 decay + causal mask + FP16 cast | 同上 `:169-210` |
| 状态搬运/布局 | 同上 `:212-231` |
| AIC/AIV 双槽流水与跨核 flag | 同上 `:1016-1211` |
| 动态 Matmul 复用 L1/L0/L0C | `.../op_kernel/mamba2_dynamic_matmul.h:6-26` |
| `Mmad` + `Fixpipe` + `FIX_M` 等待 | 同上 `:195-254` |
| 性能报告 | `docs/performance.md:47-191` |
| 结构与限制 | `STRUCTURE.md:177-206` |

固定 commit 的文件 SHA256：

```text
mamba2_ssd_chunk_mix.cpp  406dcb786054cbcbaf1659bcd2a21939a74a435730bf85e5e4d299719ba34d6b
mamba2_dynamic_matmul.h  c4528ab4af94a8f78715de5c5c618a5aa953eaf36edc8babc7919e7729894b80
design.md                 41cf64f09c21c6aae4581f61a357232902345d03a56af3101e121052397c9570
performance.md            a55543e1e8783aba2aa0e1928c63d05a83917dc291ceb46c3126fbf097d30b8b
```

## 4. 机制一：ChunkMix 把三个 GEMM 和 Vector 变换放进一个 MIX kernel

### 4.1 Cube 侧

每个 chunk 需要：

1. `CB = C @ B^T`：`T×N` 乘 `N×T`，得到 `T×T`；
2. `y_diag = W @ xdt`：`T×T` 乘 `T×P`；
3. `chunk_states = B^T @ weightedX`：`N×T` 乘 `T×P`。

`Mamba2DynamicMatmul` 的设计是“一套 L1/L0/L0C 分配，服务三种不同 K/stride 的 GEMM”（`mamba2_dynamic_matmul.h:6-26`）：

- `BaseK` 由调用方选择：ChunkMix 用完整的 64 列 tile 减少控制；StateEpilogue 用 16 列 tile 做流水；
- `kBaseM=64`、`kBaseN=64`，`MaxN` 可到 128/256；
- `ComputeBlock()` 做 `Mmad` + `Fixpipe`；`Fixpipe` 是异步的，所以 `CopyOut()` 在释放 L0C 前显式 `SetFlag/WaitFlag<FIX_M>`（`:225-232`），避免下一次 `Mmad` 覆盖还在写的 accumulator；
- `ComputeBlockToHalf()` 只把 Cube→Vector 的内部交接降为 FP16，公共输出仍走 FP32。

这减少了“每种 GEMM 各建一套 L1/L0 缓冲”的静态开销，也让同一个 kernel 可以在不同阶段复用同一份 Matmul 资源。

### 4.2 Vector 侧

Vector 负责：

- `B[T,N] -> B^T[N,T]` 的 16×16 分块转置（`mamba2_ssd_chunk_mix.cpp:80-96`）；
- `dA` 的 decay：`exp(dA_last - dA)`、`exp(dA_i - dA_j)`；
- 因果 mask：一次构建 `RowsPerAiv × 64` 的下三角 mask，后续每个 head/task 复用（`:67-78`）；
- `CB` 乘 decay、乘 mask、FP16 cast（`:169-210`）；
- `weightedX` 的按行缩放与可选转置（`:117-167`）。

两个 AIV 各负责一半 token rows；`B` 的转置按源行切分，因此两个 AIV 写的是不相交的目标列，不需要跨 AIV 同步。

### 4.3 消费者原生状态布局

`stateNpLayout` 分支（`mamba2_ssd_chunk_mix.cpp:1180-1189`）：

- `[N,P]`：`B^T[N,T] @ weightedX[T,P]` 直接写 Cube-native 布局，消费者不需要再转置；
- `[P,N]`：legacy `state_passing` 路径，需要 `weightedX^T[P,T] @ B[T,N]`，多一次布局转换；
- Key 3 还保留 legacy B slot（`design.md:44-48`）。

这是“消费者布局”而不是“生产者方便布局”的典型：StateEpilogue 后续按 `[N,P]` 消费，就在 ChunkMix 里直接写成 `[N,P]`，省掉跨 kernel 转置。

## 5. 机制二：双槽流水与跨核 flag

ChunkMix 的 AIC/AIV 不是简单的“AIV 准备完，AIC 再算”，而是双槽交错。源码 `mamba2_ssd_chunk_mix.cpp:1016-1211`：

- `weightedXtHalfOffset_[4]`、`wHalfOffset_[2]` 是 workspace 里的 slot 偏移；
- AIV 先 `PrepareHead()` 算 `weightedX`，再等 `kCubeCbReady`，然后 `BuildWeights()` 生成 `W`，`SetHeadReady(slot)`；
- AIC 先算共享的 `C@B`，`SetFlag(kCubeCbReady)`；然后对每个 head：`WaitHeadReady(slot)`、`ComputeBlock(w,x,y)`、可选 state GEMM、`SetHeadDone(slot)`；
- AIV 在 producer 循环结束后，消费 `kHeadDone0/1`，保证 slot 不被提前覆盖；
- grouped 模式还有 `kGroupedStateReady`，把多个 head 的 state 合并成一次宽 GEMM。

关键设计：

1. **weightedX 只依赖 x/dA**，所以可以在 Cube 算 `C@B` 时提前准备；
2. **W 依赖 CB**，所以必须等 `kCubeCbReady`；
3. **双槽**让 head `h` 的 W 生产和 head `h-1` 的 diag GEMM 重叠；
4. **按 head 分发**获得并行度；当 `(B,G,K)` 足以占满 AIC 时切 group-task，同核复用 B transpose 和 CB。

这不是“把所有东西塞进一个 kernel 就自动重叠”，而是把 producer/consumer 的依赖拆成明确的事件：CB ready、head ready、head done、grouped state ready。事件编号和 slot 生命周期必须与 buffer 容量一起设计。

## 6. 证据

### 6.1 同平台三层对照

固定 shape `[4,2048,16,64,128,128,4]`，启用 `D/z/dt_bias/softplus`，同一张 910B3（`docs/performance.md:140-150`）：

| NPU 实现 | p50 latency | Kernels / forward | 相对 PyTorch | 相对 Triton |
|---|---:|---:|---:|---:|
| PyTorch composition | 6.287 ms | 335 | 1.00× | 0.68× |
| Triton-Ascend | 4.305 ms | 20 | 1.46× | 1.00× |
| AscendC | 0.632 ms | 3 | 9.94× | 6.81× |

作者明确说明：这不是“AscendC 语言固定比 Triton 快 6.81×”，差异来自算子融合、kernel 数、中间 tensor 物化、Cube/Vector 映射和数据布局的共同变化（`docs/performance.md:149-150`）。本文同意这个口径。

### 6.2 大 shape 与 A100 对照

H256 training forward：`[B,L,H,P,N,chunk,G]=[8,4096,256,64,64,64,64]`，返回 final state（`docs/performance.md:51-64`）：

| 平台 | 实现 | Forward | 相对 A100 吞吐 |
|---|---|---:|---:|
| A100 80GB PCIe | `mamba_ssm` | 22.583 ms | 100% |
| Ascend 910B3 | AscendC | 31.888 ms | 70.82% |

这是同 shape、同 feature 的跨设备对照，但不是同架构；70.82% 只适用于该 workload，不是普遍结论。

### 6.3 阶段分解

| Stage | Mean latency | Cube active | AIC MAC | AIV MTE2 | 结论 |
|---|---:|---:|---:|---:|---|
| Preprocess | 6.667 ms | - | - | 26.5% | Vector/control 与输入布局 |
| ChunkMix | 14.850 ms | 99.1% | 16.1% | 27.8% | AIC 持续工作，但有效 MAC 仍低 |
| StateEpilogue | 12.303 ms | 98.3% | 2.9% | 76.5% | 数据搬运与低密度 projection 主导 |

`docs/performance.md:121-134` 明确提醒：`Cube active` 不等于有效 MAC；不同 engine 的 activity 可以重叠，不能相加为 100%。StateEpilogue 是最明显的架构瓶颈，优化方向是消费者布局、GM→UB 搬运和小矩阵映射，而不是继续减少已经只有微秒级的 kernel gap。

### 6.4 流水与中间物化

- profiler service time 33.9664 ms/step，device-event p50 31.8882 ms；profiler 自身 overhead 约 6.5%；
- kernel 间空洞 0.00313 ms/step，underfeed 0.00923%，最大间隙 0.0015 ms（`docs/performance.md:73-84`）；
- 因此这个 shape 不是 launch/host underfeed 主导，剩余差距在 device 内部；
- H256 forward 的 shape-derived 中间结果写回+重读下界约 12.06 GiB，其中 `y_diag` 与 `chunk_state` 合计 8 GiB（66.3%）（`docs/performance.md:185-191`）。这是逻辑量，不是 HBM counter，但足以解释为什么 head 增大后 latency 线性增长。

### 6.5 证据级别

| 项 | 级别 |
|---|---|
| 源码/数据流/布局 | `S+L` |
| 作者 benchmark/profiler | `O+A` |
| 本机编译/运行 | `N` |
| 跨设备 A100 对照 | `O+A`，同 shape 不同架构 |
| 950PR 结果 | `O+A`，但与 910B 分开；当前 HEAD 有回归 |

## 7. 适用条件

1. **序列/head 足够大**：chunk 内多出的三角工作和固定开销需要被摊薄；短序列、小 `H` 可能不如直接扫描或更小的 chunk。
2. **`P=N=chunk` 在支持集合内**：`(64,64,64)`、`(64,128,64)`、`(128,128,64)`；`H%G==0`。
3. **Cube/Vector 能并行**：MIX_AIC_1_2，两个 AIV 各一半 rows；需要足够的 head/task 并行度。
4. **UB 能容纳 chunk 的输入/权重/中间**：T64 每 AIV 约 41.9 KiB；T128 约 182.8 KiB，接近 910B 192 KiB 上限，因此 T128 不做第二套输入/计算 buffer。
5. **状态消费者接受 `[N,P]`**：如果下游必须 `[P,N]`，会退回 legacy 路径，多一次布局转换，收益下降。
6. **目标 SoC/CANN 与源码匹配**：910B3 用 key 2/3/7；950PR 用 arch35 key 9/10/11，路径不同，不能把 910B3 的 tiling/UB 预算直接搬到 950。

## 8. 负例、退化与未知风险

1. **不是 A100 等价**：H256 只有 70.82% 的 A100 吞吐；StateEpilogue 仍然 memory-bound（98.3% Cube active 但只有 2.9% AIC MAC）。
2. **中间物化大**：12.06 GiB 的跨 kernel 写回+重读；chunk 越大，`y_diag/chunk_state` 的物化越明显。
3. **T128 的 UB 余量很小**：182.8 KiB / 192 KiB，不能再加一套 buffer；任何额外 staging 都可能溢出。
4. **950PR 当前 HEAD 有回归**：`STRUCTURE.md:177-188` 记录 source-direct grouped fast path 在 2026-08-10 复测出现 output 回归；final state 正确、grouped token output 错误；generic fallback 正确但不能作为 grouped 性能结论。这是“同一算法不同 SoC 路径不能互证”的明确负例。
5. **backward 覆盖窄**：M1 native-core 只支持 contiguous FP32、`P=N=chunk=64`；其它情况走 M0 correctness fallback（`README.md:154-155`, `:189-198`）。
6. **profiler 有 overhead**：33.9664 vs 31.8882 ms（约 6.5%），阶段时间不能直接当正式时延。
7. **chunk 小到一定程度**：chunk 内三角工作减少，但 chunk 间状态传递次数增加，可能出现“计算少了、依赖链长了”的退化；需要扫 `chunk=32/64/128` 才能判断。

## 9. 复用建议

### 9.1 优先复用

- **chunked scan 的分解**：把长序列扫描拆成“chunk 内三角矩阵乘 + chunk 间状态传递”，是 SSM/线性注意力/长卷积都可借鉴的结构。
- **消费者原生布局**：让 producer 直接写 consumer 需要的 `[N,P]`，避免跨 kernel 转置；前提是下游布局稳定。
- **一套 Matmul 资源服务多种 GEMM**：用可变 K/stride 的 `Mamba2DynamicMatmul` 减少 L1/L0 静态分配；注意 L0C 的 `FIX_M` 生命周期。
- **双槽 + 明确事件**：CB ready / head ready / head done 的拆分，比一个 `PIPE_ALL` 更能形成重叠；但事件和 slot 必须成对。

### 9.2 需要验证后再用

- T128 的 182.8 KiB UB 预算：换 CANN/编译器/SoC 后必须重新算，不能假设还有 9 KiB 余量；
- group-task 的 `(B,G,K)` 切换阈值：需要按 AIC 数、head 数、chunk 数重新测；
- state `[N,P]` 布局：只有下游确实按这个布局消费时才省转置；
- 950PR 的 arch35 路径：必须单独验证，不能拿 910B3 的结果替代。

### 9.3 暂不使用

- 把 0.632 ms / 9.94× 当作“AscendC 比 Triton 快 6.81×”的通用结论；
- 在短序列、小 head、`P/N/chunk` 不匹配时强行套 chunked scan；
- 在 T128 上再加第二套输入/计算 buffer；
- 把 950PR 的历史 checkpoint 结果当作当前 HEAD 的结果。
