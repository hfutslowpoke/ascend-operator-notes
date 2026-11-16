# 等变张量积与 mesh 算子：分组 GEMM、有界 stencil 与容量调度

本文从 `Parity-LRX/ICTC-O3-HP-Ascend` 提炼三类可迁移机制，面向“多 path 共享输入”的等变算子、mesh/粒子沉积、以及大图推理的显存调度。源码对应固定 commit `ca9ccc4ea9e2e9d21e48daeb45925f2bc4382be8`；性能数字全部是作者环境（910B3/CANN 9.0.0）的 `O+A`，本机未构建。

## 1. 问题特征

这三类算子表面上差异很大，但共享三个结构问题：

1. **多 path/多分量共享输入**：等变张量积有多个 `(l_in,l_edge,l_out)` 耦合路径，同一 `l_in` 的 feature 被多个 path 读；朴素实现会为每个 path 单独启动 GEMM、单独写输出。
2. **stencil/邻居访问的中间物化**：mesh 静电的每个 atom 有 `order^3` 个 stencil 点（CIC 8、TSC 27、PCS 64）；通用实现容易生成 `[N,stencil,S]` 中间张量。
3. **大图推理的边中间显存**：图消息的 edge 中间 `[E,P,C]` 在 atom/edge 很大时是显存主因；全量保存无法运行，但重算又增加 FLOPs。

对应的三种机制是：按输入角阶分组 GEMM、有界 stencil + channel tile、capacity chunk 重算。

## 2. 机制 A：按输入角阶分组，把多个 path 合成一次 GEMM

### 2.1 源码

`ictc_o3/backends/ascend/packed_tensor_product.py:138-160`：

```text
by_input[i] = [(path, ni, no, offset), ...]
groups = [(i, items, sum(no)), ...]
```

`_Product.forward`（`:60-80`）：

```text
for (i, items, width), matrix, left in zip(groups, matrices, features):
    base = _Matrix.apply(matrix.reshape(rows,width,items[0][1]), left)
    for (path, ni, no, offset), block in zip(items, base.split(...)):
        value = block * path_weights[path][:,None,:]
        targets[path].copy_(value.reshape(rows,-1))
```

### 2.2 改变了什么

- 同一 `l_in` 的多个 path 共享一次 feature 读和一次 batched GEMM；
- `channels` 在 GEMM 内连续，避免 `[E, path, channel, angular]` 的转置/repack；
- 输出直接写到 packed buffer 的 path 偏移，消费者不需要再拼；
- gate adjoint（`:83-110`）不保存全部 path 输出，而是重算 base，用额外 GEMM 换显存。

### 2.3 适用条件

- path-preserving 的 ICTC/等变基，path 共享输入角阶；
- `E` 足够大、`C` 足够大，GEMM 能摊薄固定开销；
- 消费者接受 packed 输出，或愿意用 `unpack_output`；
- 训练时能接受 gate adjoint 的 base 重算。

### 2.4 代价与负例

- CANN vector BMM 只在 `E>=16384`、`C>=64`、`min(q,angular)>=4`、`q*angular>=64` 时启用；否则退回 `torch.bmm`，分组收益下降（`:26-32`）。
- 分组计划需要 `index_select + contiguous` 构造 bond projection；计划缓存 miss 时有一次成本。
- 同一 group 内 `no` 不同，`split` 后仍要小段拷贝；不是“零拷贝”。
- 不是任意 e3nn 适配器：ICTC 分量与 e3nn 球谐分量不能直接互换，需要 basis map（`docs/O3_API.md:122,205`）。
- `native` 模式不保证比厂商 GEMM 快；`docs/HP_API.md:59-60` 明确说“rewriting a GEMM does not imply higher throughput than the vendor implementation”。

### 2.5 证据

作者报告（`docs/PERFORMANCE.md:5-31`、`README.md`）：

- C128/lmax3：262,144 实例 forward 3.90× e3nn；131,072 实例 forward+VJP 3.85×；
- 四个配置整体 forward 1.87–3.90×、forward+VJP 2.19–3.85×；
- MindChemistry C64/lmax2、65,536 实例：forward 23.18×、forward+gradient 18.37×；
- 协议：eager FP32、HF32 关闭、无 `torch.compile`、至少 2 s warmup、3 轮 × 3 次同步调用取轮中位数。

本机未构建，不能给 `V`。

## 3. 机制 B：有界 stencil + channel tile，不物化 `[N,stencil,S]`

### 3.1 源码

`ictc_o3/ascendc/op_kernel/electrostatics.cpp:22-119`：

- `Assignment::Init` 按 `12*tile+32` float 分配 UB；
- `Process` 按 `task = atom*ct + tile` 切分，每个 task 只处理一个 atom 的一个 channel tile；
- `Spline` 算 base/weight/first/second derivative（`:42-49`）；
- 对 `order^3` 个 stencil 点循环（`:54-91`）；
- spread 用 `SetAtomicAdd<float>()` 写 mesh；gather 读 mesh；coordinate VJP 用 `product=src*field` 和 spline 导数算 `gx/gy/gz`；
- channel 维只保留 `count=min(tile,c-first)` 个 float。

主机选路 `ictc_o3/backends/ascend/electrostatics.py:22-62`：

- capacity/native 模式始终有界 UB；
- `N<32768` 或 `channels>4`：native；
- `channels==1 && N<=65536`：particle tile（有 Triton-Ascend 时）；
- `N>65536` 单通道：回退 CANN assignment；
- 小 first adjoint `<8192` 不用 particle。

### 3.2 改变了什么

- 避免 `[N,stencil,S]` 中间张量，stencil 权重/导数只存在于寄存器/UB；
- channel tile 让 UB 用量与 `channels` 解耦；
- spread/gather/coordinate VJP 共用同一个 stencil 循环，减少重复 stencil 计算；
- 按规模/通道/adjoint 目标选 schedule，而不是单 kernel 通吃。

### 3.3 适用条件

- reciprocal-only 的 Gaussian-screened 静电；mesh 固定、位置/cell 动态；
- PCS/CIC/TSC，channel 数能被 UB tile 容纳；
- 单通道、中等 atom 数、需要坐标/force 梯度时，particle tile 最占优；
- 多 NPU 时需要 HCCL + slab FFT。

### 3.4 代价与负例

- 每 atom 的 spline 计算是标量；PCS 64 个 stencil 点，串行循环；
- atomic 写 mesh 的浮点求和顺序不确定；作者明确说 floating atomic sums are nondeterministic；
- 不是完整 Ewald：没有 real-space、self、neutralizing background；`full_ewald` 只是保留 Gaussian screening 的名字；
- `N>65536` 单通道 auto 回退 CANN；`channels>4` 走 native；capacity/native 始终有界；换 schedule 会改变性能和显存；
- mesh/sigma/boundary 改变势能；不能拿一个配置的精度结论套另一个；
- 原生库必须匹配 CANN/torch_npu/ABI/SoC；换环境要重建。

### 3.5 证据

作者报告（`docs/ELECTROSTATICS.md:245-327`）：

- 16,384 atoms scalar q：训练额外 5.2%、E/F 推理额外 6.2%；
- particle scheduling：额外推理 37.78→18.43 ms（−51.2%），训练 78.38→46.58 ms（−40.6%）；
- q+μ+Q：训练额外 6.8%、推理额外 10.7%；
- 65,536 atoms standalone：60.64 vs 96.11 ms 推理；144.75 vs 169.63 ms 训练；峰值 395→41 MB；
- 8 NPU、131,072 atoms：q+μ+Q MD step 1,926.926→1,011.944 ms（1.90×），weak efficiency 36.8→70.5%。

## 4. 机制 C：capacity chunk，用重算换显存

### 4.1 源码

`ictc_o3/backends/ascend/capacity_message.py:7-63`：

- forward 保存输入，调用 `fast_graph(capacity_stream=...)`，只保留紧凑 node 输出；
- backward 若 `torch.is_grad_enabled()`，走原 native graph 保留高阶；
- 否则用 `first_order_backward`；再不行按 chunk 重算；
- 每 chunk：切 `src/dst/harmonics/gates/mask`，重跑 `fast_graph`，`autograd.grad`，feature 梯度 `add_` 到 node 缓冲，edge 梯度 `copy_` 到切片；
- 每 chunk 同步、释放中间。

自动 chunk（`:65-80`）：

```text
budget = min(total/8, max(0, total - allocated - gradient - 6*node_bytes)/2)
per_edge = _edge_working_bytes(...)
chunk = min(edges, 1048576, budget/per_edge)
chunk = chunk//128*128
```

分块也不够就返回 `None`，回退原图。

### 4.2 改变了什么

- 显存峰值从“全边中间”降到“node 输出 + 单 chunk”；
- 用重算 FLOPs 和 chunk 循环换显存；
- 训练/高阶/可训练权重/可训练 mask 走原图，不改变训练语义；
- 大 frozen radial 最后 Linear 可以流式放进 chunk（`:109-179`）。

### 4.3 适用条件与负例

适用：

- 推理/冻结参数、普通一阶 adjoint；
- 大 atom 数、边中间是显存主因；
- 动态 graph/atom 数，不能预分配固定容量；
- 能接受重算额外 FLOPs 和 chunk 循环开销。

负例：

- 不是训练加速；训练/高阶走原图；
- 小图不划算；
- chunk 依赖实际内存/分配器状态，性能有方差；
- weight/projection 在 fwd/bwd 之间被替换会报错；
- radial streaming 只对 `torch.nn.Sequential` 最后 Linear、无 hooks、冻结参数等条件生效；
- chunk 边界改变归约顺序，数值一致性要单独验证。

### 4.4 证据

作者报告：

- C64/lmax2 capacity inference 到 229,376 atoms，首次 OOM 262,144；40.7–42.8k atom-steps/s；
- 196,608 atoms：11.30→4.65 s（2.43×），峰值 52.68→43.55 GiB；
- C256/lmax3、16,384 atoms：5.53k atom-steps/s，2.96 s/call；
- 协议：fresh process、至少 2 s warmup、5 次同步调用中位数；显存是 PyTorch allocator 峰值。

## 5. 横向比较

| 维度 | packed TP | mesh 静电 | capacity chunk |
|---|---|---|---|
| 主要瓶颈 | 多 path GEMM 启动/输出 repack | stencil 中间物化/UB | edge 中间显存 |
| 核心动作 | 按 `l_in` 分组 + 直接写 packed | 每 atom×channel tile + 有界 stencil | 紧凑 node 输出 + chunk 重算 |
| 省掉什么 | 每 path GEMM/输出副本 | `[N,stencil,S]` | 全边中间 |
| 增加什么 | 分组计划、gate base 重算 | 标量 spline/atomic | 重算 FLOPs/chunk 循环 |
| 选路条件 | E/C 大小 | N/channel/adjoint 目标 | 内存预算/训练与否 |
| 数值风险 | FP32 严格、path 顺序 | atomic 非确定 | chunk 归约顺序 |

三个机制的共同点是：**先识别“哪个中间物/启动开销是主因”，再决定分组、tile 还是重算**；不是“手写 kernel 就一定快”。

## 6. 消融设计

1. **packed TP**：固定 C/lmax，扫 `E=1k/16k/64k/262k`、`C=32/64/128/256`；比较 grouped vs per-path；记录 GEMM 数、输出 repack 字节、forward/VJP 时间。
2. **gate adjoint**：比较“保存全部 path 输出”vs“重算 base”；记录 backward 时间、峰值显存、base GEMM 次数。
3. **mesh assignment**：固定 atom 数，扫 `channels=1/2/4/8`、`order=2/3/4`、`N=4k/16k/65k/131k`；比较 native/vectorized/particle/CANN；记录时延、显存、数值。
4. **atomic 非确定性**：同一输入重复 20 次，记录 mesh/energy/force 的 bitwise/容差差异；若任务要求确定性，测 atomic-free 路径。
5. **capacity chunk**：固定模型，扫 `chunk=4k/16k/64k/256k`；记录峰值显存、时延、重算 FLOPs、数值；确认小图/训练不启用。
6. **多 NPU**：固定总 atom 数，扫 1/2/4/8 rank；记录 FFT/HCCL/assignment 占比和 weak efficiency；空 rank、triclinic cell 单独测。

## 7. 证据边界

- 全部性能数字是作者 910B3/CANN 9.0.0 报告；本机未构建 `libictc_ascendc.so`，没有 `V`。
- 仓库有协议、图、实现说明，但没有逐点原始 benchmark JSON/CSV；不能把图上的绝对值当精确复现。
- 3.90×/4.76×/1.90× 不能与 KernelSwift/Be4ver 的加速比横比：输入、baseline、计时协议、模型都不同。
- `full_ewald` 不是完整 Ewald；`atomic` 非确定；`capacity` 不是训练加速；这些边界比数字本身更重要。
- 这些机制在 910C/950/CANN 9.1 上的可移植性未验证；ASC CMake、UB、FFT、HCCL 都可能变化。
