# Parity-LRX/ICTC-O3-HP-Ascend：O(3) 张量积、长程静电与容量调度

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/Parity-LRX/ICTC-O3-HP-Ascend` |
| 作者 | Nickel 等 |
| 场景 | 生产型 O(3)/MACE 等变算子库与分子动力学后端（非比赛） |
| 访问日期 | 2026-11-16 |
| 默认分支 | `main` |
| 固定 commit | `ca9ccc4ea9e2e9d21e48daeb45925f2bc4382be8` |
| 版本 | `ictc-o3-hp-ascend 0.1.0` |
| 主要 DSL | Ascend C、Triton-Ascend、PyTorch autograd/FX |
| 算子族 | 等变张量积、图消息、对称收缩、方向谐波、长程静电、MD 容量调度 |
| 许可证 | MIT（`LICENSE`；第三方许可见 `LICENSES/`、`LICENSE_SCOPE.md`） |
| 本地缓存 | `repository-research-cache/Parity-LRX_ICTC-O3-HP-Ascend` |
| 与既有归档关系 | 新来源；与 GroupedMatmulGelu 无重复 |
| 本报告范围 | 深读 packed tensor product、mesh electrostatics assignment、capacity message chunking 三份实现 |

### 1.1 环境

| 项 | 值 |
|---|---|
| 硬件 | Ascend 910B3，Linux ARM64 |
| CANN | 9.0.0（`docs/DEVELOPMENT.md:42-43`） |
| Python | 3.10 |
| PyTorch / torch_npu | 2.7.1 / 2.7.1.post4 |
| Triton-Ascend | 3.2.1 |
| 精度 | eager FP32，HF32 关闭，不使用 `torch.compile` |
| 计时 | 单 NPU：至少 2 s warmup，5 个同步的三调用块，取中位数；多 NPU 计时另见 LAMMPS 文档 |

本机没有构建该仓库：它需要 CANN 9.0.0 的 ASC CMake 接口、`torch_npu` 2.7.1 和完整 MACE/LAMMPS 依赖；本机是 CANN 9.1.0 + torch_npu 2.10.0。因此本报告的性能数字全部是作者文档/图表的 `O+A`，不是本机 `V`。

### 1.2 证据形态与局限

仓库有 `docs/PERFORMANCE.md`、`docs/ELECTROSTATICS.md`、`docs/USAGE.md` 和 SVG/PNG 图，但没有把逐点原始 benchmark JSON/CSV 或完整日志放进 Git。可核对的是：

- 计时协议（warmup、同步块、中位数、排除项）；
- 模型/shape/精度/邻居数/mesh 配置；
- 对比对象（e3nn 0.5.9、MACE 0.3.14、MindChemistry 0.2.0）；
- 代码中的路径选择、tiling、chunk 和 fallback。

不可核对的是每个点的原始计时样本。因此本报告把数字写成“作者报告”，不把它们当成本机复现或跨仓库可横比的加速比。

## 2. 实现一：packed O(3) tensor product

### 2.1 语义与布局

`EdgeTensorProduct` 是 path-preserving 的等变张量积：

```text
features:   [E, C*(L_in+1)^2]
harmonics:  [E, (L_edge+1)^2]
gates:      [E, P*C] 或 [E, P, C]
output:     [E, output_width]
```

每个耦合路径 `(l_in, l_edge, l_out)` 满足三角规则和 `l_in+l_edge+l_out` 为偶数；输出保留 path 维度和每个角度的 `2*l+1` 分量。`ir_mul` 把角度放在通道前，`mul_ir` 把通道放在角度前。

### 2.2 固定源码位置

| 内容 | 路径/行 |
|---|---|
| 分组计划：按输入角阶聚合路径 | `ictc_o3/backends/ascend/packed_tensor_product.py:138-160` |
| 每组一次 batched GEMM | 同上 `:72-79` |
| 直接写入 packed 输出，避免逐 path repack | 同上 `:67-78` |
| gate adjoint 重算 base，而不是保存全部 path 输出 | 同上 `:83-110` |
| CANN vector BMM 选择条件 | 同上 `:26-32` |
| 公共布局/输出拆分 | 同上 `:49-57,113-165` |
| FP32 严格检查 | `ictc_o3/backends/ascend/runtime.py`（`require_npu_fp32`） |
| 性能与协议 | `docs/PERFORMANCE.md:5-31`、`docs/USAGE.md:792-861` |
| 模式/回退说明 | `docs/HP_API.md:44-66`、`docs/O3_API.md:80-130` |

关键文件 SHA256：

```text
packed_tensor_product.py  6239902413e319d8d0481629da4da66912c39e61635e491e10bc0e29487a5ba1
PERFORMANCE.md            e6141358a1942fd7db3b1d63f1f23ff48d09f4d121264fc226f065714ca56797
USAGE.md                  698797645fa7dd450f5ac8033b9305c3a4be9b7c7ecec7a0f67cd13bf88eb53b
```

### 2.3 机制：按输入角阶分组，把多个 path 合成一次 GEMM

朴素实现会对每个 `(l_in,l_edge,l_out)` path 单独做一次 GEMM，并分别读写输出。`packed_tensor_product.py:142-149` 改成：

```text
by_input[i] = [(path, ni, no, offset), ...]
groups = [(i, items, sum(no)), ...]
```

即把共享同一输入角阶 `l_in` 的多个输出 path 合并成一组：

- 每个 group 做一次 `base = _Matrix.apply(matrix, left)`，得到 `[rows, width, items[0].ni]`；
- `base.split(...)` 按每个 path 的 `no` 切开；
- 每个 path 乘自己的 gate 权重，直接 `copy_` 到 packed 输出的对应偏移；
- 输出顺序按 `(l_out, path_offset)` 排序，消费者不需要再 repack。

这减少的是：

- 每个 path 的 GEMM 启动/调度次数；
- 输出中间张量和逐 path 的 layout 转换；
- 输入 feature 的重复读取（同一 `l_in` 的多个 path 共享一次读）。

代价是：

- 需要预先构造分组的 bond projection（`index_select` + `contiguous`），只在计划缓存 miss 时做一次；
- 同一 group 内 path 的 `no` 不同，`split` 后仍有小段拷贝；
- 只有 `E` 足够大、`C` 足够大时，GEMM 才能摊薄固定开销；小 E 可能不如逐 path 的简单实现。

### 2.4 机制：gate adjoint 重算 base，换内存

`_Product.backward`（`:83-110`）没有保存 forward 的所有 path 输出，而是：

- 保存 `right`、`weights`、`features`；
- backward 时按 group 重新算 `base = _Matrix.apply(matrix, left)`；
- 只对需要 gate 梯度的 path 计算 `dw[path] = (block * outputs[path]).sum(1)`；
- 对 feature 梯度，重算 `gated` 并做一次转置 BMM。

这是典型的“用算力换显存”：forward 少存 `P` 份 path 输出，backward 多算一次 base。适合大 `E`、大 `C`、显存受限于中间激活的训练；如果 backward 不是瓶颈或 base GEMM 很便宜，重算可能反而增加时间。

### 2.5 证据

作者报告（`docs/PERFORMANCE.md:5-31`、`README.md:1-40`）：

- C128/lmax3：262,144 个 TP 实例时 forward 3.90× e3nn；131,072 实例时 forward+VJP 3.85×；
- 四个配置（C64/lmax2、C128/lmax2、C128/lmax3、C256/lmax3）整体 forward 1.87–3.90×、forward+VJP 2.19–3.85×；
- 对 MindChemistry PyNative：C64/lmax2、65,536 实例时 forward 23.18×、forward+gradient 18.37×；
- 协议：eager FP32、HF32 关闭、无 `torch.compile`；TP 输入和 basis/layout 准备不计时；至少 2 s warmup；3 轮 × 3 次同步调用，取轮中位数；CPU FP64 小案例和全 batch 抽样校验。

这些数字是作者环境（910B3/CANN 9.0.0/PyTorch 2.7.1）；本机未复现。

### 2.6 适用条件与负例

适用：

- path-preserving ICTC 基、FP32、同 NPU；
- `E` 大、`C` 较大，grouped GEMM 能摊薄固定开销；
- 消费者接受 packed 输出，或愿意用 `unpack_output`；
- 训练时能接受 gate adjoint 的 base 重算。

负例/风险：

- **不是任意 e3nn 适配器**：`docs/O3_API.md:122,205` 明确说 ICTC 分量与 e3nn 球谐分量不能直接互换，需要 basis map；factory 的 path 系数固定为 1，不能当通用 e3nn TP。
- **小 E/小 C 退化**：CANN vector BMM 只在 `E>=16384`、`C>=64`、`min(q,angular)>=4` 且 `q*angular>=64` 时启用（`:26-32`）；否则退回 `torch.bmm`，grouped 计划的收益下降。
- **native 不一定更快**：`docs/HP_API.md:59-60` 明确写“rewriting a GEMM does not imply higher throughput than the vendor implementation”；`native` 模式是给检查和调优用的，不是默认最快。
- **输出 path 顺序/多重度**：`unpack_output` 保留 path multiplicity，不能把输出当成每角阶只有 `C` 通道；接错布局会静默算错。
- **重算的代价**：gate adjoint 重算 base 会增加 backward FLOPs；对 backward 主导或小模型可能不划算。

## 3. 实现二：mesh electrostatics assignment

### 3.1 语义

`MeshElectrostatics` 计算 Gaussian-screened reciprocal correction：

- 输入：`positions[N,3]`、`batch[N]`、`cell[B,3,3]`、`q[N,S]`，可选 `dipole[N,S,3]`、`quadrupole[N,S,3,3]`；
- 输出：`[N,1]` 原子能量贡献；
- 支持 CIC/TSC/PCS 质量分配、periodic/slab 边界、矩形 mesh、多 graph、多 source channel；
- 原生 adjoint 支持能量、力、cell 导数、force-loss mixed second derivatives；
- 功能是“reciprocal correction”，**不包含** real-space pairs、self energy、neutralizing background；`full_ewald` 的名字不代表完整 Ewald 和。

### 3.2 固定源码位置

| 内容 | 路径/行 |
|---|---|
| assignment kernel：spline 权重/导数、3D stencil、channel tile、atomic 写 mesh | `ictc_o3/ascendc/op_kernel/electrostatics.cpp:22-119` |
| 主机 schedule 选择：native/vectorized/particle | `ictc_o3/backends/ascend/electrostatics.py:22-85` |
| spread/gather/coordinate VJP 的 autograd 包装 | 同上 `:88-146` |
| mesh shape/config/边界 | 同上 `:147-191` |
| 实现说明与 schedule | `docs/ELECTROSTATICS.md:191-243` |
| 成本/协议 | 同上 `:245-327` |
| 多 NPU scaling | 同上 `:366-460` |
| 数值检查 | 同上 `:329-364` |

关键文件 SHA256：

```text
electrostatics.cpp        b8379188434cef69b28de59753ae47ef9ac8e41333b70d68eb2af69c86ffa423
electrostatics.py         334c7e2db84289e74eeea4dfb2ad539e7fb279ddb988245c295b3bd531455d00
ELECTROSTATICS.md         a7064ac0ef95ad184aecdb8549bb79e30bc130e013924c602d72c9529cfcb2d7
```

### 3.3 机制：有界 stencil + channel tile，不物化 `[N,stencil,S]`

`Assignment::Process`（`electrostatics.cpp:35-102`）按 `task = atom * channel_tiles + tile` 切分：

- 每个 task 只处理一个 atom 的一个 channel tile；
- 先算 `Spline` 的 base/weight/first/second derivative（`:42-49`），3 个维度各 `order` 个点；
- 对 `order^3` 个 stencil 点（CIC 8、TSC 27、PCS 64）循环；
- 每个点只读写 `count` 个 channel（`count = min(tile, c-first)`），用 `Muls/Add` 在 UB 上算；
- spread 用 `SetAtomicAdd<float>()` 把结果累加到 mesh；gather 对每个 stencil 点读 mesh 并累加；
- coordinate VJP 在同一个 stencil 循环里用 `product = src*field` 和 spline 导数算 `gx/gy/gz`，最后 `ReduceSum` 写 `df`。

这避免了通用实现常见的 `[N, stencil, S]` 中间张量：stencil 权重和导数只存在于寄存器/UB 的 `w[3][4]`、`d[3][4]`、`h[3][4]`，channel 维只保留一个 tile。代价是：

- 每个 atom 的 spline 计算是标量；
- stencil 循环在 AIV 上串行，PCS 64 个点；
- atomic 写 mesh 的浮点求和顺序不确定；
- channel tile 太小时 DMA/向量指令效率低。

### 3.4 机制：按规模/通道宽度选择 schedule

`prefer_native_assignment`（`electrostatics.py:22-47`）不是“永远用 native”，而是按规模和通道数选：

- capacity/native 模式：始终有界 UB assignment；
- `N < 32768` 或 `channels > 4`：native；
- `channels == 1` 且 `N <= 65536`：particle tile（有 Triton-Ascend 时）；
- `N > 65536` 的单通道：回退到 CANN assignment，因为大 force-training field 更适合原 schedule；
- `_particle_assignment`（`:50-62`）再按 adjoint 目标细分：`mode==3 && needed[0]` 不用 particle；小 first adjoint `<8192` 不用 particle。

这说明“有界 assignment”不是万能：它在中等规模、单通道、需要坐标梯度时最占优；更大规模/更宽通道/不同 adjoint 目标会换 schedule。

### 3.5 证据

作者报告（`docs/ELECTROSTATICS.md:245-327`、`docs/USAGE.md:743-752`）：

- 16,384 atoms，scalar q：训练额外时间 5.2%、E/F 推理额外 6.2%；
- 同一规模，particle scheduling 把额外推理时间从 37.78 ms 降到 18.43 ms（−51.2%），训练从 78.38 ms 降到 46.58 ms（−40.6%）；
- q+μ+Q：训练额外 6.8%、推理额外 10.7%（16,384 atoms）；
- 65,536 atoms 单通道 standalone：E/F 推理 60.64 ms vs 96.11 ms；训练 144.75 ms vs 169.63 ms；推理峰值显存约 395→41 MB；
- 8 NPU、131,072 atoms：q+μ+Q 完整 MD step 1,926.926→1,011.944 ms（1.90×），weak efficiency 36.8%→70.5%；
- 协议：同一 910B3、eager FP32、HF32 关闭、固定 C64/lmax2 backbone、56 neighbors/atom、PCS、1 Å mesh、Gaussian σ=1 Å；三调用同步块、5 块中位数、至少 2 s warmup；邻居构建和输入传输不计时。

这些是作者环境证据；本机未构建原生库。

### 3.6 适用条件与负例

适用：

- reciprocal-only 的 Gaussian-screened 静电；mesh 固定、位置/cell 动态；
- PCS/CIC/TSC，channel 数能被 UB tile 容纳；
- 单通道、中等 atom 数、需要坐标/force 梯度时，particle tile 最占优；
- 多 NPU 时能接受 HCCL + slab FFT 的通信/布局。

负例/风险：

- **不是完整 Ewald**：没有 real-space、self、neutralizing background；`full_ewald` 只是保留 Gaussian screening 的名字，不能当完整 Ewald。
- **atomic 非确定性**：mesh spread 用 `SetAtomicAdd`，浮点求和顺序随调度变化；作者也明确“floating atomic sums are nondeterministic, as with other mesh scatter implementations”（`docs/ELECTROSTATICS.md` 前文）。
- **规模/通道边界**：`N>65536` 的单通道 auto 会回退 CANN；`channels>4` 走 native；capacity/native 模式始终有界；换 schedule 会改变性能和显存，不改变数学。
- **小 N 的启动开销**：particle tile 在小 first adjoint 上不划算（`<8192` 不用）。
- **mesh/sigma 改变势能**：`smearing_sigma`、mesh spacing、boundary 改变都会改变结果；不能拿一个配置的精度结论套另一个。
- **多 NPU 依赖**：slab FFT、HCCL、空 rank、triclinic cell 都有额外边界；单 NPU 结果不能外推。
- **原生库依赖**：`load_library` 需要匹配 CANN/torch_npu/ABI/SoC 的 `libictc_ascendc.so`；换环境要重建。

## 4. 实现三：capacity message chunking

### 4.1 语义

容量推理模式（capacity mode）目标是在大 atom 数下把显存压到可运行，同时不改变模型/权重/FP32 算术：

- forward 只保留紧凑 node 输出，边中间按 chunk 处理；
- backward 对普通一阶 adjoint 逐 chunk 重算边中间，累加 node 梯度；
- 高阶梯度（`torch.is_grad_enabled()` 且需要 create_graph）走原 native graph，保留递归高阶；
- 大 frozen radial 输出 Linear 可以流式放进 chunk，避免完整 gate 存储。

### 4.2 固定源码位置

| 内容 | 路径/行 |
|---|---|
| `_Message.forward`：保存输入，调用 `fast_graph(capacity_stream=...)` | `ictc_o3/backends/ascend/capacity_message.py:7-20` |
| `_Message.backward`：高阶走原图，一阶走 chunk | 同上 `:21-63` |
| 自动 chunk：按设备内存、已分配、node 输出、per-edge 工作估算 | 同上 `:65-80` |
| dispatch：训练/可训练权重/可训练 mask 走原图 | 同上 `:83-106` |
| radial projection streaming | 同上 `:109-179` |
| 容量模式说明 | `docs/HP_API.md:67-101` |
| 性能/显存协议 | `docs/USAGE.md:764-849`、`docs/PERFORMANCE.md:31-66` |

关键文件 SHA256：

```text
capacity_message.py  7abb5d959bc8056094d1c19137ad4ce8d07de0b47e235ee2813c3570f2681efc
HP_API.md            d3d128c52b2ba31f4f355de9dfc60609e4b100a4ce7197d99dc3017093c395d3
```

### 4.3 机制：用重算换显存

`capacity_message.py:28-63` 的 backward：

- 如果 `torch.is_grad_enabled()`：用原 native graph 建图，保留高阶导数；
- 否则用 `capacity_adjoint.first_order_backward`；
- 如果仍不行，就按 `chunk` 循环：
  - 取 `src[begin:end]`、`dst[begin:end]`、对应 harmonics/gates/mask；
  - 重新跑 `fast_graph` 得到 chunk 输出；
  - 对需要的输入做 `torch.autograd.grad`；
  - feature 梯度 `add_` 到 node 缓冲，edge 梯度 `copy_` 到对应切片；
  - 每 chunk 同步并释放中间。

自动 chunk（`:65-80`）：

```text
budget = min(total/8, max(0, total - allocated - gradient - 6*node_bytes)/2)
per_edge = _edge_working_bytes(...)
chunk = min(edges, 1048576, budget/per_edge)
chunk = chunk//128*128
```

如果 `budget < max(per_edge, 2*node_bytes)`，返回 `None`，回退原图。即“内存不够就分块，分块也不够就不启用”。

### 4.4 证据

作者报告：

- C64/lmax2 capacity inference 支持到 229,376 atoms，首次 OOM 262,144；performance inference 首次 OOM 49,152（`docs/USAGE.md:777-783`）；
- 229,376 atoms 时 40.7–42.8k atom-steps/s（`README.md`、`docs/PERFORMANCE.md`）；
- 196,608 atoms：capacity update 把延迟从 11.30 s 降到 4.65 s（2.43×），峰值显存 52.68→43.55 GiB（`docs/PERFORMANCE.md:47-52`）；
- C256/lmax3、16,384 atoms：5.53k atom-steps/s，2.96 s/次 energy-and-force（`docs/PERFORMANCE.md:24-29`）；
- 协议：fresh process、至少 2 s warmup、5 次同步调用中位数；显存是 PyTorch allocator 峰值，不含 CANN/driver 外部分配。

### 4.5 适用条件与负例

适用：

- 推理/冻结参数、普通一阶 adjoint；
- 大 atom 数、边中间是显存主因；
- 动态 graph/atom 数，不能预分配固定容量；
- 能接受重算的额外 FLOPs 和 chunk 循环开销。

负例/风险：

- **不是训练加速**：训练/可训练权重/可训练 mask 走原图；高阶梯度也走原图，不能把 capacity 的 2.43× 用到训练。
- **小图不划算**：chunk 循环和重算有固定成本；小 atom 数应直接用 performance 模式。
- **chunk 依赖实际内存**：同一 shape 在不同进程/分配状态下 chunk 可能不同，性能有方差。
- **权重替换守卫**：forward/backward 之间替换 owner.weight 或 projection 会报错；这是有意的安全边界。
- **不是所有模型都支持**：radial streaming 只对 `torch.nn.Sequential` 的最后 Linear、无 hooks、冻结参数等条件生效；其它情况回退。
- **数值**：重算路径保持 FP32 和原权重，但 chunk 边界改变归约顺序；作者用全尺寸 energy/force 对比和 CPU FP64 小案例验证，不能假设任意模型都一致。

## 5. 证据与判断

### 5.1 作者证据强度

| 实现 | 作者证据 | 原始日志 | 本机 |
|---|---|---|---|
| packed TP | 3.90×/3.85×、1.87–3.90×/2.19–3.85×；协议完整 | 仓库只有文档/图，无逐点 JSON/CSV | 未构建 |
| mesh electrostatics | 5.2%/6.2%、40.6%/51.2%、1.90×；数值检查列表 | 同上 | 未构建 |
| capacity chunk | 2.43×、52.68→43.55 GiB、229,376 atoms | 同上 | 未构建 |

因此级别是 `S+L+O+A`（源码、固定 commit、作者报告/图、作者声明），没有本机 `V`。不能把 3.90× 与 KernelSwift 的 2.478× 等横比，因为输入、baseline、计时协议和模型完全不同。

### 5.2 与第一批机制的关系

- **布局/缓冲生命周期**：packed TP 的“channels contiguous inside GEMM”和 mesh 的 channel tile 都是消费者布局优先；和 Mamba 的 `[N,P]` 状态布局同一类。
- **去中间张量**：packed TP 直接写 packed 输出、mesh 不物化 `[N,stencil,S]`、capacity 不保存全边中间，都是减少中间物化。
- **按规模选路**：mesh 的 native/vectorized/particle、capacity 的 chunk/fallback、packed TP 的 CANN BMM/torch.bmm，都是“按 shape/规模/内存选 schedule”，不是单一 kernel 通吃。
- **数值风险**：mesh 的 atomic scatter、capacity 的 chunk 归约顺序、packed TP 的 FP32 严格模式，都需要在目标模型上重新验证。

## 6. 复用建议

### 6.1 优先复用

- **按输入角阶分组的 packed TP**：把共享 `l_in` 的多个 path 合成一次 GEMM，直接写消费者布局；适合等变网络、球谐卷积、任何“多 path 共享输入”的算子。
- **有界 stencil + channel tile**：每个 atom 一个 task，stencil 权重/导数只在寄存器/UB，channel 分 tile；适合 mesh spread/gather、粒子沉积、图聚合。
- **按规模/通道/adjoint 目标选 schedule**：把“什么时候用 particle tile、什么时候回退 CANN/native”写成显式策略，而不是一个 kernel 硬扛。
- **capacity 的 chunk 重算**：大图推理用“紧凑 node 输出 + 逐 chunk 重算边中间”换显存；训练/高阶保持原图。
- **UI/边界守卫**：权重替换检查、chunk 大小 128 对齐、空图/空 rank、channel tail、mask 形状检查，都是可复用的工程边界。

### 6.2 需要验证后再用

- **910B3/CANN 9.0.0 → 910C/CANN 9.1.0**：ASC CMake、UB、FFT、HCCL 都可能变；本机没有构建。
- **packed TP 的 grouped GEMM**：小 E/小 C 是否仍优于逐 path；gate adjoint 重算的额外 FLOPs 是否被显存收益抵消。
- **mesh 的 particle tile**：单通道、中等 N 的 crossover 是否在目标模型/邻接度下成立；`N>65536` 的回退是否仍是最优。
- **capacity chunk**：目标模型的 chunk 大小、重算 FLOPs、显存峰值和数值一致性；不能直接套 2.43×。
- **多 NPU**：slab FFT/HCCL 的 scaling 依赖 mesh 形状、rank 数、空 rank；单 NPU 结果不能外推。

### 6.3 暂不使用

- 把 3.90×/4.76×/1.90× 当作本机或 910C 的保证；
- 在没有原始逐点日志的情况下，把作者图的绝对值当精确性能；
- 把 `full_ewald` 当完整 Ewald；
- 把 capacity 2.43× 用到训练或高阶梯度；
- 在未满足 `E>=16384, C>=64` 时强行走 CANN vector BMM；
- 把 ICTC 球谐分量与 e3nn 分量直接互换。
