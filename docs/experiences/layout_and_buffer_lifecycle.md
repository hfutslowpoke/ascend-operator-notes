# 布局与缓冲生命周期：从“能算”到“少搬、少等、不覆盖”

## 1. 问题

在 910B/910C 上，很多算子不是算不动，而是：

- 输入是 AoS（结构体数组），逐字段跨步访问，向量化效率低；
- 生产者写“自己方便”的布局，消费者又要转置/重排；
- 双缓冲/多槽的 slot 被提前复用，或者等待事件只等了一半；
- UB/CBUF/L0C 容量接近上限，任何额外 staging 都溢出；
- 固定 UB 绝对偏移和 SDK 版本绑定，换编译器就错位。

本轮四份深读都涉及布局/生命周期，只是侧重不同：

| 来源 | 布局/生命周期要点 |
|---|---|
| Fire | AoS 6 字段 → 6 个 SoA 平面 → AoS 5 字段输出；静态 UB 偏移；选择性修复复用缓冲 |
| Tree | AoS 6 字段 → 6 个 SoA 平面；点/段双平面；UB 分 tier 绝对偏移 |
| Spline | 边表/伪坐标/权重预取；目标节点链表；4C 聚合平面 + 4 个权重矩阵 |
| Mamba-2 | `B[T,N] → B^T[N,T]`；状态 `[N,P]` 消费者原生；双槽 + 跨核 flag；L0C `FIX_M` 等待 |
| Linear+SwiGLU | 两个 FP32 accumulator 同驻 CBUF；只写 `[M,N]`；M>128 回退 |

## 2. AoS/SoA：什么时候值得转

### 2.1 Fire：6 字段 AoS → 6 平面

源码：`Hansenwuwuwu/ascend-c-competition-archive`，`FireCandidateDecodeFusion/rank2_submission_1176342/code/op_kernel/device_fire.h:145-160,272-365`。

输入 `raw[B,N,6]` 是 AoS：每个候选的 6 个字段连续。如果直接按字段做向量化，会遇到 stride=6 的跨步访问；如果逐候选标量，又浪费向量单元。

rank2 的做法：

1. 把 `raw` 的 `count*6` 个元素连续搬入 UB；
2. 在 UB 里对 6 个字段一起做 sigmoid；
3. `GatherChannelPlanes<6>` 用 `vgather` 把 AoS 重排成 6 个 `pitch` 平面；
4. 在平面上做坐标/宽高/分数；
5. 用 `scatter_vnchwconv_b32` / `GatherMask` 恢复 `[x,y,w,h,score]` 顺序，写回 AoS。

代价：

- 需要 6 个输入平面 + 5 个 prior 平面 + 输出/临时缓冲；
- `vgather` / `scatter_vnchwconv_b32` 是平台内建，不是通用接口；
- 平面化的收益取决于字段数、`N` 大小、UB 容量；`N` 很小或字段很少时，gather/scatter 可能不划算。

### 2.2 Tree：6 字段线段 + 点平面

源码：`TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_pairs.h:340-398`。

`line_segments[B,M,2,3]` 的 6 个字段（A.x,A.y,A.z,C.x,C.y,C.z）也是 AoS。rank1 先 Gather 成 6 个 SoA 平面，再算：

```text
ux=Cx-Ax, uy=Cy-Ay, uz=Cz-Az
L=ux²+uy²+uz²
vx=ux/L, vy=uy/L, vz=uz/L
```

点 `P` 也按 3 个平面 Gather。这样逐点计算时，`P-A`、点积、投影都是连续向量，不是跨步访问。

### 2.3 什么时候值得 AoS→SoA

判断标准：

- 字段数 ≥3，且每个字段都要做相同/相近的向量操作；
- `N`/`M` 足够大，能摊薄 gather/scatter；
- 目标 SoC 有高效 gather/scatter 或 `Brcb`/`Transpose`；
- UB 能容纳多个平面；
- 输出顺序可以恢复，且恢复成本低于重复跨步访问。

负例：

- 字段数 2、`N` 很小：直接 AoS 向量化可能更便宜；
- gather 的索引表需要每 shape 生成/搬运；如果索引表本身很大，可能抵消收益；
- 平台没有高效 gather 时，SoA 转换会变成标量循环。

## 3. 消费者原生布局：让生产者直接写下游要的格式

### 3.1 Mamba-2 的 `[N,P]` 状态

源码：`So-cean/mamba-ascendc`，`mamba2_ssd_chunk_mix.cpp:1180-1189`，`design.md:42-57`。

ChunkMix 的状态有两种布局：

- `[N,P]`：`B^T[N,T] @ weightedX[T,P]` 直接写 Cube-native 布局，StateEpilogue 按 `[N,P]` 消费；
- `[P,N]`：legacy `state_passing` 路径，需要 `weightedX^T[P,T] @ B[T,N]`，多一次布局转换；
- Key 3 还保留 legacy B slot。

这里的关键不是“哪个布局更自然”，而是**下游消费者要什么**。如果 StateEpilogue 最终要 `[N,P]`，就在 ChunkMix 里直接写 `[N,P]`，省掉跨 kernel 转置。代价是：

- `[N,P]` 路径要求 `B^T` 已经准备好；
- legacy 路径仍需保留，用于不支持 `[N,P]` 的消费者；
- 两套布局增加 tiling key/workspace 分支。

### 3.2 B 转置：共享给多个 head

`mamba2_ssd_chunk_mix.cpp:80-96` 把 `B[T,N]` 按 16×16 tile 转成 `B^T[N,T]`；两个 AIV 按源行切分，写不相交的目标列，不需要跨 AIV 同步。转置后的 `B^T` 在多个 head 之间复用；group-task 还复用 `CB`。

这说明“转置”不一定是坏事：如果转置结果能被多个消费者复用，成本可以摊薄；如果每个 head 都转一次，就不值得。

### 3.3 消费者布局的判断表

| 问题 | 是 | 否 |
|---|---|---|
| 下游是否固定消费某种布局？ | 让生产者直接写该布局 | 保留通用布局 + 转换 |
| 转置结果是否被多个 head/任务复用？ | 值得转置 | 可能不值得 |
| 转换是否能与计算重叠？ | 可隐藏 | 会计入关键路径 |
| 是否有 fallback 布局？ | 双路径 + key | 单一布局风险高 |

## 4. 双槽、事件与生命周期

### 4.1 Mamba-2 的双槽

源码：`mamba2_ssd_chunk_mix.cpp:1016-1211`。

- `weightedXtHalfOffset_[4]`、`wHalfOffset_[2]` 是 workspace slot；
- AIV 准备 `weightedX`，等 `kCubeCbReady`，再 `BuildWeights()`，`SetHeadReady(slot)`；
- AIC 算共享 `C@B`，`SetFlag(kCubeCbReady)`；对每个 head `WaitHeadReady(slot)`、`ComputeBlock(w,x,y)`、`SetHeadDone(slot)`；
- AIV 消费 `kHeadDone0/1`，保证 slot 不被提前覆盖；
- grouped 模式还有 `kGroupedStateReady`。

关键原则：

1. **每个 slot 有明确的 producer/consumer**；
2. **等待事件对应“数据真的可用”**，不是“某个 pipe 空闲”；
3. **复用 slot 前要等所有读者完成**，包括 DMA/Vector/Cube；
4. **双槽不是越多越好**：T128 的 UB 已经 182.8 KiB，没有空间做第二套输入/计算 buffer（`design.md:50-57`）。

### 4.2 L0C 的 `FIX_M` 等待

源码：`mamba2_dynamic_matmul.h:211-254`。

`Fixpipe` 是异步的；`CopyOut()` 在释放 L0C 前显式：

```cpp
SetFlag<HardEvent::FIX_M>(eventId);
WaitFlag<HardEvent::FIX_M>(eventId);
```

否则下一次 `Mmad` 可能覆盖还在写 GM 的 accumulator。这是“buffer 生命周期”最容易被忽略的一类：不是等 Vector 读完，而是要等 Fixpipe 写完。

### 4.3 Fire 的静态 UB 偏移

源码：`device_fire.h:199-216`。

`PadSmallDecode` 用 `StaticUBBuffer<Offset,Bytes>` 把 `priorBuf/planeBuf/rawBuf/halfBuf/priorPackedBuf/outBuf/onesBuf/maskBuf/indexBuf` 固定在编译期偏移。好处：

- 没有动态分配/对齐开销；
- 每个缓冲的地址在编译期已知，便于生成地址表；
- 容量固定，容易算 UB 峰值。

代价：

- 换 `K`/`TILE`/dtype 就要重新算偏移；
- 任何新增缓冲都可能撞到已有区域；
- 与编译器保留区/UB 版本绑定，不能跨 SDK 直接复制。

## 5. UB/CBUF/L0C 容量边界

### 5.1 Mamba-2 T128

`design.md:50-57`：T64 每 AIV 约 41.9 KiB；T128 约 182.8 KiB，接近 910B 192 KiB 上限。因此 T128 不做第二套输入/计算 buffer。任何额外 staging 都可能溢出。

### 5.2 Linear+SwiGLU

`linear_swiglu_kernel.py:35-66`：两个 `[BLOCK_M,BLOCK_N]` FP32 accumulator。M=128、N tile=128 时，两个 accumulator 约 128 KiB，再加 A/权重/流水，接近 CBUF 上限。报告说 256 列 fused tile 超过 CBUF；M>128 回退。

### 5.3 Tree 的 UB 分 tier

`tree_pairs.h:230-264` 把 UB 分成 Tier 0/1/2，并用绝对偏移安排 SoA、点、offset、reduced、工作平面。这样做能把 192 KiB 用到极限，但也意味着：

- 任何新增平面都要重新排布；
- bank/对齐（Even/Odd 256）是手工设计的；
- 不同 `PATH` 的偏移不同，不能混用。

## 6. 负例与风险

1. **转置不一定值得**：如果转置结果只被一个消费者用一次，转置的读写可能超过收益。Mamba-2 的 B 转置值得，是因为多个 head 复用；不要把这个结论外推到单 head。
2. **双槽可能挤爆 UB**：T128 没有第二套 buffer；在 192 KiB 边缘再加 slot，可能编译失败或溢出。
3. **事件等待不完整**：只等 Vector 读完、不等 Fixpipe 写完，会导致输出错值；L0C 复用必须等 `FIX_M`。
4. **静态偏移/SDK 耦合**：Fire/Tree 的绝对偏移和私有内建与 CANN/编译器版本强绑定，不能直接复制。
5. **消费者布局需要 fallback**：Mamba-2 同时保留 `[N,P]` 和 `[P,N]`；如果只写一种，遇到不支持的下游就错。
6. **AoS→SoA 的 gather 成本**：字段少、N 小时可能不划算；索引表本身也要占 UB/GM。

## 7. 可证伪的验证设计

### 7.1 布局消融

- 对同一算子，分别实现 AoS 直接向量化、AoS→SoA→AoS、纯标量三版；
- 记录 GM 读写字节、UB 峰值、gather/scatter 指令数、时延；
- 如果 SoA 版的字节数下降但时延不变，说明瓶颈不在布局；如果时延下降但 UB 峰值逼近上限，说明收益不可扩展。

### 7.2 转置收益

- 测“每个 head 转一次” vs “转一次共享给所有 head”；
- 记录转置字节、复用次数、转置与计算的 overlap；
- 如果复用次数 <2，转置大概率不划算。

### 7.3 双槽/事件

- 用 profiler 看每个 slot 的 producer/consumer 时间线；
- 故意减少一个 slot，观察是否出现 stall/覆盖；
- 如果双槽没有减少 stall，说明瓶颈不在等待，而在计算/带宽。

### 7.4 容量边界

- 扫 `TILE/BLOCK_N/M`，记录编译失败点、UB/CBUF 峰值、时延；
- 找到“能编译但变慢”和“编译失败”的边界；
- 不要把某个 SoC 的容量常数直接搬到另一个 SoC。

## 8. 证据边界

- 本文的布局/生命周期结论来自固定 commit 的源码；没有本机编译/运行。
- 容量数字（41.9/182.8/192 KiB、128 KiB accumulator）来自作者文档或按 shape 估算，不是本机 profiler。
- 事件/双槽的正确性需要设备验证；静态源码只能证明“设计上有这些等待”，不能证明“运行时没有 race”。
