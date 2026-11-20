# 制造专场：最容易被忽略的 10 个“小点”

制造专场的 15 题里，很多失败不是“不会写 kernel”，而是下面这些小点：

- 比较符是 `<` 还是 `<=`；
- 空段/空行/零命中填什么；
- 并列时取第一个还是最后一个；
- 输出是动态长度还是固定槽位；
- `rot` 是左乘还是右乘、转置还是不转置；
- 索引是 int32 还是 int64；
- 归一化因子乘在哪一步。

这些点通常只改一行代码，但会改变整个输出。本文只记录有源码/官方题面/本机证据的结论；不把 950 或作者成绩当 910B 结果。

## 1. 速查表

| # | 小点 | 正确做法 | 错误直觉 | 证据 |
|---:|---|---|---|---|
| 1 | SegmentMinCsr 空段 | `out=0`、`arg_out=E` | 填 dtype 最大值 | pyg-lib `segment_csr_kernel.cpp:279-404`；官方题面 |
| 2 | SegmentMinCsr 并列 | 严格 `<`，保留首遇 | `<=` 保留最后一次 | 同上 |
| 3 | FPS 并列 | 最小输入索引 | `index % 256` lane tie | torch_cluster `argmax` 首最大值；官方题面 |
| 4 | Radius 边界 | `d² <= r²` | 和 Ball/Cylinder 共用 `<` | 官方 Radius 题面 |
| 5 | Ball/Cylinder 边界 | 严格 `d² < r²` | 用 `<=` | AnyGrasp `ball_query_gpu.cu` / `cylinder_query_gpu.cu` |
| 6 | Ball/Cylinder 不足 | 尾部重复**首个命中** | 重复最后一个/填 0 | 同上 |
| 7 | Cylinder 旋转 | `rot^T·Δ` | `rot·Δ` 或先求逆 | AnyGrasp `x_rot=r0*dx+r3*dy+r6*dz` |
| 8 | Spspmm 输出 | COO 固定槽位 `U=m·n`，未命中 `(-1,-1)/0` | CSR 动态 nnz | 官方 Spspmm 题面 |
| 9 | Spspmm 零和 | 保留显式零 | 值为 0 就丢弃 | 官方题面；cuSPARSE 语义 |
| 10 | FFT inverse | 乘 `1/(N·M)` | 只做共轭/反向 | 官方 FourierTransform2D 题面 |

## 2. 三个“只改一行但影响很大”的案例

### 2.1 GroupedTopk：把 256 次 exp 降到 8 次

源码：`mantuoluozk/kernelswift-2026-triton-kernels`，`task01_grouped_topk/solution.py:14-70`。

小点是两个数学事实：

1. softmax 单调，选组/选专家可以直接在 logits 上做，不需要先 softmax；
2. Top-K 重新归一化时，256 路公共分母 `Σ_j exp(z_j)` 抵消，只需对选中的 8 个 logits 做 `exp(z_i-z_0)`。

结果：每行从 256 次 `exp` + 256 次加和，降到 8 次 `exp` + 8 路归一化。本机 910B3 复测：`0.5935→0.2223 ms`，2.670×。

适用条件：`renormalize=True`、scoring 单调。负例：`renormalize=False` 时分母不抵消；sigmoid/温度/非单调 scoring 不成立；tie 行为要与 `torch.topk` 对齐。

### 2.2 TreeLineMinDistance：外提不变量、sqrt 后置、`L=0` 分支

源码：`Hansenwuwuwu/ascend-c-competition-archive`，`TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_pairs.h:340-515`。

小点：

- `D=C-A`、`L=D·D`、`1/L` 只依赖线段，按 batch 算一次，所有点复用；
- 每个候选只算平方距离，最后对最小值做一次 `sqrt`，而不是每个候选 `sqrt`；
- 比较用严格 `<`，保留最小索引 tie；
- `L=0` 必须显式走“点到端点距离”，不能只写 `max(L, FLT_MIN)`；后者会得到接近 0 的投影，不是端点距离。

注意：同一份实现的 `M=1024` half Top-2 粗筛**没有 no-miss 证明**，不能当精确 argmin 模板；这是“小点优化”的反例，不是可复用经验。

### 2.3 Concat：窄行整块拷贝，以及三个“优化反而更慢”

源码：`cann/cann-ops-competitions`，`Concat`（BatchLions）。

小点：单路非空且行 ≤128B 时，把 `rows` 并进列，走整块连续拷贝；作者实测 1.54×–674×（910B1）。

但同一份实现也给出了负结果：

- 均衡列带（`S9_BAL=1`）在 246 路上慢 **1.35%**：关键路径由最宽 band 决定；
- 边界路裁剪（`S9_TRIM`）在三个形状上慢 8.0%/7.9%/1.0%：少读的字节抵不上跨步 DMA；
- 整块取 TilingData 头慢 **8%**：连续标量头本来就在同一 cache line；
- 整块拷贝只对窄行有利，≥256B 行会慢 11%–19%。

经验：小点优化必须带 **crossover 条件** 和 **负例**。不是“少读字节”就一定快，也不是“多一个 fast path”就一定赚。

## 3. 制造专场的边界规则

### 3.1 SegmentMinCsr

```text
out[r,c]     = min(src[indptr[r]:indptr[r+1], c])
arg_out[r,c] = 段内全局行号（首遇 tie）
空段：out=0，arg_out=E
E=0 全空图：out=0，arg_out=0
```

`indptr` 是 int64；`arg_out` 是 int64。`HIT1920/torch_scatter_npu` 的补充骨架是 int32、无 arg_out、空段填最大值，只能当构建/调用骨架，不能当官方语义。

### 3.2 FPS

```text
首点 = 0
每轮：dist = min(dist, ||x - x[selected]||²)
selected = argmax(dist)  # 首最大值 = 最小索引
npoint 轮
```

torch_cluster 的 `argmax` 与官方最小索引 tie 一致；但它的 `random_start`/`ratio`/`ceil` 不是官方 `npoint` 接口。FP16 的逐维 half 舍入 vs FP32 累加仍需用官方 `mx_driving` 逐位确认。

### 3.3 Radius / BallQuery / CylinderQuery

| 算子 | 比较符 | 输出 | 不足/未命中 |
|---|---|---|---|
| Radius | `d² <= r²` | 定槽位 `[2,M*K]` | `(-1,-1)` |
| BallQuery | `d² < r²` | `[B,M,nsample]` | 尾部重复首个命中；零命中全 0 |
| CylinderQuery | `y_l²+z_l² < r²` 且 `hmin<x_l<hmax` | `[B,M,nsample]` | 尾部重复首个命中；零命中全 0 |

CylinderQuery 的局部坐标是 `rot^T·Δ`；AnyGrasp 源码写成 `x_rot=r0*dx+r3*dy+r6*dz`，不要写成 `rot·Δ`。`rot` 不要求正交，不要做转置归一化/求逆。

### 3.4 Spspmm / FFT

- Spspmm：COO 输入/输出，固定槽位 `U=m·n`，未命中 `(-1,-1)/0`；值为 0 的坐标要保留；CSR SpGEMM 的输出不能直接当官方输出。
- FFT：`inverse=true` 要乘 `1/(N·M)`；`fft2_d` 只覆盖 32/64/128，官方要求 2..8192；`fft1_d` 可补单维，但仍需 2D 组合和归一化。

## 4. 小点检查表

写任何制造专场算子前，先回答：

1. 比较符是 `<` 还是 `<=`？三个点云算子是否混用了？
2. 空段/空行/零命中/未命中，填 0、`(-1,-1)` 还是哨兵 `E`？
3. 并列时首遇还是末遇？是否与参考库的 `argmax/argmin` 一致？
4. 输出是动态长度还是固定槽位？槽位大小怎么算？
5. 索引是 int32 还是 int64？有没有静默截断？
6. 布局是 AoS 还是 SoA？消费者要什么布局？
7. 归一化/缩放乘在哪一步？inverse/mean 是否重复乘？
8. 小 shape 的 fast path 是否有校验/回退？换 shape 会不会错？
9. 这个“小优化”的 crossover 在哪里？有没有负例？
10. 有没有本机 910B3 日志？没有就保持 `not-run`。

## 5. 证据边界

- GroupedTopk/Sinkhorn/52Xayah 有本机 910B3 结果；FusedMoE 本机编译失败；Hansen/Mamba/vLLM-HUST 未在本机运行。
- 制造专场的 SegmentMinCsr 补充骨架有 BUILD+RUN PASS，但只是 `8×2048` int32 示例，不是官方语义；主 `ops-gnn` 实现仍未跑。
- 制造专场官方题面、上游语义核对和 CPU 参考测试见 [`competition/mfg-2026/`](../../competition/mfg-2026/)；本文不重复长表。
- 所有 950 结果、作者成绩、平台分数都不能当 910B 本机结果；没有 910C 数据就保持 `N`。
