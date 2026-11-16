# 工作量与固定开销：什么时候“少算”，什么时候“多算但更规则”

## 1. 问题

同一份 910B/910C 代码，性能瓶颈可能在三类位置：

1. **真的算多了**：重复投影、重复 sqrt、重复读先验、重复做 softmax；
2. **固定开销**：kernel 启动、动态分发、布尔 mask、小 GEMM、权重转换副本、中间张量写回；
3. **依赖链**：串行扫描、跨核等待、单核归约、L0C/UB 复用等待。

“优化”不是永远减少 FLOPs。有时多算 4 倍，反而因为去掉了固定开销而快 10 倍；有时少算一半，却因为引入了 gather/转置/分支而变慢。本文把本轮看到的几种结构放在同一个判断框架里。

## 2. 三类杠杆

### 2.1 减少真实工作量

#### 例 A：SplineConv 先按权重索引聚合

源码：`Hansenwuwuwu/ascend-c-competition-archive`，`SplineConv/rank1_submission_1174398/code/op_kernel/spline_conv_compact.h:200-225`。

通用逐边：

```text
for each edge e:
    for each slot s:
        out[dst[e]] += basis[e,s] * (x[src[e]] @ W[wi[e,s]])
```

rank1：

```text
for each target i:
    for each incoming edge e:
        for each slot s:
            agg[wi[e,s]] += basis[e,s] * x[src[e]]
    for each active wi:
        out[i] += agg[wi] @ W[wi]
```

设入度 `d`、通道 `C`、输出 16：

- 逐边投影约 `4*C*16*d` MAC；
- 先聚合约 `4*C*d + 4*C*16` MAC；
- 比值约 `16d/(d+16)`：`d=1` 0.94，`d=4` 3.2，`d=8` 5.3。

**条件**：权重索引只由离散 cell 决定、basis 对边线性、目标节点有多个入边、聚合平面能驻留 UB。

**负例**：`d=1` 时多一次清零/聚合，反而略慢；basis 依赖非离散属性或权重索引随边变化时，代数前提不成立。

#### 例 B：GroupedTopk 省掉 256 路 softmax

源码：`mantuoluozk/kernelswift-2026-triton-kernels`，`platforms/ascend910b/task01_grouped_topk/solution.py:14-70`。

两个事实：

- softmax 单调，选组/选专家可直接在 logits 上做；
- Top-K 重新归一化时，256 路公共分母抵消，只需对 8 个选中 logits 做 `exp`。

所以每行从 256 次 `exp` + 256 次加和，降到 8 次 `exp` + 8 路归一化。代价是 12 次 `argmax`（4 次组 + 8 次专家）和掩码。

**条件**：softmax/单调 scoring + `renormalize=True`。

**负例**：`renormalize=False` 时分母不抵消；`E` 很大时 12 次 `E` 宽归约变成 O(E*K) 串行选择。

#### 例 C：TreeLineMinDistance 外提不变量、sqrt 后置

源码：`Hansenwuwuwu/ascend-c-competition-archive`，`TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_pairs.h:340-515`。

- `D=C-A`、`L=D·D`、`1/L` 只依赖线段，按 batch 算一次，复用于所有点；
- 每个候选只算平方距离，最后对最小值做一次 `sqrt`，而不是每个候选 `sqrt`；
- chunk 内归约 + 跨 chunk 成对归约，比较用严格 `<` 保留最小索引。

**条件**：线段在 batch 内共享；归约维度可分离；tie 规则明确。

**负例**：`M=1024` 的 half Top-2 粗筛没有 no-miss 证明（见数值经验）；`max(L, FLT_MIN)` 不等价于 `L==0` 分支。

### 2.2 用更多计算换掉固定开销

#### 例 D：FusedMoE 稠密全专家

源码：`mantuoluozk/kernelswift-2026-triton-kernels`，`task02_fused_moe/solution.py:8-87`；参考 `reference.py:42-75`。

- 参考：每个 token 只算 Top-2 专家，8 次小 GEMM + 布尔 mask + Python 循环 + 完整权重 cast；
- 候选：8 个专家全算，3 个规则 kernel，权重按 tile cast。

纯 MAC 约 `3*E*T*H*I` vs `3*topk*T*H*I`，`E=8, topk=2` 时多 4 倍；但 6.942960 ms → 0.606584 ms（11.446×，作者 910B1 官方评测器）。这说明在该 shape 下，固定开销/小 GEMM 低效/权重副本远大于 4 倍 MAC。

**条件**：`E` 小、`T` 小、`H/I` 小、Top-K 固定、允许计算未选中专家。

**负例**：`E=64` 时纯 MAC 比 32×，稠密全专家几乎必然不划算；`T` 大时 `[E,T,I]` 中间缓冲和 4 倍计算会反噬。

#### 例 E：Sinkhorn 标量化

源码：`mantuoluozk/kernelswift-2026-triton-kernels`，`task08_hc_split_sinkhorn/solution.py:41-89`。

4×4 矩阵、20 轮迭代，参考实现每轮产生多个 PyTorch 小 kernel 和中间 tensor；候选把 16 个元素放进寄存器，一个 kernel 完成 20 轮。算术量近似不变，3.032301 ms → 0.329475 ms（9.203×）。

**条件**：矩阵极小且固定、每行独立、行数不太少。

**负例**：`hc_mult=8/16` 时 64/256 个标量，指令膨胀；行数很大时一个 program 一行、1 warp 可能填不满。

### 2.3 减少中间张量与 kernel 数

#### 例 F：Linear+SwiGLU

源码：`vLLM-HUST/vllm-hust-opset`，`linear_swiglu_kernel.py:35-66`。

基线 GEMM 写 `[M,2N]`，再启动 `npu_swiglu` 读它；候选在同一 kernel 里保留 gate/up 两个 FP32 accumulator，只写 `[M,N]`。GEMM MAC 不变，省的是中间写回/重读和一次 kernel。

**条件**：`M<=128`、unquantized、BF16/FP16、CBUF 能放两个 accumulator。

**负例**：M=256 的 fused tile 超 CBUF；大 M 多 tile 出现错误值；冷启动 JIT 可能为负；端到端 TPOT 只降 3.82%，远小于算子 16–18%。

#### 例 G：Mamba-2 ChunkMix

源码：`So-cean/mamba-ascendc`，`mamba2_ssd_chunk_mix.cpp:1016-1211`，`design.md:20-57`。

把长度 `L` 的串行扫描拆成 `L/chunk` 个 chunk 的状态传递 + chunk 内三角矩阵乘；把 335 个 PyTorch kernel / 20 个 Triton kernel 压到 3 个设备 kernel。代价是 chunk 内多出 `O(L*chunk)` 工作、12.06 GiB 中间物化、StateEpilogue memory-bound。

**条件**：序列/head 足够大、`P=N=chunk` 匹配、Cube/Vector 能并行、UB 能容纳 chunk。

**负例**：短序列/小 head 时分块开销大于收益；T128 的 UB 已到 182.8 KiB；950PR 当前 HEAD 有 grouped fast-path 回归。

## 3. 判断表

| 现象 | 优先怀疑 | 先做的实验 |
|---|---|---|
| 算子 FLOPs 不高但很慢 | 固定开销/launch/小 GEMM | 数 kernel 数、launch 数、每个 kernel 的平均工作量 |
| 中间张量很大、带宽高 | 生产者消费者物化 | 算中间字节数，尝试融合/consumer layout |
| 多算反而更快 | 动态分发/权重副本/mask | 对比 kernel-only 与端到端；做“只去掉分发”的消融 |
| 少算反而更慢 | gather/转置/分支/UB 压力 | 对比“少算但多搬”和“原路径”的字节/停顿 |
| 大 shape 收益消失 | 算法工作量/容量 | 扫 E/T/L/chunk，找收益拐点 |
| 小 shape 收益消失 | 启动/占用率 | 扫行数/批大小，测空 kernel/固定开销 |

## 4. 负例汇编

1. **FusedMoE 的 4 倍 MAC**：只在 E/T/H/I 小的固定 shape 成立；E 大时不要用。
2. **Linear+SwiGLU 的 M>128**：CBUF 溢出/错误值；必须回退。
3. **Sinkhorn 的 4×4 标量化**：不能扩到 8×8/16×16。
4. **Spline 的 d=1**：聚合开销大于收益。
5. **Tree 的 half Top-2**：无 no-miss 证明，不能当精确 argmin。
6. **52Xayah 的 Softmax/Matmul**：朴素 Triton 比厂商库慢 4×/4.3×；标准 shape 上厂商库是强基线。
7. **Task04/Task10**：超大融合触发 watchdog/UB 超限；融合不是无条件收益。

## 5. 可证伪的验证设计

### 5.1 分离“kernel 本身”和“固定开销”

对同一算子做三组计时：

1. 纯 kernel（如果工具支持，排除 Python/launch）；
2. 单次 forward（包含 launch，逐 sample 同步）；
3. 端到端（包含前后处理/模型步）。

如果 2 快而 1 不快，收益来自 launch/分发；如果 3 不快而 2 快，目标 op 在总时延中占比太低。

### 5.2 扫规模找拐点

- FusedMoE：`E=8/16/32/64`、`T=83/512/2048/8192`；
- Mamba-2：`L=512/2048/8192/16384`、`chunk=32/64/128`；
- Linear+SwiGLU：`M=1/32/128/256/512`；
- Sinkhorn：`hc_mult=4/8/16`；
- Spline：按入度分桶 `d=1/2/4/8/16`。

记录：MAC 数、kernel 数、中间字节数、UB/CBUF 峰值、时延。收益拐点比单点加速比更有用。

### 5.3 消融而不是归因

对 FusedMoE，至少拆成：

- 只把 Python 循环换成规则 kernel；
- 只去掉完整权重 cast；
- 只把 Top-2 分发改成全专家；
- 三者组合。

对 Linear+SwiGLU，至少拆成：

- 只融合 swiglu（仍写 `[M,2N]`）；
- 只去掉中间张量（不融合激活）；
- 两者都做。

没有消融，就不能把 11.446× 或 16% 归因到某一个技巧。

## 6. 证据边界

- 本文所有数字来自原作者/原报告；本机未复现。
- 评测口径不同：KernelSwift 是 forward 级逐 sample 同步；Mamba-2 是 device event p50；vLLM-HUST 是算子级 10/50 + 端到端 3 轮；Hansen 只有公示分数。
- 不能跨来源比较加速比；只能比较“机制在各自条件下是否成立”。
