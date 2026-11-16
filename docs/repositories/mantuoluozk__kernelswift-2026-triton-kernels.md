# mantuoluozk/kernelswift-2026-triton-kernels：Ascend 910B Triton 三题深读

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/mantuoluozk/kernelswift-2026-triton-kernels` |
| 作者 | mantuoluozk |
| 场景 | 2026 KernelSwift 算子创新大赛，赛道一（Triton 算子优化） |
| 访问日期 | 2026-11-16 |
| 默认分支 | `main` |
| 固定 commit | `ac937591f5517b0dfc7c172252f0f27caf772d34` |
| 主要 DSL | Triton / Triton-Ascend |
| 算子族 | GroupedTopk、FusedMoE、FlexAttention、SPLADE、RotaryEmbedding、MMEncoderAttention、mhc_post、Sinkhorn、CentreRandomAugmentation、head_compute_mix_bwd |
| 许可证 | 仓库未声明 LICENSE；`evaluator/LICENSE` 是官方评测器上游许可证 |
| 本地缓存 | `repository-research-cache/mantuoluozk_kernelswift-2026-triton-kernels` |
| 与既有归档关系 | 新来源；与 GroupedMatmulGelu 无重复 |
| 本报告范围 | 深读 Ascend 910B 的 Task01 GroupedTopk、Task02 FusedMoE、Task08 hc_split_sinkhorn |

### 1.1 环境与评测口径

仓库按四个平台组织：海光 BW1000、沐曦 C500、昇腾 910B、燧原 S60。本报告只看 `platforms/ascend910b`。

| 项 | 值 |
|---|---|
| 设备 | Ascend 910B1，单卡 64 GiB HBM |
| CANN | 9.0.0 |
| PyTorch / torch_npu | 2.10.0 / 2.10.0 |
| Triton-Ascend | 3.2.1 |
| Python | 3.11.15 (aarch64) |
| 评测器 | `evaluator/auto_bench.py`（官方评测器副本） |
| warmup / repeat | 200 / 500 |
| 统计 | 每个 sample 单独 `perf_counter` + device synchronize，取中位数 |
| 精度 | `atol=1e-2`、`rtol=1e-2` |

评测器 `time_forward()` 的实现是：先 warmup，再对每个 repeat 执行一次 `forward`，每次前后同步设备，记录单次时间，最后取中位数（`evaluator/auto_bench.py:434-450`）。因此这些数字是**一次完整 forward 调用**的时延，包含 Python 调度、kernel launch 和同步，不是纯 kernel 时间。对于“减少 kernel 数/启动开销”的优化，这个口径反而更贴近真实调用。

仓库 README 还明确记录了 910B 的三个限制（`platforms/ascend910b/README.md:93-98`）：

- UB 容量会直接限制单个 program 的 tile；Task10 的 8192 元素单块约 320 KiB，超过后端报告的 192 KiB；
- 超大融合不一定更快：Task04 的 decoder+pool 单 kernel 会触发 AICore watchdog，拆成 `torch_npu` GEMM + Triton pool 后才稳定；
- 首次执行包含 Triton 编译/CANN 代码生成，不能算进 kernel 性能。

这三条是本文判断“融合边界”的重要负例。

### 1.2 本机 910B3 复测

2026-11-16 在本机 Ascend 910B3（CANN 9.1.0、torch 2.10.0、torch_npu 2.10.0、Triton-Ascend 3.2.0）上复测：

| 任务 | 本机结果 | 本机数字 | 作者数字 | 判断 |
|---|---|---|---:|---|
| GroupedTopk | PASS | 0.593500→0.222255 ms，2.670× | 0.701666→0.283144 ms，2.478× | 方向一致 |
| FusedMoE | 编译失败 | UB 2359296 bits > 1572864 | 6.942960→0.606584 ms，11.446× | 当前 TA/CANN 不可移植 |
| Sinkhorn | PASS | 3.616230→0.273420 ms，13.226×；max diff 6.765e-06 | 3.032301→0.329475 ms，9.203× | 方向一致 |

完整命令/日志见 [本机 910B3 复测记录](../research/LOCAL_910B_VALIDATION_20261116.md)。

## 2. Task01 GroupedTopk：利用 Softmax 单调性，省掉 256 路归一化

### 2.1 语义

输入 `gating_output[rows,256]`（FP32）、`hidden_states[rows,7168]`（只用于 batch 对齐）。输出 `weights[rows,8]`（FP32）、`ids[rows,8]`（INT32）。

参考实现（`reference.py:31-65`）：

1. 对 256 个专家做 softmax；
2. 每个专家组 32 个专家，取组内最大值作为组分数；
3. 选 Top-4 组；
4. 在选中组的 128 个专家中选 Top-8；
5. 对 Top-8 权重重新归一化，乘 `routed_scaling_factor`。

固定配置：`topk=8`、`renormalize=True`、`num_expert_group=8`、`topk_group=4`、`scoring_func="softmax"`。

### 2.2 固定源码位置

| 内容 | 路径/行 |
|---|---|
| 单 kernel 入口、256 专家加载 | `platforms/ascend910b/task01_grouped_topk/solution.py:8-12` |
| 8×32 reshape、组内 max、Top-4 组 | 同上 `:14-26` |
| 选中组掩码、Top-8 专家 | 同上 `:28-59` |
| 只对 8 个 logits 做 exp + 归一化 | 同上 `:61-70` |
| 输出写回 | 同上 `:72-89` |
| 参考实现 | `.../task01_grouped_topk/reference.py:24-65` |

固定 commit 的文件 SHA256：

```text
task01_grouped_topk/solution.py  69261f900b091b2f40e2741f5b45a4f52b06374ec6610fecf351b388816cd227
```

### 2.3 机制：单调性 + 分母抵消

两个数学事实：

1. **Softmax 单调**：`softmax(z)_i` 与 `z_i` 同序（分母对所有 i 相同），所以选组、选专家都可以直接在 logits 上做，不需要先算 256 路 softmax。
2. **Top-K 重新归一化时分母抵消**：

```text
p_i = exp(z_i) / Σ_j exp(z_j)
w_i = p_i / Σ_{k∈TopK} p_k
    = exp(z_i) / Σ_{k∈TopK} exp(z_k)
```

所以只需对选中的 8 个 logits 做 `exp(z_i - z_0)`，再除以 8 项和；256 路 softmax 的 `exp` 和 `sum` 完全不需要。

源码对应：

- `group_scores = tl.max(tl.reshape(logits,(8,32)), axis=1)`（`:15-16`）；
- 连续 4 次 `argmax` + `where(...,-inf)` 选 Top-4 组（`:20-26`）；
- `selected_group` 掩码后，连续 8 次 `argmax` + `where` 选 Top-8 专家（`:35-59`）；
- `e0=exp(v0-v0)`、`e1=exp(v1-v0)` ... `denom=sum`（`:62-70`）；
- 写回 `e_i/denom*scale` 和 `i_i`（`:72-89`）。

### 2.4 工作量与代价

对一个 row：

- 参考：256 次 exp、256 次加和、一次 256 路 softmax、两次 topk（8 组、256 专家）、一次 8 路归一化；
- 候选：4 次组 `argmax`（每次 8 个组分数）+ 8 次专家 `argmax`（每次 256 个候选）+ 8 次 exp + 8 路归一化。

它省掉的是 256 次 exp/加和，但增加了“连续 argmax + where 掩码”的选择链。对 256 个专家、Top-8，这个选择链约 12 次 256 宽归约；如果专家数更大（比如 512/1024）或 Top-K 更大，反复 `argmax` 会变成 O(E*K) 的串行选择，可能不如排序/堆/分段归约。

另一个代价是**输出顺序/tie 语义**：连续 `argmax` 得到的是按分数从大到小（每次取当前最大）的 Top-K；`torch.topk` 默认也返回排序后的值/索引。若题目要求“按专家索引升序”或特定 tie 规则，需要额外验证。`tl.argmax` 的 tie 行为也要与参考对齐。

### 2.5 证据与判断

| 项 | 内容 |
|---|---|
| 正确性 | 910B1 官方评测器 PASS；本机 910B3 PASS |
| 时延 | 作者 0.701666 ms → 0.283144 ms，2.478×；本机 0.593500→0.222255 ms，2.670× |
| 关键配置 | `num_warps=1` |
| 证据级别 | `O+A+V`（作者官方评测器 + 本机 910B3 复测；设备/版本不同） |
| 复用判断 | **优先复用“单调性 + 分母抵消”的代数简化；选择链按 E/K 重新设计** |

适用条件：

- scoring 是 softmax 或其它单调变换；
- 最终权重要在 Top-K 内重新归一化，公共分母确实抵消；
- 专家数/组数固定且不大，连续 argmax 的串行成本可接受；
- 输出顺序/tie 规则与参考一致。

负例/风险：

- 如果 `renormalize=False`，公共分母不抵消，不能直接省 softmax；
- 如果 scoring 是 sigmoid 或带温度的非单调变换，单调性不成立；
- `E` 很大时，12 次 256 宽归约变成 O(E*K)，可能不如一次排序/分段归约；
- 该 kernel 一个 program 处理一行，`rows=83` 时只有 83 个 program；行数很少时 launch/占用率是主问题，行数很大时又要看是否受 UB/寄存器限制；
- 本机 2.670× 只在 910B3/CANN 9.1.0/TA 3.2.0 上成立；不能把 2.478× 或 2.670× 外推到 910C/其它 CANN。

## 3. Task02 FusedMoE：用“稠密全专家”换掉动态分发

### 3.1 语义

输入 `hidden_states[83,128]`（FP16）、`router_logits[83,8]`（FP32）；权重 `w1[8,128,128]`（gate+up 拼接）、`w2[8,128,64]`。每个 token 选 Top-2 专家，对两个专家输出加权求和：

```text
scores = softmax(router_logits)
top2 = topk(scores, 2)
for e in top2:
    gate_up = x @ w1[e].T          # [T, 2I]
    gate, up = gate_up.chunk(2)
    y_e = (silu(gate) * up) @ w2[e].T
out = Σ_e weight_e * y_e
```

参考实现（`reference.py:42-75`）是标准的动态分发：

- `torch.softmax` + `torch.topk`；
- `flat_ids`/`flat_w` 把 `(token,k)` 展平；
- `for e in range(8)`：布尔 mask，取出该专家的 token，做小 GEMM，写回 `expert_out[mask]`；
- 最后乘权重并按 top-k 求和。

### 3.2 固定源码位置

| 内容 | 路径/行 |
|---|---|
| gate/up 投影 + SiLU + 逐元素乘 | `platforms/ascend910b/task02_fused_moe/solution.py:8-24` |
| down 投影 | 同上 `:27-42` |
| Top-2 路由 + 两专家加权 | 同上 `:45-63` |
| 三个 kernel 的 launch 与缓冲 | 同上 `:76-87` |
| 参考动态分发 | `.../task02_fused_moe/reference.py:42-75` |

固定 commit 的文件 SHA256：

```text
task02_fused_moe/solution.py  8dc46f0ac534fb1d203dac7681245264f71d66fab680c6c00b03fa966d403021
```

### 3.3 机制：规则化三阶段，计算所有专家

候选实现不按 top-2 分发，而是：

1. `_gate_up_kernel`：grid `(ceil(83/32), 8)`，每个 program 处理一个 32-token tile 和一个专家；加载 `x[32,128]`、`w1[e]` 的 gate/up 两个 64 列块，做两次 `tl.dot`，融合 `gate*sigmoid(gate)*up`，写 `act[8,83,64]`；
2. `_down_kernel`：grid `(ceil(83/32), 8, 2)`，每个 program 处理 32 token、一个专家、64 个输出列；读 `act`，做 `act[32,64] @ w2[e][64,64]`，写 `expert_out[8,83,128]`；
3. `_route_kernel`：grid `(83,)`，每个 token 读 8 个 logits，取 Top-2，算 `ratio=exp(l1-l0)`、`w0=1/(1+ratio)`、`w1=ratio/(1+ratio)`，从 `expert_out` 取两个专家向量加权写回。

关键点：

- 三个 kernel 的 shape 都是规则的，没有布尔 mask、没有 Python 循环、没有动态 shape 分发；
- `w1`/`w2` 是 FP32 参数，但 kernel 内按 tile `tl.load(...).to(tl.float16)`，避免参考实现里 `w1.to(dtype)` 的完整权重转换副本；
- 计算所有 8 个专家，而不是只算 Top-2；`_route_kernel` 最后只消费两个。

### 3.4 工作量：4 倍专家 FLOPs 换掉固定开销

设 `T=83`、`H=128`、`I=64`、`E=8`、`topk=2`。

- 参考：每个 token 只算 2 个专家，gate/up 约 `2*T*H*(2I)`，down 约 `2*T*I*H`，合计约 `6*T*H*I` MAC（忽略 bias/激活）；
- 候选：所有 8 个专家，gate/up 约 `E*T*H*(2I)`，down 约 `E*T*I*H`，合计约 `3*E*T*H*I = 24*T*H*I` MAC；
- 纯 MAC 比约 `24/6 = 4×`。

但候选把“8 次小 GEMM + 布尔 mask + Python 循环 + 完整权重 cast”换成 3 个规则 kernel，最终 6.942960 ms → 0.606584 ms（11.446×）。这说明在该 shape 下，**固定开销/小 GEMM 低效/权重转换**远大于多算 4 倍的专家 FLOPs。

这也给出一个明确边界：如果 `E` 从 8 增到 64，纯 MAC 比变成 `3*64*T*H*I / (3*2*T*H*I) = 32×`，稠密全专家几乎必然不划算；如果 `T` 从 83 增到 8192，规则 kernel 的吞吐会提高，但 8 个专家的 `[E,T,I]` 中间缓冲和 4 倍计算仍在，收益会快速下降。

### 3.5 证据与判断

| 项 | 内容 |
|---|---|
| 正确性 | 910B1 官方评测器 PASS（`rtol=1e-2, atol=1e-2`）；本机未通过编译 |
| 时延 | 作者 6.942960 ms → 0.606584 ms，11.446×；本机无有效时延 |
| 关键配置 | 32-token tile，`num_warps=4` |
| 证据级别 | `O+A` + 本机编译失败（不是 `V` 性能证据） |
| 复用判断 | **在 E 小、T 小、H/I 小的固定 shape 下优先验证“稠密全专家 + 三阶段规则 kernel”；E/T 大时不要用** |

适用条件：

- 专家数 `E` 很小（本例 8），top-k 固定（2）；
- token 数 `T` 很小（83），动态分发的固定开销占比高；
- `H/I` 小，单个专家的 GEMM 小，规则 tile 仍能填满；
- 权重可以按 tile 转换，不需要完整副本；
- 允许计算未被选中的专家（不违反题目“不得调用参考实现”的规则）。

负例/风险：

- `E` 大时计算浪费线性放大；
- `T` 大时 `[E,T,I]` 中间缓冲和 4 倍计算会反噬；
- `_route_kernel` 只从 logits 取 Top-2，依赖“softmax 单调 + Top-2 重新归一化分母抵消”；若 `renormalize=False` 或 scoring 改变，权重公式要重算；
- 该实现假定 `T=83`、`H=128`、`I=64`、`E=8`、`topk=2`，`forward` 里的 `83`、`128`、`64` 是硬编码；
- 权重 FP32→FP16 的按 tile cast 会重复读取权重；`T` 增大时，每个 M tile 都重新读同一专家权重，可能变成带宽瓶颈；
- 本机 TA 3.2.0/CANN 9.1.0 在原始 BLOCK_M=32（num_warps=1/4）和本地 BLOCK_M=16 副本上都 UB 溢出；11.446× 不能外推到其它 shape 或当前环境。

## 4. Task08 hc_split_sinkhorn：把 4×4 Sinkhorn 留在寄存器

### 4.1 语义

输入 `mixes[B,S,24]`、`hc_scale[3]`、`hc_base[24]`，`hc_mult=4`、`sinkhorn_iters=20`、`eps=1e-6`。输出：

```text
pre  = sigmoid(mixes[...,0:4]*s0 + base[0:4]) + eps
post = 2*sigmoid(mixes[...,4:8]*s1 + base[4:8])
comb = softmax_rows(mixes[...,8:24]*s2 + base[8:24])   # 4x4
for 20 iterations:
    comb = comb / (row_sum(comb)+eps)
    comb = comb / (col_sum(comb)+eps)
```

参考实现（`reference.py:26-52`）用 PyTorch 的 `amax`、`exp`、`sum`、逐元素除法，20 轮会产生大量小 kernel 和中间 tensor。

### 4.2 固定源码位置

| 内容 | 路径/行 |
|---|---|
| 单 program 处理一个 `(b,s)` 行 | `platforms/ascend910b/task08_hc_split_sinkhorn/solution.py:19-22` |
| pre/post 门控 | 同上 `:28-37` |
| 16 个 comb 值加载 + 行 max/exp | 同上 `:41-65` |
| 20 轮行/列归一化（标量） | 同上 `:67-81` |
| 三个输出写回 | 同上 `:83-89` |
| 参考实现 | `.../task08_hc_split_sinkhorn/reference.py:26-52` |

固定 commit 的文件 SHA256：

```text
task08_hc_split_sinkhorn/solution.py  79bc380ac1a6dd40036ab57ee83a34b85c4202f5479e9b8bb6686f0767c96f2a
```

### 4.3 机制：16 个标量 + 20 轮循环全部在寄存器

代码把 4×4 矩阵的 16 个元素显式命名成 `m00..m33`（`:41-56`），然后：

- 每行取 max、减 max、exp（`:58-65`）；
- 20 轮循环中，行和、行除、列和、列除都用标量表达式（`:67-81`）；
- 最后 16 个 `tl.store` 写回（`:86-89`）。

这不是“向量化 4×4”，而是**故意标量化**：4×4 太小，二维 tile/归约的索引和掩码开销会超过计算本身；把它变成 16 个寄存器标量，20 轮循环没有 kernel 边界、没有中间 tensor、没有跨 program 同步。

### 4.4 工作量与代价

参考：20 轮 ×（行 sum + 行 div + 列 sum + 列 div），每轮至少 4 个 PyTorch 算子；加上初始 softmax/exp、pre/post，整体是几十个 kernel + 多个 `[B,S,4,4]` 中间张量。

候选：1 个 kernel，16 个标量，20 轮循环；每个 `(b,s)` 一个 program，`num_warps=1`。

收益主要来自：

- 消除 20 轮的小 kernel 启动和中间 tensor 写回；
- 避免每轮重新 materialize `comb`；
- 把 pre/post/comb 三个输出融合到同一个 kernel。

代价：

- 16 个标量在 Triton 里会展开成大量标量指令；对 4×4 是优点，对 8×8/16×16 会迅速变成指令膨胀；
- `num_warps=1`、一个 program 一行；测试 shape 只有 `B=2,S=8`（16 行），program 数很少，launch/占用率是主问题；
- `eps` 加在每轮行/列分母上，和参考的 `+eps` 位置必须逐项一致；否则 20 轮后会累积差异；
- 没有 mask/边界处理，固定 `mix_hc=24`、`hc_mult=4`。

### 4.5 证据与判断

| 项 | 内容 |
|---|---|
| 正确性 | 910B1 PASS；作者 max abs err `1.19209290e-07`；本机 910B3 PASS，max abs err `6.76512718e-06` |
| 时延 | 作者 3.032301 ms → 0.329475 ms，9.203×；本机 3.616230→0.273420 ms，13.226× |
| 关键配置 | `num_warps=1`，20 轮，`eps=1e-6` |
| 证据级别 | `O+A+V`（作者 + 本机 910B3；本机误差更大但仍过 1e-2 容差） |
| 复用判断 | **固定 4×4/小矩阵 Sinkhorn 优先复用“标量化 + 单 kernel + 寄存器驻留”；矩阵变大时改用二维 tile** |

适用条件：

- 矩阵极小且固定（4×4），迭代次数固定；
- 每行独立，不需要跨行同步；
- 行数不太少（否则 launch/占用率主导），但也不能多到标量指令成为吞吐瓶颈；
- pre/post/comb 可以在同一个 kernel 里融合。

负例/风险：

- `hc_mult=8/16` 时 64/256 个标量，代码量和寄存器压力不适合照搬；
- 行数很大时，一个 program 一行、`num_warps=1` 可能无法填满 AI Core；
- 如果参考的 `eps` 位置、初始 row softmax 与 20 轮顺序不同，数值会偏；
- 该实现只覆盖固定 shape，不能当通用 Sinkhorn。

## 5. 三题横向观察

| 维度 | GroupedTopk | FusedMoE | Sinkhorn |
|---|---|---|---|
| 主要收益 | 代数简化（单调性/分母抵消） | 规则化 + 消除动态分发/权重副本 | 单 kernel + 寄存器驻留 |
| 算术量 | 减少 256 路 exp/sum | **增加 4× 专家 MAC** | 算术量近似不变 |
| 固定开销 | 减少 softmax/topk kernel | 减少 8 次小 GEMM/loop/mask | 减少 20 轮小 kernel/中间 tensor |
| 关键前提 | softmax + renormalize | E/T/H/I 小 | 4×4、20 轮、固定 eps |
| 主要负例 | E 大时 O(E*K) 选择 | E/T 大时稠密计算反噬 | 矩阵变大时标量爆炸 |
| 证据 | 作者 2.478×；本机 2.670× | 作者 11.446×；本机编译失败 | 作者 9.203×；本机 13.226× |

共同点：

1. **先判断瓶颈是“算术”还是“固定开销”**：GroupedTopk 省算术，FusedMoE 反而加算术但省固定开销，Sinkhorn 主要省 kernel/中间张量。
2. **固定 shape 的常量要显式写进 kernel**，但必须配 Host 侧 shape 校验/回退；否则换 shape 就错。
3. **不要用总加速比归因单个技巧**：FusedMoE 的 11.446× 同时包含“去掉 Python 循环/布尔 mask”“避免完整权重 cast”“规则 GEMM tile”“三阶段融合”，没有消融就不能把全部收益归给“稠密全专家”。

## 6. 证据边界与复用建议

### 6.1 证据边界

- 作者数字来自 910B1/CANN 9.0.0/TA 3.2.1；本机数字来自 910B3/CANN 9.1.0/TA 3.2.0。两者只能比较机制方向，不能把绝对值或加速比横比。
- 本机 GroupedTopk/Sinkhorn 通过；FusedMoE 在 TA 3.2.0/CANN 9.1.0 上 UB 溢出，未得到性能数字。
- 评测是 forward 级、每次 sample 同步，包含 Python/launch；不是纯 kernel 微基准。
- 正确性阈值是 `atol=1e-2, rtol=1e-2`，比很多算子宽松；不能据此推断更高精度场景也通过。
- 没有 910C 数据；910B1/910B3 与 910C 的核数、UB、CANN 版本差异需要重新验证。
- 仓库未声明 LICENSE；本库不复制其源码。

### 6.2 优先复用

- **GroupedTopk 的单调性 + 分母抵消**：这是纯代数，最容易迁移到其它 MoE 路由/专家选择。
- **FusedMoE 的“规则三阶段”**：在 E/T 小的固定 shape 下，用规则 kernel 替代动态分发；先做端到端消融，确认收益来自 launch 还是 GEMM。
- **Sinkhorn 的标量化**：只用于 4×4 或类似极小矩阵；用“每行一个 program + 寄存器驻留 + 单 kernel”消除迭代边界。

### 6.3 需要验证后再用

- FusedMoE 的稠密全专家：必须扫 `E` 和 `T`，确认收益随规模的变化；如果 `E≥32` 仍想用，考虑分组/容量因子/稀疏分发。
- GroupedTopk 的连续 `argmax`：在 `E=256/Top-8` 下够用；`E=1024` 时要换选择算法。
- Sinkhorn 的 `num_warps=1`：行数很大时要重新调 grid/warp。

### 6.4 暂不使用

- 把 11.446× 直接外推到其它 MoE shape；
- 在 `renormalize=False` 或非 softmax scoring 下复用 GroupedTopk 的权重公式；
- 把 4×4 标量化 Sinkhorn 直接扩展到 8×8/16×16；
- 忽略 Task04 的 watchdog 和 Task10 的 UB 超限负例。
