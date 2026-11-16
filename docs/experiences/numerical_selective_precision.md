# 数值与选择性精度：别为了少数 lane 把整张量升精度

## 1. 问题

高性能算子经常遇到两组矛盾：

- 题目要求 FP32 中间/严格容差，但全程 FP32 会慢；
- 用 FP16/BF16 或近似筛选能快，但可能改变结果、破坏 tie、或在边界输入上超差。

本轮看到的有效做法不是“全量高精度”或“全量低精度”，而是：

1. **主路径用足够精度**（通常是 FP32 向量）；
2. **用廉价判据找出少数敏感 lane**；
3. **只对敏感 lane 做高精度/精确回退**；
4. **对不能证明的近似筛选，明确列为风险，不作为精确模板**。

## 2. Fire：选择性高精度修复

### 2.1 问题

中心坐标：

```text
x = (2*sigmoid(tx) - 0.5 + gx) * stride
```

当 `2*sigmoid(tx) ≈ 0.5 - gx` 时，两个相近数相减发生灾难性抵消；`stride` 越大，绝对误差越大。题面要求 FP32 中间，不能直接用低精度 exp。

### 2.2 做法

源码：`Hansenwuwuwu/ascend-c-competition-archive`，`FireCandidateDecodeFusion/rank2_submission_1176342/code/op_kernel/device_fire.h:105-143`，调用 `:320-327`。

1. 先看 `priors` 中 `stride` 的最大值；若不超过 FP16 4096 / FP32 32 的阈值，直接返回（`:105-118`）；
2. 否则对 x/y 两个通道计算 `abs(2*s - 0.5 + gx)`（`:119-125`）；
3. 只对 `abs(...) < 1/16` 且 `stride` 超阈值的 lane，读取原始 logit，用软件 53 位 exp + 正确舍入倒数重算（`:126-139`）；
4. 把修复后的倒数写回对应 lane。

主路径仍然是向量化 `exp` + `1/(1+·)`；只有少数 lane 走标量高精度。

### 2.3 适用条件

- 主路径是 FP32 向量，且大多数 lane 的误差可接受；
- 敏感条件可以用廉价判据（`stride` 最大值、`abs(2s-0.5+gx)`）先筛；
- 敏感 lane 比例低；
- 题目允许逐 lane 分支/标量回退；
- 软件 exp/倒数的结果能覆盖目标范围（这里 `original∈(-20,0)`）。

### 2.4 负例/风险

- 敏感 lane 比例高时，标量循环会反噬；
- `stride` 阈值和 `1/16` 是题目/实现特定常数，不能照搬；
- 软件 53 位 exp 的代码体积/指令数不小；
- 该路径仍然改变舍入顺序，不能保证与参考逐位一致；
- 没有本机正确性/性能数据。

### 2.5 验证

- 构造 `gx` 使 `2*sigmoid(tx)-0.5+gx` 落在 0 附近，统计触发比例；
- 对比“无修复”“全量高精度”“选择性修复”的误差分布和时延；
- 如果触发比例 >5–10%，选择性修复可能不划算；如果无修复仍满足容差，修复是多余。

## 3. Tree：half Top-2 粗筛的风险

### 3.1 问题

`TreeLineMinDistance` 的 `M=1024` 路径要把每个点到 1024 条线段的距离归约到最小。全 FP32 计算 1024 次/点很贵。

### 3.2 做法

源码：`TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_pairs.h:752-854`。

1. 把点和线段缩放到 half；
2. 按 8 个 128 段 chunk 计算 half 平方距离；
3. 每个 chunk 用 `WholeReduceMin` + `GatherMask` 取 Top-2（共 16 个候选）；
4. 把这 16 个候选索引搬回 FP32 平面，用 FP32 重算精确距离；
5. 在 16 个候选中选最小。

### 3.3 为什么不能当精确模板

- 没有“被筛掉的候选不可能更优”的证明；
- half 舍入误差可能让真实最小段落出 Top-2；
- 坐标范围 `[-1000,1000]` 时，half 相对误差约 `2^-11`，平方距离误差会放大；
- 固定 Top-2 是启发式，不是保守下界；
- 题目要求精确 argmin 和最小索引 tie；一旦漏候选，后续 FP32 重算也救不回来。

### 3.4 如果要复用

应该改成：

- 用可证明的下界（AABB、投影分离、Lipschitz/误差界）做保守剪枝；
- 对边界/近 tie 候选保留更多候选或精确回退；
- 用 FP32 误差界决定 half 是否可用；
- 给出 no-miss 证明或至少给出覆盖率/反例测试。

### 3.5 另一个数值风险

通用路径用 `Maxs(L, FLT_MIN)` 避免除零（`tree_pairs.h:383`）。题面要求 `L=0` 时用 `||P-A||`；`max(L, FLT_MIN)` 会得到接近 0 的投影，不等于端点距离。复用时应显式 `L==0` 分支。

## 4. GroupedTopk：单调性与分母抵消

### 4.1 数学

源码：`mantuoluozk/kernelswift-2026-triton-kernels`，`task01_grouped_topk/solution.py:14-70`。

- softmax 单调：`softmax(z)_i` 与 `z_i` 同序，所以选组/选专家可以直接在 logits 上做；
- Top-K 重新归一化：

```text
w_i = exp(z_i) / Σ_{k∈TopK} exp(z_k)
```

256 路公共分母 `Σ_j exp(z_j)` 抵消，只需对 8 个选中 logits 做 `exp(z_i-z_0)`。

### 4.2 条件

- scoring 是 softmax 或单调变换；
- `renormalize=True`；
- Top-K 权重只在选中的 K 个内归一化；
- 输出顺序/tie 与参考一致。

### 4.3 负例

- `renormalize=False`：公共分母不抵消，不能省 softmax；
- sigmoid/温度/非单调 scoring：单调性不成立；
- 连续 `argmax` 的 tie 行为与 `torch.topk` 可能不同；
- `E` 很大时，O(E*K) 选择链可能比排序慢。

### 4.4 验证

- 构造 tie（多个专家同分）和极端 logits（很大/很小），对比参考的索引/权重；
- 关闭 `renormalize` 或换 scoring，确认实现会回退/报错，而不是静默算错。

## 5. Sinkhorn：20 轮迭代的 eps 与顺序

### 5.1 数学

源码：`mantuoluozk/kernelswift-2026-triton-kernels`，`task08_hc_split_sinkhorn/solution.py:41-89`。

```text
comb = softmax_rows(raw)
for 20 iterations:
    comb = comb / (row_sum(comb)+eps)
    comb = comb / (col_sum(comb)+eps)
```

参考实现先做一次 row normalize，再做一次 col normalize，然后循环 19 次 row+col（`reference.py:39-46`）；候选在 exp 后直接进入 20 轮 row+col。两者在实数上等价，但：

- `eps` 加在每轮分母上，位置/次数必须一致；
- 初始 row softmax 与第一轮 row normalize 的舍入顺序不同；
- 20 轮后误差会累积，不能只看单轮。

### 5.2 条件

- 固定 4×4、20 轮、`eps=1e-6`；
- 每行独立；
- FP32 标量/寄存器；
- 输出 `pre/post/comb` 的顺序与参考一致。

### 5.3 负例

- `hc_mult=8/16` 时标量爆炸；
- 行数很大时一个 program 一行、1 warp 可能不够；
- `eps` 位置/初始归一化不一致会导致 `1e-7` 级甚至更大误差；
- 只适用于固定 4×4，不能当通用 Sinkhorn。

### 5.4 验证

- 用全 0、全 1、极端大/小、近似置换矩阵等输入对比参考；
- 检查 20 轮后的行/列和、`pre/post` 误差；
- 如果误差随轮数增长，检查 eps/顺序，而不是盲目加精度。

## 6. 通用检查表

| 检查 | 问题 | 证据 |
|---|---|---|
| 主路径精度 | 是否满足题目 FP32 中间/容差？ | 源码 + 误差测试 |
| 敏感 lane | 是否有廉价判据？比例多少？ | 触发比例统计 |
| 高精度回退 | 回退是否覆盖所有敏感情况？ | 反例/边界测试 |
| 近似筛选 | 是否有 no-miss 证明？ | 下界/覆盖测试 |
| tie | 是否保留最小索引/参考顺序？ | 对抗 tie 测试 |
| eps/顺序 | 迭代归一化的 eps 位置/顺序是否一致？ | 逐轮对比 |
| 低精度 cast | FP16/BF16 是否只在允许的输出/中间？ | dtype/容差测试 |
| 溢出/下溢 | 平方、exp、倒数是否可能溢出？ | 极值输入 |

## 7. 负例汇编

1. **Fire 的选择性修复**：敏感 lane 多时标量反噬；阈值/软件 exp 不能照搬。
2. **Tree 的 half Top-2**：无 no-miss 证明，不能用于精确 argmin。
3. **Tree 的 `max(L, FLT_MIN)`**：不等价于 `L==0` 分支。
4. **GroupedTopk 的分母抵消**：只在 softmax + renormalize 下成立。
5. **Sinkhorn 的 eps/顺序**：20 轮累积，位置错了会系统性偏。
6. **52Xayah 的 `torch.allclose`**：只验证了一个规则 shape，不能证明一般 shape/边界。

## 8. 证据边界

- 本文的数值结论来自源码和数学推导；没有本机误差测试。
- Fire/Tree 的官方分数不是数值正确性的证明；Tree 的 half 粗筛尤其不能被“通过比赛”反推为精确。
- 任何选择性精度方案都需要在目标 SoC/CANN 上重新验证 FP32 中间、舍入模式和容差。
