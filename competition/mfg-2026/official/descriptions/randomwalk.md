## 算子说明

### 算子功能

图表示学习的基础采样算子：DeepWalk/node2vec 以游走迹替代邻接矩阵做无监督训练，GraphSAINT-RW、PinSAGE 用随机游走构造 mini-batch 邻域。从每个 seed 出发，沿 CSR 图（rowptr/col）游走 walk_length 步得到轨迹。均匀路径（p = q = 1）与 pyg_lib.sampler.random_walk 对应；p/q 偏置为赛题扩展语义（同 node2vec 定义 / torch_cluster.random_walk），见维度约束。

计算规则：

```
out[i, 0] = seed[i]
第 j 步（j = 1..walk_length）：当前节点 v 的邻居区间为 [rowptr[v], rowptr[v+1])
  p == q == 1.0（均匀）：v = col[start + floor(rand * (end - start))]
  node2vec 偏置：候选 x 权重  x==prev(上一跳) -> 1/p；x 与 prev 相邻 -> 1；否则 1/q，
    按权重 CDF 与 rand 选出下一节点（第一步无 prev，权重全 1）
prev = 移动前的当前节点；孤立点（出度 0）停留，v 与 prev 均不变，该步仍消费 rand[i, j-1]
```

### 输入输出规格

| 张量 | 形状 | 数据类型 | 含义 |
|------|------|---------|------|
| rowptr | (R) | int64 | CSR 行指针，R = N+1 |
| col | (E) | int64 | CSR 列索引（邻接表） |
| seed | (B) | int64 | 游走起点，取值 [0, N) |
| rand | (B, walk_length) | float32 | 确定性校验辅助输入（必选），取值 [0,1) |
| out | (B, L) | int64 | 游走轨迹，L = walk_length+1，首列为 seed |

维度约束：N ∈ [1, 10000000]，E ∈ [0, 20000000]（E = 0 即全孤立图：col 为空、rowptr 全零，每步均停留且仍消费 rand），B ∈ [1, 100000]，walk_length ∈ [1, 64]。图含孤立点（停留语义；含 seed 全部为孤立点、每步都停留的用例）；随机性是本算子正确性校验的核心难点：rand 为必选输入，使参考轨迹与待测轨迹逐位可比（性能测试亦提供 rand，内容可为任意 [0,1) 随机流）。p、q ≠ 1 的 node2vec 偏置为赛题在 pyg-lib 之上的扩展语义（按 node2vec 定义实现）；pyg-lib 内核 TORCH_CHECK 仅支持 p = q = 1，偏置需参赛者自行实现。

### 属性参数

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| walk_length | int | 16 | 游走步数 |
| p | float | 1.0 | node2vec 返回参数 |
| q | float | 1.0 | node2vec in-out 参数 |

### 输入数据约束（结构化张量生成要求）

CSR 类输入（indptr/rowptr/ptr_*）必须单调不减且末位等于对应张量长度；索引类输入（index/col/COO）必须落在合法区间；COO 输入需按行主序排列且无重复条目。随机图测试数据为幂律度分布（含 hub 节点与孤立点）。

### 语义核对说明

本算子语义已对照参考库源码核对：pyg-lib（`pyg_lib/csrc/`）与 torch_scatter / torch_sparse（`projects/pytorch_scatter`、`projects/sparse1/pytorch_sparse`）。规格以赛题规格表为准，与参考库实现的一致性及差异点已在维度约束与计算规则中注明。

### 参考实现说明

参考实现以 numpy 编写（`random_walk.py` 的 `impl` 函数），float32 / float16 原生支持；bfloat16 用例由测试框架以 float32 golden 做精度转换后比对。索引类输出（int64）要求与 golden 逐位一致。

### 功能示例

rowptr = [0,2,3,5]，col = [1,2,0,0,2]，seed = [0]，walk_length = 3，rand = [[0.1, 0.7, 0.4]]，p = q = 1.0

输出：

```
[[0 1 0 1]]
```

### 任务要求

针对不同 shape 维度，设计算子逻辑，保证算子精度正确（结果满足测试用例中规定的 atol / rtol；索引类输出需逐位一致）。

充分发挥系统带宽能力和并行计算能力，算子性能更优。控制流完全数据相关（下一跳取决于当前节点的邻接表），是 NPU/向量核上最难向量化的算子：可沿 B 维展开 walker 并行、预取邻接区间；p/q 偏置的权重计算需在片上完成 CDF；孤立点分支不允许多余同步。

**禁止事项**：禁止回退到 CPU；精度测试禁止使用未指定分布的自带随机源（必须消费 rand 输入）；禁止改变轨迹首列为 seed 的约定。
