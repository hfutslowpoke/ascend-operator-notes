## 算子说明

### 算子功能

GNN 邻居聚合中的 min-pooling：MessagePassing 的 aggr=min、图级池化 global_min_pool 都以分段最小值实现。CSR 分段布局下行 r 消费连续切片 src[indptr[r]:indptr[r+1]]，arg_out 记录每列最小值来源位置，用于反池化与梯度路由。本算子对应 pyg_lib.segment_min_csr（torch_scatter.segment_min_csr 同族）。

计算规则：

```
out[r, c]     = min(src[indptr[r]:indptr[r+1], c])
arg_out[r, c] = 段内 argmin（首遇 tie-break：严格小于才更新）
空段（indptr[r+1] == indptr[r]）：out 槽位 = 0，arg_out = E（哨兵值）
```

### 输入输出规格

| 张量 | 形状 | 数据类型 | 含义 |
|------|------|---------|------|
| src | (E, K) | float32/float16 | 分段源数据 |
| indptr | (R) | int64 | CSR 行指针，R = N+1，单调不减，indptr[N]=E |
| out | (N, K) | 与 src 相同 | 段内逐列最小值 |
| arg_out | (N, K) | int64 | 段内 argmin 的 src 行号（全局行号，非段内偏移） |

维度约束：N ∈ [1, 1000000]，E ∈ [0, 5000000]（E = 0 即全空段图：src 为空、indptr 全零，所有段 out = 0、arg_out = 0（= 哨兵 E）），K ∈ [1, 4096]。测试数据允许空段（含连续多空段、全空段图）；tie-break 必须首遇（含取值仅 {-2..2}、全段同值密集的 tie-break 压力用例）；arg_out 输出全局 src 行号。

### 属性参数

无

### 输入数据约束（结构化张量生成要求）

CSR 类输入（indptr/rowptr/ptr_*）必须单调不减且末位等于对应张量长度；索引类输入（index/col/COO）必须落在合法区间；COO 输入需按行主序排列且无重复条目。随机图测试数据为幂律度分布（含 hub 节点与孤立点）。

### 语义核对说明

本算子语义已对照参考库源码核对：pyg-lib（`pyg_lib/csrc/`）与 torch_scatter / torch_sparse（`projects/pytorch_scatter`、`projects/sparse1/pytorch_sparse`）。规格以赛题规格表为准，与参考库实现的一致性及差异点已在维度约束与计算规则中注明。

### 参考实现说明

参考实现以 numpy 编写（`segment_min_csr.py` 的 `impl` 函数），float32 / float16 原生支持；bfloat16 用例由测试框架以 float32 golden 做精度转换后比对。索引类输出（int64）要求与 golden 逐位一致。

### 功能示例

src = [[5,1],[2,8],[7,3]]，indptr = [0, 2, 2, 3]（第 1 行为空段）

输出：

```
out =
[[2. 1.]
 [0. 0.]
 [7. 3.]]
arg_out =
[[1 0]
 [3 3]
 [2 2]]
```

### 任务要求

针对不同 shape 维度，设计算子逻辑，保证算子精度正确（结果满足测试用例中规定的 atol / rtol；索引类输出需逐位一致）。

充分发挥系统带宽能力和并行计算能力，算子性能更优。难点在变长段的负载均衡：真实图度分布为幂律（少量千度 hub + 大量度 1 节点），按 N 均分会导致部分核空转，需按段长或前缀和工作量切分；K 维可并行；fp16/bf16 需按数值比较而非位比较；out 与 arg_out 双写。

**禁止事项**：禁止回退到 CPU；禁止把 tie-break 实现为任遇或末遇；禁止把空段输出为 NaN 或保持未初始化。
