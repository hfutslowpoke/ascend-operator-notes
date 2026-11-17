## 算子说明

### 算子功能

GNN 消息传递的核心算子：GCN/GraphSAGE 等的邻接聚合（归一化邻接矩阵 × 特征矩阵）在稀疏存储下就是 CSR SpMM。稀疏图 (M, N) 与稠密特征 (N, K) 相乘并按行求和聚合。本算子对应 torch_sparse.spmm_sum（SparseTensor @ dense 的 sum 归约）。

计算规则：

```
out[r, :] = sum_{k in [rowptr[r], rowptr[r+1])} value[k] * mat[col[k], :]
空行（rowptr[r+1] == rowptr[r]）out 槽位 = 0
累加建议 float32 精度，输出回转输入 dtype
```

### 输入输出规格

| 张量 | 形状 | 数据类型 | 含义 |
|------|------|---------|------|
| rowptr | (R) | int64 | CSR 行指针，R = M+1，单调不减，rowptr[M]=E |
| col | (E) | int32 | CSR 列索引，取值 [0, N)，行内升序 |
| value | (E) | float32/float16 | 非零值 |
| mat | (N, K) | 与 value 相同 | 稠密矩阵 |
| out | (M, K) | 与 value 相同 | 输出 |

维度约束：M, N ∈ [1, 10000000]，E ∈ [1, 20000000]，K ∈ [1, 4096]。测试数据允许空行；行内 col 升序（确定性累加序）；float32 精度 atol = rtol = 1e-4，float16 精度 atol = rtol = 1e-2。注：torch_sparse 参考实现内核 col 为 int64、value 会转成 mat 的 dtype；本规格 col 采用 int32，value 与 mat 同 dtype。

### 属性参数

无

### 输入数据约束（结构化张量生成要求）

CSR 类输入（indptr/rowptr/ptr_*）必须单调不减且末位等于对应张量长度；索引类输入（index/col/COO）必须落在合法区间；COO 输入需按行主序排列且无重复条目。随机图测试数据为幂律度分布（含 hub 节点与孤立点）。

### 语义核对说明

本算子语义已对照参考库源码核对：pyg-lib（`pyg_lib/csrc/`）与 torch_scatter / torch_sparse（`projects/pytorch_scatter`、`projects/sparse1/pytorch_sparse`）。规格以赛题规格表为准，与参考库实现的一致性及差异点已在维度约束与计算规则中注明。

### 参考实现说明

参考实现以 numpy 编写（`spmm_sum.py` 的 `impl` 函数），float32 / float16 原生支持；bfloat16 用例由测试框架以 float32 golden 做精度转换后比对。索引类输出（int64）要求与 golden 逐位一致。

### 功能示例

rowptr = [0,2,3]，col = [0,1,2]，value = [1,2,3]，mat = [[1,2],[3,4],[5,6]]

输出：

```
[[ 7. 10.]
 [15. 18.]]
```

### 任务要求

针对不同 shape 维度，设计算子逻辑，保证算子精度正确（结果满足测试用例中规定的 atol / rtol；索引类输出需逐位一致）。

充分发挥系统带宽能力和并行计算能力，算子性能更优。经典 SpMM：难点在 E ≤ 2e7 的幂律度分布负载均衡（按行均分会使 hub 行独占单核），需按前缀和非零数切分；K ≤ 4096 时列内并行与 UB 分块；rowptr/col/value/mat 四路数据流读取；fp16/bf16 用 fp32 累加保精度。

**禁止事项**：禁止回退到 CPU；禁止改变行内累加顺序导致的非确定性（按 col 升序累加）；禁止空行输出未初始化。
