## 算子说明

### 算子功能

图神经网络（GNN）消息传递与子图采样中，需要按边/采样索引把节点特征重排到新的槽位：GraphSAINT、Cluster-GCN 等采样完成后用 gather 恢复子图特征，scatter/segment 聚合的反向传播也依赖对称的 gather 原语。本算子对应 pyg_lib.ops.gather_coo（torch 等价 src[index]）。

计算规则：

```
out[i, :] = src[index[i], :],  i in [0, E)
```

### 输入输出规格

| 张量 | 形状 | 数据类型 | 含义 |
|------|------|---------|------|
| src | (M, K) | float32/float16/int32/int8 | 源特征张量 |
| index | (E) | int32 | gather 索引，取值 [0, M) |
| out | (E, K) | 与 src 相同 | 输出，out[i]=src[index[i]] |

维度约束：M ∈ [1, 1000000]，K ∈ [1, 4096]，E ∈ [1, 4000000]。index 保证合法（无需越界检查），允许重复索引（重复读同一段地址）。五种 dtype 都需支持；越界行为未定义。

### 属性参数

无

### 输入数据约束（结构化张量生成要求）

CSR 类输入（indptr/rowptr/ptr_*）必须单调不减且末位等于对应张量长度；索引类输入（index/col/COO）必须落在合法区间；COO 输入需按行主序排列且无重复条目。随机图测试数据为幂律度分布（含 hub 节点与孤立点）。

### 语义核对说明

本算子语义已对照参考库源码核对：pyg-lib（`pyg_lib/csrc/`）与 torch_scatter / torch_sparse（`projects/pytorch_scatter`、`projects/sparse1/pytorch_sparse`）。规格以赛题规格表为准，与参考库实现的一致性及差异点已在维度约束与计算规则中注明。

### 参考实现说明

参考实现以 numpy 编写（`gather_coo.py` 的 `impl` 函数），float32 / float16 原生支持；bfloat16 用例由测试框架以 float32 golden 做精度转换后比对。索引类输出（int64）要求与 golden 逐位一致。

### 功能示例

src = [[1,2],[3,4],[5,6]]（float32），index = [2,0,2]

输出：

```
[[5. 6.]
 [1. 2.]
 [5. 6.]]
```

### 任务要求

针对不同 shape 维度，设计算子逻辑，保证算子精度正确（结果满足测试用例中规定的 atol / rtol；索引类输出需逐位一致）。

充分发挥系统带宽能力和并行计算能力，算子性能更优。纯带宽型算子：E·K 连续写 + 按 index 的随机读，K ≤ 4096 时单行最大 16 KB，需考虑 L2/UB 缓冲与多核按 E 切分的负载均衡；int8 行宽最小时注意搬运粒度与向量化效率；E = 4e6、K = 4096 为规模上限。

**禁止事项**：禁止回退到 CPU；禁止对 index 排序或改变输出行顺序；禁止改变 gather 语义（越界检查不需实现）。
