## 算子说明

### 算子功能

点云 voxel 化：3D 目标检测/分割（PointNet++、VoxelNet 类）把连续坐标量化到均匀体素网格，相同体素的点归为同一 cluster，是稀疏卷积（SparseConv3d）与voxel pooling 的前置步骤。本算子对应 pyg_lib.ops.grid_cluster（torch_geometric.nn.pool.grid_cluster 同语义）。

计算规则：

```
cluster[i] = sum_d trunc((pos[i, d] - start[d]) / size[d]) * stride[d]
num_voxels[d] = trunc((end[d] - start[d]) / size[d]) + 1
stride：stride[0] = 1，stride[d] = prod(num_voxels[0:d])
（stride[0] = 1 即第 0 维变化最快、最后一维最慢）
```

### 输入输出规格

| 张量 | 形状 | 数据类型 | 含义 |
|------|------|---------|------|
| pos | (N, D) | float32/float16 | 点坐标 |
| size | (D) | float32/float16 | 每维体素尺寸，各维 > 0 |
| start | (D) | float32/float16 | 网格起点（必选，pos - start ≥ 0） |
| end | (D) | float32/float16 | 网格终点（必选，end ≥ start） |
| out | (N) | int64 | 每点的 cluster 下标（不做连续 id 压缩） |

维度约束：N ∈ [1, 10000000]，D ∈ {2, 3}，prod(num_voxels) < 2^28（线性化不溢出）。start/end 为必选输入；pos - start ≥ 0（除法非负，trunc 与 floor 一致）；输出为原始线性化下标，无需 unique/重映射。

### 属性参数

无

### 输入数据约束（结构化张量生成要求）

CSR 类输入（indptr/rowptr/ptr_*）必须单调不减且末位等于对应张量长度；索引类输入（index/col/COO）必须落在合法区间；COO 输入需按行主序排列且无重复条目。随机图测试数据为幂律度分布（含 hub 节点与孤立点）。

### 语义核对说明

本算子语义已对照参考库源码核对：pyg-lib（`pyg_lib/csrc/`）与 torch_scatter / torch_sparse（`projects/pytorch_scatter`、`projects/sparse1/pytorch_sparse`）。规格以赛题规格表为准，与参考库实现的一致性及差异点已在维度约束与计算规则中注明。

### 参考实现说明

参考实现以 numpy 编写（`grid_cluster.py` 的 `impl` 函数），float32 / float16 原生支持；bfloat16 用例由测试框架以 float32 golden 做精度转换后比对。索引类输出（int64）要求与 golden 逐位一致。

### 功能示例

pos = [[0.1,0.1],[0.6,0.3],[1.1,0.9]]，size = [0.5, 0.5]，start = [0, 0]，end = [2, 2]

输出：

```
[0 1 7]
```

### 任务要求

针对不同 shape 维度，设计算子逻辑，保证算子精度正确（结果满足测试用例中规定的 atol / rtol；索引类输出需逐位一致）。

充分发挥系统带宽能力和并行计算能力，算子性能更优。逐点 trunc 量化 + 乘加，纯带宽型；N ≤ 1e7 时注意 fp32 除法与 int64 乘法精度；D = 2/3 两种维度都要支持，避免模板分支导致的两倍代码体量。

**禁止事项**：禁止回退到 CPU；禁止做 unique/连续 id 压缩（输出原始线性化下标）；禁止改变线性化顺序（stride[0] = 1，第 0 维变化最快）。
