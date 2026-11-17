## 算子说明

### 算子功能

点云处理的基础邻域算子：PointNet++ 集合抽象层的球查询（ball query）、KPConv/3D 目标检测与分割的局部图构建都以 radius 搜索为前置步骤；也用于分子建模与物理模拟的近邻配对。对每个查询点 y[i] 收集距离 r 内的 x 邻居。本算子与 pyg_lib.ops.radius 对应（邻居成员语义一致）；输出顺序由赛题扫描序契约定义，见维度约束。

计算规则：

```
对每个查询点 y[i]（按 i 升序）：
  按 j = x 批内起始序号升序扫描，d2 = ||x[j] - y[i]||^2（float32 计算）
  若 d2 <= r*r 且未被 ignore_same_index 剔除，收集 (i, j)；剔除条件 = x 的全局序号等于查询全局序号 i（同一 batch 内即 j = i - ptr_x[b]，多 batch 时批内 j 不等于批内 i）
  收满 max_num_neighbors 个后停止（first-K 截断；命中数超过 max_num_neighbors 时按扫描序保留前 K——不是按距离取最近 K）
查询 i 的索引对写入其专属槽位区间 [i·max_num_neighbors, (i+1)·max_num_neighbors)（扫描序排列）
out[0, s] = 查询索引 i，out[1, s] = 参考索引 j；未命中槽位填充 (-1, -1)
（定槽位输出：每查询命中数 ≤ max_num_neighbors，总槽位 M×max_num_neighbors 与数据无关）
```

### 输入输出规格

| 张量 | 形状 | 数据类型 | 含义 |
|------|------|---------|------|
| x | (N, D) | float32/float16 | 参考点集 |
| y | (M, D) | float32/float16 | 查询点集 |
| ptr_x | (Rx) | int64 | x 的 batch 边界，Rx = Bx+1，单调不减且 ptr_x[Bx] = N；单 batch 传 [0, N] |
| ptr_y | (Ry) | int64 | y 的 batch 边界，Ry = By+1，单调不减且 ptr_y[By] = M；单 batch 传 [0, M] |
| out | (2, E) | int64 | 定槽位索引对，E = M×max_num_neighbors（查询 i 独占槽位区间 [i·max_num_neighbors, (i+1)·max_num_neighbors)）：row 0 = 查询索引，row 1 = 参考 x 索引；未命中槽位 = (-1, -1) |

维度约束：N, M ∈ [1, 1000000]，M × max_num_neighbors ≤ 4000000（输出槽位上界），D ∈ {2, 3}，r ≥ 0（r = 0 时无命中，该查询槽位全为 (-1,-1)）。ptr_x/ptr_y 为必选输入，允许空 batch（ptr 相邻元素相等）；输出契约：查询 i 的命中写入其专属槽位区间、批内按 x 扫描序（升序）截断，未命中槽位为 (-1, -1)，与 golden 参考实现逐位可比（该扫描序为赛题确定性契约；pyg-lib CPU 内核为 KD-tree 检索且 sorted = false，不保证此顺序）。ignore_same_index 用于 x 与 y 为同一点集时剔除自身配对。

### 属性参数

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| r | float | 1.0 | 搜索半径 |
| max_num_neighbors | int | 32 | 每个查询点最多收集的邻居数 |
| ignore_same_index | bool | False | 跳过 j == i 的索引对 |

### 输入数据约束（结构化张量生成要求）

CSR 类输入（indptr/rowptr/ptr_*）必须单调不减且末位等于对应张量长度；索引类输入（index/col/COO）必须落在合法区间；COO 输入需按行主序排列且无重复条目。随机图测试数据为幂律度分布（含 hub 节点与孤立点）。

### 语义核对说明

本算子语义已对照参考库源码核对：pyg-lib（`pyg_lib/csrc/`）与 torch_scatter / torch_sparse（`projects/pytorch_scatter`、`projects/sparse1/pytorch_sparse`）。规格以赛题规格表为准，与参考库实现的一致性及差异点已在维度约束与计算规则中注明。

### 参考实现说明

参考实现以 numpy 编写（`radius.py` 的 `impl` 函数），float32 / float16 原生支持；bfloat16 用例由测试框架以 float32 golden 做精度转换后比对。索引类输出（int64）要求与 golden 逐位一致。

### 功能示例

x = [[0,0],[1,0],[5,5]]，y = [[0,0]]，r = 1.5，max_num_neighbors = 2，ptr_x = [0, 3]，ptr_y = [0, 1]

输出：

```
[[0 0]
 [0 1]]
```

### 任务要求

针对不同 shape 维度，设计算子逻辑，保证算子精度正确（结果满足测试用例中规定的 atol / rtol；索引类输出需逐位一致）。

充分发挥系统带宽能力和并行计算能力，算子性能更优。算力压力主要在 O(M·N) 距离计算（D=3 时每对 9 次乘加），N, M ≤ 1e6 需分块复用 x/y 的片上驻留与向量化距离计算；first-K 截断允许早退；定槽位输出（每查询专属槽位区间）可按查询直接并行写，无需原子写或两段式；batch 边界要避免跨 batch 配对。

**禁止事项**：禁止回退到 CPU；禁止按距离排序或改变扫描序输出契约；禁止跨 batch 配对；禁止未命中槽位填充 (-1, -1) 以外的值（含 0）。
