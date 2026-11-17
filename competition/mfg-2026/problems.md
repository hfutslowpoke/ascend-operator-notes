# 15 道赛题逐题说明与源码缺口

本页只写“能从公开源码/文档确认的语义”和“下一步要验证什么”。没有官方题面的题，明确写“未知”，不补造 shape/容差。

## 01 ComplexAbs（仿真）

- **语义**：复数逐元素模长，`y = sqrt(re² + im²)`。
- **dtype**：COMPLEX32→FLOAT16、COMPLEX64→FLOAT、COMPLEX128→DOUBLE；输出类型由输入推导，`Tout` 属性保留但不用。
- **约束**：ND，秩≤8，不支持空 Tensor；A2 不支持，950/A3 支持。
- **源码**：[`ops-math/math/complex_abs`](ops-math/README.md)；`op_kernel/arch35/complex_abs_dag.h`、`op_host/arch35/complex_abs_tiling_arch35.cpp`。
- **910B 结论**：官方实现不能直接上 910B；可参考“先平方和再 sqrt”的 DAG 写法，但需要自己写 arch22/Vector 版本。
- **下一步**：确认题面是否要求 COMPLEX128（910B 可能无 FP64 向量支持）、空 Tensor/特殊值语义。

## 02 ComplexDiv（仿真）

- **语义**：复数逐元素除法，`(a+bi)/(c+di)`；题面可能还要求特殊值/极值处理。
- **源码参考**：`ops-solver/src/utils/complex_solver_kernel.hpp:49-80` 的 `Quotient/Divide`：按 `max(|c|,|d|)` 缩放，FMA 计算分子/分母误差，两次补偿商。
- **约束**：该函数在 `__NPU_ARCH__ == 3510`（950）守卫内，且是 LU/QR 的内部 helper，不是独立 elementwise 算子；910B 不能直接编译。
- **910B 结论**：算法可迁移，代码不可直接复用；需要写 910B Vector 版本并确认 `float`/`complex64` 布局。
- **下一步**：问题面是否要求 COMPLEX32/128、NaN/Inf/0 除、是否允许 Newton 修正。

## 03 ComplexExp（仿真）

- **语义（推测，需题面）**：`exp(a+bi) = exp(a)·(cos b + i·sin b)`。
- **公开源码**：**未找到**专门的 Ascend C ComplexExp 实现。
- **可组合的官方件**：`ops-math` 的 `exp`、`sin`、`cos`（math 类）与 `experimental/math/complex_v2` 的实虚交错布局；需要确认这些算子在 910B 的 dtype/精度。
- **风险**：极值 `exp(a)` 溢出、`b` 的范围规约、复数内存布局（实虚交错 vs 分离平面）都会改变结果；没有题面不能定。
- **下一步**：拿到题面后，决定是“一个融合 kernel”还是“exp + sincos + ComplexV2 组合”，再写 910B 版本。

## 04 FusedPropStep（仿真）

- **语义**：**未知**。名称指向“融合传播步”（propagation step），但没有公开公式、输入输出或测试。
- **公开源码**：**未找到**。`ops-fft`、`ops-math`、`ops-solver`、`mat-chem-sim-pred` 都没有同名算子；`mat-chem-sim-pred` 的 FNO/PINN 是 PDE 推理，不是本题。
- **风险**：不能根据名字猜“传播”是图传播、物理场传播还是数值积分；猜错会直接改变数学语义。
- **下一步**：必须拿到官方题面/参考实现；在此之前只保留缺口，不写虚假候选。

## 05 FourierTransform2D（仿真）

- **语义**：二维 C2C DFT，输入/输出复数按行主序、实虚交错；`aclfftPlan2d(batch,nx,ny,ACLFFT_C2C)` + `aclfftExecC2C`。
- **约束**：文档明确仅 Ascend 910B 支持，950 不支持；`nx,ny` 只能取 {32,64,128} 的组合；batch>0；C2C，不含 R2C/C2R。
- **源码**：[`ops-fft/src/fft2_d`](ops-fft/README.md)；`arch32/dd/fft2_d_dd.cpp:17-29` 尺寸表、`:31-62` P/Q 矩阵、`:64-112` 入口；`arch32/fft2_d_exec_api.cpp:18-60` 校验/分派。
- **机制**：2D DFT 拆成 P·X·Q 两次矩阵乘（Cube），P/Q 由 Host 生成；复数实虚分离用 `vreducev2`；batch 循环 + 多核切分 + L1/L0 双缓冲。
- **910B 结论**：**本机 PASS（32×32）**。CANN 9.1.0 构建成功；32×32 batch=1、冲激输入与解析 DFT 对照，5 次 best 0.1537 ms、max_abs=9.086799e-08、bad=0；64/128、batch>1、backward 未跑。
- **负例**：不支持任意 N/质数尺寸；不支持 950；Host 端 P/Q 是 O(N²) 内存；作者没有性能数据；本机只覆盖 32×32 单 batch forward。

## 06 GatherCoo（图）

- **语义**：按有序 COO 索引扩展源行，`out[..., e, ...] = src[..., index[..., e], ...]`；index 必须 int64、非降序。
- **源码**：`ops-gnn/csrc/npu/gather_coo/op_kernel/arch35/gather_coo_kernel.h:24-110`（slab 快路径/通用 tile/输出缓存）。
- **约束**：`ops-gnn` 的 `gather_coo` 仅 arch35（950）；910B 没有 arch22 实现。
- **替代**：`ops-sparse/sparse/gather` 有 arch22，但它是“dense vector 按稀疏索引 gather”，不是 COO 行扩展；不能直接替代。
- **910B 结论**：参考算法（重复行 slab、按 tile 输出），但需自己写 arch22；`ops-sparse` 的 gather 只覆盖部分语义。
- **下一步**：问题面是否要求重复索引、空行、rank 1–8、非连续输入；这些决定能否复用 slab 优化。

## 07 SegmentMinCsr（图）

- **语义**：沿 CSR `indptr` 指定的分段做最小值归约；空段填充 dtype 最小值（省略 `out` 时）。
- **源码**：`ops-gnn/csrc/npu/segment_csr/op_kernel/arch22/segment_csr.cpp:22-61`；同一 `SegmentReduceDevice<T, SegmentReduce, HasArg>` 覆盖 min/max/sum；`segment_csr_common.h:35-61` 的 int64 行拷贝。
- **支持**：`ops-gnn` 明确 A2/A3（arch22）与 950；segment 系列在 910B 可用。
- **机制**：按 `tileRows × tileK` 流式读 CSR 行，`tileK` 由 UB/容量规划决定；多 AIV 按 unit 切分；min/max 极值路径与 sum 的有序浮点路径分开；空段填充。
- **负例**：`indptr` 必须非降序 int64；极值类型/NaN 语义要按题面确认；不要假设 `tileK` 对所有 shape 最优。
- **下一步**：在 910B3 构建 `ops-gnn` 的 segment 测试，跑 min/max/sum 与空段。

## 08 RandomWalk（图）

- **语义**：COO 图上的均匀或 node2vec 偏置随机游走；输入 row/col/start，输出 `[S, walk_length+1]` 节点序列，可选边序列。
- **源码**：`ops-gnn/csrc/npu/random_walk/op_kernel/arch35/random_walk.cpp`（本次 raw 拉取 403，路径已由 tree 确认）、`random_walk.h:18-33`、`common/philox.h`。
- **支持**：arch35（950）only；910B 没有 arch22 实现。
- **机制**：Philox 计数器 RNG；按源节点分组/排序；node2vec 的 p/q 权重；边序列可选。
- **负例**：随机性必须固定种子；自环/重复边/未排序输入的语义要按题面；不能把 950 实现直接搬 910B。
- **下一步**：拿到题面确认 p/q、coalesced、return_edge_indices；再决定是否移植 Philox + 分组采样。

## 09 SpmmSum（图）

- **语义**：CSR 稀疏矩阵 × 稠密矩阵，`C = alpha·A·B + beta·C`；本题只关心 sum 聚合，但库支持 sum/max/min/mean。
- **源码**：
  - `ops-gnn/csrc/npu/common/spmm/spmm_utils.h:210-241` 的 `SpmmReduce`/`SpmmReducer`；`spmm`/`bspmm` 在 A2/A3 可用。
  - `ops-sparse/sparse/spmm/arch22/spmm_host.cpp:201-330` 的 dtype/行分块；`kernels/spmm_kernel_group_axpy.h` 的 group axpy；fp32 Kahan 路径。
- **支持**：`ops-gnn` 与 `ops-sparse` 都明确 A2/A3。
- **机制**：CSR 行分块 + 按行组 axpy；稠密 B 的 N/K 布局决定快路径；fp32 高精度用 Kahan 补偿；complex64 有单独路径。
- **负例**：A2 的 dtype 组合比 950 少；`CSR_ALG3` 只支持非转置；零维/空行语义要按 README；不要假设所有 alg 都快。
- **下一步**：在 910B3 构建 `ops-sparse` SpMM 测试，先 fp32 再 fp16/bf16；记录 `nnz`/N/K 对行分块的影响。

## 10 Spspmm（图）

- **语义**：CSR × CSR，`C' = alpha·A·B + beta·C`；输出 CSR 列索引严格升序、无重复坐标；数值抵消的显式零保留。
- **源码**：`ops-sparse/sparse/spgemm/README.md:1-40`（多阶段 API）；`arch22/spgemm_host.cpp:654-...`（symbolic）、`:1066-...`（compute）；`spgemm_symbolic_kernel.cpp`、`spgemm_numeric_kernel.cpp`；`spgemm_merge_row_fastpaths.h`。
- **支持**：A2/A3/950；本库对 A2 的 `spgemm` 有 arch22 实现。
- **机制**：WorkEstimation 统计中间乘积 → EstimateMemory 规划 workspace → Compute 做 symbolic（rowOffsets）+ numeric（merge/去重）→ Copy 写输出；输出按行合并，重复坐标相加。
- **负例**：`numProds` 不是最终 `nnz(C)`；输出 CSR 需要调用方分配/动态更新；大 `nnz` 可能超 INT32；多阶段状态机不能跳步。
- **下一步**：先用小图（对角/链/随机）验证输出结构，再跑 910B3；记录 `numProds/nnz/workspace`。

## 11 Radius（点云）

- **语义**：对 Y 中每个点，找同 batch 内 X 中距离 ≤ r 的点；超过 `max_num_neighbors` 时保留扫描顺序前 K 个；输出 `[2,E]` int64。
- **源码**：`DrivingSDK/kernels/radius/op_host/radius.cpp:1-100`、`op_kernel/radius.cpp`、`docs/zh/api/context/radius.md`。
- **支持**：文档明确 A2/A3/950；Ndim<9、batch≤1024、单 batch 点数≤512。
- **机制**：按 batch 分核；半径平方比较；每 batch 限 512 点以便 UB 驻留；输出邻居对；超过 K 截断。
- **负例**：不支持任意大 batch/点数；Ndim≥9 不行；输出 E 动态、需要 workspace；排序/去重语义按文档。
- **下一步**：910B3 构建/调用 `mx_driving.radius`，与 CPU 暴力搜索对照。

## 12 GridCluster（点云）

- **语义**：把点坐标量化到规则网格，输出线性 int64 网格 ID（torch_cluster 兼容）。
- **源码**：`ops-gnn/csrc/npu/grid_cluster/op_kernel/arch35/grid_cluster.cpp:15-120`（subnormal/缩放/索引）、`:135-221`（3D SIMT/宽维）；`grid_cluster.h`。
- **支持**：arch35（950）only；910B 没有 arch22。
- **机制**：按 `floor((x-origin)/size)` 计算每维索引，再按各维 radix 组合成线性 ID；对 subnormal/极小 size 做缩放和倒数快路径；多核按点切分。
- **负例**：size 极小/非均匀、负坐标、int64 溢出、空点集；950-only，不能直接上 910B。
- **下一步**：题面确认网格定义/顺序；再决定是否移植到 arch22。

## 13 FurthestPointSample（点云）

- **语义**：从 `[B,N,3]` 点云中迭代选 `num_points` 个最远点，输出 `[B,num_points]` int32 索引。
- **源码**：`DrivingSDK/kernels/furthest_point_sampling/op_kernel/furthest_point_sampling.cpp:115-208`（初始化/迭代/距离更新）；`op_host/furthest_point_sampling.cpp:1-100`（tiling/UB）；`docs/zh/api/context/furthest_point_sampling.md`。
- **支持**：文档明确 A2/A3/950；总元素 `B*N*3 ≤ 383166`。
- **机制**：首个采样点固定（索引 0）；维护每个点到已选集合的最近距离；每轮计算到新选中点的距离、取 min、reduce max 选下一个；点云大于 UB 时按 `pieces` 分片；fp16/bf16 先转 fp32 算距离；多核按 batch。
- **负例**：总元素上限；首点约定（0）会影响结果；分片会多次读；没有作者性能数据。
- **下一步**：910B3 构建/调用，和 CPU 参考逐索引对照。

## 14 BallQuery（点云）

- **语义**：对每个 query center，在球壳 `[min_radius, max_radius)`（或 d²=0）内按点序取前 `sample_num` 个索引；不足用首个命中填充；零命中全 0。
- **源码**：
  - 直接版：`ops-math/conversion/ball_query/op_kernel/arch35/ball_query_simt.h`（SIMT grid-stride）；`op_host/arch35/ball_query_tiling.cpp`；README 声称 A2 √，但代码/config 只有 arch35，存在冲突。
  - 910B 可用版：`ops-math/conversion/stack_ball_query`，`op_kernel/stack_ball_query.cpp:17-40`、`stack_ball_query.h:28-140`、`op_host/config/ascend910b/`。
- **支持**：`stack_ball_query` 明确 A2/A3；直接 `ball_query` 的 A2 支持需本机构建确认。
- **机制**：Stack 版用 2D 堆叠布局 + batch counts，按 center 分核，流式读 xyz 片段，计算距离并收集索引；直接版用 SIMT 每线程多 query，距离平方比较，首命中填充，无 atomic。
- **负例**：零命中输出全 0 与“命中第 0 点”不可区分；`ball_query` README/代码支持矩阵冲突；`stack_ball_query` 输入是堆叠 2D + batch counts，不是 `[B,3,N]`。
- **下一步**：优先构建 `stack_ball_query`（有 910B 配置）；再核对直接 `ball_query` 是否真的能编 A2。

## 15 CylinderQuery（机器人抓取）

- **语义**：对每个 query point，用 3×3 旋转把点云变到局部坐标，在圆柱 `radius`、高度 `[hmin,hmax]` 内取前 `nsample` 个点；Python 层排序并截断，未命中用首索引填充。
- **源码**：`DrivingSDK/kernels/cylinder_query/op_host/cylinder_query.cpp:1-120`（tiling/UB/属性）、`op_kernel/cylinder_query.cpp:60-102`（任务/分块/流水）、`:300-350`（旋转/距离/compare/select）、`mx_driving/ops/cylinder_query.py:29-50`（sort/truncate/fill）。
- **支持**：DrivingSDK 面向 A2/A3/950；CylinderQuery 没有 arch35-only 分支，host/kernel 走通用路径；文档未单独列支持表，**A2 需本机构建确认**。
- **机制**：每核处理若干 query；点云按 UB 分块（双缓冲）；`x' = R·(p-center)`，用 `y'²+z'² < r²` 与 `hmin < x' < hmax` 生成 mask；`Select` 写候选索引；输出后 Python `sort` + 取前 nsample + 用首索引填充 miss。
- **负例**：`nsample` 受 UB 限制；rot 必须是 9 元素 3×3；输出先给全量候选再由 Python 截断，端到端包含排序开销；没有作者性能数据。
- **下一步**：910B3 构建 `cylinder_query`，与 CPU/文档公式对照；确认输出 `[B,M,N]` 的语义和 Python 截断行为。
