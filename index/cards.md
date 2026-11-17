# 阅读卡

由 `tools/build_index.py` 从 `index/catalog.json` 生成；不要手改。
每张卡给出“一句话机制 + 先读文件 + 本机状态 + 边界”，用于 agent 快速定位，不替代条目 README。

## competition

### `GITCODE-RECOMPUTE-WU`

- **recompute_w_u (GDN chunked attention)** · `attention-recompute-gemm` · 本机 `fail`
- 一句话：MIX kernel: AIV preprocesses A/beta/g and packs right matrices, AIC does Matmul or hand-written Mmad+Fixpipe. Group head/chunk tasks (up to 4 heads x 4 chunks), reuse GQA K in L0B, multi-slot workspace (depth 5/3), D128 four-slot path, staged output for non-16
- 先读：[README](../competition/cann-ops-competitions/README.md) · [source.json](../competition/cann-ops-competitions/source.json)
- 源码入口：`01_official/cann-ops-ladder-2026/July/recompute_w_u/submissions/Delicate02/code/op_kernel/recompute_wu.h:19-83,143-199,228-268,591-700,1025-1105,1313-1395`；`01_official/cann-ops-ladder-2026/July/recompute_w_u/submissions/Delicate02/code/op_host/recompute_wu_tiling.cpp:46-67,204-345,346-390,408-441`；`01_official/cann-ops-ladder-2026/July/recompute_w_u/submissions/Delicate02/README.md:60-76`
- 适用条件：B=1；BT=64；K/V<=256；GQA H%Hg==0
- 负例/风险：compile success != runnable: 507057 on CANN 9.1.0 for 5 shapes；B!=1 rejected by host；BT!=64 rejected

### `GITCODE-LINALG-LSTSQ`

- **LinalgLstsq (batched least squares, driver=gels)** · `batched-least-squares` · 本机 `partial`
- 一句话：Shape-specific tiling keys (tiny15/16, packed50/64, shared QR 200/256, generic square/rectangular). FP32 TwoSum/TwoProduct split high/low, compensated tile products and residual reductions; iterative refinement using the original equation; pivot/rank/finite gu
- 先读：[README](../competition/elec-ops-inspection/README.md) · [source.json](../competition/elec-ops-inspection/source.json)
- 源码入口：`2026_elec_operator_competition/preliminary/submissioins/XDU_XDU-BTD/code/lstsq/op_host/linalg_lstsq.cpp:24-91`；`2026_elec_operator_competition/preliminary/submissioins/XDU_XDU-BTD/code/lstsq/op_kernel/linalg_lstsq.cpp:25-90,160-335`；`2026_elec_operator_competition/preliminary/submissioins/XDU_XDU-BTD/code/lstsq/op_kernel/lstsq_compensated_tile_product.h:17-40`
- 适用条件：FP32；column full rank；M>=N；M<=1024
- 负例/风险：no author per-case timing/baseline；CANN 9.1 direct build fails on two ReinterpretCast dependent template names；multi-batch (16x16 b2, 50x50 b4) fails 507035

### `HANSEN-2026POWER-FIRE-R2`

- **FireCandidateDecodeFusion (rank2)** · `candidate-decode-fusion` · 本机 `not-run`
- 一句话：Host 按 B*N/核数选 tile；设备把 AoS 6 字段转 SoA 平面，一次 sigmoid 后完成 5 字段解码；对中心坐标抵消的少数 lane 用软件 53 位 exp + 正确舍入倒数修复；静态地址/索引表减少 scalar
- 先读：[README](../competition/hansen-2026-power/README.md) · [source.json](../competition/hansen-2026-power/source.json)
- 源码入口：`archives/2026-electric-power/extracted/FireCandidateDecodeFusion/rank2_submission_1176342/code/op_host/fire_host_plan.h:10-120`；`archives/2026-electric-power/extracted/FireCandidateDecodeFusion/rank2_submission_1176342/code/op_kernel/fire_candidate_decode_fusion.cpp:77-195`；`archives/2026-electric-power/extracted/FireCandidateDecodeFusion/rank2_submission_1176342/code/op_kernel/device_fire.h:105-143,272-365`
- 适用条件：小字段 AoS 逐元素变换；敏感 lane 比例低；UB 能容纳多个平面/静态表；目标 SoC 支持 vgather/scatter 内建
- 负例/风险：敏感 lane 多时标量修复反噬；N/B 很小时平面化/静态表准备不划算；私有内建/静态表与 CANN/SoC 绑定

### `MFG2026-MATH-COMPLEX-ABS`

- **ComplexAbs** · `complex-elementwise` · 本机 `not-run`
- 一句话：实虚平方和再 sqrt；超大值先 SCALE_UP/SCALE_DOWN；Compare/Select 处理 |re|/|im|；按 dtype pack 输出
- 先读：[README](../competition/mfg-2026/ops-math/README.md) · [source.json](../competition/mfg-2026/ops-math/source.json)
- 源码入口：`competition/mfg-2026/ops-math/source/math/complex_abs/README.md:1-60`；`competition/mfg-2026/ops-math/source/math/complex_abs/op_kernel/arch35/complex_abs_dag.h:23-178`
- 适用条件：950/A3 only；complex32/64/128；no empty tensor；rank<=8
- 负例/风险：A2 not supported；COMPLEX128 on 910B unclear；NaN/Inf semantics

### `MFG2026-SOLVER-COMPLEX-DIV`

- **ComplexDiv (reference)** · `complex-elementwise` · 本机 `not-run`
- 一句话：按 max(|c|,|d|) 取 2 的幂缩放；归一化后算分母/分子；FMA 记录乘积/分母误差；remainder 修正商；虚部按 bc-ad
- 先读：[README](../competition/mfg-2026/ops-solver/README.md) · [source.json](../competition/mfg-2026/ops-solver/source.json)
- 源码入口：`competition/mfg-2026/ops-solver/source/src/utils/complex_solver_kernel.hpp:49-80`；`competition/mfg-2026/ops-solver/source/docs/zh/complex_solver_migration.md:1-20`
- 适用条件：950 SIMT only；complex64 helper；not an elementwise op；no aclnn API
- 负例/风险：not 910B-buildable；no standalone op；complex32/128 not covered

### `GITCODE-CONCAT`

- **S9 Concat (dynamic tensor list)** · `concat-layout` · 本机 `offline-only`
- 一句话：Host normalizes dynamic inputs to rows x n_i, then chooses row/column split, DMA/Gather assembly, input split or output linear split. Gather path packs contiguous rows per input, builds a byte-offset table, gathers into a full row, writes once. Column bands re
- 先读：[README](../competition/cann-ops-competitions/README.md) · [source.json](../competition/cann-ops-competitions/source.json)
- 源码入口：`01_official/operator-challenge/cann-ops-challenge-S9/submissions/Concat/BatchLions/op_host/concat.cpp:55-75,240-330,440-532,625-680`；`01_official/operator-challenge/cann-ops-challenge-S9/submissions/Concat/BatchLions/op_kernel/concat.cpp:264-630,909-1071,1075-1445`；`01_official/operator-challenge/cann-ops-challenge-S9/submissions/Concat/BatchLions/op_host/s9_concat_layout.h:96-117,163-241,355-435,603-696`
- 适用条件：dynamic tensor list <=256；CANN 8.5.0 target；UB budget/alignment satisfied；Gather path needs 16/32-bit dtype and nOut limit
- 负例/风险：balanced column bands slower 1.35% on G16 246-input case；edge trimming slower 8.0%/7.9%/1.0% on three shapes；bulk TilingData header fetch slower 8%

### `MFG2026-FFT2D`

- **FourierTransform2D (fft2_d C2C)** · `fourier-transform` · 本机 `pass`
- 一句话：Host 生成 P/Q DFT 矩阵；设备按 P·X·Q 做两次 Cube 矩阵乘；复数实虚用 vreducev2 分离，矩阵乘后合并写回；尺寸专用 batch 循环 + 多核/L1/L0C 双缓冲
- 先读：[README](../competition/mfg-2026/ops-fft/README.md) · [source.json](../competition/mfg-2026/ops-fft/source.json)
- 源码入口：`competition/mfg-2026/ops-fft/source/docs/zh/API_Reference/FFT_2D.md:1-70`；`competition/mfg-2026/ops-fft/source/src/fft2_d/arch32/fft2_d_exec_api.cpp:18-60`；`competition/mfg-2026/ops-fft/source/src/fft2_d/arch32/dd/fft2_d_dd.cpp:17-29,31-62,64-112`
- 适用条件：Ascend 910B only；nx,ny ∈ {32,64,128}；C2C only；batch>0
- 负例/风险：950 unsupported；arbitrary/prime sizes unsupported；no R2C/C2R

### `HANSEN-2026POWER-TREE-R1`

- **TreeLineMinDistance (rank1)** · `geometric-argmin` · 本机 `not-run`
- 一句话：Host 按 M/N 退化维度选 PATH；线段 AoS→SoA，D/L/1/L 按 batch 外提；点 tile 向量化平方距离，64 段 chunk 归约 + 跨 chunk 成对归约，sqrt 后置；M=1024 路径用 half Top-2 粗筛 + FP32 重算 16 候选
- 先读：[README](../competition/hansen-2026-power/README.md) · [source.json](../competition/hansen-2026-power/source.json)
- 源码入口：`archives/2026-electric-power/extracted/TreeLineMinDistance/rank1_submission_1175341/code/op_host/tree_line_min_distance.cpp:438-496`；`archives/2026-electric-power/extracted/TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_line_min_distance.cpp:550-659`；`archives/2026-electric-power/extracted/TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_pairs.h:221-515,752-854`
- 适用条件：存在 M=1/N=1 等退化维度；线段不变量可在 batch 内复用；tie 需最小索引；尾块非 32 对齐
- 负例/风险：half Top-2 无 no-miss 证明，不能用于精确 argmin；max(L,FLT_MIN) 不等价 L==0；大量固定 PATH/UB 绝对偏移

### `MFG2026-GNN-GATHER-COO`

- **GatherCoo** · `graph-gather` · 本机 `not-run`
- 一句话：slab 快路径处理连续重复行；通用 tile 逐行拷贝；输出 slab 缓存/批量写出；纯前向位拷贝
- 先读：[README](../competition/mfg-2026/ops-gnn/README.md) · [source.json](../competition/mfg-2026/ops-gnn/source.json)
- 源码入口：`competition/mfg-2026/ops-gnn/source/csrc/npu/gather_coo/op_kernel/arch35/gather_coo_kernel.h:24-110`；`competition/mfg-2026/ops-gnn/source/docs/zh/api_reference.md:533-563`
- 适用条件：950/arch35 only；index int64 nondecreasing；no backward；rank 1-8
- 负例/风险：not 910B-buildable；no COO alternative in ops-sparse (dense gather only)；index ordering required

### `MFG2026-SPARSE-GATHER`

- **Gather (dense gather)** · `graph-gather` · 本机 `not-run`
- 一句话：按索引从稠密向量取值写 sparse values；raw 快路径跳过描述符创建/销毁；idxBase 0/1
- 先读：[README](../competition/mfg-2026/ops-sparse/README.md) · [source.json](../competition/mfg-2026/ops-sparse/source.json)
- 源码入口：`competition/mfg-2026/ops-sparse/source/sparse/gather/README.md:1-60`；`competition/mfg-2026/ops-sparse/source/sparse/gather/arch22/gather_host.cpp:1-120`
- 适用条件：A2/A3；dense gather only；idx int32；idxBase 0/1
- 负例/风险：not GatherCoo；no COO row expansion；index range/duplicates

### `MFG2026-GNN-RANDOM-WALK`

- **RandomWalk** · `graph-random-walk` · 本机 `not-run`
- 一句话：Philox 计数器 RNG；按源节点分组/排序；node2vec p/q 权重；输出节点/边序列
- 先读：[README](../competition/mfg-2026/ops-gnn/README.md) · [source.json](../competition/mfg-2026/ops-gnn/source.json)
- 源码入口：`competition/mfg-2026/ops-gnn/source/csrc/npu/random_walk/op_kernel/arch35/random_walk.h:18-33`；`competition/mfg-2026/ops-gnn/source/docs/zh/api_reference.md:688-730`
- 适用条件：950/arch35 only；fixed seed for reproducibility；coalesced input semantics；p/q>0
- 负例/风险：not 910B-buildable；stochastic result；self-loop/duplicate edge semantics

### `MFG2026-GNN-SEGMENT-MIN-CSR`

- **SegmentMinCsr** · `graph-segment-reduction` · 本机 `not-run`
- 一句话：模板 SegmentReduce 同时覆盖 min/max/sum；按 tileRows×tileK 流式读 CSR 行；多 AIV 按 unit 切分；极值与有序浮点路径分开；空段填充
- 先读：[README](../competition/mfg-2026/ops-gnn/README.md) · [source.json](../competition/mfg-2026/ops-gnn/source.json)
- 源码入口：`competition/mfg-2026/ops-gnn/source/csrc/npu/segment_csr/op_kernel/arch22/segment_csr.cpp:22-61`；`competition/mfg-2026/ops-gnn/source/csrc/npu/segment_csr/op_kernel/arch22/segment_csr_common.h:35-61`；`competition/mfg-2026/ops-gnn/source/docs/zh/api_reference.md:73-140`
- 适用条件：A2/A3 arch22；int64 indptr nondecreasing；min/max/sum shared template；empty segment fill
- 负例/风险：NaN/extreme semantics per题面；tileK tuning；empty segment fill

### `HANSEN-2026POWER-SPLINE-R1`

- **SplineConv (rank1)** · `graph-spline-conv` · 本机 `not-run`
- 一句话：把逐边 x@W[wi] 重排为按离散 wi 先聚合 x*basis，再对 4 个权重矩阵做一次 4C×16 向量乘；目标节点链表遍历；Host 按 UB/稀疏族选 single/sparse/generic
- 先读：[README](../competition/hansen-2026-power/README.md) · [source.json](../competition/hansen-2026-power/source.json)
- 源码入口：`archives/2026-electric-power/extracted/SplineConv/rank1_submission_1174398/code/op_host/spline_conv.cpp:14-145`；`archives/2026-electric-power/extracted/SplineConv/rank1_submission_1174398/code/op_kernel/spline_conv_compact.h:51-94,183-225`
- 适用条件：权重索引只由离散 cell 决定；basis 对边线性；目标节点入度较高（d>=2，d>=4 更稳）；4C 聚合与权重能驻留 UB
- 负例/风险：d=1 时聚合开销大于收益；basis 依赖非离散属性或索引随边变化时代数不成立；浮点求和顺序改变

### `KS2026-ASC910B-FUSEDMOE`

- **FusedMoE** · `moe-expert-compute` · 本机 `fail`
- 一句话：不按 Top-2 动态分发，而是用三个规则 kernel 计算所有 8 个专家：gate/up+SiLU*up、down、Top-2 加权；权重按 tile FP32->FP16，避免完整转换副本
- 先读：[README](../competition/kernelswift-2026/README.md) · [source.json](../competition/kernelswift-2026/source.json)
- 源码入口：`platforms/ascend910b/task02_fused_moe/solution.py:8-87`；`platforms/ascend910b/task02_fused_moe/reference.py:42-75`
- 适用条件：E 小（8）；T 小（83）；H/I 小（128/64）；topk 固定 2
- 负例/风险：E 大时 4x 浪费线性放大（E=64 约 32x）；T 大时 [E,T,I] 中间缓冲/计算反噬；route 只对 softmax+renormalize 成立

### `KS2026-ASC910B-GROUPEDTOPK`

- **GroupedTopk** · `moe-routing` · 本机 `pass`
- 一句话：利用 softmax 单调性直接在 logits 上选组/选专家；Top-K 重新归一化时 256 路公共分母抵消，只对 8 个 logits 做 exp+归一化；一个 kernel 完成选择/归一化/写回
- 先读：[README](../competition/kernelswift-2026/README.md) · [source.json](../competition/kernelswift-2026/source.json)
- 源码入口：`platforms/ascend910b/task01_grouped_topk/solution.py:8-89`；`platforms/ascend910b/task01_grouped_topk/reference.py:24-65`
- 适用条件：softmax 或单调 scoring；renormalize=True；E/K 固定且不大；输出顺序/tie 与参考一致
- 负例/风险：renormalize=False 时分母不抵消；E 大时 O(E*K) 选择链可能变慢；tying/输出顺序需验证

### `KS2026-ASC910B-SINKHORN`

- **hc_split_sinkhorn** · `normalization` · 本机 `pass`
- 一句话：每个 program 处理一个 (b,s) 行；4x4 的 16 个元素显式标量化常驻寄存器；pre/post/20 轮 row/col 归一化/三个输出全部融合在一个 kernel
- 先读：[README](../competition/kernelswift-2026/README.md) · [source.json](../competition/kernelswift-2026/source.json)
- 源码入口：`platforms/ascend910b/task08_hc_split_sinkhorn/solution.py:19-89`；`platforms/ascend910b/task08_hc_split_sinkhorn/reference.py:26-52`
- 适用条件：固定 4x4；固定 20 轮；每行独立；行数不太少
- 负例/风险：hc_mult=8/16 时标量爆炸；行数很大时 1 program/行、1 warp 可能填不满；eps/初始归一化顺序需与参考一致

### `MFG2026-GNN-GRID-CLUSTER`

- **GridCluster** · `point-cloud-clustering` · 本机 `not-run`
- 一句话：每维 floor((x-origin)/size) 得索引；按 radix 组合成线性 ID；subnormal/极小 size 用缩放与倒数快路径；3D SIMT/宽维分支
- 先读：[README](../competition/mfg-2026/ops-gnn/README.md) · [source.json](../competition/mfg-2026/ops-gnn/source.json)
- 源码入口：`competition/mfg-2026/ops-gnn/source/csrc/npu/grid_cluster/op_kernel/arch35/grid_cluster.cpp:15-120,135-221`
- 适用条件：950/arch35 only；int64 linear ID；3D/wide dims；size>0
- 负例/风险：not 910B-buildable；subnormal/negative coords；int64 overflow

### `MFG2026-CYLINDER-QUERY`

- **CylinderQuery** · `point-cloud-query` · 本机 `not-run`
- 一句话：每核处理若干 query；点云 tile 进 UB 双缓冲；p-center 后用 R 旋转；CompareScalar 生成半径/高度 mask；Select 写候选；Python sort/truncate/fill
- 先读：[README](../competition/mfg-2026/driving-sdk/README.md) · [source.json](../competition/mfg-2026/driving-sdk/source.json)
- 源码入口：`competition/mfg-2026/driving-sdk/source/kernels/cylinder_query/op_host/cylinder_query.cpp:1-120`；`competition/mfg-2026/driving-sdk/source/kernels/cylinder_query/op_kernel/cylinder_query.cpp:60-102,300-350`；`competition/mfg-2026/driving-sdk/source/mx_driving/ops/cylinder_query.py:29-50`
- 适用条件：3×3 rot (9 floats)；nsample fits UB；A2-targeted generic path；output post-processed in Python
- 负例/风险：A2 support not explicitly documented；nsample UB limit；Python sort cost

### `MFG2026-MATH-STACK-BALL-QUERY`

- **StackBallQuery** · `point-cloud-query` · 本机 `not-run`
- 一句话：按 center 分段；batch 的 xyz 片段流式搬 UB；距离平方与 maxRadius² 比较；命中写结果；达到 sample_num 提前停；不足用首索引补齐
- 先读：[README](../competition/mfg-2026/ops-math/README.md) · [source.json](../competition/mfg-2026/ops-math/source.json)
- 源码入口：`competition/mfg-2026/ops-math/source/conversion/stack_ball_query/README.md:1-58`；`competition/mfg-2026/ops-math/source/conversion/stack_ball_query/op_kernel/stack_ball_query.h:28-140,281-397`；`competition/mfg-2026/ops-math/source/conversion/stack_ball_query/op_kernel/stack_ball_query.cpp:17-40`
- 适用条件：A2/A3；2D stacked layout；batch counts；fp16/fp32
- 负例/风险：not raw [B,3,N]；zero-hit all-0 indistinguishable；fp16 radius² precision

### `MFG2026-RADIUS`

- **Radius** · `point-cloud-query` · 本机 `not-run`
- 一句话：按 batch 分核；每 batch 点数上限 512 以驻留 UB；半径平方比较；命中写邻居对；超过 max_num_neighbors 截断；输出动态 [2,E]
- 先读：[README](../competition/mfg-2026/driving-sdk/README.md) · [source.json](../competition/mfg-2026/driving-sdk/source.json)
- 源码入口：`competition/mfg-2026/driving-sdk/source/kernels/radius/op_host/radius.cpp:1-100`；`competition/mfg-2026/driving-sdk/source/kernels/radius/op_kernel/radius.cpp:1-120`；`competition/mfg-2026/driving-sdk/source/docs/zh/api/context/radius.md:1-50`
- 适用条件：Ndim<9；batch<=1024；per-batch points<=512；output [2,E]
- 负例/风险：hard caps；dynamic output workspace；ordering/truncation semantics

### `MFG2026-FPS`

- **FurthestPointSampling** · `point-cloud-sampling` · 本机 `not-run`
- 一句话：维护 nearestDist[N]；每轮计算到新选中点的距离、取 min、reduce max 选下一个；点云大于 UB 时按 pieces 分片；多核按 batch；fp16/bf16 cast fp32 算距离
- 先读：[README](../competition/mfg-2026/driving-sdk/README.md) · [source.json](../competition/mfg-2026/driving-sdk/source.json)
- 源码入口：`competition/mfg-2026/driving-sdk/source/kernels/furthest_point_sampling/op_kernel/furthest_point_sampling.cpp:115-208`；`competition/mfg-2026/driving-sdk/source/kernels/furthest_point_sampling/op_host/furthest_point_sampling.cpp:1-100`；`competition/mfg-2026/driving-sdk/source/docs/zh/api/context/furthest_point_sampling.md:1-39`
- 适用条件：B*N*3<=383166；num_points>0；first index 0；fp32/fp16/bf16
- 负例/风险：total element cap；first-index convention；piece re-reads

### `MFG2026-GNN-SPMM`

- **SpmmSum (CSR aggregation)** · `sparse-dense-matmul` · 本机 `not-run`
- 一句话：SpmmReduce 枚举 sum/max/min/mean；Reducer 给初值/apply；CSR 行分组 + copy_lhs；形状/索引校验
- 先读：[README](../competition/mfg-2026/ops-gnn/README.md) · [source.json](../competition/mfg-2026/ops-gnn/source.json)
- 源码入口：`competition/mfg-2026/ops-gnn/source/csrc/npu/common/spmm/spmm_utils.h:210-241`；`competition/mfg-2026/ops-gnn/source/docs/zh/api_reference.md:197-199`
- 适用条件：A2/A3 arch22；CSR；sum/max/min/mean；dtype combos
- 负例/风险：empty segment semantics；dtype combos；row grouping layout

### `MFG2026-SPARSE-SPMM`

- **SpmmSum (aclsparse SpMM)** · `sparse-dense-matmul` · 本机 `not-run`
- 一句话：按 dtype/布局/opA/opB 选行分块；行组 axpy；ALG1 通用/ALG2 行主序快路径/ALG3 确定性；fp32 Kahan
- 先读：[README](../competition/mfg-2026/ops-sparse/README.md) · [source.json](../competition/mfg-2026/ops-sparse/source.json)
- 源码入口：`competition/mfg-2026/ops-sparse/source/sparse/spmm/README.md:1-120`；`competition/mfg-2026/ops-sparse/source/sparse/spmm/arch22/spmm_host.cpp:201-330`；`competition/mfg-2026/ops-sparse/source/sparse/spmm/arch22/kernels/spmm_kernel_group_axpy.h:1-120`
- 适用条件：A2/A3；CSR A；A2 dtype combos；ALG1/2/3
- 负例/风险：dtype combos narrower than 950；ALG3 non-transpose only；layout/alg performance varies

### `MFG2026-SPARSE-SPGEMM`

- **Spspmm (CSR × CSR)** · `sparse-sparse-matmul` · 本机 `not-run`
- 一句话：WorkEstimation 统计中间乘积 → EstimateMemory 规划 workspace → Compute symbolic(rowOffsets)+numeric(merge/去重) → Copy 写输出 CSR
- 先读：[README](../competition/mfg-2026/ops-sparse/README.md) · [source.json](../competition/mfg-2026/ops-sparse/source.json)
- 源码入口：`competition/mfg-2026/ops-sparse/source/sparse/spgemm/README.md:1-40`；`competition/mfg-2026/ops-sparse/source/sparse/spgemm/arch22/spgemm_host.cpp:654-720,1066-1120`；`competition/mfg-2026/ops-sparse/source/sparse/spgemm/arch22/spgemm_symbolic_kernel.cpp:1-120`
- 适用条件：A2/A3/950；CSR；non-transpose only；sorted unique output
- 负例/风险：numProds != nnz(C)；dynamic output allocation；INT32 nnz limit

## general

### `PARITY-O3-PACKED-TP`

- **Packed path-preserving O(3) tensor product** · `equivariant-tensor-product` · 本机 `not-run`
- 一句话：按输入角阶 l_in 分组多个 path，每组一次 batched GEMM；channels contiguous inside GEMM；直接写 packed 输出的 path 偏移；gate adjoint 重算 base 而不是保存全部 path 输出；CANN vector BMM 仅在 E>=16384,C>=64 等条件启用，否则 torch.bmm
- 先读：[README](../general/equivariant/parity-ictc-o3/README.md) · [source.json](../general/equivariant/parity-ictc-o3/source.json)
- 源码入口：`ictc_o3/backends/ascend/packed_tensor_product.py:1-165`；`docs/PERFORMANCE.md:5-31,47-66`；`docs/USAGE.md:792-861`
- 适用条件：path-preserving ICTC basis；FP32；E large and C>=64 for CANN BMM；consumer accepts packed output/unpack_output
- 负例/风险：not an arbitrary e3nn TP adapter; components need basis maps；E<16384 or C<64 falls back to torch.bmm；native mode not automatically faster than vendor GEMM

### `XAYAH-ASC910B-MATMUL-NEG`

- **Matmul (negative)** · `gemm` · 本机 `pass-negative`
- 一句话：1D grid + 四组 autotune tile + K 循环 tl.dot；无 split-K/L2 swizzle/Ascend 专用流水
- 先读：[README](../general/elementwise/52xayah-triton-kernels/README.md) · [source.json](../general/elementwise/52xayah-triton-kernels/source.json)
- 源码入口：`src/matmul.py:8-89`
- 适用条件：规则大 shape；FP16；只用教学 tile
- 负例/风险：K 大时单 program 串行；无 split-K/L2 swizzle；厂商 matmul 更强

### `VLLMHUST-LINEAR-SWIGLU`

- **Linear + SwiGLU Graph fusion** · `linear-activation-fusion` · 本机 `not-run`
- 一句话：同一 kernel 保留 gate/up 两个 FP32 accumulator，A 每个 K block 只加载一次；最后 gate*sigmoid(gate)*up，只写 [M,N]；M<=128 融合，否则 native GEMM+npu_swiglu
- 先读：[README](../general/transformer/vllm-hust-opset/README.md) · [source.json](../general/transformer/vllm-hust-opset/source.json)
- 源码入口：`src/vllm_hust_operator_optimizations/optimizations/linear_swiglu_kernel.py:35-107`；`src/vllm_hust_operator_optimizations/optimizations/linear_swiglu_graph.py:34-124`；`src/vllm_hust_operator_optimizations/optimizations/linear_swiglu_fusion_pass.py:12-48`
- 适用条件：M<=128；unquantized；BF16/FP16；Graph/Triton path
- 负例/风险：M=256 fused tile 超 CBUF；大 M 多 tile 出现错误值，必须回退；冷启动 JIT 可能为负

### `PARITY-MESH-ELECTROSTATICS`

- **Bounded mesh assignment for reciprocal electrostatics** · `long-range-electrostatics` · 本机 `not-run`
- 一句话：每 atom x channel tile 一个 task；Spline 权重/一阶/二阶导数只存寄存器/UB；order^3 stencil 循环内做 Muls/Add/atomic mesh 写；gather/coordinate VJP 同循环；按 N/channel/adjoint 目标选 native/vectorized/particle/CANN
- 先读：[README](../general/equivariant/parity-ictc-o3/README.md) · [source.json](../general/equivariant/parity-ictc-o3/source.json)
- 源码入口：`ictc_o3/ascendc/op_kernel/electrostatics.cpp:22-119`；`ictc_o3/backends/ascend/electrostatics.py:22-85`；`docs/ELECTROSTATICS.md:245-327,366-455`
- 适用条件：reciprocal-only Gaussian-screened functional；PCS/CIC/TSC and mesh fixed while positions/cell live；channel tile fits UB；multi-NPU needs HCCL + slab FFT
- 负例/风险：not full Ewald: no real-space/self/neutralizing background；floating atomic scatter nondeterministic；N>65536 single-channel auto falls back to CANN

### `PARITY-CAPACITY-CHUNK`

- **Capacity-mode chunked message recomputation** · `memory-scheduled-graph-message` · 本机 `not-run`
- 一句话：forward fast_graph(capacity_stream) 只留紧凑 node 输出；backward 若高阶走原图，否则按 chunk 重算边中间并累加；自动 chunk 由设备内存/已分配/node 输出/per-edge 工作估算；大 frozen radial 最后 Linear 流式放入 chunk
- 先读：[README](../general/equivariant/parity-ictc-o3/README.md) · [source.json](../general/equivariant/parity-ictc-o3/source.json)
- 源码入口：`ictc_o3/backends/ascend/capacity_message.py:7-80,83-179`；`docs/HP_API.md:67-101`；`docs/PERFORMANCE.md:47-66`
- 适用条件：inference/frozen params；ordinary first-order adjoint；large atom count and edge intermediates dominate memory；dynamic graph
- 负例/风险：not training/higher-order; falls back to original graph；small graphs not worth chunk loop；chunk depends on live memory/allocator state

### `BE4VER-QDWH-SDC-FUSED`

- **Fused QDWH polar decomposition + device SDC + leaf Jacobi** · `polar-eigendecomposition` · 本机 `pass`
- 一句话：一次 host malloc/launch；H 直接写 SDC arena slot0；Phase1 QDWH + Phase2 SDC 在 kernel 内用 PhaseBarrier 衔接；同层节点 batch QDWH（kSdcBatchMax 由 UB 预算 static_assert 推导）；H±=Q^T H Q 一次 GEMM；跳过数学冗余对称化；lambda 跨 AIV 分布式归约
- 先读：[README](../general/linear-algebra/be4ver-svd/README.md) · [source.json](../general/linear-algebra/be4ver-svd/source.json)
- 源码入口：`qdwh_sdc_fused_jacobi/fused_qdwh_sdc.hpp:89-188,386-500,733-825`；`qdwh_sdc_fused_jacobi/sdc_node.hpp:331-397`；`qdwh_sdc_fused_jacobi/vec.hpp:397-453`
- 适用条件：FP32；mPad+nPad<=6144；threshold 16..64；blockDim 16..64 even
- 负例/风险：1000x800 seed42 奇异值误差 1.5e-5~3.1e-4 且逐次不同（atomic + 特征值间隙放大）；cond=1e6 小奇异值相对误差约1.5%；3072^2 batch 8->15 无收益，已计算受限

### `XAYAH-ASC910B-SOFTMAX-NEG`

- **Softmax (negative)** · `softmax` · 本机 `pass-negative`
- 一句话：行内 max/exp/sum/div 融合，一个 program 一行；固定 1024，无 mask/autotune
- 先读：[README](../general/elementwise/52xayah-triton-kernels/README.md) · [source.json](../general/elementwise/52xayah-triton-kernels/source.json)
- 源码入口：`src/softmax.py:8-41`
- 适用条件：N=1024 规则 shape；FP32；只需单行 softmax
- 负例/风险：N!=1024 越界/错误；无 mask/autotune；朴素 Triton 比厂商库慢 4x

### `MAMBA2-ASCENDC-CHUNKMIX`

- **Mamba2SsdChunkMix** · `ssm-chunked-scan` · 本机 `not-run`
- 一句话：chunked scan 把长序列扫描拆成 chunk 内三个 GEMM + chunk 间状态；MIX_AIC_1_2 两个 AIV 各一半 rows；B 转置共享；状态 [N,P] 消费者原生；双槽 workspace + 跨核 flag；一套 L1/L0/L0C 服务多种 GEMM
- 先读：[README](../general/attention/mamba-ascendc/README.md) · [source.json](../general/attention/mamba-ascendc/source.json)
- 源码入口：`mamba_ascendc_ops/mamba2_ssd_chunk_mix/design.md:1-69`；`mamba_ascendc_ops/mamba2_ssd_chunk_mix/op_kernel/mamba2_ssd_chunk_mix.cpp:37-231,1016-1211,1293-1339`；`mamba_ascendc_ops/mamba2_ssd_chunk_mix/op_kernel/mamba2_dynamic_matmul.h:6-26,195-254`
- 适用条件：序列/head 足够大；P=N=chunk 在支持集合；H%G==0；Cube/Vector 能并行
- 负例/风险：H256 只有 A100 70.82%；StateEpilogue memory-bound (98.3% Cube active, 2.9% AIC MAC)；12.06 GiB 中间物化

### `BE4VER-PACO-BATCHED-JACOBI`

- **PACO batched one-sided Jacobi SVD** · `svd` · 本机 `pass`
- 一句话：小 batch 一矩阵一核；大 batch 按 K 个矩阵行交错到 UB，合并 Mul/WholeReduceSum；旋转用 Muls/Axpy，避免多次 Adds 拷回；AIC 只做 Gram
- 先读：[README](../general/linear-algebra/be4ver-svd/README.md) · [source.json](../general/linear-algebra/be4ver-svd/source.json)
- 源码入口：`paco-jacobi-svd/jacobi_svd_custom.cpp:130-170,215-303,435-538`；`paco-jacobi-svd/docs/64x64-optimization-study-2026-11-12.md:1-18,62-100`
- 适用条件：FP32；rows<=128/cols<=256；batch 大到 perCore>1 才走 K 合并；UB 能容纳 K 个矩阵
- 负例/风险：K 合并的矢量角度链只在 count=64 时正确，代码仍用标量角度；列 pad 到 64 会比 32 慢；UB>192KB 直接 507015/输出全 0
