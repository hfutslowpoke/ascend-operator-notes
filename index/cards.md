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

### `GITCODE-CONCAT`

- **S9 Concat (dynamic tensor list)** · `concat-layout` · 本机 `offline-only`
- 一句话：Host normalizes dynamic inputs to rows x n_i, then chooses row/column split, DMA/Gather assembly, input split or output linear split. Gather path packs contiguous rows per input, builds a byte-offset table, gathers into a full row, writes once. Column bands re
- 先读：[README](../competition/cann-ops-competitions/README.md) · [source.json](../competition/cann-ops-competitions/source.json)
- 源码入口：`01_official/operator-challenge/cann-ops-challenge-S9/submissions/Concat/BatchLions/op_host/concat.cpp:55-75,240-330,440-532,625-680`；`01_official/operator-challenge/cann-ops-challenge-S9/submissions/Concat/BatchLions/op_kernel/concat.cpp:264-630,909-1071,1075-1445`；`01_official/operator-challenge/cann-ops-challenge-S9/submissions/Concat/BatchLions/op_host/s9_concat_layout.h:96-117,163-241,355-435,603-696`
- 适用条件：dynamic tensor list <=256；CANN 8.5.0 target；UB budget/alignment satisfied；Gather path needs 16/32-bit dtype and nOut limit
- 负例/风险：balanced column bands slower 1.35% on G16 246-input case；edge trimming slower 8.0%/7.9%/1.0% on three shapes；bulk TilingData header fetch slower 8%

### `HANSEN-2026POWER-TREE-R1`

- **TreeLineMinDistance (rank1)** · `geometric-argmin` · 本机 `not-run`
- 一句话：Host 按 M/N 退化维度选 PATH；线段 AoS→SoA，D/L/1/L 按 batch 外提；点 tile 向量化平方距离，64 段 chunk 归约 + 跨 chunk 成对归约，sqrt 后置；M=1024 路径用 half Top-2 粗筛 + FP32 重算 16 候选
- 先读：[README](../competition/hansen-2026-power/README.md) · [source.json](../competition/hansen-2026-power/source.json)
- 源码入口：`archives/2026-electric-power/extracted/TreeLineMinDistance/rank1_submission_1175341/code/op_host/tree_line_min_distance.cpp:438-496`；`archives/2026-electric-power/extracted/TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_line_min_distance.cpp:550-659`；`archives/2026-electric-power/extracted/TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_pairs.h:221-515,752-854`
- 适用条件：存在 M=1/N=1 等退化维度；线段不变量可在 batch 内复用；tie 需最小索引；尾块非 32 对齐
- 负例/风险：half Top-2 无 no-miss 证明，不能用于精确 argmin；max(L,FLT_MIN) 不等价 L==0；大量固定 PATH/UB 绝对偏移

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
