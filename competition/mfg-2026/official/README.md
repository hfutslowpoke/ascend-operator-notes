# 官方赛题规格（2026 制造行业算子专场）

> 来源：CANN Judge 公开题面（`https://cannjudge.cn/public/gcc_2026_mfg/<problem>`）与其公开 API `GET https://cannjudge.cn/api/problems/contest/6aed4926a5348dcef5dd8d0c`，抓取日期 2026-11-17。

本目录只保存公开题面，不包含隐藏测试数据。`/api/testcases/<id>` 返回 403，未绕过。

## 覆盖分级

| 级别 | 含义 | 数量 |
|---|---|---:
| `direct` | 源码语义与官方一致，且存在 A2/910B 路径；只需调用/薄适配 | 2 |
| `adapt` | 有 A2/910B 目标源码，但接口/layout/shape/后处理需改 | 6 |
| `reference` | 只有 950/arch35 或算法 helper，可读但不可直接上 910B | 5 |
| `none` | 未找到公开单 kernel 实现（03/04 已有 910B 基础件可组合） | 2 |

## 15 题速查

| # | 赛题 | 类别 | 输入 → 输出 | 关键属性/约束 | 容差 | 源码级别 |
|---:|---|---|---|---|---|---|
| 01 | [`complexabs`](descriptions/complexabs.md) | 仿真计算 | src [N,M] complex64 → out [N,M] float32 | 无；N,M ∈ [1,65536]；输入分量有限，不支持 NaN/Inf；分量绝对值 ≤ 10 | atol=rtol=1e-4 | `reference` |
| 02 | [`complexdiv`](descriptions/complexdiv.md) | 仿真计算 | dividend [N,M] float16/float32/complex64；divisor [N,M] 与 dividend 同 dtype → out [N,M] 与 dividend 同 dtype | 无；N,M ∈ [1,65536]；分量绝对值 ≤ 10；除数非零；除零行为未定义；支持 NumPy/torch 广播；不做隐式类型提升 | float32/complex64: atol=rtol=1e-4；float16: atol=rtol=1e-3 | `reference` |
| 03 | [`complexexp`](descriptions/complexexp.md) | 仿真计算 | x [N,M] float16/float32/complex64 → out [N,M] 与 x 同 dtype | 无；N,M ∈ [1,65536]；实部、虚部均在 [-10,10]；输入有限，不支持 NaN/Inf | float32/complex64: atol=rtol=1e-4；float16: atol=rtol=1e-3 | `none` |
| 04 | [`fusedpropstep`](descriptions/fusedpropstep.md) | 仿真计算 | field [B,N,M] float16/float32/complex64；h [Bh,N,M] 与 field 同 dtype → out [B,N,M] 与 field 同 dtype | norm bool 默认 False；B ∈ [1,256]；N,M ∈ [2,8192] 且均为 2 的幂；取值 [-1,1]，输入有限 | complex64: atol=rtol=1e-4；float32/float16: atol=rtol=1e-3 | `none` |
| 05 | [`fouriertransform2d`](descriptions/fouriertransform2d.md) | 仿真计算 | x [B,N,M] complex64 → y [B,N,M] complex64 | inverse bool 默认 False；B ∈ [1,256]；N,M ∈ [2,8192] 且均为 2 的幂；输入分量有限，取值 [-1,1] | atol=rtol=1e-3 | `adapt` |
| 06 | [`gathercoo`](descriptions/gathercoo.md) | 图计算 | src (M,K) float32/float16/int32/int8；index (E) int32，取值 [0,M) → out (E,K) 与 src 同 dtype | 无；M ∈ [1,1000000]；K ∈ [1,4096]；E ∈ [1,4000000] | 索引/拷贝精确；dtype 原样 | `reference` |
| 07 | [`segmentmincsr`](descriptions/segmentmincsr.md) | 图计算 | src (E,K) float32/float16；indptr (R) int64，R=N+1，单调不减，indptr[N]=E → out (N,K) 与 src 同 dtype；arg_out (N,K) int64，段内全局行号 | 无；N ∈ [1,1000000]；E ∈ [0,5000000]；K ∈ [1,4096] | values 按题面 atol/rtol；arg_out 逐位一致 | `direct` |
| 08 | [`randomwalk`](descriptions/randomwalk.md) | 图计算 | rowptr (R) int64；col (E) int64；seed (B) int64；rand (B,walk_length) float32，必选 → out (B,L) int64，L=walk_length+1 | walk_length int 默认 16，范围 [1,64]；p float 默认 1.0；q float 默认 1.0；N ∈ [1,10000000]；E ∈ [0,20000000]；B ∈ [1,100000] | 轨迹逐位一致 | `reference` |
| 09 | [`spmmsum`](descriptions/spmmsum.md) | 图计算 | rowptr (R) int64；col (E) int32，行内升序；value (E) float32/float16；mat (N,K) 与 value 同 dtype → out (M,K) 与 value 同 dtype | 无；M,N ∈ [1,10000000]；E ∈ [1,20000000]；K ∈ [1,4096] | float32: atol=rtol=1e-4；float16: atol=rtol=1e-2 | `adapt` |
| 10 | [`spspmm`](descriptions/spspmm.md) | 图计算 | indexA (2,E1) int64；valueA (E1) float32/float16；indexB (2,E2) int64；valueB (E2) 与 valueA 同 dtype → indexC (2,U) int64，U=m·n；valueC (U) 与 valueA 同 dtype | m int 默认 64；k int 默认 64；n int 默认 64；M,N,K ∈ [1,10000000]；E1,E2 ∈ [1,5000000]；U=m·n ≤ 250000 | float32: atol=rtol=1e-4；float16: atol=rtol=1e-2；indexC 逐位一致 | `adapt` |
| 11 | [`radius`](descriptions/radius.md) | 点云计算 | x (N,D) float32/float16；y (M,D) 与 x 同 dtype；ptr_x (Rx) int64；ptr_y (Ry) int64 → out (2,E) int64，E=M×max_num_neighbors | r float 默认 1.0，≥0；max_num_neighbors int 默认 32；ignore_same_index bool 默认 False；N,M ∈ [1,1000000]；M×max_num_neighbors ≤ 4000000；D ∈ {2,3} | 索引逐位一致 | `adapt` |
| 12 | [`gridcluster`](descriptions/gridcluster.md) | 点云计算 | pos (N,D) float32/float16；size (D)；start (D)；end (D) → out (N) int64 | 无；N ∈ [1,10000000]；D ∈ {2,3}；size 各维 >0；end≥start；pos-start≥0 | int64 逐位一致 | `reference` |
| 13 | [`furthestpointsample`](descriptions/furthestpointsample.md) | 点云计算 | points [B,N,3] float32/float16 → idx [B,npoint] int32 | npoint int 默认 128，1≤npoint≤N；points 最后一维 3，取值 [-1,1]；B,N,npoint 为正整数；输入有限，不支持 NaN/Inf | 索引逐位一致 | `direct` |
| 14 | [`ballquery`](descriptions/ballquery.md) | 点云计算 | xyz [B,N,3] float32/float16；new_xyz [B,M,3] 与 xyz 同 dtype → idx [B,M,nsample] int32 | radius float 默认 0.4，>0；nsample int 默认 16，>0；B,N,M 正整数；N≥1；xyz/new_xyz 最后一维 3，常规 [-1,1]；d² < radius² 严格小于；first-K 扫描序 | int32 索引逐元素一致 | `adapt` |
| 15 | [`cylinderquery`](descriptions/cylinderquery.md) | 机器人抓取 | xyz [B,N,3] float32/float16；new_xyz [B,M,3] 与 xyz 同 dtype；rot [B,M,3,3] 与 xyz 同 dtype → idx [B,M,nsample] int32 | radius float 默认 0.05，>0；hmin float 默认 -0.02；hmax float 默认 0.04，>hmin；nsample int 默认 32，1≤nsample≤N；B,N,M 正整数；N≥1；rot row-major，一般有限 3×3，不要求正交；局部坐标 = rot^T @ delta（不是 rot @ delta） | int32 索引逐元素一致 | `adapt` |

## 910B 补充基础件（不是完整题解）

| 题 | 基础件 | 能做什么 | 仍缺什么 |
|---|---|---|---|
| 03 ComplexExp | `experimental/math/exp` + `sin` + `cos` + `complex_v3`（均 910B arch22/A2） | 组合 `exp(a)·cos(b)`、`exp(a)·sin(b)`，再由 ComplexV3 拼成 complex64/complex32；ComplexV3 支持广播 | 单 kernel 融合、fp16 极值/精度、官方 4 组 dtype 的完整验证 |
| 04 FusedPropStep | `ops-fft/fft1_d`（910B C2C 1D）+ `fft2_d`（2D，32/64/128）+ `rfft1_d`/`irfft1_d`（910B FP32 R2C/C2R 1D） | `fft1_d` 单维覆盖 n≤256 DFT、512..16384 FFTB；实数输入可用 R2C/C2R 按行/列组合 | 2..8192 全尺寸 2D 组合、FP16 原生 R2C/C2R、单 kernel 融合、`norm` 归一化、大 batch |
| 05 FourierTransform2D | 同上 `fft1_d` + `fft2_d` | `fft2_d` 32/64/128 直接可用；`fft1_d` 可补 256..8192 的单维 FFT | 2D 组合、inverse 的 `1/(N·M)`、数据布局与 18 组官方 shape |

这些基础件来自同一固定 commit，源码已复制到对应条目的 `source/`；`source_match` 仍按“完整题解”评估为 `none`（03/04）或 `adapt`（05），不把基础件冒充完整实现。

## 与源码映射的关键差异

| 赛题 | 候选源码 | 关键差异 | 复用判断 |
|---|---|---|---|
| `complexabs` | `MFG2026-MATH-COMPLEX-ABS` | ops-math complex_abs 仅 arch35（950/A3），A2 明确不支持；官方输入仅 complex64，源码支持 complex32/64/128，属于更宽的实现 | 只复用平方和/sqrt/缩放算法；910B 需另写 Vector 版本，不能直接构建 arch35 代码。 |
| `complexdiv` | `MFG2026-SOLVER-COMPLEX-DIV` | ops-solver 只有 complex64 的 LU/SIMT 内部 helper，不是独立 elementwise 算子；不覆盖 float16/float32 与广播；SIMT 仅 950 可编译 | 复用缩放 + FMA 补偿商算法；910B 需写 Vector 版并补 float16/float32/广播。 |
| `complexexp` | `—` | 未找到公开的 Ascend C ComplexExp 实现；GitCode 全站搜索 ComplexExp 仅命中无关词法分析仓库 | 无公开源码；可按官方公式用 exp/cos/sin 组合，或写单 kernel 融合。需先确认 float16 中间精度。 |
| `fusedpropstep` | `—` | 未找到公开的 Ascend C FusedPropStep 实现；ops-fft/fft2_d 只覆盖 complex64、32/64/128，且不融合逐点乘，无法覆盖 2..8192 与实数 dtype | 无公开源码；可借鉴 ops-fft 的 P/Q 分解与复数拆分，但必须扩展尺寸、dtype、逆变换归一化和融合逐点乘。 |
| `fouriertransform2d` | `MFG2026-FFT2D` | ops-fft/fft2_d 只接受 32/64/128 的 C2C，官方要求 2..8192 的 2 的幂；fft2_d 的方向参数可做 forward/backward，但需核对 1/(N·M) 归一化；源码只支持 complex64，官方也是 complex64，这一点匹配 | 复用 P/Q 矩阵乘分解、复数实虚分离、Cube 双缓冲；尺寸/归一化需扩展。本机仅 32×32 通过，不能覆盖官方大 shape。 |
| `gathercoo` | `MFG2026-GNN-GATHER-COO` | ops-gnn gather_coo 仅 arch35（950），910B 无 arch22；源码 index 为 int64、支持 rank 1–8；官方 index 为 int32、2D [M,K]；ops-sparse gather 是 dense gather，不是 COO 行扩展 | 只复用重复行 slab / tile 写出思路；910B 需按官方 int32 2D 契约重写。 |
| `segmentmincsr` | `MFG2026-GNN-SEGMENT-MIN-CSR` | 需确认 Python 入口返回 (values, indices) 与官方两个输出张量的绑定方式；源码还支持 sum/max/mean 等，官方只要求 min | ops-gnn `segment_min_csr` 在 A2/A3 arch22 上已实现 arg_out、空段 0/哨兵、首遇 tie-break，语义高度匹配；优先本机构建验证。 |
| `randomwalk` | `MFG2026-GNN-RANDOM-WALK` | ops-gnn random_walk 仅 arch35（950），910B 无 arch22；源码用 Philox 计数器 RNG；官方要求外部 rand 输入，保证逐位可比；官方 p/q 为非 1 的 node2vec 扩展，源码接口也需核对 | 只参考 Philox/分组采样/孤立点处理；官方确定性 rand 契约必须重写。 |
| `spmmsum` | `MFG2026-SPARSE-SPMM / MFG2026-GNN-SPMM` | ops-sparse SpMM 是通用 alpha·A·B+beta·C，需要 alpha=1/beta=0 并构造 CSR 描述符；ops-gnn spmm 是 PyTorch 层聚合接口，需把 value/mat 映射到 op/reduce；官方 col 为 int32，源码部分路径为 int64 | 复用 A2/arch22 的行分块/group axpy/确定性累加；写薄适配层并验证空行与 dtype。 |
| `spspmm` | `MFG2026-SPARSE-SPGEMM` | ops-sparse SpGEMM 输入/输出为 CSR、动态 nnz；官方为 COO + 固定槽位 U=m·n；官方要求未命中槽位 (-1,-1)/0，且保留零和坐标；源码输出 CSR 排序去重；官方维度属性 m/k/n 可大于 index 上限 | 复用 symbolic/numeric/merge 多阶段 SpGEMM 内核；需加 COO↔CSR 转换与固定槽位输出层。 |
| `radius` | `MFG2026-RADIUS` | DrivingSDK radius 输出动态 [2,E]，官方要求定槽位 [2,M×K] 且未命中 (-1,-1)；源码约束 Ndim<9、batch≤1024、单 batch 点数≤512；官方 D∈{2,3}、N/M≤1e6；官方有 ignore_same_index；源码需核对该属性 | 复用 batch 分核、平方距离、扫描序截断；需重写定槽位输出、2/3D 与更大 shape。 |
| `gridcluster` | `MFG2026-GNN-GRID-CLUSTER` | ops-gnn grid_cluster 仅 arch35（950），910B 无 arch22；源码 stride 顺序与官方一致（dim0 最快），但 SIMT 路径需重写；个人仓 flying_whj/CANN_grid_cluster 也是 arch35，但有设计文档/golden 可读 | 复用网格量化、stride 组合、subnormal/倒数快路径；910B 需写 Vector/SIMT 版本，不能直接构建 arch35。 |
| `furthestpointsample` | `MFG2026-FPS` | DrivingSDK 文档还支持 bfloat16，官方只要求 float32/float16；源码有总元素 B*N*3≤383166 的约束；官方测试规模远小于此；需本机确认 half 路径与并列 tie-break 逐位一致 | 官方题面明确以 mx_driving Ascend `furthest_point_sampling` 为参考，DrivingSDK 源码应为主候选；优先构建并跑官方 18 组 shape。 |
| `ballquery` | `MFG2026-MATH-STACK-BALL-QUERY` | stack_ball_query 输入是 2D 堆叠布局 + batch counts，官方是 [B,N,3]/[B,M,3]；stack_ball_query 在 float16 路径把 radius² 先转 half，官方要求 float32 阈值语义；ops-math 直接版 ball_query 有 min_radius/max_radius 语义，官方只有单 radius；且直接版只有 arch35 | 复用按 center 分段、流式点云、提前停；需改 [B,*,3] 布局、float32 阈值，并确认单半径语义。 |
| `cylinderquery` | `MFG2026-CYLINDER-QUERY` | DrivingSDK kernel 的旋转方向是 rot^T·Δ，与官方一致；但 Python 层 `sortRes` 会排序后截断，官方要求扫描序 first-K；DrivingSDK 输出全量候选，再由 Python 排序；官方直接输出 [B,M,nsample]；float16 输入需确认 host/kernel 是否先转 float32 | 复用旋转、双 mask、严格比较；必须去掉排序，改为扫描序 first-K + 填充规则。 |

## 题面文件

每题完整公开题面在 [`descriptions/`](descriptions/)；机器可读的 shape/dtype/约束摘要在 [`specs.json`](specs.json)，原始公开 API 快照在 [`problems.json`](problems.json)。

## 重要校正

- **FourierTransform2D** 官方要求 `B∈[1,256]`、`N,M∈[2,8192]` 的 2 的幂、`inverse` 属性；`ops-fft/fft2_d` 只覆盖 32/64/128，不能用 32×32 本机 PASS 代表全题通过。`fft1_d` 提供 910B 单维基础件（n≤256 DFT、512..16384 FFTB、更大 FFTN），但仍需 2D 组合与逆变换归一化。
- **Spspmm** 官方是 COO + 固定槽位 `U=m·n`；`ops-sparse` SpGEMM 是 CSR + 动态 nnz，必须加转换/槽位层。
- **Radius** 官方是定槽位 `[2,M×K]`、未命中 `(-1,-1)`、`D∈{2,3}`；DrivingSDK 是动态输出且有单 batch 512 点上限。
- **CylinderQuery** 官方要求 `rot^T·Δ`、扫描序 first-K；DrivingSDK kernel 旋转方向一致，但 Python 层会排序后截断。
- **BallQuery** 官方只有单 `radius`、严格 `<`、float16 先转 float32；`stack_ball_query` 是堆叠布局，且 fp16 路径把 `radius²` 转 half。
- **FurthestPointSample** 官方明确以 `mx_driving` Ascend FPS 为参考，且 float16 必须保持 half 精度；DrivingSDK 是最强候选，但仍需本机逐位验证。
- **SegmentMinCsr** 官方要求 `arg_out`、空段 `out=0/arg_out=E`、首遇 tie-break；`ops-gnn segment_min_csr` 的 A2/arch22 路径已覆盖这些语义，是最接近可直接复用的实现。
