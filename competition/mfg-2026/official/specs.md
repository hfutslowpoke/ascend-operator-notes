# 官方题面与源码匹配逐题表

由 `tools/build_mfg_official.py` 从 `official/specs.json` 生成；完整题面见 `descriptions/`。

## 01 仿真计算：ComplexAbs

- 官方题面：[complexabs](descriptions/complexabs.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/complexabs)
- 公式：out = sqrt(re² + im²)
- 输入：src [N,M] complex64
- 输出：out [N,M] float32
- 属性：无
- 约束：N,M ∈ [1,65536]；输入分量有限，不支持 NaN/Inf；分量绝对值 ≤ 10
- 容差：atol=rtol=1e-4
- 源码级别：`reference`；候选：`MFG2026-MATH-COMPLEX-ABS`
- 源码差异：ops-math complex_abs 仅 arch35（950/A3），A2 明确不支持；官方输入仅 complex64，源码支持 complex32/64/128，属于更宽的实现
- 复用判断：只复用平方和/sqrt/缩放算法；910B 需另写 Vector 版本，不能直接构建 arch35 代码。

## 02 仿真计算：ComplexDiv

- 官方题面：[complexdiv](descriptions/complexdiv.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/complexdiv)
- 公式：(a+bi)/(c+di) = (a+bi)(c-di)/(c²+d²)
- 输入：dividend [N,M] float16/float32/complex64；divisor [N,M] 与 dividend 同 dtype
- 输出：out [N,M] 与 dividend 同 dtype
- 属性：无
- 约束：N,M ∈ [1,65536]；分量绝对值 ≤ 10；除数非零；除零行为未定义；支持 NumPy/torch 广播；不做隐式类型提升
- 容差：float32/complex64: atol=rtol=1e-4；float16: atol=rtol=1e-3
- 源码级别：`reference`；候选：`MFG2026-SOLVER-COMPLEX-DIV`
- 源码差异：ops-solver 只有 complex64 的 LU/SIMT 内部 helper，不是独立 elementwise 算子；不覆盖 float16/float32 与广播；SIMT 仅 950 可编译
- 复用判断：复用缩放 + FMA 补偿商算法；910B 需写 Vector 版并补 float16/float32/广播。

## 03 仿真计算：ComplexExp

- 官方题面：[complexexp](descriptions/complexexp.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/complexexp)
- 公式：实数: exp(a)；复数: exp(a)·(cos b + i·sin b)
- 输入：x [N,M] float16/float32/complex64
- 输出：out [N,M] 与 x 同 dtype
- 属性：无
- 约束：N,M ∈ [1,65536]；实部、虚部均在 [-10,10]；输入有限，不支持 NaN/Inf
- 容差：float32/complex64: atol=rtol=1e-4；float16: atol=rtol=1e-3
- 源码级别：`none`；候选：`—`
- 源码差异：未找到公开的 Ascend C ComplexExp 实现；GitCode 全站搜索 ComplexExp 仅命中无关词法分析仓库；已有 910B 基础件可组合，但没有官方/公开的单 kernel ComplexExp 实现
- 补充基础件：experimental/math/exp（910B arch22）；experimental/math/sin（910B）；experimental/math/cos（910B）；experimental/math/complex_v3（910B arch22，real/imag→complex，支持广播）
- 复用判断：没有单 kernel ComplexExp；可用 910B 的 exp/sin/cos + ComplexV3 组合出 e^a·cos b / e^a·sin b 再拼成复数。代价是 4 个算子、额外 GM 往返，需验证 fp16 精度与极值。

## 04 仿真计算：FusedPropStep

- 官方题面：[fusedpropstep](descriptions/fusedpropstep.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/fusedpropstep)
- 公式：S=FFT2(field); P=S·h; out=IFFT2(P)；norm=True 时逆变换含 1/(N·M)，False 时归一化因子预乘入 h
- 输入：field [B,N,M] float16/float32/complex64；h [Bh,N,M] 与 field 同 dtype
- 输出：out [B,N,M] 与 field 同 dtype
- 属性：norm bool 默认 False
- 约束：B ∈ [1,256]；N,M ∈ [2,8192] 且均为 2 的幂；取值 [-1,1]，输入有限；Bh = 1 或 Bh = B；h 的 N,M 与 field 相同
- 容差：complex64: atol=rtol=1e-4；float32/float16: atol=rtol=1e-3
- 源码级别：`none`；候选：`—`
- 源码差异：未找到公开的 Ascend C FusedPropStep 实现；ops-fft/fft2_d 只覆盖 complex64、32/64/128，且不融合逐点乘，无法覆盖 2..8192 与实数 dtype；已有 910B 的 1D/2D FFT 基础件，但需要组合/融合，且 fft2_d 仅 32/64/128
- 补充基础件：ops-fft/fft1_d（910B C2C 1D，n≤256 DFT / 512..16384 FFTB）；ops-fft/fft2_d（910B C2C 2D，仅 32/64/128）
- 复用判断：没有公开 FusedPropStep；可用 fft1_d 的行/列 1D FFT 组合 2D FFT，再做逐点乘与逆变换。仍缺单 kernel 融合、2..8192 全尺寸、实数 dtype 和 norm 归一化。

## 05 仿真计算：FourierTransform2D

- 官方题面：[fouriertransform2d](descriptions/fouriertransform2d.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/fouriertransform2d)
- 公式：正变换 exp(-2πi(kn/N+lm/M))；inverse=True 时 exp(+...) 并乘 1/(N·M)
- 输入：x [B,N,M] complex64
- 输出：y [B,N,M] complex64
- 属性：inverse bool 默认 False
- 约束：B ∈ [1,256]；N,M ∈ [2,8192] 且均为 2 的幂；输入分量有限，取值 [-1,1]
- 容差：atol=rtol=1e-3
- 源码级别：`adapt`；候选：`MFG2026-FFT2D`
- 源码差异：ops-fft/fft2_d 只接受 32/64/128 的 C2C，官方要求 2..8192 的 2 的幂；fft2_d 的方向参数可做 forward/backward，但需核对 1/(N·M) 归一化；源码只支持 complex64，官方也是 complex64，这一点匹配；fft1_d 可作为行/列 1D FFT 基础件覆盖 2..8192 的单维；但 2D 组合、inverse 归一化和数据布局仍需实现
- 补充基础件：ops-fft/fft1_d（910B C2C 1D，n≤256 DFT / 512..16384 FFTB / 更大 FFTN）
- 复用判断：fft2_d 只覆盖 32/64/128；fft1_d 提供 910B 1D 基础件（n≤256 DFT、512..16384 FFTB、更大 FFTN），可按行/列组合覆盖官方 2..8192。仍缺 2D 组合/逆变换归一化/大 batch 的实现与验证。

## 06 图计算：GatherCoo

- 官方题面：[gathercoo](descriptions/gathercoo.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/gathercoo)
- 公式：out[i,:] = src[index[i],:]
- 输入：src (M,K) float32/float16/int32/int8；index (E) int32，取值 [0,M)
- 输出：out (E,K) 与 src 同 dtype
- 属性：无
- 约束：M ∈ [1,1000000]；K ∈ [1,4096]；E ∈ [1,4000000]；允许重复 index；越界行为未定义；无需越界检查
- 容差：索引/拷贝精确；dtype 原样
- 源码级别：`reference`；候选：`MFG2026-GNN-GATHER-COO`
- 源码差异：ops-gnn gather_coo 仅 arch35（950），910B 无 arch22；源码 index 为 int64、支持 rank 1–8；官方 index 为 int32、2D [M,K]；ops-sparse gather 是 dense gather，不是 COO 行扩展
- 复用判断：只复用重复行 slab / tile 写出思路；910B 需按官方 int32 2D 契约重写。

## 07 图计算：SegmentMinCsr

- 官方题面：[segmentmincsr](descriptions/segmentmincsr.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/segmentmincsr)
- 公式：out[r,c]=min(src[indptr[r]:indptr[r+1],c])；arg_out 为全局 src 行号
- 输入：src (E,K) float32/float16；indptr (R) int64，R=N+1，单调不减，indptr[N]=E
- 输出：out (N,K) 与 src 同 dtype；arg_out (N,K) int64，段内全局行号
- 属性：无
- 约束：N ∈ [1,1000000]；E ∈ [0,5000000]；K ∈ [1,4096]；允许空段/连续空段/全空段；tie-break 首遇（严格小于才更新）；空段 out=0，arg_out=E（哨兵）
- 容差：values 按题面 atol/rtol；arg_out 逐位一致
- 源码级别：`direct`；候选：`MFG2026-GNN-SEGMENT-MIN-CSR`
- 源码差异：需确认 Python 入口返回 (values, indices) 与官方两个输出张量的绑定方式；源码还支持 sum/max/mean 等，官方只要求 min
- 复用判断：ops-gnn `segment_min_csr` 在 A2/A3 arch22 上已实现 arg_out、空段 0/哨兵、首遇 tie-break，语义高度匹配；优先本机构建验证。

## 08 图计算：RandomWalk

- 官方题面：[randomwalk](descriptions/randomwalk.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/randomwalk)
- 公式：out[i,0]=seed[i]；每步按 rowptr/col 与 rand 选下一节点；孤立点停留
- 输入：rowptr (R) int64；col (E) int64；seed (B) int64；rand (B,walk_length) float32，必选
- 输出：out (B,L) int64，L=walk_length+1
- 属性：walk_length int 默认 16，范围 [1,64]；p float 默认 1.0；q float 默认 1.0
- 约束：N ∈ [1,10000000]；E ∈ [0,20000000]；B ∈ [1,100000]；孤立点停留且每步仍消费 rand；p=q=1 为均匀；p/q≠1 为 node2vec 扩展
- 容差：轨迹逐位一致
- 源码级别：`reference`；候选：`MFG2026-GNN-RANDOM-WALK`
- 源码差异：ops-gnn random_walk 仅 arch35（950），910B 无 arch22；源码用 Philox 计数器 RNG；官方要求外部 rand 输入，保证逐位可比；官方 p/q 为非 1 的 node2vec 扩展，源码接口也需核对
- 复用判断：只参考 Philox/分组采样/孤立点处理；官方确定性 rand 契约必须重写。

## 09 图计算：SpmmSum

- 官方题面：[spmmsum](descriptions/spmmsum.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/spmmsum)
- 公式：out[r,:] = sum_{k∈[rowptr[r],rowptr[r+1])} value[k] * mat[col[k],:]
- 输入：rowptr (R) int64；col (E) int32，行内升序；value (E) float32/float16；mat (N,K) 与 value 同 dtype
- 输出：out (M,K) 与 value 同 dtype
- 属性：无
- 约束：M,N ∈ [1,10000000]；E ∈ [1,20000000]；K ∈ [1,4096]；允许空行，空行输出 0；按 col 升序确定性累加
- 容差：float32: atol=rtol=1e-4；float16: atol=rtol=1e-2
- 源码级别：`adapt`；候选：`MFG2026-SPARSE-SPMM / MFG2026-GNN-SPMM`
- 源码差异：ops-sparse SpMM 是通用 alpha·A·B+beta·C，需要 alpha=1/beta=0 并构造 CSR 描述符；ops-gnn spmm 是 PyTorch 层聚合接口，需把 value/mat 映射到 op/reduce；官方 col 为 int32，源码部分路径为 int64
- 复用判断：复用 A2/arch22 的行分块/group axpy/确定性累加；写薄适配层并验证空行与 dtype。

## 10 图计算：Spspmm

- 官方题面：[spspmm](descriptions/spspmm.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/spspmm)
- 公式：C[i,n]=sum_k A[i,k]·B[k,n]；输出模式交集，行主序
- 输入：indexA (2,E1) int64；valueA (E1) float32/float16；indexB (2,E2) int64；valueB (E2) 与 valueA 同 dtype
- 输出：indexC (2,U) int64，U=m·n；valueC (U) 与 valueA 同 dtype
- 属性：m int 默认 64；k int 默认 64；n int 默认 64
- 约束：M,N,K ∈ [1,10000000]；E1,E2 ∈ [1,5000000]；U=m·n ≤ 250000；COO 行主序、无重复；输出前 E3 个真实条目，其余 (-1,-1)/0；累加和为 0 的坐标保留
- 容差：float32: atol=rtol=1e-4；float16: atol=rtol=1e-2；indexC 逐位一致
- 源码级别：`adapt`；候选：`MFG2026-SPARSE-SPGEMM`
- 源码差异：ops-sparse SpGEMM 输入/输出为 CSR、动态 nnz；官方为 COO + 固定槽位 U=m·n；官方要求未命中槽位 (-1,-1)/0，且保留零和坐标；源码输出 CSR 排序去重；官方维度属性 m/k/n 可大于 index 上限
- 复用判断：复用 symbolic/numeric/merge 多阶段 SpGEMM 内核；需加 COO↔CSR 转换与固定槽位输出层。

## 11 点云计算：Radius

- 官方题面：[radius](descriptions/radius.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/radius)
- 公式：d2=||x[j]-y[i]||²；d2≤r² 且未被 ignore_same_index 剔除则收集 (i,j)，first-K 截断
- 输入：x (N,D) float32/float16；y (M,D) 与 x 同 dtype；ptr_x (Rx) int64；ptr_y (Ry) int64
- 输出：out (2,E) int64，E=M×max_num_neighbors
- 属性：r float 默认 1.0，≥0；max_num_neighbors int 默认 32；ignore_same_index bool 默认 False
- 约束：N,M ∈ [1,1000000]；M×max_num_neighbors ≤ 4000000；D ∈ {2,3}；固定槽位：查询 i 写 [i·K,(i+1)·K)，未命中 (-1,-1)；按 x 扫描序 first-K，不排序；禁止跨 batch
- 容差：索引逐位一致
- 源码级别：`adapt`；候选：`MFG2026-RADIUS`
- 源码差异：DrivingSDK radius 输出动态 [2,E]，官方要求定槽位 [2,M×K] 且未命中 (-1,-1)；源码约束 Ndim<9、batch≤1024、单 batch 点数≤512；官方 D∈{2,3}、N/M≤1e6；官方有 ignore_same_index；源码需核对该属性
- 复用判断：复用 batch 分核、平方距离、扫描序截断；需重写定槽位输出、2/3D 与更大 shape。

## 12 点云计算：GridCluster

- 官方题面：[gridcluster](descriptions/gridcluster.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/gridcluster)
- 公式：cluster[i]=Σ_d trunc((pos[i,d]-start[d])/size[d])·stride[d]；stride[0]=1，stride[d]=prod(num_voxels[0:d])
- 输入：pos (N,D) float32/float16；size (D)；start (D)；end (D)
- 输出：out (N) int64
- 属性：无
- 约束：N ∈ [1,10000000]；D ∈ {2,3}；size 各维 >0；end≥start；pos-start≥0；prod(num_voxels)<2^28；stride[0]=1，第 0 维变化最快
- 容差：int64 逐位一致
- 源码级别：`reference`；候选：`MFG2026-GNN-GRID-CLUSTER`
- 源码差异：ops-gnn grid_cluster 仅 arch35（950），910B 无 arch22；源码 stride 顺序与官方一致（dim0 最快），但 SIMT 路径需重写；个人仓 flying_whj/CANN_grid_cluster 也是 arch35，但有设计文档/golden 可读
- 复用判断：复用网格量化、stride 组合、subnormal/倒数快路径；910B 需写 Vector/SIMT 版本，不能直接构建 arch35。

## 13 点云计算：FurthestPointSample

- 官方题面：[furthestpointsample](descriptions/furthestpointsample.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/furthestpointsample)
- 公式：首点索引 0；每轮更新到已选集合的最小平方距离，选最大者；并列取最小索引
- 输入：points [B,N,3] float32/float16
- 输出：idx [B,npoint] int32
- 属性：npoint int 默认 128，1≤npoint≤N
- 约束：points 最后一维 3，取值 [-1,1]；B,N,npoint 为正整数；输入有限，不支持 NaN/Inf；float16 必须保持 half 精度，不得提升到 float32；并列距离取最小输入索引；不额外排除已选点
- 容差：索引逐位一致
- 源码级别：`direct`；候选：`MFG2026-FPS`
- 源码差异：DrivingSDK 文档还支持 bfloat16，官方只要求 float32/float16；源码有总元素 B*N*3≤383166 的约束；官方测试规模远小于此；需本机确认 half 路径与并列 tie-break 逐位一致
- 复用判断：官方题面明确以 mx_driving Ascend `furthest_point_sampling` 为参考，DrivingSDK 源码应为主候选；优先构建并跑官方 18 组 shape。

## 14 点云计算：BallQuery

- 官方题面：[ballquery](descriptions/ballquery.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/ballquery)
- 公式：d²=(x_k-x_q)²+(y_k-y_q)²+(z_k-z_q)²；d²<radius² 命中；按 k 升序取前 nsample
- 输入：xyz [B,N,3] float32/float16；new_xyz [B,M,3] 与 xyz 同 dtype
- 输出：idx [B,M,nsample] int32
- 属性：radius float 默认 0.4，>0；nsample int 默认 16，>0
- 约束：B,N,M 正整数；N≥1；xyz/new_xyz 最后一维 3，常规 [-1,1]；d² < radius² 严格小于；first-K 扫描序；命中不足 nsample 时尾部重复首个命中；零命中全 0；float16 输入必须先转 float32 再算距离和阈值
- 容差：int32 索引逐元素一致
- 源码级别：`adapt`；候选：`MFG2026-MATH-STACK-BALL-QUERY`
- 源码差异：stack_ball_query 输入是 2D 堆叠布局 + batch counts，官方是 [B,N,3]/[B,M,3]；stack_ball_query 在 float16 路径把 radius² 先转 half，官方要求 float32 阈值语义；ops-math 直接版 ball_query 有 min_radius/max_radius 语义，官方只有单 radius；且直接版只有 arch35
- 复用判断：复用按 center 分段、流式点云、提前停；需改 [B,*,3] 布局、float32 阈值，并确认单半径语义。

## 15 机器人抓取：CylinderQuery

- 官方题面：[cylinderquery](descriptions/cylinderquery.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/cylinderquery)
- 公式：Δ=xyz[b,k]-new_xyz[b,j]；[x_l,y_l,z_l]^T=rot[b,j]^T·Δ；命中条件 y_l²+z_l²<radius² 且 hmin<x_l<hmax
- 输入：xyz [B,N,3] float32/float16；new_xyz [B,M,3] 与 xyz 同 dtype；rot [B,M,3,3] 与 xyz 同 dtype
- 输出：idx [B,M,nsample] int32
- 属性：radius float 默认 0.05，>0；hmin float 默认 -0.02；hmax float 默认 0.04，>hmin；nsample int 默认 32，1≤nsample≤N
- 约束：B,N,M 正整数；N≥1；rot row-major，一般有限 3×3，不要求正交；局部坐标 = rot^T @ delta（不是 rot @ delta）；严格比较：y_l²+z_l²<radius² 且 hmin<x_l<hmax；按 xyz 输入索引升序 first-K；不足重复首个命中；零命中全 0；float16 输入必须先转 float32 再变换/比较
- 容差：int32 索引逐元素一致
- 源码级别：`adapt`；候选：`MFG2026-CYLINDER-QUERY`
- 源码差异：DrivingSDK kernel 的旋转方向是 rot^T·Δ，与官方一致；但 Python 层 `sortRes` 会排序后截断，官方要求扫描序 first-K；DrivingSDK 输出全量候选，再由 Python 排序；官方直接输出 [B,M,nsample]；float16 输入需确认 host/kernel 是否先转 float32
- 复用判断：复用旋转、双 mask、严格比较；必须去掉排序，改为扫描序 first-K + 填充规则。
