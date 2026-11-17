# 制造专场：6 个 adapt 题的适配蓝图

本页把官方接口与现有 Ascend 源码之间的“最后一层”写成可执行步骤。它不是提交代码，也不是本机结果；用途是让后续实现者知道**必须改什么、不能改什么、怎么验**。

适用条目：

| 题 | 官方候选 | 现有源码 | 适配级别 |
|---|---|---|---|
| 05 `FourierTransform2D` | `ops-fft/fft2_d` + `fft1_d` | 2D 仅 32/64/128；1D C2C 覆盖更宽 | `adapt` |
| 09 `SpmmSum` | `ops-sparse/spmm` 或 `ops-gnn/spmm` | CSR SpMM，通用 alpha/beta 或 PyTorch 层接口 | `adapt` |
| 10 `Spspmm` | `ops-sparse/spgemm` | CSR 输入/输出、动态 nnz | `adapt` |
| 11 `Radius` | `DrivingSDK/radius` | 动态输出、KD-tree/单 batch 上限 | `adapt` |
| 14 `BallQuery` | `ops-math/stack_ball_query` | 堆叠布局、fp16 阈值转 half | `adapt` |
| 15 `CylinderQuery` | `DrivingSDK/cylinder_query` | kernel 语义一致，Python 层排序 | `adapt` |

统一原则：

- 只按官方 shape/dtype/layout/属性特化，**不改 golden、容差、计时协议**；
- 不把 950/arch35 代码直接搬到 910B；不伪造 910C 结果；
- 所有 NPU 运行走唯一资源锁；
- 索引类输出（`arg_out`、FPS idx、Radius/Ball/Cylinder idx、Spspmm indexC）必须逐位一致，不能只用 atol/rtol。

## 1. FourierTransform2D（05）

官方：`x [B,N,M] complex64`，`y` 同形；`B∈[1,256]`，`N,M∈[2,8192]` 且为 2 的幂；`inverse bool`；容差 `atol=rtol=1e-3`。

现有：`fft2_d` 仅 32/64/128 的 C2C；`fft1_d` 是 1D C2C，覆盖 n≤256 DFT、512..16384 FFTB、更大 FFTN。

适配步骤：

```text
for b in range(B):
    # 行变换：对每个 n，沿最后一维做长度 M 的 1D FFT
    for n in range(N):
        fft1_d(M, x[b,n,:], tmp[b,n,:], direction=forward/inverse)
    # 列变换：对每个 m，沿第一维做长度 N 的 1D FFT
    for m in range(M):
        fft1_d(N, tmp[b,:,m], y[b,:,m], direction=forward/inverse)
    if inverse:
        y[b] *= 1.0 / (N * M)
```

要点：

- `fft1_d` 的 `direction` 与 `inverse` 对应；官方 inverse 含 `1/(N·M)`，必须显式缩放；
- 行/列访问需要转置或 stride 布局；`fft1_d` 的纵向 plan 在 910B 上支持 C2C（Stride kernel），但仍需核对 non-contiguous 输入；
- 2..8192 的每个 2 的幂都要跑；不能只跑 32×32；
- `inverse=false` 与 `true` 分别验证；复数实虚交错、行主序；
- 精度对照用解析冲激或独立 CPU DFT，不依赖 `torch.fft` 自身版本。

验收：

```text
B=1, N=M ∈ {2,4,8,16,32,64,128,256,512,1024,2048,4096,8192}
forward + inverse
max_abs/max_rel <= 1e-3 (atol=rtol); 记录 best/median 时延与形状
```

## 2. SpmmSum（09）

官方：`rowptr (M+1) int64`，`col (E) int32` 行内升序，`value (E) fp32/fp16`，`mat (N,K)` 同 dtype，`out (M,K)` 同 dtype；空行输出 0；M,N≤1e7，E≤2e7，K≤4096；fp32 容差 1e-4，fp16 1e-2。

现有：`ops-sparse` 是 `C = alpha·op(A)·op(B)+beta·C` 的 CSR SpMM；`ops-gnn` 是 PyTorch 层聚合接口。

适配步骤：

```text
assert rowptr.dtype == int64 and col.dtype == int32
assert rowptr[0] == 0 and rowptr[M] == E
assert col 行内升序; value.dtype == mat.dtype
# ops-sparse:
#   A = CSR(rowptr, col, value)  (M x N)
#   B = mat                      (N x K)
#   C = out                      (M x K)
#   alpha = 1, beta = 0, opA/opB = NON_TRANSPOSE
#   调用 SpMM，输出写回 out
```

要点：

- `ops-sparse` 的索引/描述符可能是 int32；官方 `rowptr` 是 int64。若描述符只收 int32，必须做边界检查/转换，不能静默截断；
- 空行必须显式写 0；`beta=0` 时不要让 `out` 保留旧值；
- 行内 `col` 升序是确定性累加序，不能改；
- fp16 用 fp32 累加，最后转回 fp16；对照官方容差 1e-2；
- 不要把 `ops-gnn` 的 `spmm(op,reduce)` 直接当官方接口；它需要额外的 PyTorch 张量/层语义。

验收：

```text
小图：M=N=4, K=2, 手工 golden
边界：空行、单行、hub 行、全空图（E=0 是否允许以题面为准）
精度：fp32 1e-4 / fp16 1e-2；空行必须逐位 0
```

## 3. Spspmm（10）

官方：`indexA (2,E1) int64`、`valueA (E1)`、`indexB (2,E2) int64`、`valueB (E2)`；属性 `m,n,k` 默认 64；输出 `indexC (2,U) int64`、`valueC (U)`，`U=m·n`，`U≤250000`；前 E3 个真实条目，其余 `(-1,-1)/0`；零和坐标保留；COO 行主序、无重复。

现有：`ops-sparse/spgemm` 是 CSR SpGEMM，输出规范化 CSR、动态 nnz。

适配步骤：

```text
# 1) COO -> CSR
A = coo_to_csr(indexA, valueA, shape=(M,K))
B = coo_to_csr(indexB, valueB, shape=(K,N))
# 2) 调 CSR SpGEMM：C = A @ B
C = spgemm(A, B, alpha=1, beta=0)
# 3) CSR -> COO，按行主序展开
coo = csr_to_coo(C)
# 4) 固定槽位 U=m*n
indexC = full((2,U), -1, int64); valueC = zeros(U)
for t, (i,j,v) in enumerate(sorted(coo)):
    assert t < U
    indexC[:,t] = (i,j); valueC[t] = v
# 剩余槽位保持 (-1,-1)/0
```

要点：

- 官方 `m,n,k` 是**属性**，不是从 index 推断；`U=m·n` 可大于实际 nnz；
- 零和坐标必须保留（结构由稀疏模式决定，不能因 `v==0` 丢弃）；
- 官方要求 COO 行主序、无重复；CSR SpGEMM 的输出通常已按行/列排序，但转换后仍要显式排序/去重；
- `indexA/indexB` 是 int64，而 `ops-sparse` 的 CSR 索引可能是 int32；必须检查 N/K/M ≤ int32 或做安全转换；
- `U≤250000` 是输出槽位上限，不能按动态 nnz 分配后越界。

验收：

```text
手工 2x2/2x3/3x2；空行/空列；无交集；数值抵消为 0；U 边界 250000
indexC 逐位一致；valueC 按 fp32 1e-4 / fp16 1e-2
```

## 4. Radius（11）

官方：`x (N,D)`、`y (M,D)` fp32/fp16；`ptr_x (Rx) int64`、`ptr_y (Ry) int64`；`out (2, M*K) int64`，`K=max_num_neighbors`；`r>=0`；`D∈{2,3}`；`N,M≤1e6`，`M*K≤4e6`；`ignore_same_index bool`。

规则：

```text
for i in 0..M-1:
    base = i * K
    cnt = 0
    for j in x 的批内扫描序:
        if ignore_same_index and global_j == i: continue
        d2 = ||x[j]-y[i]||^2   # float32 计算
        if d2 <= r*r:
            out[0, base+cnt] = i
            out[1, base+cnt] = j
            cnt += 1
            if cnt == K: break
    # 剩余槽位保持 (-1,-1)
```

要点：

- 官方是 `d² <= r²`（含边界），不是 `<`；
- 扫描序 first-K，不是按距离排序；不能复用 torch_cluster KD-tree 的输出顺序；
- 输出是定槽位 `[2, M*K]`，每查询独占 `[i*K,(i+1)*K)`；未命中填 `(-1,-1)`，不能填 0；
- `y` 按 i 升序；`x` 按批内 j 升序；`ptr_x/ptr_y` 只用于 batch 边界，禁止跨 batch；
- `ignore_same_index` 比较全局序号；多 batch 时不要误用批内局部索引；
- fp16 输入按 float32 计算距离（官方明确）；
- 源码的 `Ndim<9`、单 batch≤512、动态输出都要替换/放宽。

验收：

```text
D=2/3；r=0（全 (-1,-1)）；边界 d²==r²（必须命中）
空 batch；单 batch；多 batch；ignore_same_index 开/关
索引逐位一致；不能按距离排序
```

## 5. BallQuery（14）

官方：`xyz [B,N,3]`、`new_xyz [B,M,3]` fp32/fp16；`idx [B,M,nsample] int32`；`radius>0`、`nsample>0`；严格 `d²<radius²`；扫描序 first-K；不足重复首个命中；零命中全 0；fp16 先转 fp32。

适配步骤：

```text
for b, j:
    cnt = 0
    for k in 0..N-1:
        d2 = sum((xyz[b,k]-new_xyz[b,j])**2)   # fp32
        if d2 < radius*radius:
            if cnt == 0: idx[b,j,:] = k          # 先填满首个命中
            idx[b,j,cnt] = k
            cnt += 1
            if cnt == nsample: break
    # 零命中：idx[b,j,:] 保持 0
```

要点：

- 官方是单 `radius`；`ops-math/ball_query` 的 `min_radius/max_radius` 和 `d²==0` 命中要裁掉；
- `stack_ball_query` 是 2D 堆叠布局 + batch counts；官方是 `[B,N,3]/[B,M,3]`，需要改布局；
- `stack_ball_query` 的 fp16 路径把 `radius²` 转 half；官方要求 fp32 阈值，必须改；
- 严格 `<`，不是 `<=`；
- 填充规则：不足 nsample 时尾部重复**第一个命中**；零命中全 0。

验收：

```text
B=1/多 batch；N=1；命中 0/1/nsample-1/nsample/>nsample
d²==r² 边界（不命中）；fp16 输入
idx 逐位一致
```

## 6. CylinderQuery（15）

官方：`xyz [B,N,3]`、`new_xyz [B,M,3]`、`rot [B,M,3,3]` fp32/fp16；`idx [B,M,nsample] int32`；`radius>0`、`hmin<hmax`、`nsample≤N`；局部坐标 `rot^T·Δ`；严格 `y_l²+z_l²<radius²` 且 `hmin<x_l<hmax`；扫描序 first-K；不足重复首个命中；零命中全 0；fp16 先转 fp32。

适配步骤：

```text
for b, j:
    R = rot[b,j]  # row-major 3x3
    cnt = 0
    for k in 0..N-1:
        d = xyz[b,k] - new_xyz[b,j]
        x_l = R[0,0]*d0 + R[1,0]*d1 + R[2,0]*d2   # rot^T 第一列 = rot[:,0]
        y_l = R[0,1]*d0 + R[1,1]*d1 + R[2,1]*d2
        z_l = R[0,2]*d0 + R[1,2]*d1 + R[2,2]*d2
        if y_l*y_l + z_l*z_l < radius*radius and hmin < x_l < hmax:
            if cnt == 0: idx[b,j,:] = k
            idx[b,j,cnt] = k
            cnt += 1
            if cnt == nsample: break
```

要点：

- `rot` 是一般有限 3×3，不要求正交；不要做转置归一化/求逆；
- `rot^T·Δ` 的写法不要写成 `rot·Δ`；AnyGrasp 源码用 `r0*dx+r3*dy+r6*dz` 等，正好是 `rot^T`；
- 必须去掉 DrivingSDK Python 层的 `sortRes`；官方是扫描序 first-K，不是按距离排序；
- 严格 `<` 两个边界；`hmin<x_l<hmax` 不含端点；
- fp16 输入先转 fp32 再变换/比较；`radius²/hmin/hmax` 用 fp32；
- 输出 `[B,M,nsample]`，不是动态候选列表。

验收：

```text
rot=单位矩阵/非正交矩阵/负行列式
边界 y²+z²==r²（不命中）、x_l==hmin/hmax（不命中）
命中 0/1/nsample-1/nsample/>nsample；fp16 输入
idx 逐位一致；禁止排序
```

## 7. 两个 direct 候选的验证清单

### SegmentMinCsr（07）

```text
官方：src (E,K) fp32/fp16, indptr (R) int64, out (N,K), arg_out (N,K) int64
规则：out[r,c]=min(src[indptr[r]:indptr[r+1],c])
      arg_out=段内全局行号；首遇 tie（严格 <）
      空段 out=0, arg_out=E；E=0 全空图也合法
```

验证：

- `ops-gnn` arch22 `segment_min_csr` 构建；先小 shape，再空段/全空图；
- 首遇 tie：构造全同值段；`arg_out` 必须是段内第一个全局行号；
- 空段 `out=0/arg_out=E`；`E=0` 时 `arg_out=0`；
- fp16/fp32 分别测；`indptr int64`；
- 与 pyg-lib CPU 逐位对照（只对照索引/值，不把 pyg-lib 当 Ascend golden）。

### FPS（13）

```text
官方：points [B,N,3] fp32/fp16, idx [B,npoint] int32
规则：首点 0；每轮更新到已选集合的最小平方距离；选最大者
      并列取最小输入索引；不额外排除已选点；npoint<=N
```

验证：

- 18 组官方 shape；`npoint=1`、`npoint=N`；
- 并列：正方形四点、圆上对称点、全点重合（应重复 0）；
- fp16：逐维 half 舍入 vs FP32 累加两种候选，逐位对照官方 mx_driving；
- 首点固定 0；不要把 torch_cluster 的 `random_start`/ratio 带进来。

## 8. 统一验收表

| 项 | 要求 |
|---|---|
| 形状 | 官方 18 组（每题以题面/测试规模为准） |
| dtype | 官方列出的全部 dtype；fp16 中间精度按题面 |
| 索引 | 逐位一致；不能只报 atol/rtol |
| 数值 | 官方 atol/rtol；同时记录 max_abs/max_rel |
| 空段/空行/零命中/未命中 | 按题面填充，不能复用其他题的习惯 |
| 计时 | 官方计时口径；记录 warmup/repeat/median/mean；不以 Profile 代替 |
| 设备 | 只报 910B3；910C 另取证 |
| 负例 | 失败日志保留；不删、不改 golden |

## 9. 参考

- 官方题面：[`official/descriptions/`](official/descriptions/)
- 官方规格：[`official/specs.json`](official/specs.json)
- 上游语义核对：[`../../docs/research/MFG2026_ALGORITHM_REFERENCES_20261117.md`](../../docs/research/MFG2026_ALGORITHM_REFERENCES_20261117.md)
- 独立 CPU 参考：[`reference/`](reference/)（9 项语义测试；不是官方隐藏 golden）
- 本机验证协议：[`../../skills/ascend-910b-local-validation/SKILL.md`](../../skills/ascend-910b-local-validation/SKILL.md)
