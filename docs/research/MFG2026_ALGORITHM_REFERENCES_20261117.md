# 制造专场：上游算法参考核对（2026-11-17）

本页用公开上游 CPU/CUDA 参考实现核对官方题面里的边界语义。它**不是 Ascend 实现**，也不冒充题解；用途是回答三个问题：

1. 官方题面的比较符、tie-break、空段/未命中填充是否与业界参考一致；
2. 现有 Ascend 源码的差异是“实现风格不同”还是“语义真的变了”；
3. 写适配层时哪些细节不能凭经验猜。

证据级别：`O`（官方题面）+ `S`（上游源码/行号）+ `A`（本文归纳）。没有本机 910B3 运行结果。

## 1. 参考清单

| 题 | 上游参考 | 固定 commit | 许可证 | 核对结论 |
|---|---|---|---|---|
| 07 `SegmentMinCsr` | pyg-lib `pyg_lib/csrc/ops/cpu/segment_csr_kernel.cpp` | `d13e20662aaa075c82ab9c814f8c39789425bf7a` | MIT | 与官方完全一致：`arg_out` 哨兵 `E`、空段 `out=0`、严格 `<` 首遇 |
| 13 `FurthestPointSample` | torch_cluster `csrc/cpu/fps_cpu.cpp` | `e9a855c284b45edcbf0282cf70ac09bee0ce4e49` | MIT | `argmax` 首最大值 = 最小索引 tie；`random_start`/ratio 不是官方接口 |
| 11 `Radius` | torch_cluster `csrc/cpu/radius_cpu.cpp` | `e9a855c284b45edcbf0282cf70ac09bee0ce4e49` | MIT | `ignore_same_index` 语义一致；KD-tree 动态输出/顺序与官方定槽位/扫描序不同 |
| 14 `BallQuery` | AnyGrasp SDK `pointnet2/_ext_src/src/ball_query_gpu.cu` | `b8eaafc9eca7babd5208e7a5ade3c561060be4c5` | 文件头 MIT | 严格 `d² < r²`、扫描序 first-K、不足重复首个命中、零命中全 0 |
| 15 `CylinderQuery` | AnyGrasp SDK `pointnet2/_ext_src/src/cylinder_query_gpu.cu` | `b8eaafc9eca7babd5208e7a5ade3c561060be4c5` | 文件头 MIT | `rot^T·Δ`、严格 `d²<r²` 且 `hmin<x<hmax`、扫描序 first-K、不足重复首个命中 |
| 05/04 FFT | `cann/ops-fft` 的 `fft1_d`/`rfft1_d`/`irfft1_d` 文档 | `65f5ff1dee32802562a9412789b0c7719ef99269` | CANN EULA 2.0 | 尺寸/FP32/归一化限制已在 `ops-fft/README.md` §3.1 记录 |
| 10 `Spspmm` | 官方题面 + cuSPARSE 语义；未找到可固定的同接口开源参考 | — | — | COO + 固定槽位 `U=m·n` 仍以官方题面为准 |

链接：

- pyg-lib segment：<https://github.com/pyg-team/pyg-lib/blob/d13e20662aaa075c82ab9c814f8c39789425bf7a/pyg_lib/csrc/ops/cpu/segment_csr_kernel.cpp#L256-L404>
- torch_cluster FPS：<https://github.com/rusty1s/pytorch_cluster/blob/e9a855c284b45edcbf0282cf70ac09bee0ce4e49/csrc/cpu/fps_cpu.cpp#L7-L53>
- torch_cluster Radius：<https://github.com/rusty1s/pytorch_cluster/blob/e9a855c284b45edcbf0282cf70ac09bee0ce4e49/csrc/cpu/radius_cpu.cpp#L31-L110>
- AnyGrasp BallQuery：<https://github.com/graspnet/anygrasp_sdk/blob/b8eaafc9eca7babd5208e7a5ade3c561060be4c5/pointnet2/pointnet2/_ext_src/src/ball_query_gpu.cu#L14-L58>
- AnyGrasp CylinderQuery：<https://github.com/graspnet/anygrasp_sdk/blob/b8eaafc9eca7babd5208e7a5ade3c561060be4c5/pointnet2/pointnet2/_ext_src/src/cylinder_query_gpu.cu#L14-L61>

## 2. CylinderQuery：`rot^T·Δ` 已由 AnyGrasp 源码确认

AnyGrasp 的 kernel 按 row-major 读 `rot[9]` 为 `r0..r8`，对 `Δ=xyz-new_xyz` 计算：

```text
x_rot = r0*Δx + r3*Δy + r6*Δz
y_rot = r1*Δx + r4*Δy + r7*Δz
z_rot = r2*Δx + r5*Δy + r8*Δz
```

这正是 `rot^T·Δ`：`rot` 的第一列 `(r0,r3,r6)` 与 `Δ` 点积得到 `x_rot`。官方题面写的“`rot` 的列与 `delta` 做内积”与源码一致；`rot` 不要求正交，直接使用原始 9 个元素，不要做转置归一化/求逆。

命中与填充：

```text
d2 = y_rot² + z_rot²
命中 = (d2 < radius²) && (hmin < x_rot < hmax)
k 从 0 到 N-1 升序扫描；cnt==nsample 立即停
第一次命中时先把整行 idx 填成 k；之后按 cnt 覆盖
零命中：idx 保持 torch::zeros 的全 0
```

所以官方“扫描序 first-K、不足重复首个命中、零命中全 0”不是新规则，而是 AnyGrasp 的原语义。DrivingSDK kernel 的旋转/严格比较与它一致；**唯一必须改的是 Python 层 `sortRes`**：官方不允许按距离排序后截断。

浮点：AnyGrasp 的接口只接受 float；官方额外允许 float16 输入，并明确“float16 输入必须先转 float32 再变换/比较”。因此 `radius²`、`hmin/hmax`、`rot^T·Δ` 都应在 FP32 中算，不能把阈值先截成 half。

## 3. BallQuery：严格 `<` 与 first-K 已确认

AnyGrasp `ball_query_gpu.cu` 的核心：

```text
d2 = (new_x-x)² + (new_y-y)² + (new_z-z)²
if (d2 < radius²):
    第一次命中时把整行 idx 填成 k
    然后 idx[cnt] = k; cnt++
k 升序扫描，cnt==nsample 停止
零命中：idx 保持全 0
```

这与官方“`d² < radius²`、扫描序 first-K、不足重复首个命中、零命中全 0”完全一致。

注意两个变体：

- 原始 PointNet++ 的 `tf_grouping_g.cu` 先 `sqrt` 再与 `radius` 比较，并用 `max(...,1e-20f)` 避免零距离；对正半径，`d²<r²` 与 `sqrt(d²)<r` 等价，但官方明确用 `d²`，AnyGrasp 的写法更贴近。
- `ops-math/stack_ball_query` 的 FP16 路径把 `radius²` 先转 half；官方要求 float16 输入先转 float32 再算距离/阈值，因此这是**语义差异**，不是单纯的精度差异。

## 4. Radius：参考实现的顺序不能直接搬

torch_cluster `radius_cpu.cpp` 用 nanoflann KD-tree：

```text
radiusSearch(y_i, r*r, ..., params.sorted=false)
每个 query 取前 max_num_neighbors 个命中
ignore_same_index 时比较全局索引 x_start + ret_matches[j].first != i
输出是动态 [E,2]，最后转置成 [2,E]
```

可与官方对齐的部分：

- `r*r` 作为半径平方阈值；
- `ignore_same_index` 比较的是**全局输入索引**，不是 batch 内局部索引；
- 查询点 `y` 与参考点 `x` 的方向：输出 `(x_index, y_index)`，与官方 `(i,j)` 语义一致（以题面为准）。

不能直接搬的部分：

- KD-tree 的 `sorted=false` 返回顺序不是输入扫描序；官方要求按 `x` 索引升序 first-K；
- 输出是动态 `[2,E]`；官方要求定槽位 `[2, M*K]`，未命中 `(-1,-1)`；
- 源码单 batch 点数上限 512、`Ndim<9`；官方 `D∈{2,3}`、`N,M≤1e6`、`M*K≤4e6`；
- 官方是 `d² ≤ r²`（包含边界），BallQuery/CylinderQuery 是严格 `<`；三者不要共用一个比较符。

## 5. FPS：torch_cluster 的 tie 是最小索引，但 count/ratio 不是官方接口

torch_cluster `fps_cpu.cpp`：

```text
get_dist(x, idx) = (x - x[idx]).pow_(2).sum(1)
out[0] = start_idx（random_start 时 rand()%N，否则 0）
每轮 argmax = dist.argmax()   # PyTorch argmax 返回第一个最大值
dist = min(dist, get_dist(x, argmax))
```

`argmax` 的首最大值语义就是官方“距离并列取最小输入索引”。官方制造题固定首点 0、显式 `npoint`，因此对应 `random_start=false` 且直接采样 `npoint` 轮；torch_cluster 的 `ratio`/`ceil`/`rand()` 都不属于官方契约。

FP16 距离有两种可能实现，必须逐位对照官方 `mx_driving`：

1. **逐维 half 舍入**：`delta = (x-x[idx]).half()`，`square = (delta*delta).half()`，`dist = (dist + square).half()`；950 社区仓的 golden 采用这种写法；
2. **FP32 累加后转回**：`delta.float()`、平方和累加用 FP32，最后再转 half。

官方题面只说“按输入数据类型计算平方欧氏距离”“float16 必须保持 half 精度”，没有写明逐维舍入还是 FP32 累加；这两种写法在边界点会选出不同索引，不能凭经验选。正确做法是用官方算子跑 18 组 shape，对两种候选逐位比对。

## 6. SegmentMinCsr：pyg-lib 源码与官方逐条一致

pyg-lib `segment_min_csr_kernel`（`segment_csr_kernel.cpp:279-404`）：

```text
arg_out 初始化为 sentinel = src.size(dim)（即 E）
out 无 optional_out 时初始化为 numeric_limits::max()
内层 e 从 row_start 到 row_end 顺序扫描，严格 if (v < *out_slot)
  -> tie 保留第一次出现
空行：out.masked_fill_(arg_out == sentinel, 0)
       -> out=0，arg_out 保持 E
arg_out dtype 为 int64（indptr.options()）
```

这与官方题面完全一致：`out=min`、`arg_out` 全局行号、首遇 tie、空段 `out=0`/`arg_out=E`、`indptr int64`。因此 `ops-gnn/segment_min_csr` 的 arch22 实现可以继续标为 `direct` 候选；但仍需本机 910B3 构建来证明它能编译/运行，不能只凭源码相似。

补充：`HIT1920/torch_scatter_npu` 的 `SegmentMinCsr` 不是这个语义（int32 indptr、无 arg_out、空段填最大值）；它的 `Segment_Coo` min/max 相等极值保留最后一次出现。两条都不能直接当官方实现。

## 7. 对仓库条目的影响

| 条目 | 本次核对后的写法 |
|---|---|
| `MFG2026-GNN-SEGMENT-MIN-CSR` | `direct` 更有依据；引用 pyg-lib 行号；仍标 `not-run`，下一步 910B3 构建 |
| `MFG2026-FPS` | tie=最小索引已由 torch_cluster `argmax` 确认；FP16 逐维舍入 vs FP32 累加列为待验证 |
| `MFG2026-RADIUS` | `ignore_same_index` 语义一致；KD-tree 顺序/动态输出不能复用；官方是 `≤` |
| `MFG2026-MATH-STACK-BALL-QUERY` | 严格 `<`/first-K 有 AnyGrasp 依据；FP16 `radius²` 转 half 是明确差异 |
| `MFG2026-CYLINDER-QUERY` | `rot^T·Δ`/严格边界/first-K 有 AnyGrasp 依据；只差 Python 层排序 |

本页只引用上游仓库的链接和短公式，不复制第三方源码；AnyGrasp 相关文件头标注 MIT，pyg-lib/torch_cluster 为 MIT，PointNet++ 为 MIT。实际复用代码前仍应阅读对应仓库的 LICENSE。

## 8. 复现

```bash
# pyg-lib
git clone --depth 1 https://github.com/pyg-team/pyg-lib
# torch_cluster
git clone --depth 1 https://github.com/rusty1s/pytorch_cluster
# AnyGrasp SDK（cylinder/ball query 参考）
git clone --depth 1 https://github.com/graspnet/anygrasp_sdk
# PointNet++（原始 ball query 变体）
git clone --depth 1 https://github.com/charlesq34/pointnet2
```

核对时以本文表中的 commit 为准；上游后续可能修改接口或语义。
