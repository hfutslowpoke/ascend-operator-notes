# Hansenwuwuwu/ascend-c-competition-archive：2026 电力行业算子源码深读

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/Hansenwuwuwu/ascend-c-competition-archive` |
| 作者/维护者 | Hansenwuwuwu |
| 场景 | 昇腾 AI 行业创新大赛——2026 电力行业算子专场 |
| 访问日期 | 2026-11-16 |
| 默认分支 | `main` |
| 固定 commit | `37e07aba5273e280d2d227d8764c5e49b7eb3eae` |
| 主要 DSL | Ascend C（C++，`__aicore__` / `__global__`） |
| 算子族 | 检测框解码融合、图样条卷积、几何最近距离、SwiGLU、线性代数（pinv/det/lstsq/eigh/solve） |
| 许可证 | 仓库未声明 LICENSE/COPYING；`THIRD_PARTY_NOTICES.md` 说明公开可读不等于可再分发 |
| 本地缓存 | `repository-research-cache/Hansenwuwuwu_ascend-c-competition-archive` |
| 与既有归档关系 | 新来源；题目与 GroupedMatmulGelu 不同，未重复 |
| 本报告范围 | 深读 FireCandidateDecodeFusion rank2、SplineConv rank1、TreeLineMinDistance rank1 |

### 1.1 归档内容

该仓库不是单一比赛提交，而是对 2026 电力行业算子专场公示材料的整理：

- 10 道题；2026-11-08 公示快照含 9 道题的 27 份提交；
- 2026-11-09 另收到 SwiGLU 前三名 3 份 ZIP，作为补充材料；
- 271 个源码文件、139,460 行文本；每份 ZIP、源码文件有 SHA-256 / 覆盖账本；
- 题面 PDF、下载清单、逐文件审读记录、设计复盘都在仓库内；
- 维护者明确声明：**没有重新编译或运行任何归档代码**。

因此本文能使用的证据是：

| 证据 | 级别 | 用途 |
|---|---|---|
| 固定 commit 的源码 | `S+L` | 机制、数据流、行号 |
| 公示页分数/名次 | `O` | 证明该提交在当届公开结果中的相对位置 |
| 维护者的逐行审计 | `A` | 交叉核对，但不作为独立测量 |
| 逐 case 时延、本机编译/运行 | 无 | 不能声称加速比或通用正确性 |

公示分数（来自 `archives/2026-electric-power/README.md` 表格）：

| 题目 | 公示顺序 | 原榜名次 | 队伍 | 分数 | 提交 ID |
|---|---:|---:|---|---:|---:|
| FireCandidateDecodeFusion | 2 | 2 | Delicate02 的队伍 | 83.26 | 1176342 |
| SplineConv | 1 | 1 | Zhanng-Whu 的队伍 | 89.16 | 1174398 |
| TreeLineMinDistance | 1 | 1 | gcw_OTnEiQgj 的队伍 | 91.85 | 1175341 |

这些分数不是同一题内的“加速比”，也不跨题可比；本文只用它们说明源码来自当届公开获奖提交。

比赛环境记录为 CANN 9.0.0.0、Ascend 910B4、20 AIC。该环境来自题面/公示记录，不是本机验证；源码本身不能证明在其它 910B/910C 或 CANN 版本上可编译。

## 2. FireCandidateDecodeFusion rank2：AoS→SoA + 选择性高精度修复

### 2.1 语义与契约

输入：

- `raw_pred[B,N,6]`，FP16/FP32，字段 `[tx,ty,tw,th,obj_logit,fire_logit]`；
- `priors[N,5]`，FP32，字段 `[gx,gy,stride,anchor_w,anchor_h]`，batch 共享。

对每个 `(b,n)`：

```text
sx = sigmoid(tx), sy = sigmoid(ty), sw = sigmoid(tw), sh = sigmoid(th)
x = (2*sx - 0.5 + gx) * stride
y = (2*sy - 0.5 + gy) * stride
w = (2*sw)^2 * anchor_w * stride
h = (2*sh)^2 * anchor_h * stride
fire_score = sigmoid(obj_logit) * sigmoid(fire_logit)
```

输出 `[B,N,5]`，顺序固定 `[x,y,w,h,fire_score]`，dtype 与 `raw_pred` 相同；FP16 输入也必须用 FP32 完成 sigmoid、平方、坐标和分数中间，最后再转 FP16。约束：`B∈[1,8]`、`N∈[1,32768]`、尾块合法、不得过滤/排序/NMS。

### 2.2 固定源码位置

| 内容 | 路径/行 |
|---|---|
| Host 分路与 tiling 计划 | `.../rank2_submission_1176342/code/op_host/fire_host_plan.h:10-120` |
| 64-bit 打包 tiling 的解包 | `.../code/op_kernel/fire_candidate_decode_fusion.cpp:104-127` |
| 小/中/大 tile 的 kernel 分派 | `.../code/op_kernel/fire_candidate_decode_fusion.cpp:128-195` |
| 静态 UB 缓冲布局 | `.../code/op_kernel/device_fire.h:199-216` |
| 单 tile 融合主路径 | `.../code/op_kernel/device_fire.h:272-365` |
| 选择性高精度修复 | `.../code/op_kernel/device_fire.h:105-143`，调用 `:320-327` |
| AoS→通道平面 Gather | `.../code/op_kernel/device_fire.h:145-160` |
| 静态地址/索引表 | `.../code/op_kernel/fire_static_metadata.h`（1100 行，4032 个 u64 表项） |

固定 commit 的文件 SHA256：

```text
fire_host_plan.h                  f4612e6bfe1bfdcff44a79067bdd16ef6daee8cd11972cfa617b8c113ad85f45
fire_candidate_decode_fusion.cpp  e5c444ea25a0b5a9fb751bae2e102409b31c304867f649413868b9f431e1bcdd
device_fire.h                     7d201791e790ba54a88f6ef3911179e8c177fd7fb4ed55fd7404245f4ab79fe2
fire_static_metadata.h            9c58bd934b0838fc05b6514d614ed560c27f34b17a02c71a6674c9b6874020c3
```

### 2.3 主机分路：先按总行数选资源族

`fire_make_plan()` 只依赖 shape/dtype/可用核数，不读张量值。核心分支：

1. `totalRows = B*N <= 8192`：走 `compactSingleTile`，tile 取 32/64/128，block 数取 `min(cores, tiles)`，至少 8 个 block；
2. `totalRows >= cores*1664*3`：走大 shape，tile 取 1640（FP32 且对齐）或 1664，block 数按 `B*ceil(N/tile)`；
3. 否则若 `batchOne` 条件成立：按每 batch 的 worker 数选 512/528/640/896/1024 的 tile；
4. 否则：按总行数除以 512 选 worker 数，8 的倍数对齐，tile 取 512/528/640/1024。

对应源码：`fire_host_plan.h:14-60`。这套分路的目的是让“准备/索引/搬运”的固定成本被足够的行数摊薄，同时避免单个 block 处理过大的 tile 导致 UB 或尾部浪费。

### 2.4 设备路径：一次装载、一次写回

小 tile 的 `ProcessSingle()`（`device_fire.h:272-365`）数据流：

1. 把 `raw` 的 `count*6` 个元素、`prior` 的 `count*5` 个元素连续搬入 UB；
2. 对 6 个字段一起做 `-x`、`exp`、`+1`、`1/(1+·)`，得到 6 个 sigmoid 平面；
3. 用 `vgather` 把 AoS 的 6 字段拆成 6 个 SoA 平面（`GatherChannelPlanes<6>`）；
4. 把 `priors` 拆成 5 个平面（`GatherChannelPlanes<5>`）；
5. 在 SoA 上做 `2*s-0.5+gx`、平方、乘 anchor、乘 stride、score 乘；
6. 用 `scatter_vnchwconv_b32` / `GatherMask` 按 `[x,y,w,h,score]` 重排，转回 dtype 后一次写出。

这条路径的“融合”不是把多个数学公式塞进一个循环，而是**把 AoS 的 6 个字段变成 6 个可向量化的平面，在平面内完成 5 个输出的计算，再恢复原候选顺序**。它减少了：

- 每字段单独读 `raw` 的 6 次全局访问；
- 中间 `[B,N,5]` 张量的写回/重读；
- 多次 kernel 启动；
- 因 AoS 跨步导致的低效向量化。

代价是：

- 需要 6 个输入平面 + 5 个 prior 平面 + 输出/临时缓冲，UB 预算固定；
- `vgather` / `scatter_vnchwconv_b32` / `set_va_reg_sb` 是平台相关内建，不是通用 Ascend C 接口；
- `fire_static_metadata.h` 为每种容量预生成地址/索引表，增加编译体积和版本耦合。

### 2.5 选择性高精度修复：只修“会抵消”的 lane

题面要求 FP32 中间精度，而中心坐标公式 `2*sx - 0.5 + gx` 在 `2*sx ≈ 0.5 - gx` 时会发生灾难性抵消。rank2 没有把整张量升到更高精度，而是：

1. 先看 `priors` 中 `stride` 的最大值；若不超过 FP16 4096 / FP32 32 的阈值，直接返回（`device_fire.h:105-118`）；
2. 否则对 x/y 两个通道计算 `abs(2*s - 0.5 + gx)`（`:119-125`）；
3. 只对 `abs(...) < 1/16` 且 `stride` 超阈值的 lane，读取原始 logit，调用 `FireSoft53::evaluate()` 的软件 53 位 exp，再用 `CenterReciprocalBits()` 做正确舍入的倒数（`:126-139`）；
4. 把修复后的倒数写回 `values` 的对应 lane。

这是一个**数值/性能折中**：正常 lane 保持向量化 `exp` + 倒数；敏感 lane 才走标量软件高精度。它假设敏感 lane 很少；如果某个 shape 的 `gx` 恰好让大量 lane 落在抵消区，标量修复会成为新瓶颈。

### 2.6 证据与判断

| 项 | 内容 |
|---|---|
| 官方公开结果 | 83.26 分，公示顺序 2，原榜 2 |
| 逐 case 时延 | 仓库未提供 |
| 本机编译/运行 | 未做 |
| 机制证据 | `S+L`：Host 分路、SoA gather、选择性修复均可定位 |
| 复用判断 | **优先复用“AoS→SoA→AoS + 选择性高精度”设计；不直接复制私有内建和静态表** |

适用条件：

- 小字段 AoS（如 5–6 个字段）逐元素变换，输出字段数接近输入；
- 字段可平面化，且平面化成本低于重复跨步访问；
- 存在少量数值敏感 lane，且能用廉价判据先筛出来；
- UB 能容纳多个平面和静态表；目标 SoC 支持相应的 gather/scatter 指令。

负例/风险：

- `N` 很小、`B` 很小时，平面化和静态表准备可能超过直接标量/逐字段向量化；
- 如果敏感 lane 比例高，选择性修复的标量循环会反噬；
- `arch 2201` 守卫、`set_va_reg_sb`、`vgather`、`scatter_vnchwconv_b32` 都是平台/编译器绑定；
- 该实现的 FP32 舍入顺序与参考实现不同，不能用实数恒等式推断逐位一致；
- 没有本机正确性/性能证据，不能把 83.26 分当作通用加速比。

## 3. SplineConv rank1：先按权重索引聚合，再乘权重

### 3.1 语义

输入节点特征 `x[N,Ci]`、边表 `edge_index[2,E]`、边伪坐标 `edge_attr[E,dim]`、样条核权重 `weight[K,Ci,Co]`，可选 `root_weight[Ci,Co]`、`bias[Co]`。对每条边 `e`、每个基函数槽 `s=0..S-1`（`S=(degree+1)^dim`）：

```text
k = s
for d in 0..dim-1:
    k_mod = k mod (degree+1)
    k = k div (degree+1)
    v_d = pseudo[e,d] * (kernel_size[d] - degree*is_open_spline[d])
    wi += ((floor(v_d) + k_mod) mod kernel_size[d]) * off
    off *= kernel_size[d]
    b *= B_degree(frac(v_d), k_mod)

message[e,:] = Σ_s basis[e,s] * (x[src[e],:] @ weight[wi[e,s],:,:])
out[i,:] = Σ_{e:dst[e]=i} message[e,:] / deg(i)   # mean；add 则不除
out += x @ root_weight + bias
```

`root_weight` 与显式自环是两条独立路径；`aggr` 只支持 add/mean；`dim∈{2,3}`、`degree∈{1,2,3}`、`kernel_size[d]∈{2,3,4}`、`Ci,Co∈[8,128]`。

### 3.2 固定源码位置

| 内容 | 路径/行 |
|---|---|
| Host shape/attr 校验、tiling、路由 | `.../SplineConv/rank1_submission_1174398/code/op_host/spline_conv.cpp:14-145` |
| UB 预算与 `streaming/cacheWeights/cacheGraph/fusedDot` 选择 | 同上 `:86-105` |
| `single/sparse` 专用 key 选择 | 同上 `:109-141` |
| 目标节点链表遍历 + 按 `wi` 聚合 | `.../code/op_kernel/spline_conv_compact.h:183-225` |
| 4 个权重矩阵的向量乘 + 行归约 | 同上 `:86-94` |
| 单边 `SingleDot` 特化 | 同上 `:51-85` |
| 输出写回/零行处理 | 同上 `:231-260` |

固定 commit 的文件 SHA256：

```text
spline_conv.cpp         677b9e8d032f66bd4c7d51e052df6a133d444f78d2215f220803cf69681d38d8
spline_conv_compact.h   e5919e0ee59371d454076e0c1866be35f189b9973981aeadf8d383115fefb176
```

### 3.3 算法重排：从“逐边投影”到“按权重索引聚合”

通用公式是每条边先做 `x[src] @ W[wi]`，再乘 basis 并聚合。rank1 利用线性性，把同一个目标节点、同一个权重索引 `wi` 的贡献先聚合：

```text
agg[wi,:] += basis[e,s] * x[src[e],:]
out[dst,:] += agg[wi,:] @ W[wi]
```

源码 `spline_conv_compact.h:200-225` 正是这个结构：

- `Duplicate(agg, 0, 4*C)` 清零 4 个 slot 的聚合平面；
- 对目标节点的每条入边，算 4 个 basis 因子，按 `wi` 做 4 次 `Axpy(agg[wi*C], xv, factor, C)`；
- 边遍历结束后 `Dot()` 一次性做 `4C × 16` 的矩阵向量乘；
- `mean` 时按入边数除以 `count`。

### 3.4 工作量模型

设目标节点入度 `d`，通道 `C`，输出 `Co=16`，4 个 slot：

- 逐边投影：约 `4 * C * 16 * d` MAC，权重装载 `4d` 次矩阵；
- 先聚合：约 `4 * C * d` MAC 做聚合 + `4 * C * 16` MAC 做一次投影，权重装载 4 次。

两者比值（忽略搬运/控制）：

```text
speedup_mac ≈ (64*C*d) / (4*C*d + 64*C) = 16d / (d + 16)
```

- `d=1`：16/17 ≈ 0.94，重排反而略多；
- `d=2`：32/18 ≈ 1.78；
- `d=4`：64/20 ≈ 3.2；
- `d=8`：128/24 ≈ 5.3；
- `d→∞`：趋近 16。

这解释了两件事：

1. **为什么 rank1 只在特定稀疏族启用**：Host 的 `sparse` 条件要求 `e<n`、`e≤64`、`nodesPerBlock≤ci`、`nodesPerBlock*e≤512`、`ci=32/128`、`co=16`、`dim=2`、`degree=1`、`kernel=2`，并且 UB 能放下权重、图、聚合和点积缓冲（`spline_conv.cpp:109-115`）。入度太低或图太大时，收益不足以抵消额外缓冲。
2. **为什么单边/低入度要走 `SingleDot`**：`spline_conv_compact.h:51-85` 对只有一条边的目标直接做 4 个权重矩阵的 `Muls/Axpy`，不走聚合，避免“d=1 反而多一次清零+聚合”。

### 3.5 证据与判断

| 项 | 内容 |
|---|---|
| 官方公开结果 | 89.16 分，公示顺序 1，原榜 1 |
| 逐 case 时延 | 仓库未提供 |
| 本机编译/运行 | 未做 |
| 机制证据 | `S+L`：聚合顺序、权重索引、目标链表、Dot 均可定位 |
| 复用判断 | **优先复用“离散索引先聚合”的代数重排；不复制固定 ci/co/key 常量** |

适用条件：

- 权重只由离散索引决定，且索引不随被聚合对象变化；
- 操作对聚合线性（加权和、均值等），可以先聚合再乘；
- 目标节点入度较高（经验上 `d≥2`，`d≥4` 更稳）；
- 聚合平面和权重能常驻 UB；目标节点可用链表/桶遍历。

负例/风险：

- `d=1` 时聚合开销大于收益；
- 如果 basis 依赖边的非离散属性、或权重索引在聚合维上变化，代数前提不成立；
- 浮点求和顺序改变，可能超出 `atol/rtol`；
- 聚合专用路径主要是 `ci=32/128, co=16, dim=2, degree=1, kernel=2`；compact 文件另有 `ci=8` 的 single 特化。其它形状走通用逐边路径；
- Host 条件、tiling key、kernel 编译区必须闭环；只看到某个模板存在，不等于实际评测 shape 会走它。

## 4. TreeLineMinDistance rank1：按退化维度换并行轴 + 共享线段不变量

### 4.1 语义

输入 `tree_points[B,N,3]`、`line_segments[B,M,2,3]`，对每个点 `P` 和每条线段 `(A,C)`：

```text
D = C - A
R = P - A
L = D·D
if L > 0:
    t = clamp((R·D)/L, 0, 1)
    Q = A + t*D
    dist = ||P - Q||_2
else:
    dist = ||P - A||_2
min_distance[b,i] = min_j dist
nearest_segment_idx[b,i] = argmin_j dist  # 完全 tie 取最小 j
```

`B∈[1,8]`、`N∈[1,16384]`、`M∈[1,1024]`、坐标建议 `[-1000,1000]`、FP16/FP32、连续 ND；`N/M` 可非 32 对齐。

### 4.2 固定源码位置

| 内容 | 路径/行 |
|---|---|
| Host shape→path 路由 | `.../TreeLineMinDistance/rank1_submission_1175341/code/op_host/tree_line_min_distance.cpp:438-496` |
| kernel 入口与各 PATH 分派 | `.../code/op_kernel/tree_line_min_distance.cpp:550-659` |
| `TreePairs` 类、UB 布局 | `.../code/op_kernel/tree_pairs.h:221-289` |
| 线段 SoA + 不变量 `D/L/1/L` | 同上 `:340-398` |
| 点 tile 遍历、平方距离、分块归约 | 同上 `:420-515` |
| 1024 段专用路径的 FP16 Top-2 粗筛 | 同上 `:752-787` |
| 16 个候选的 FP32 重算 + 最终 argmin | 同上 `:789-854` |

固定 commit 的文件 SHA256：

```text
tree_line_min_distance.cpp  ad652b92592fe84acbbe28f16f1ff18e27b7c30c19072d846a09dbf6fa6dcff6
tree_pairs.h                3bb699614c775f2900bf17401f7e232854cd6df8adddf5772a9f6829911664cf
```

### 4.3 按 shape 换并行轴

Host 路由不是“一个通用 kernel + tile 参数”，而是显式分族（`tree_line_min_distance.cpp:446-484`）：

| 条件 | path | 策略 |
|---|---:|---|
| `points≤3 && M≤6` 或 `M==1 && points≤32` | 7–12 | tiny/固定形状，单核或固定块 |
| `M==1` | 1/20 | 沿点向量化；`N=16384` 专用 path 20 |
| `N==1` | 4/14 | 沿线段向量化；`M=1024` 专用 path 14 |
| `M≤32 && points≤64` | 5/6/13/16/17 | short 路径，按 8/32 宽向量化 |
| `M≥256 && N≥64` | 3/15/19/21/22/23 | 大任务，按点 tile/批固定；`M=1024` 走 64 段 chunk |
| 其它 | 2 | 通用 `TreePairs` |

这体现了“按退化维度改变并行轴”：

- `M=1`：没有 argmin 意义，直接沿 `N` 个点并行，省掉线段归约；
- `N=1`：没有点并行，沿 `M` 个线段并行，最后归约；
- `M/N` 都大：按点 tile 分组，线段分块 64，块内归约后再跨块合并；
- 小 shape：直接标量/固定长度，避免启动通用 kernel 的固定开销。

### 4.4 共享线段不变量

`TreePairs::run()` 对每个 batch 先做：

1. 把 AoS 的 `[A,C]` 6 个字段 Gather 成 6 个 SoA 平面（`tree_pairs.h:369-374`）；
2. 计算 `ux=Cx-Ax`、`uy`、`uz`，`L=ux²+uy²+uz²`，`vx=ux/L`、`vy=uy/L`、`vz=uz/L`（`:375-386`）；
3. 把 `D`、`L`、`1/L` 留在 UB，供该 batch 的所有点复用。

逐点计算时只做 `P-A`、点积、clamp、投影、平方距离；`D/L` 不再重复算。这减少了每条线段在每个点上的固定工作，尤其是 `L`、倒数、SoA 转换。

### 4.5 归约与 sqrt 后置

对一个点 tile：

- 每个 64 段 chunk 内，向量化计算平方距离，`WholeReduceMin` 得到 chunk 最小值和对应索引；
- 用 `GatherMask` 把每个 chunk 的 `(value,index)` 收拢；
- 跨 chunk 用 `Compare` + `SelectCal` 成对更新，比较用严格 `<`，因此完全 tie 保留较小索引；
- 所有候选比较完后才做一次 `Sqrt(distance)`，而不是每个候选都 sqrt；
- index 和 distance 分开发布，index 可以先写。

这减少了 `N*M` 次 sqrt 到 `N` 次，并把“距离最小”与“索引 tie”放在同一条归约链上。

### 4.6 1024 段专用路径：FP16 Top-2 粗筛（高风险）

`TreePairs::run_1024()`（`tree_pairs.h:752-854`）对 `M=1024` 的做法：

1. 把点和线段缩放到 half，按 8 个 128 段 chunk 计算 half 平方距离；
2. 每个 chunk 用 `WholeReduceMin` + `GatherMask` 取 Top-2（共 16 个候选）；
3. 把这 16 个候选的索引搬回 FP32 平面，用 FP32 重算精确距离；
4. 在 16 个候选中选最小，输出。

这能把 FP32 距离计算从 1024 次/点降到 16 次/点，但**没有“被筛掉的候选不可能更优”的证明**：

- half 的舍入误差可能让真实最小段落出 Top-2；
- 坐标范围 `[-1000,1000]` 时，half 的相对误差约 `2^-11`，平方距离误差会放大；
- 固定 Top-2 是启发式，不是保守下界；
- 因此这条路径可以解释“为什么快”，但不能作为精确 argmin 的通用模板。若未来复用，应改成“可证明下界 + 候选精算”，而不是固定 Top-2。

另外，通用路径用 `Maxs(L, FLT_MIN)` 避免除零（`tree_pairs.h:383`）。这对 `L=0` 的零长度线段会改变结果：题面要求 `L=0` 时用 `||P-A||`，而 `max(L, FLT_MIN)` 会得到接近 0 的投影，不等于端点距离。复用时应显式分支 `L==0`，而不是用最小正数替换。

### 4.7 证据与判断

| 项 | 内容 |
|---|---|
| 官方公开结果 | 91.85 分，公示顺序 1，原榜 1 |
| 逐 case 时延 | 仓库未提供 |
| 本机编译/运行 | 未做 |
| 机制证据 | `S+L`：shape 路由、SoA、不变量、归约、half 粗筛均可定位 |
| 复用判断 | **复用“退化维度换轴 + 不变量外提 + sqrt 后置 + 成对归约”；不复制固定 Top-2 粗筛** |

适用条件：

- 归约维度与点维度可分离，且存在退化形状（`M=1`、`N=1`）可专门化；
- 线段/模板不变量能在 batch 内复用；
- 归约需要稳定 tie（最小索引）；
- 尾块非 32 对齐，必须用有效长度而不是对齐长度做归约。

负例/风险：

- 固定 Top-2 half 粗筛无 no-miss 证明；
- `max(L, FLT_MIN)` 不等价于 `L==0` 分支；
- 大量固定 PATH/UB 绝对偏移，和 192 KiB UB、SDK 版本强绑定；
- 大 `M=1024` 路径的 half/FP32 转换、索引位模式技巧需要逐项验证符号、溢出和 tie；
- 没有本机性能数据，不能把 91.85 分归因到某一条技巧。

## 5. 三份实现的横向观察

| 维度 | Fire rank2 | Spline rank1 | Tree rank1 |
|---|---|---|---|
| 主要收益来源 | 融合 + AoS/SoA + 选择性高精度 | 代数重排（先聚合后乘权重） | 形状专用化 + 不变量外提 + sqrt 后置 |
| 是否改变工作量 | 减少中间张量和 kernel 数；算术量近似不变 | 高入度时减少投影 MAC | 减少重复 `D/L`、sqrt 和无效 argmin |
| 是否改变布局 | AoS→SoA→AoS | 目标节点链表 + 权重索引平面 | AoS→SoA + 点/段平面 |
| 关键前提 | 小字段、少量敏感 lane、私有 gather | 离散权重索引、线性聚合、入度高 | 退化维度、tie 规则、尾块 |
| 最大风险 | 私有内建/静态表/标量修复 | `d=1` 退化、浮点顺序 | half Top-2 无证明、`FLT_MIN` 替换 |
| 可迁移性 | 中（设计可迁移，代码强绑定） | 高（代数是通用方法） | 中（路由/归约可迁移，粗筛不可照搬） |

共同点：

1. **Host 先算 shape 族，再选 kernel**，而不是一个 kernel 吃所有 shape；
2. **把固定 shape 的索引/掩码/地址尽量编译期化**，减少设备端 scalar 和分支；
3. **把“清零/写回/边界”单独设计**，避免为了一个 tail 把整条向量路径拖慢；
4. **所有三份都没有逐 case 时延**，所以只能把它们当结构证据，不能当性能复现。

## 6. 复用建议

### 6.1 优先复用

- **Fire 的 AoS→SoA→AoS 融合**：适合 5–6 个字段的逐元素变换，尤其是输出字段数接近输入、且原布局跨步访问低效的场景。
- **Fire 的选择性高精度修复**：适合主路径 FP32、只有少量 lane 发生抵消的场景；先做廉价筛选，再对标量敏感 lane 精算。
- **Spline 的离散索引先聚合**：适合图卷积/稀疏聚合/专家聚合等“权重只由离散索引决定”的算子；先用入度分布判断是否值得。
- **Tree 的退化维度换轴**：`M=1`/`N=1` 时直接删掉无意义的归约/广播工作；这是最通用、最低风险的一条。
- **Tree 的不变量外提 + sqrt 后置 + 成对归约**：适合最近邻/距离/argmin 类算子。

### 6.2 需要验证后再用

- Fire 的静态地址表/私有 `vgather`：只有在目标 CANN/SoC 支持且能用生成器复现时才考虑；
- Spline 的 `sparse` key 和 UB 预算：必须按目标 910B/910C 的 UB 容量、AIV 数、CANN 版本重新算；
- Tree 的 shape 专用 PATH：先确认实际评测 shape 会命中，而不是只看模板存在；
- Tree 的 half 粗筛：除非能给出可证明下界和 no-miss 证书，否则不要用于精确 argmin。

### 6.3 暂不使用

- Tree 的 `max(L, FLT_MIN)` 替换 `L==0` 分支；
- 任何固定 Top-2/Top-K 启发式筛选，除非题目允许近似或能证明候选完整；
- 直接复制 `set_va_reg_sb`、`scatter_vnchwconv_b32`、`vgather` 等平台内建；
- 把公示分数当作加速比或跨题比较。

## 7. 证据边界

- 本报告只做静态源码阅读和固定 commit 核对；没有编译、运行、profiling 或平台提交。
- 官方分数来自归档的公示页快照；公示期后的名次变化、人工合规审查、评测环境差异都未核实。
- 源码中的注释、README、维护者审计只作为线索；结论以源码行号和公式为准。
- 第三方源码未复制进本库；使用前需核对原作者授权和比赛规则。
