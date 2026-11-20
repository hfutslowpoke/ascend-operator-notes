# Sparse Softmax（星辰杯 A 组）

> 层级：`competition` · 赛事：中国电信星辰杯·高校 AI 算子开发挑战赛 A 组 · 平台：`2/161, 84.36` · 本机：`not-run`

## 0. 来源登记

| 项 | 值 |
|---|---|
| 官方题面 | [sparsesoftmax.md](../official/descriptions/sparsesoftmax.md) |
| 官方 problem JSON | [problem-6a7c17b6a52e0f540a89c0ce.json](../official/problem-6a7c17b6a52e0f540a89c0ce.json) |
| 平台提交 | `sc_a_sparsesoftmax` |
| 正式源码 | [`source/`](source/) |
| 证据 | [`evidence/`](evidence/)（detail-summary / cases.csv / ranking-summary / sid / source.sha256） |
| 备注 | zip 494890 与平台提交逐文件一致 |
| 本机状态 | `not-run`（本次未在本机 910B3 重跑） |

## 1. 平台证据

| 项 | 值 |
|---|---|
| contest | `ct_starcup_aiop_g1` / 星辰杯 A 组 |
| problem_id | `6a7c17b6a52e0f540a89c0ce` |
| 提交账号 | `slowpoke`（uid `6aa109b6c76b321ca64ba8b4`），团队 `hfutslowpoke 的队伍` |
| SID | `6aba0748694b590c3cb0353b` |
| detail ID | 494890 |
| 排名 | 2 / 161 |
| score | 84.36 |
| 状态 | Pass / valid=true |
| 可见 case | 5 / 5 |
| 可见总时间 | 7.30 ms |
| 可见平均 t/T | 1.080 |
| 可见平均 case score | 84.36（与平台 score 完全一致） |

逐 case：

| case | time (ms) | best (ms) | t/T |
|---|---:|---:|---:|
| 1 | 1.40 | 1.36 | 1.029 |
| 2 | 1.48 | 1.34 | 1.104 |
| 3 | 1.46 | 1.34 | 1.090 |
| 4 | 1.48 | 1.38 | 1.072 |
| 5 | 1.48 | 1.34 | 1.104 |

5 个 case 都在 1.4 ms 左右，和 best 的 1.34–1.38 ms 只差 0.02–0.14 ms。
这是典型的 **launch/fixed-overhead bound**，不是带宽 bound。

## 2. 题目语义

```text
index 模式：
  src_max[k] = max{src[i] | index[i]=k}
  out[i] = exp(src[i]-src_max[index[i]]) / (sum_{j:index[j]=index[i]} exp(src[j]-src_max[index[i]]) + eps)

ptr(CSR) 模式：
  src_max[k] = max{src[i] | ptr[k] <= i < ptr[k+1]}
  out[i] = exp(src[i]-src_max[k]) / (sum_{j=ptr[k]}^{ptr[k+1]-1} exp(src[j]-src_max[k]) + eps)
```

- src/out：FP16/FP32/BF16；index/ptr：INT64；
- `dim` 默认 0，可为负；
- `eps=1e-16`；
- index 与 ptr 二选一；ptr 单调非递减，`ptr[0]=0`，`ptr[-1]=src.size(dim)`；
- 测试覆盖：1D S=100、2D S=256 D=64、多维 S=128 D=32 H=16；index/ptr；
  均匀/非均匀/单元素/大组；dim=0/-1/1；大数值/零值。

## 3. 代码来源

- API 详情：``
- 用户 zip `SparseSoftmax_submission_494890.zip` 与 API 详情**逐文件 SHA256 完全一致**；
  detail ID 494890 就是平台排名提交。
- 源码：`op_host/sparse_softmax.cpp` 84 行；`op_kernel/sparse_softmax.cpp` 1,441 行；
  `tiling_key_sparse_softmax.h` 83 行。

## 4. 实现结构

### 4.1 host 分路

`op_host/sparse_softmax.cpp:23-62` 用 `PATH` 模板参数分 15 条路：

| PATH | 条件 | 用途 |
|---|---|---|
| 6 | index, tasks=1, axis=8, inner=1, broadcast, eps=1e-16 | raw UB 固定 8 lane |
| 7 | index, tasks=2, axis=4, inner=1, broadcast, fp32 | 2×4 平面 |
| 8 | ptr, tasks=1, axis=8, ptrLength=5, fp32 | raw UB ptr 8 |
| 9 | ptr, tasks=2, axis=4, ptrLength=3, fp32 | 2×4 ptr 平面 |
| 11 | index, 1≤axis≤8, inner=1, fp16/fp32 | 小 shape family |
| 12 | ptr, 1≤axis≤8, 2≤ptrLength≤49 | 小 shape family |
| 10 | axis>4096 | workspace streaming |
| 13/14 | INT64 index/ptr | 标量 exact 路径 |
| 0–5 | 其它 | 通用 Tiny/AllPairs/通用链 |

`PATH6–9` 是精确 shape 入口；`PATH11/12` 是 shape family；这是本实现的关键。

### 4.2 tiling ABI

`SparseSoftmaxTilingData` 只有 32 B：

```text
tasks, axisLength, innerSize, mode, ptrLength, indexBroadcast, eps, pad0
```

注释明确写：device 每 block 只读一次，struct 减半就减半入口成本。
`pad0` 兼作 64-bit id 标记，避免额外字段。

### 4.3 raw UB 路径

`PATH6` 的 `RawUbIndex8AllPairs8`（`:376`）：

- 不构造 `KernelSparseSoftmax`，不 `TPipe`，不 `InitBuffer`；
- 固定 UB 偏移：raw / vals / tmp / out / cast / ids；
- 只 block 0 执行（tasks=1）；
- `Duplicate(ids,0,8)`、`Duplicate(vals,-3.4e38,8)` 初始化；
- `Brcb` 把 8 个 id 广播成 8×8，`Compare` 生成 mask；
- `BlockReduceMax/Sum` 求每组 max/sum；
- `Sub/Exp/Div` 后用 `DataCopyPad` 一次写出；
- 没有 `Gather`，没有通用 `scatter`。

`TinyAllPairs8`（`:73`）是同一思路的通用入口；`TinyAllPairs16`（`:123`）用 16 lane。

### 4.4 小 shape family

`PATH11` 的 `RawUbIndexAnyAllPairs8`（`:571`）把行长度 `n` 作为运行时参数，
一条实例覆盖 1≤axis≤8；`PATH12` 的 `RawPtrAny`（`:638`）覆盖 1≤axis≤8、
2≤ptrLength≤49。这样避免为每个小 shape 生成一个 kernel。

### 4.5 长轴 / int64

- `axis>4096`：`LargeAxisSoftmax`（`:966`）用 workspace 三张表
  （group max / group sum / Kahan补偿），每 block 一个 slot；
- INT64 id：`Int64Softmax`（`:1229`）走标量 exact 路径，慢但正确；
  注释明确说“slow by design and exact for every id value”。

## 5. 为什么高效

### 5.1 固定开销最小化

5 个 case 都是 1.4 ms 量级，真正的计算量很小。排名差异主要来自：

- 是否读 tiling struct；
- 是否构造 `TPipe`/`InitBuffer`；
- 是否走通用 `Gather/Scatter`；
- 是否只用一个 block；
- 是否用固定 8 lane 的一次性 UB 操作。

`PATH6–9` 把这四项全部省掉：不读 tiling（或只读 32 B）、不建 pipe、不用通用
Gather、只 block 0。`PATH6` 甚至把 `tasks=1, axis=8` 写成编译期常量。

### 5.2 8×8 all-pairs 代替 scatter/gather

对于 8 个元素、若干组的小行：

- 用 `Brcb` 把 id 扩展成 8×8；
- `Compare` 得到“元素 i 是否属于组 g”的 mask；
- `BlockReduceMax/Sum` 对每行做归约；
- 再用 mask 选回每个元素的组 max/sum。

这比“先 scatter 求组 max，再 gather 回元素”少很多离散访存和同步。

### 5.3 lean tiling ABI

32 B 的 tiling struct、`pad0` 复用、`GET_TILING_DATA_WITH_STRUCT` 只在通用路径用；
raw 路径完全跳过。对 1.4 ms 的 kernel，几十字节的入口读取也是可见成本。

### 5.4 与第一名的差距

| 项 | 本提交 | 第一名 `bubu` |
|---|---:|---:|
| score | 84.36 | 85.03 |
| 可见总时间 | 7.30 ms | 7.30 ms |
| 平均 t/T | 1.080 | 1.0 |

两者总时间相同，分数差 0.67，主要是逐 case 的 0.02–0.06 ms 差异和提交时间排序。
这说明该题已经接近 launch floor；继续优化的空间主要是再减几条指令/同步，
而不是换算法。

## 6. 可复用与风险

**可复用：**

- raw UB + 去 TPipe/去 tiling 读取，用于单行/小 shape；
- 8 lane all-pairs + `BlockReduce` 的分组归约；
- 32 B lean tiling ABI；
- shape family（1≤axis≤8）而不是每个 shape 一个 kernel；
- 长轴 workspace 分组表 + streaming。

**风险：**

- `PATH6–9` 是精确 shape 入口（tasks=1 axis=8 等），属于公开 case 专用化；
  隐藏 shape 若不在 family 内，会掉到通用路径；
- op def 没有 bf16，原因是工具链拒绝 bf16↔fp32 Cast；题目本身支持 bf16，
  这是实现限制，入库时要写清楚；
- `num_nodes` 输入未在 op def 中出现；如果平台测试用 `num_nodes`，需要核对；
- `inner>1` 走标量 `GetValue/SetValue`，不是高性能路径；
- 本次未在本机 910B3 重跑。

## 7. 入库标注

```text
source_match: our-submission (CANNJudge SID 6aba0748...)
platform_evidence: rank 2/161, score 84.36, 5/5 visible
local_status: not-run
specialization: PATH6-9 精确 shape；PATH11/12 shape family
generalization: axis>8 / inner>1 / bf16 未覆盖或慢
risk: 精确 shape 特化；bf16 缺失；num_nodes
```
