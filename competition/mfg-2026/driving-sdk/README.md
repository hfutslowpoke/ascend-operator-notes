# DrivingSDK：Radius / FurthestPointSample / CylinderQuery

> 层级：`competition` · 赛事：2026 制造行业算子专场 · 对应赛题：11 `Radius`、13 `FurthestPointSample`、15 `CylinderQuery` · 本机 910B3：`not-run`

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | https://gitcode.com/Ascend/DrivingSDK |
| 维护者 | Ascend / 华为 |
| 固定 commit | `d10b13c95ec7f25a54a4aa9677c436cde30ff6e2`（2026-11-17） |
| 默认分支 | `master` |
| 许可证 | Apache-2.0 |
| 本地源码 | `competition/mfg-2026/driving-sdk/source/` |
| 去重 | 与 ops-gnn 的 radius/fps 实现不同：DrivingSDK 是自动驾驶生产 SDK，文档明确 A2/A3/950 |

## 2. 赛题与接口

### 2.0 官方题面校正（2026-11-17）

| 赛题 | 官方规格 | 源码匹配 |
|---|---|---|
| 11 `Radius` | `x/y` float32/float16，`ptr_x/ptr_y` int64，`D∈{2,3}`，`N,M≤1e6`，`out (2,M×K)` 定槽位、未命中 `(-1,-1)`，`ignore_same_index` | `adapt`：DrivingSDK 是动态输出、单 batch≤512、`Ndim<9`，需重写输出/约束 |
| 13 `FurthestPointSample` | `points [B,N,3]` float32/float16，`npoint≤N`，首点 0，并列取最小索引，float16 保持 half 精度 | `direct` 候选：官方题面明确以 `mx_driving` Ascend FPS 为参考；仍需本机逐位验证 |
| 15 `CylinderQuery` | `xyz/new_xyz/rot` float32/float16，`rot^T·Δ`，严格边界，扫描序 first-K，不足重复首个命中 | `adapt`：kernel 的 `rot^T` 与严格比较一致；但 Python 层会排序后截断，需改为 first-K |


### 2.1 Radius（11）

```python
mx_driving.radius(x, y, ptr_x, ptr_y, r, max_num_neighbors) -> Tensor
```

- `x`: `[numpoints_x, Ndim]` FP32；`y`: `[numpoints_y, Ndim]` FP32；
- `ptr_x/ptr_y`: `[batch_size+1]` int，CSR 式 batch 切分；
- 对每个 y，找同 batch 内 `dist² <= r²` 的 x，按扫描顺序保留前 `max_num_neighbors`；
- 输出 `[2, num_neighbors]` int，第一行是 y 索引、第二行是 x 索引（按文档/测试为准）。

源码约束：`Ndim < 9`、`batch_size <= 1024`、单 batch 点数 `<= 512`；支持 A2/A3/950。官方约束是 `D∈{2,3}`、`N,M≤1e6`、`M×max_num_neighbors≤4e6`、定槽位输出，二者不可混用。

### 2.2 FurthestPointSample（13）

```python
mx_driving.furthest_point_sampling(points, num_points) -> Tensor
```

- `points`: `[B,N,3]`，FP32/FP16/BF16；输出 `[B,num_points]` int32；
- 迭代选最远点；源码约束 `B*N*3 <= 383166`；支持 A2/A3/950。官方测试规模为 `N≤2048`、`B≤4`、`npoint≤128`，远小于源码上限。

#### 2.2.1 950 社区任务的 FPS 踩坑（只能作精度对照）

`gcw_i0yFZkCV/ops-gnn-fps-950`（commit `9d8da03b18d0b7bc70bfbab47f10c9361c942b91`，CANN EULA 2.0）是 950 社区任务实现，接口是 `torch_cluster.fps(src, ratio, random_start, ptr)`，**不是**官方制造的 `furthest_point_sampling(points, npoint)`。它记录了三个容易踩的坑：

1. **FP16 逐维舍入**：torch_cluster CUDA/C++ 路径按每个特征维先算 `delta.half()`、平方后再 `half()`、再累加，和“先转 FP32 累加再转回”会在边界选点上产生不同结果；官方题面要求“按输入数据类型计算平方欧氏距离”且“float16 必须保持 half 精度”，实现时必须明确逐维舍入还是 FP32 累加，并用官方算子逐位核对。
2. **并列 tie-break 不是简单最小索引**：torch_cluster CUDA 的 256-lane 归约等价于“先选 `index % 256` 最小的 lane，再选该 lane 先遇到的点”；官方制造题明确要求**最小输入索引**。不能照搬 torch_cluster 的 lane 规则。
3. **计数/ratio 语义不同**：950 任务用 `ceil(degree.float() * ratio_stored_dtype)`，并存在 FP16 标量 ratio 的 3000/3001 边界差异；官方制造题直接给 `npoint`，不做 ratio/ceil。不要把 950 任务的 count 契约带进官方实现。

本地副本：`supporting/ops-gnn-fps-950/`（LICENSE、测试 README、compatibility README、golden、frozen oracle test）；来源/边界见 [`PROVENANCE.md`](supporting/ops-gnn-fps-950/PROVENANCE.md)。这些文件只用于理解 torch_cluster 的 FP16/tie 行为；官方候选仍是 DrivingSDK 的 `furthest_point_sampling`，并按官方 `npoint`/首点 0/最小索引 tie 改写。

### 2.3 CylinderQuery（15）

```python
mx_driving.cylinder_query(radius, hmin, hmax, nsample, new_xyz, xyz, rot)
```

- `new_xyz`: 查询中心；`xyz`: 点云；`rot`: 每个 query 的 3×3 旋转（9 元素）；
- 在局部坐标下按 `y²+z² < radius²` 且 `hmin < x < hmax` 选点；
- 输出先为全量候选，Python `sort` 后取前 `nsample`，miss 用首索引填充。

## 3. 源码定位

| 内容 | 路径/行 |
|---|---|
| Cylinder host：属性/UB/tiling | `kernels/cylinder_query/op_host/cylinder_query.cpp:1-120` |
| Cylinder kernel：任务/分块/流水 | `kernels/cylinder_query/op_kernel/cylinder_query.cpp:60-102` |
| Cylinder 旋转/距离/compare/select | 同上 `:300-350` |
| Cylinder Python sort/truncate/fill | `mx_driving/ops/cylinder_query.py:29-50` |
| Cylinder 测试/形状 | `tests/torch/test_cylinder_query.py` |
| FPS kernel：初始化/迭代/距离更新 | `kernels/furthest_point_sampling/op_kernel/furthest_point_sampling.cpp:115-208` |
| FPS host：tiling/UB | `kernels/furthest_point_sampling/op_host/furthest_point_sampling.cpp:1-100` |
| FPS 语义/约束 | `docs/zh/api/context/furthest_point_sampling.md:1-40` |
| Radius host：batch/UB/输出 | `kernels/radius/op_host/radius.cpp:1-100` |
| Radius kernel | `kernels/radius/op_kernel/radius.cpp:1-120` |
| Radius 语义/约束 | `docs/zh/api/context/radius.md:1-50` |

机器索引：[source.json](source.json)。

本地源码副本：[`source/`](source/)（按上游路径保存，字节与固定 commit 一致；`source.json` 的 `local_path` 指向对应文件。上游原文的相对链接假设整仓，本地不保证可解析）。

## 4. 机制

### 4.1 Radius

- 按 batch 分核；源码每个 batch 点数上限 512，便于把 X/Y 点集放进 UB；官方允许 `N,M≤1e6`，需要重新分块。
- 距离用平方比较，避免 sqrt；
- 每个 y 扫描同 batch 的 x，命中写入邻居对；超过 `max_num_neighbors` 截断；
- 输出是动态长度 `[2,E]`，需要 workspace/输出计数；顺序由扫描顺序决定（确定性）。

### 4.2 FPS

- 维护 `nearestDist[N]`（每个点到已选集合的最近距离）；初始选索引 0；
- 每轮：计算所有点到“新选中点”的距离，`nearestDist = min(nearestDist, d)`，再 reduce max 得到下一个最远点；
- 点云大于 UB 时按 `pieces` 分片，每片更新部分距离；多核按 batch 切分；
- FP16/BF16 先 cast 到 FP32 算距离，输出索引 int32；首点固定为 0 会影响结果。

### 4.3 CylinderQuery

- 每个 query 有 3×3 旋转；kernel 把点云 tile 搬入 UB（双缓冲），计算 `p - center`，再用 `R` 旋转；
- `CompareScalar` 生成 `y'²+z'² < r²` 与 `hmin < x' < hmax` 两个 mask，`Select` 写候选索引；
- 每核处理若干 query，点云按 UB 容量分块；`nsample` 参与 UB 预算；
- Python 层 `sort` 候选、取前 `nsample`、把 miss 填成首索引，再转 int32。**这与官方扫描序 first-K 不一致**；官方禁止排序。

## 5. 910B 适用性与证据

- Radius/FPS 文档明确列出 **Atlas A2 训练/推理**；CylinderQuery 走通用 host/kernel 路径，未见 arch35-only 分支，但文档没有单独支持表，**A2 需本机构建确认**。官方题面进一步表明 FPS 就是 `mx_driving` 参考语义，是最强候选；Radius/Cylinder 仍需适配。
- 本批未构建/未运行：`local_status=not-run`；无作者性能数字。
- 构建方式：DrivingSDK 是 CMake/setup.py 工程，`kernels/CMakeLists.txt` 自动扫描所有算子；全量构建较重。可先只编译/安装 `mx_driving` 的 Python 扩展，再调用单个 op；若不能单算子构建，记录为构建成本负例。

## 6. 负例/风险

- **Radius**：源码 `Ndim>=9`、batch>1024、单 batch>512 直接不支持；输出 E 动态，官方要求定槽位 `[2,M×K]` 和 `(-1,-1)`；`ignore_same_index` 需补。
- **FPS**：源码总元素上限 383166；官方要求 float16 保持 half 精度，不能提升到 fp32；并列取最小索引需逐位验证；分片会多次读点云。950 社区仓证明 torch_cluster 的 `index % 256` lane tie 与 FP16 逐维舍入会和“最小索引/FP32 累加”产生不同结果，必须按官方口径单独验证。
- **CylinderQuery**：`nsample` 受 UB 限制；rot 必须是 9 元素；kernel 的 `rot^T·Δ` 与官方一致，但 Python 层排序违反 first-K；官方要求 `nsample≤N`、float16→float32 变换/比较。
- **通用**：DrivingSDK 依赖 torch_npu/自定义 C++ 扩展，版本匹配要求高；不要把“文档支持 A2”当成本机通过。

## 7. 复用判断

**优先复用**：Radius 的 batch 分核 + 半径平方 + 截断；FPS 的最近距离数组 + 迭代 reduce max + 分片；CylinderQuery 的 3×3 旋转 + 双 mask + 候选输出。

**需要验证**：910B3 上单算子构建/调用；FPS 的 half 精度（逐维舍入 vs FP32 累加）与最小索引 tie-break；Radius 的定槽位/ignore_same_index；CylinderQuery 去掉 Python 排序后的 first-K。

**暂不使用**：超出约束的 shape；需要 950 专属路径的场景；未确认 A2 的 CylinderQuery 直接当 910B 模板。

## 8. 下一步

1. 先构建/安装 `mx_driving`，跑 `tests/torch/test_radius.py`、`test_furthest_point_sampling.py`、`test_cylinder_query.py`；
2. 与官方样例/CPU 暴力参考对照：Radius 定槽位、FPS 逐位索引、CylinderQuery 扫描序 first-K；
3. 记录 ACLNN/event 时延、输入 shape、workspace、首轮 setup；
4. 若全量构建过重，记录“单算子构建不可行”的成本，并改用源码级移植评估。
