# 2026 制造行业算子专场：GitCode 源码补充调研（2026-11-17）

状态：已完成赛事主页/15 题目录核对、15 份官方公开题面抓取、6 个官方/生产仓固定 commit、15 份实现映射；`ops-fft/fft2_d` 已在本机 910B3 通过 32×32 C2C 子集，其余 14 份未跑。本文记录搜索范围、筛选、去重、覆盖和缺口；逐题细节见 [competition/mfg-2026/problems.md](../../competition/mfg-2026/problems.md) 和 [official/specs.md](../../competition/mfg-2026/official/specs.md)。

## 1. 赛事与 15 题

来源：<https://cann-mfg.gccorg.com/>（访问 2026-11-17）。

| 类别 | 赛题 |
|---|---|
| 仿真计算 | ComplexAbs、ComplexDiv、ComplexExp、FusedPropStep、FourierTransform2D |
| 图计算 | GatherCoo、SegmentMinCsr、RandomWalk、SpmmSum、Spspmm |
| 点云 | Radius、GridCluster、FurthestPointSample、BallQuery |
| 机器人抓取 | CylinderQuery |

赛事时间：2026-11-09 至 2026-12-07（预赛），2026-12-20 决赛；官方算力为云端 910B；评测为 CANN Judge 精度 + 性能/耗时。

## 2. 搜索范围与方法

- 赛事主页：解析 15 题卡片、赛制、开源要求；
- GitCode CANN 组织/仓库：`cann/ops-fft`、`cann/ops-gnn`、`cann/ops-sparse`、`cann/ops-math`、`cann/ops-solver`、`Ascend/DrivingSDK`；
- 制造 SIG 文档：`cann/community/CANN/sigs/manufacturing`（列出 `mfg-ops-cax`、`mfg-ops-vision`，但 2026-11-17 `git ls-remote` 无 HEAD，尚未公开）；
- 竞赛仓：`cann/cann-ops-competitions` 的 `02_enterprise` 当前只有 `regional`，未找到制造专场提交目录；
- 官方题面：`https://cannjudge.cn/public/gcc_2026_mfg/<problem>`；公开 API `GET https://cannjudge.cn/api/problems/contest/6aed4926a5348dcef5dd8d0c` 返回 15 题完整描述、shape/dtype、测试规模；隐藏测试数据未公开，`/api/testcases/<id>` 返回 403，未绕过；
- GitCode/AtomGit 匿名仓库搜索：`GET https://api.atomgit.com/api/v5/search?type=repo&q=<q>`；
- 搜索词（实际使用或组合）：
  - `ComplexAbs`、`ComplexDiv`、`ComplexExp`、`FusedPropStep`、`FourierTransform2D`
  - `GatherCoo`、`SegmentMinCsr`、`RandomWalk`、`SpmmSum`、`Spspmm`
  - `Radius`、`GridCluster`、`FurthestPointSample`、`BallQuery`、`CylinderQuery`
  - `grid_cluster`、`segment_csr`、`spgemm`、`spmm_sum`、`fft2`、`ops-gnn`、`ops-fft`、`DrivingSDK`、`ops-math`
  - `2026制造行业算子`、`制造行业算子`、`mfg-ops`、`gcc_2026_mfg`
- 固定方式：`git ls-remote` 取 HEAD，`raw.gitcode.com/<repo>/raw/<commit>/<path>` 拉取选中文件，逐文件 SHA256；选中文件已复制到各条目 `source/`，未复制整仓。

## 3. 入选仓库

| 仓库 | 固定 commit | 许可 | 角色 | 覆盖赛题 |
|---|---|---|---|---|
| `cann/ops-fft` | `65f5ff1dee32802562a9412789b0c7719ef99269` | CANN EULA 2.0 | 910B 2D C2C FFT | 05 |
| `Ascend/DrivingSDK` | `d10b13c95ec7f25a54a4aa9677c436cde30ff6e2` | Apache-2.0 | 点云 Radius/FPS/CylinderQuery | 11、13、15 |
| `cann/ops-gnn` | `ea0b18a1d3e97f1ae2315df69338bb51c9f96466` | CANN EULA 2.0 | Segment/Spmm A2；Gather/RandomWalk/GridCluster 950 | 06、07、08、09、12 |
| `cann/ops-sparse` | `8fae00b8832bf31f1ce26059680bce67c6026c4f` | CANN EULA 2.0 | A2 SpMM/SpGEMM/Gather | 06、09、10 |
| `cann/ops-math` | `3b2baac63624c32e03781bd9f0290f1a4584f0c7` | CANN EULA 2.0 | ComplexAbs/BallQuery/StackBallQuery | 01、14 |
| `cann/ops-solver` | `739775a145d7789693f97758f6771bb1b3eb6a8f` | CANN EULA 2.0 | ComplexDiv 算法参考（950 SIMT） | 02 |

## 4. 覆盖结论

| 覆盖 | 数量 | 赛题 |
|---|---:|---|
| `direct`：语义高度匹配 + A2/910B 路径 | 2 | 07 SegmentMinCsr、13 FurthestPointSample（待本机逐位验证） |
| `adapt`：有 A2/910B 源码但需改接口/布局/后处理 | 6 | 05、09、10、11、14、15 |
| `reference`：只有 950/arch35 或算法 helper | 5 | 01、02、06、08、12 |
| `none`：未找到公开源码 | 2 | 03 ComplexExp、04 FusedPropStep |

> 官方题面抓取：`https://cannjudge.cn/public/gcc_2026_mfg/<problem>` 与 `GET /api/problems/contest/6aed4926a5348dcef5dd8d0c`（2026-11-17）。隐藏测试数据未公开，`/api/testcases/<id>` 返回 403，未绕过。

关键负例/冲突：

- `ops-math/conversion/ball_query` README 写 A2 √，但代码/config 只有 arch35（ascend350/950）；不能把 README 当本机可构建证据；`stack_ball_query` 有 `ascend910b` 配置，是更现实的 910B 入口。
- `ops-gnn/gather_coo`、`random_walk`、`grid_cluster`、`radius`、`fps` 是 arch35-only；`ops-sparse/gather` 是 dense gather，不是 COO 行扩展。
- `ops-solver` 的复数除法是 950 SIMT 内部 helper，不是独立 elementwise 算子。
- `ops-fft/fft2_d` 明确 910B √、950 ×，但尺寸仅 {32,64,128}²、只支持 C2C；官方要求 `N,M∈[2,8192]` 的 2 的幂和 `inverse`，源码只覆盖子集。
- `ops-sparse/spgemm` 是 CSR + 动态 nnz；官方 `Spspmm` 是 COO + 固定槽位 `U=m·n`，需要转换层。
- `DrivingSDK/cylinder_query` 的 kernel 旋转方向是 `rot^T·Δ`，与官方一致；但 Python 层 `sortRes` 会排序后截断，官方要求扫描序 first-K。
- `ops-math/stack_ball_query` 的 fp16 路径把 `radius²` 转 half；官方要求 float16 输入先转 float32 再比较。

## 5. 证据级别

| 项 | 级别 |
|---|---|
| 赛事 15 题目录/赛制/开源要求 | `O`（官方页面） |
| 15 题完整公开题面/shape/dtype/约束 | `O`（CANN Judge 公开 API，2026-11-17） |
| 6 仓固定 commit、源码路径/行号/哈希 | `S+L` |
| 支持矩阵/接口/约束 | `A`（官方 README/文档） |
| 本机 910B3 构建/精度/时延 | `N`（仅 FFT 32×32 PASS；其余未跑） |
| 910C | `N` |

## 6. 未纳入/待跟踪

- `cann/mfg-ops-cax`、`cann/mfg-ops-vision`：制造 SIG 文档提到，但 2026-11-17 无公开 HEAD；等公开后再评估。
- `cann/cann-ops-competitions` 制造专场提交：当前 `02_enterprise` 未见目录；赛事未截止，后续可能出现。
- `mat-chem-sim-pred`：有 FNO/PINN/PDE 算子，但没有 `FusedPropStep` 同名/同公式实现；官方题面已确认 `FusedPropStep` 是 `IFFT2(FFT2(field)·h)`，仍无公开 Ascend C 实现。
- `flying_whj/CANN_grid_cluster`、`2403_87496454/ops-gnn-grid-cluster-950`：聚焦 grid_cluster 的个人/验收仓，含设计文档/golden/测试，但仍是 arch35（950）；官方 `cann/ops-gnn` 已覆盖同一架构，未重复纳入。
- `gcw_4VCbQSde/huawei-spgemm-task`：A2/A3 SpGEMM 社区任务工程，含设计、Self-test 和 arch22 内核；核心与 `cann/ops-sparse/spgemm` 重叠，未重复纳入，但可作为 Spspmm 的补充阅读。
- `2403_86959433/ops-gnn-segment-csr`、`WSCII/ops-gnn-segment-coo-950`、`gcw_i0yFZkCV/ops-gnn-fps-950`、`hyjw6/ops-gnn-knn-950`：个人 950 社区任务仓，未提供 A2 路径或新语义。
- `Ppsjq/spmspm_pim`、`Eason2013/Random_Walk`：与 Ascend 制造算子无关。
- 其它个人仓：未找到可固定 commit、可定位源码的制造专场参赛实现；不把搜索结果/文章当源码。
- 搜索结论：除 `ComplexExp` 命中一个无关词法分析仓、`RandomWalk`/`SpmmSum`/`Spspmm` 命中通用算法仓外，15 个精确算子名在 GitCode/AtomGit 上没有新的制造专场 Ascend C 实现；`FusedPropStep`、`CylinderQuery`、`BallQuery`、`GatherCoo`、`SegmentMinCsr`、`FurthestPointSample` 精确搜索均为 0。

## 7. 下一步

1. `ops-fft/fft2_d` 已完成 32×32 C2C（best 0.1537 ms、max_abs 9.086799e-08）；继续 64/128、batch>1、backward；
2. 构建/调用 `DrivingSDK` 的 Radius/FPS/CylinderQuery（A2 文档最明确）；
3. 构建 `ops-math/stack_ball_query`（有 910B 配置）；
4. 构建 `ops-gnn/segment_csr`、`ops-sparse/spmm`/`spgemm` 的 arch22，跑小图/小点云；
5. 拿到 ComplexExp/FusedPropStep 题面后，补候选实现；在此之前保持缺口。
