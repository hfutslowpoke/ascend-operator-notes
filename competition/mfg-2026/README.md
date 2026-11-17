# 2026 制造行业算子专场（15 题）源码地图

> 层级：`competition` · 赛事：昇腾 AI 行业创新大赛——2026 制造行业算子专场 · 访问日期：2026-11-17

## 1. 赛事概况

| 项 | 值 |
|---|---|
| 赛事主页 | https://cann-mfg.gccorg.com/ |
| 赛题数 | 15（仿真计算 5、图计算 5、点云 4、机器人抓取 1） |
| 预赛 | 2026-11-09 至 2026-12-07 12:00（北京时间） |
| 决赛 | 2026-12-20，广州 |
| 官方算力 | 云端 Ascend 910B |
| 评测 | CANN Judge 精度校验 + 运行性能/耗时排名 |
| 开源要求 | 获奖者须在指定 AtomGit 仓开源并合入 PR，同时发布技术总结 |

本条目不是“参赛提交”，而是把公开可用的官方/生产源码映射到 15 道题，记录哪些能直接参考、哪些只能看算法、哪些还没有公开实现。

## 2. 15 题与源码覆盖

| # | 赛题 | 类别 | 本库候选源 | 910B 状态 | 关键说明 |
|---:|---|---|---|---|---|
| 01 | `ComplexAbs` | 仿真 | [ops-math/math/complex_abs](ops-math/README.md) | 参考（950/350） | A2 明确不支持；公式 `sqrt(re²+im²)`，dtype 映射固定 |
| 02 | `ComplexDiv` | 仿真 | [ops-solver/complex_solver_kernel.hpp](ops-solver/README.md) | 参考（950 SIMT） | 不是独立 elementwise 算子；有缩放 + FMA 补偿的复数除法 |
| 03 | `ComplexExp` | 仿真 | **未找到公开实现** | 缺口 | 可用 `exp(a)*(cos b + i sin b)` 组合；需官方题面确认 dtype/范围 |
| 04 | `FusedPropStep` | 仿真 | **未找到公开实现** | 缺口 | 名称指向融合传播步；没有题面/公式，不能猜实现 |
| 05 | `FourierTransform2D` | 仿真 | [ops-fft/fft2_d](ops-fft/README.md) | **PASS（本机 910B3，32×32）** | 仅 C2C、尺寸 {32,64,128}²；950 不支持；64/128/batch>1 未跑 |
| 06 | `GatherCoo` | 图 | [ops-gnn/gather_coo](ops-gnn/README.md) + [ops-sparse/gather](ops-sparse/README.md) | 部分 | gnn 版 arch35-only；sparse 版是 dense gather，不是 COO 行扩展 |
| 07 | `SegmentMinCsr` | 图 | [ops-gnn/segment_csr arch22](ops-gnn/README.md) | **强（A2/A3）** | 同一模板覆盖 min/max/sum；int64 indptr；空段填充语义 |
| 08 | `RandomWalk` | 图 | [ops-gnn/random_walk](ops-gnn/README.md) | 参考（950） | Philox RNG + node2vec；arch35-only |
| 09 | `SpmmSum` | 图 | [ops-gnn/spmm](ops-gnn/README.md)、[ops-sparse/spmm](ops-sparse/README.md) | **强（A2/A3）** | CSR × dense，sum/max/min/mean；fp32 Kahan 路径 |
| 10 | `Spspmm` | 图 | [ops-sparse/spgemm](ops-sparse/README.md) | **强（A2/A3）** | CSR × CSR；多阶段 symbolic/numeric/merge |
| 11 | `Radius` | 点云 | [DrivingSDK/radius](driving-sdk/README.md) | **强（A2/A3/950）** | `Ndim<9`、batch≤1024、单 batch 点数≤512 |
| 12 | `GridCluster` | 点云 | [ops-gnn/grid_cluster](ops-gnn/README.md) | 参考（950） | 网格量化 + int64 线性 ID；arch35-only |
| 13 | `FurthestPointSample` | 点云 | [DrivingSDK/furthest_point_sampling](driving-sdk/README.md) | **强（A2/A3/950）** | 迭代最远点；总元素数≤383166 |
| 14 | `BallQuery` | 点云 | [ops-math/stack_ball_query](ops-math/README.md)（A2）/ `ball_query`（950） | 部分/强 | 直接 `ball_query` 代码 arch35-only，README 却写 A2 √；`stack_ball_query` 有 910B 配置 |
| 15 | `CylinderQuery` | 机器人 | [DrivingSDK/cylinder_query](driving-sdk/README.md) | **强（A2-targeted，待构建）** | 3×3 旋转 + 半径/高度筛选；输出后 Python 排序截断 |

覆盖口径：

- **8 题有 910B/A2 目标实现**：05（已本机 PASS 32×32）、07、09、10、11、13、14（经 StackBallQuery）、15；
- **5 题只有 950/arch35 参考或部分覆盖**：01、02、06、08、12；
- **2 题未找到公开源码**：03、04。

“强”只表示源码/文档明确面向 A2 或存在 A2 构建路径，**不等于本机已跑通**；截至 2026-11-17，仅 05 `fft2_d` 在本机 910B3 通过 32×32 C2C，其余 14 份仍 `local_status=not-run`，见第 5 节。

## 3. 子条目

| 子条目 | 仓库 | 覆盖赛题 | 说明 |
|---|---|---|---|
| [ops-fft](ops-fft/README.md) | `cann/ops-fft@65f5ff1` | 05 | 910B 的 2D C2C FFT；P/Q 矩阵 + Cube |
| [driving-sdk](driving-sdk/README.md) | `Ascend/DrivingSDK@d10b13c` | 11、13、15 | 点云 Radius/FPS/CylinderQuery；A2 文档齐全 |
| [ops-gnn](ops-gnn/README.md) | `cann/ops-gnn@ea0b18a` | 06、07、08、09、12 | A2/A3 与 950 分架构；segment/spmm 可上 910B |
| [ops-sparse](ops-sparse/README.md) | `cann/ops-sparse@8fae00b` | 06、09、10 | A2/A3 的 SpMM/SpGEMM/Gather |
| [ops-math](ops-math/README.md) | `cann/ops-math@3b2baac` | 01、14 | ComplexAbs/BallQuery/StackBallQuery；有 README/代码冲突 |
| [ops-solver](ops-solver/README.md) | `cann/ops-solver@739775a` | 02 | 复数除法算法参考；950 SIMT only |

机器索引：各子条目 `source.json`；全库索引：[index/catalog.json](../../index/catalog.json)。

## 4. 为什么先做“源码地图”而不是直接改算子

- 赛事仍在进行（截止 2026-12-07），公开参赛 PR 可能后续才出现；先固定官方/生产源码，后续可增量补充。
- 部分赛题（03、04）没有公开公式；没有题面就实现，容易把语义做错。
- 官方仓的 README 与代码架构可能不一致（例如 `ball_query` 的 A2 支持）；必须用源码/配置/本机构建交叉验证。
- 竞赛目标是 910B 性能，但本库不能把 950/arch35 实现直接标成 910B 可复用；硬件分支要单独写清。

## 5. 本机验证状态

本批 15 份实现中，**1 份本机 PASS、14 份未跑**：

| 实现 | 状态 | 证据 |
|---|---|---|
| 05 `FourierTransform2D` / `ops-fft/fft2_d` | **PASS**：CANN 9.1.0 构建成功；32×32 batch=1 C2C，best 0.1537 ms，max_abs=9.086799e-08，bad=0 | [构建日志](evidence/ops-fft_fft2d_build_20261117.log)、[运行日志](evidence/ops-fft_fft2d_32x32_run_20261117.log)、[驱动](evidence/fft2d_verify.cpp) |
| 其余 14 份 | `not-run` | [evidence/README.md](evidence/README.md) |

后续优先级：

1. `ops-fft/fft2_d` 扩到 64×64、128×128、batch>1、backward；
2. `DrivingSDK/radius`、`furthest_point_sampling`、`cylinder_query`：文档明确 A2，需确认单算子构建方式；
3. `ops-gnn/segment_csr`、`ops-sparse/spmm`/`spgemm`：arch22 路径存在，但仓库构建较重；
4. `ops-math/stack_ball_query`：有 `ascend910b` 配置，可直接验证；
5. 950-only 的 01/02/06/08/12 只做算法参考，不在 910B 上强构建。

验证协议见 [skills/ascend-910b-local-validation/SKILL.md](../../skills/ascend-910b-local-validation/SKILL.md)。构建/运行日志统一放 `competition/mfg-2026/evidence/`。

## 6. 证据与许可

- 作者/官方文档：`A`；源码路径/行号/哈希：`S+L`；本机：`N`。
- `cann/ops-*` 为 CANN Open Software License Agreement Version 2.0；`Ascend/DrivingSDK` 为 Apache-2.0。
- 本库把 36 个唯一选中源码文件（39 条引用）复制到各子条目的 `source/`，保留上游路径、固定 commit、URL 和 SHA256；未复制整仓，`source.json` 的 `local_path` 可直接打开。副本是上游原样快照，原文里的相对链接按整仓解析，本地不保证可点开。
- 不把官方仓的“支持 A2”直接当成本机通过；不把 950-only 实现写成 910B 方案。

## 7. 后续增量

- 跟踪 `cann/cann-ops-competitions` 的 `02_enterprise` 制造专场提交目录；
- 跟踪 `cann/mfg-ops-cax`、`cann/mfg-ops-vision` 是否公开（2026-11-17 `git ls-remote` 无 HEAD）；
- 找到 03/04 的官方题面/公式后，补 `ComplexExp` 与 `FusedPropStep` 的候选实现；
- 05 已完成 32×32 C2C；继续对 07/09/10/11/13/14/15 做 910B3 构建/精度/计时，并扩展 05 的 shape/backward。
