# 制造专场 × CANN Judge 联合地图

由 `tools/build_mfg_official.py` 从 `official/specs.json`、`official/problems.json` 和 `index/catalog.json` 生成；不要手改。

## 赛事

- contest_id：`6aed4926a5348dcef5dd8d0c`
- 标题：昇腾AI行业创新大赛——2026制造行业算子专场（预赛）
- 题数：15
- 官方题面：<https://cannjudge.cn/public/gcc_2026_mfg/>
- 抓取日期：2026-11-17

## 15 题联合表

| # | CANN Judge 题目 | 题面 | 官方分级 | 候选实现（候选级别 / 本机） | 测试点（总/排名） |
|---:|---|---|---|---|---|
| 01 | [仿真计算：ComplexAbs](https://cannjudge.cn/public/gcc_2026_mfg/complexabs)<br>`complexabs` | [desc](official/descriptions/complexabs.md) | `reference` | [`MFG2026-MATH-COMPLEX-ABS`](ops-math/README.md) `reference` / `not-run` | 15 / 8 |
| 02 | [仿真计算：ComplexDiv](https://cannjudge.cn/public/gcc_2026_mfg/complexdiv)<br>`complexdiv` | [desc](official/descriptions/complexdiv.md) | `reference` | [`MFG2026-SOLVER-COMPLEX-DIV`](ops-solver/README.md) `reference` / `not-run` | 18 / 8 |
| 03 | [仿真计算：ComplexExp](https://cannjudge.cn/public/gcc_2026_mfg/complexexp)<br>`complexexp` | [desc](official/descriptions/complexexp.md) | `none` | [`MFG2026-MATH-COMPLEX-EXP-BLOCKS`](ops-math/README.md) `building-block` / `not-run` | 18 / 8 |
| 04 | [仿真计算：FusedPropStep](https://cannjudge.cn/public/gcc_2026_mfg/fusedpropstep)<br>`fusedpropstep` | [desc](official/descriptions/fusedpropstep.md) | `none` | — | 18 / 8 |
| 05 | [仿真计算：FourierTransform2D](https://cannjudge.cn/public/gcc_2026_mfg/fouriertransform2d)<br>`fouriertransform2d` | [desc](official/descriptions/fouriertransform2d.md) | `adapt` | [`MFG2026-FFT1D`](ops-fft/README.md) `building-block` / `not-run`<br>[`MFG2026-FFT2D`](ops-fft/README.md) `adapt` / `pass` | 15 / 8 |
| 06 | [图计算：GatherCoo](https://cannjudge.cn/public/gcc_2026_mfg/gathercoo)<br>`gathercoo` | [desc](official/descriptions/gathercoo.md) | `reference` | [`MFG2026-GNN-GATHER-COO`](ops-gnn/README.md) `reference` / `not-run`<br>[`MFG2026-SPARSE-GATHER`](ops-sparse/README.md) `not-applicable` / `not-run` | 24 / 8 |
| 07 | [图计算：SegmentMinCsr](https://cannjudge.cn/public/gcc_2026_mfg/segmentmincsr)<br>`segmentmincsr` | [desc](official/descriptions/segmentmincsr.md) | `direct` | [`MFG2026-GNN-SEGMENT-MIN-CSR`](ops-gnn/README.md) `direct` / `not-run` | 15 / 8 |
| 08 | [图计算：RandomWalk](https://cannjudge.cn/public/gcc_2026_mfg/randomwalk)<br>`randomwalk` | [desc](official/descriptions/randomwalk.md) | `reference` | [`MFG2026-GNN-RANDOM-WALK`](ops-gnn/README.md) `reference` / `not-run` | 15 / 8 |
| 09 | [图计算：SpmmSum](https://cannjudge.cn/public/gcc_2026_mfg/spmmsum)<br>`spmmsum` | [desc](official/descriptions/spmmsum.md) | `adapt` | [`MFG2026-GNN-SPMM`](ops-gnn/README.md) `adapt` / `not-run`<br>[`MFG2026-SPARSE-SPMM`](ops-sparse/README.md) `adapt` / `not-run` | 15 / 8 |
| 10 | [图计算：Spspmm](https://cannjudge.cn/public/gcc_2026_mfg/spspmm)<br>`spspmm` | [desc](official/descriptions/spspmm.md) | `adapt` | [`MFG2026-SPARSE-SPGEMM`](ops-sparse/README.md) `adapt` / `not-run` | 15 / 8 |
| 11 | [点云计算：Radius](https://cannjudge.cn/public/gcc_2026_mfg/radius)<br>`radius` | [desc](official/descriptions/radius.md) | `adapt` | [`MFG2026-RADIUS`](driving-sdk/README.md) `adapt` / `not-run` | 15 / 8 |
| 12 | [点云计算：GridCluster](https://cannjudge.cn/public/gcc_2026_mfg/gridcluster)<br>`gridcluster` | [desc](official/descriptions/gridcluster.md) | `reference` | [`MFG2026-GNN-GRID-CLUSTER`](ops-gnn/README.md) `reference` / `not-run` | 15 / 8 |
| 13 | [点云计算：FurthestPointSample](https://cannjudge.cn/public/gcc_2026_mfg/furthestpointsample)<br>`furthestpointsample` | [desc](official/descriptions/furthestpointsample.md) | `direct` | [`MFG2026-FPS`](driving-sdk/README.md) `direct` / `not-run` | 18 / 8 |
| 14 | [点云计算：BallQuery](https://cannjudge.cn/public/gcc_2026_mfg/ballquery)<br>`ballquery` | [desc](official/descriptions/ballquery.md) | `adapt` | [`MFG2026-MATH-STACK-BALL-QUERY`](ops-math/README.md) `adapt` / `not-run` | 18 / 8 |
| 15 | [机器人抓取：CylinderQuery](https://cannjudge.cn/public/gcc_2026_mfg/cylinderquery)<br>`cylinderquery` | [desc](official/descriptions/cylinderquery.md) | `adapt` | [`MFG2026-CYLINDER-QUERY`](driving-sdk/README.md) `adapt` / `not-run` | 18 / 8 |

## 三条链路

| 层 | 作用 | 入口 |
|---|---|---|
| 知识库 | 源码映射、语义分级、适配蓝图、CPU 参考 | `competition/`、`general/`、`index/catalog.json` |
| 官方题面 | 题目 ID/slug、shape/dtype、测试点数、容差 | `competition/mfg-2026/official/` |
| CANN Judge | 登录、模板、提交、回读、榜单、额度 | `tools/competition/`（便携）或 ACW `bin/cann-*` |

## 联合工作流

1. **发现**：`bin/cann-client resolve gcc_2026_mfg`，再 `bin/cann-client problems gcc_2026_mfg`；确认 `problemId`/slug。
2. **对齐**：在本页找到 slug → 官方题面 → 源码级别 → 候选实现/适配蓝图。
3. **开发**：只按官方 shape/dtype/属性特化；不按测试用例编号/hash 分支。
4. **本机验证**：经 `bin/npu-lock` 串行；记录源码/产物 SHA、设备、CANN、shape、误差、时延。
5. **打包**：`bin/cann-package --problem-id <PID> --account <A> --source-dir <src> --out <zip>`；只生成包，不 POST。
6. **dry-run**：`bin/cann-submit --problem-id <PID> --account <A> --source-dir <src> --tag <tag>`；核对模板路径、文件 SHA、payload。
7. **提交/回读**：仅在用户明确授权后 `ACW_ALLOW_SUBMIT=1 bin/cann-submit ... --submit`；拿 SID 后只 GET，保存 detail/ranking。

便携工具在 `tools/competition/tools/`，ACW 包装入口是 `bin/cann-client`、`bin/cann-package`、`bin/cann-submit`；两者语义相同，按运行环境选择。

## 当前状态与边界

- 15 题公开题面、结构化 specs、15 份 `descriptions/` 已落库；隐藏测试数据未获取，`/api/testcases/<id>` 返回 403，未绕过。
- 本页不包含任何 Cookie、Token、账号私有配置；账号文件只在私有目录/`0600`。
- 本库未记录制造专场的平台提交；本轮没有 POST。
- 本机 910B3 证据仍以条目 `local_status` 为准；`pass` 只代表本机测试，不等于平台通过。
- 官方题面/测试点数来自 CANN Judge 公开 API；平台后续更新时重新抓取并重生成。

## 安全红线

- 不把 Cookie/账号写入仓库、日志或提交包；
- 不绕过隐藏测试数据、额度或重复提交限制；
- 不把本地 `pass` 写成平台通过；不伪造 SID/榜单；
- 提交默认 dry-run；POST 必须用户显式授权（`ACW_ALLOW_SUBMIT=1` + `--submit`）。
