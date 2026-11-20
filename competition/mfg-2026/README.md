# 2026 制造行业算子专场（15 题）源码地图

> 层级：`competition` · 赛事：昇腾 AI 行业创新大赛——2026 制造行业算子专场 · 官方题面抓取：2026-11-17

## 1. 赛事概况

| 项 | 值 |
|---|---|
| 赛事主页 | https://cann-mfg.gccorg.com/ |
| 官方题面 | https://cannjudge.cn/public/gcc_2026_mfg/<problem>（15 题） |
| 赛题数 | 15（仿真计算 5、图计算 5、点云 4、机器人抓取 1） |
| 预赛 | 2026-11-09 至 2026-12-07 12:00（北京时间） |
| 决赛 | 2026-12-20，广州 |
| 官方算力 | 云端 Ascend 910B |
| 评测 | CANN Judge 精度校验 + 运行性能/耗时排名 |
| 开源要求 | 获奖者须在指定 AtomGit 仓开源并合入 PR，同时发布技术总结 |

本条目不是“参赛提交”，而是把公开官方/生产源码映射到 15 道题，并用官方题面校正“源码能做什么、不能做什么”。题目 ID/测试点数/源码分级的联合表见 [cannjudge-map.md](cannjudge-map.md)；两层库与 CANNJudge 工具/提交的关系见 [CANNJUDGE_INTEGRATION](../../docs/CANNJUDGE_INTEGRATION.md)。制造专场最容易错的比较符/tie/空段/固定槽位小点见 [10 个小点](../../docs/experiences/manufacturing_operator_edges.md)。6 个 adapt 题的具体改造步骤见 [adapters.md](adapters.md)；上游 tie/空段/first-K 语义见 [算法参考核对](../../docs/research/MFG2026_ALGORITHM_REFERENCES_20261117.md)。完整公开题面已保存到 [official/descriptions/](official/descriptions/)，机器可读摘要在 [official/specs.json](official/specs.json)。

## 2. 15 题与源码覆盖

匹配级别定义：

- `direct`：源码语义与官方一致，且存在 A2/910B 路径；只需调用/薄适配；
- `adapt`：有 A2/910B 目标源码，但接口、layout、shape 或后处理需改；
- `reference`：只有 950/arch35 或算法 helper，可读但不可直接构建到 910B；
- `none`：未找到公开完整实现；
- `building-block`：可组合的 910B 基础件，不是完整题解（不计入 15 题分级）。

| # | 赛题 | 官方题面 | 本库候选源 | 级别 | 关键校正 |
|---:|---|---|---|---|---|
| 01 | `ComplexAbs` | [题面](official/descriptions/complexabs.md) | [ops-math](ops-math/README.md) | `reference` | 源码 arch35（950/A3），A2 明确不支持；官方只需 complex64→float32 |
| 02 | `ComplexDiv` | [题面](official/descriptions/complexdiv.md) | [ops-solver](ops-solver/README.md) | `reference` | 源码是 950 SIMT 的 complex64 LU helper，不覆盖 float16/float32/广播 |
| 03 | `ComplexExp` | [题面](official/descriptions/complexexp.md) | 未找到单 kernel；[ops-math](ops-math/README.md) 有 910B 基础件 | `none` | 官方公式/dtype 已明确；可用 exp/sin/cos + ComplexV3 组合，非单 kernel |
| 04 | `FusedPropStep` | [题面](official/descriptions/fusedpropstep.md) | 未找到单 kernel；[ops-fft](ops-fft/README.md) 有 1D/2D FFT + R2C/C2R 基础件 | `none` | 官方是 `IFFT2(FFT2(field)·h)`；fft1_d/fft2_d 可组合；实数输入可用 FP32 rfft1_d/irfft1_d，但融合/norm/FP16 未实现 |
| 05 | `FourierTransform2D` | [题面](official/descriptions/fouriertransform2d.md) | [ops-fft](ops-fft/README.md) | `adapt` | fft2_d 只支持 32/64/128；fft1_d 提供单维 2..8192 基础件；官方 2D+inverse 仍需组合 |
| 06 | `GatherCoo` | [题面](official/descriptions/gathercoo.md) | [ops-gnn](ops-gnn/README.md)、[ops-sparse](ops-sparse/README.md) | `reference` | gnn 版 arch35、int64、rank1–8；官方 int32、`[M,K]→[E,K]`；sparse gather 不是 COO 行扩展 |
| 07 | `SegmentMinCsr` | [题面](official/descriptions/segmentmincsr.md) | [ops-gnn](ops-gnn/README.md) | `direct` | arch22 `segment_min_csr` 已覆盖 arg_out、空段 `0/E`、首遇 tie-break |
| 08 | `RandomWalk` | [题面](official/descriptions/randomwalk.md) | [ops-gnn](ops-gnn/README.md) | `reference` | 源码 arch35 用 Philox；官方要求外部 `rand` 输入、p/q、孤立点停留 |
| 09 | `SpmmSum` | [题面](official/descriptions/spmmsum.md) | [ops-sparse](ops-sparse/README.md)、[ops-gnn](ops-gnn/README.md) | `adapt` | A2/arch22 有 CSR SpMM；官方接口是 rowptr/col/value/mat/out，需薄适配 |
| 10 | `Spspmm` | [题面](official/descriptions/spspmm.md) | [ops-sparse](ops-sparse/README.md) | `adapt` | 源码 CSR + 动态 nnz；官方 COO + 固定槽位 `U=m·n`，需转换层 |
| 11 | `Radius` | [题面](official/descriptions/radius.md) | [DrivingSDK](driving-sdk/README.md) | `adapt` | 源码动态输出、单 batch≤512；官方定槽位、`(-1,-1)`、`D∈{2,3}`、`ignore_same_index` |
| 12 | `GridCluster` | [题面](official/descriptions/gridcluster.md) | [ops-gnn](ops-gnn/README.md) | `reference` | 源码 arch35；stride 顺序与官方一致（dim0 最快），但无 A2 |
| 13 | `FurthestPointSample` | [题面](official/descriptions/furthestpointsample.md) | [DrivingSDK](driving-sdk/README.md) | `direct` | 官方题面明确以 `mx_driving` Ascend FPS 为参考；仍需本机逐位验证 |
| 14 | `BallQuery` | [题面](official/descriptions/ballquery.md) | [ops-math](ops-math/README.md) | `adapt` | 官方单 `radius`、严格 `<`、fp16→fp32 阈值；`stack_ball_query` 是堆叠布局，fp16 把 `radius²` 转 half |
| 15 | `CylinderQuery` | [题面](official/descriptions/cylinderquery.md) | [DrivingSDK](driving-sdk/README.md) | `adapt` | kernel 的 `rot^T·Δ` 与官方一致；Python 层排序后截断，官方要求扫描序 first-K |

覆盖口径：

- **2 题有语义高度匹配的 A2/910B 实现**：07 `SegmentMinCsr`、13 `FurthestPointSample`（13 仍待本机逐位验证）；
- **6 题有 A2/910B 目标源码但需适配**：05、09、10、11、14、15；
- **5 题只有 950/arch35 或算法 helper**：01、02、06、08、12；
- **2 题未找到公开源码**：03、04。

“有 A2 路径”不等于“可直接提交”。例如 FFT 源码只覆盖 32/64/128，官方要求到 8192；Spspmm 源码是 CSR 动态输出，官方是 COO 固定槽位；CylinderQuery 源码 Python 层会排序，官方要求扫描序 first-K。

### 2.1 910B 补充基础件（不计入 15 题分级）

| 实现 | 基础件 | 用途 | 仍缺 |
|---|---|---|---|
| `MFG2026-FFT1D` | `ops-fft/fft1_d` + `rfft1_d`/`irfft1_d` | C2C 1D：n≤256 DFT、512..16384 FFTB、更大 FFTN；R2C/C2R 1D（FP32）：2 的幂到 2^29/2^27，适合实数 FusedPropStep | 行/列 2D 组合、inverse `1/(N·M)`、FP16 R2C/C2R、norm |
| `MFG2026-MATH-COMPLEX-EXP-BLOCKS` | `exp/sin/cos + ComplexV3` | 910B 实数基础件 + real/imag→complex；可组合 ComplexExp | 单 kernel 融合、fp16 极值/精度、4 个算子的 GM 往返 |

这两项在 `index/catalog.json` 中标记为 `source_match=building-block`，不计入 15 题的 `direct/adapt/reference/none` 数量。

## 3. 子条目

| 子条目 | 仓库 | 覆盖赛题 | 说明 |
|---|---|---|---|
| [ops-fft](ops-fft/README.md) | `cann/ops-fft@65f5ff1` | 05、04 的基础件 | 910B 2D C2C FFT（32/64/128）+ 1D FFT（n≤256 DFT、512..16384 FFTB） |
| [driving-sdk](driving-sdk/README.md) | `Ascend/DrivingSDK@d10b13c` | 11、13、15 | Radius/FPS/CylinderQuery；A2 文档齐全，但接口需适配 |
| [ops-gnn](ops-gnn/README.md) | `cann/ops-gnn@ea0b18a` | 06、07、08、09、12 | Segment/Spmm 有 arch22；Gather/RandomWalk/GridCluster 仅 arch35 |
| [ops-sparse](ops-sparse/README.md) | `cann/ops-sparse@8fae00b` | 06、09、10 | A2/A3 SpMM/SpGEMM/Gather；SpGEMM 是 CSR 而非官方 COO |
| [ops-math](ops-math/README.md) | `cann/ops-math@3b2baac` | 01、14 | ComplexAbs/BallQuery/StackBallQuery；有 README/代码冲突 |
| [ops-solver](ops-solver/README.md) | `cann/ops-solver@739775a` | 02 | 复数除法算法参考；950 SIMT only |

机器索引：各子条目 `source.json`；全库索引：[index/catalog.json](../../index/catalog.json)。

## 4. 为什么先做“源码地图”而不是直接改算子

- 赛事仍在进行（截止 2026-12-07），公开参赛 PR 可能后续才出现；先固定官方/生产源码，后续可增量补充。
- 官方题面已抓到，但隐藏测试数据和参考实现不公开；不能把公开题面里的示例当成 golden。
- 官方仓的 README 与代码架构可能不一致（例如 `ball_query` 的 A2 支持、`fft2_d` 的尺寸限制）；必须用源码/配置/本机构建交叉验证。
- 竞赛目标是 910B 性能，但本库不能把 950/arch35 实现直接标成 910B 可复用；硬件分支要单独写清。
- 同一数学对象不等于同一接口：Spspmm 的 COO 固定槽位、Radius 的定槽位输出、CylinderQuery 的 first-K 都会改变实现方案。

## 5. 本机验证状态

本批 15 题映射 + 2 个 910B 基础件。15 题中 **1 份有本机 PASS（仅 32×32 子集）、14 份未跑**；两个基础件也未跑：

| 实现 | 状态 | 证据 |
|---|---|---|
| 05 `FourierTransform2D` / `ops-fft/fft2_d` | **PASS（子集）**：CANN 9.1.0 构建成功；32×32 batch=1 C2C，best 0.1537 ms，max_abs=9.086799e-08，bad=0 | [构建日志](evidence/ops-fft_fft2d_build_20261117.log)、[运行日志](evidence/ops-fft_fft2d_32x32_run_20261117.log)、[驱动](evidence/fft2d_verify.cpp) |
| 其余 14 份 | `not-run` | [evidence/README.md](evidence/README.md) |

注意：官方 `FourierTransform2D` 要求 `B∈[1,256]`、`N,M∈[2,8192]` 的 2 的幂和 `inverse` 属性；当前 PASS 只证明 32×32 forward 可运行，**不能代表官方 18 组 shape 通过**。

后续优先级：

1. `ops-fft/fft2_d` 扩到 64/128/256/512/1024/2048/4096/8192、batch>1、inverse；
2. `DrivingSDK/furthest_point_sampling`：官方 18 组 shape、float16 half 精度、并列 tie-break；
3. `ops-gnn/segment_min_csr`：官方 arg_out/空段/首遇 tie-break 的 910B3 对照；
4. `DrivingSDK/radius`、`cylinder_query`、`ops-math/stack_ball_query`：按官方定槽位/first-K/float32 阈值改适配层；
5. `ops-sparse/spmm`、`spgemm`：SpmmSum 薄适配、Spspmm 的 COO/固定槽位转换；
6. 950-only 的 01/02/06/08/12 只做算法参考，不在 910B 上强构建。

验证协议见 [skills/ascend-910b-local-validation/SKILL.md](../../skills/ascend-910b-local-validation/SKILL.md)。构建/运行日志统一放 `competition/mfg-2026/evidence/`。

## 6. 证据与许可

- 官方题面/规格：`O`（CANN Judge 公开页面/API，2026-11-17）；源码路径/行号/哈希：`S+L`；本机：`N`。
- `cann/ops-*` 为 CANN Open Software License Agreement Version 2.0；`Ascend/DrivingSDK` 为 Apache-2.0。
- 本库把 79 个唯一选中源码文件（82 条引用）复制到各子条目的 `source/` 或 `supporting/`，保留上游路径、固定 commit、URL 和 SHA256；未复制整仓，`source.json` 的 `local_path` 可直接打开。副本是上游原样快照，原文里的相对链接按整仓解析，本地不保证可点开。
- 官方题面只保存公开描述，不包含隐藏测试数据；`/api/testcases/<id>` 返回 403，未绕过。
- 不把官方仓的“支持 A2”直接当成本机通过；不把 950-only 实现写成 910B 方案。

### 6.1 上游算法参考核对

除了 Ascend 源码，本库还用公开 CPU/CUDA 参考实现核对了官方边界语义，详见 [MFG2026_ALGORITHM_REFERENCES_20261117.md](../../docs/research/MFG2026_ALGORITHM_REFERENCES_20261117.md)：

| 题 | 参考 | 核对结论 |
|---|---|---|
| 07 `SegmentMinCsr` | pyg-lib `segment_csr_kernel.cpp` | `arg_out=E`、空段 `out=0`、严格 `<` 首遇，与官方逐条一致 |
| 13 `FPS` | torch_cluster `fps_cpu.cpp` | `argmax` 首最大值 = 最小索引 tie；ratio/count 不是官方接口 |
| 11 `Radius` | torch_cluster `radius_cpu.cpp` | `ignore_same_index` 一致；KD-tree 顺序/动态输出不能复用 |
| 14 `BallQuery` | AnyGrasp `ball_query_gpu.cu` | 严格 `d²<r²`、first-K、不足重复首个、零命中全 0 |
| 15 `CylinderQuery` | AnyGrasp `cylinder_query_gpu.cu` | `rot^T·Δ`、严格边界、first-K、不足重复首个、零命中全 0 |

这些参考只用于核对语义，不是 Ascend 实现；实际复用代码前阅读对应 LICENSE。语义已整理成可运行的 CPU 参考：[`reference/`](reference/)（9 项测试 PASS，非官方隐藏 golden）。

## 7. 后续增量

- 跟踪 `cann/cann-ops-competitions` 的 `02_enterprise` 制造专场提交目录；
- 跟踪 `cann/mfg-ops-cax`、`cann/mfg-ops-vision` 是否公开（2026-11-17 `git ls-remote` 无 HEAD）；
- 03/04 官方题面已补；仍无公开单 kernel，但已加入 910B 基础件（exp/sin/cos + ComplexV3；fft1_d/rfft1_d/irfft1_d），可先做组合参考实现；
- 对 07/09/10/11/13/14/15 做 910B3 构建/精度/计时，并扩展 05 到官方 shape/inverse；
- 评估 GitCode 上的个人聚焦仓（如 `flying_whj/CANN_grid_cluster`）是否值得作为补充阅读材料，避免与官方仓重复。
