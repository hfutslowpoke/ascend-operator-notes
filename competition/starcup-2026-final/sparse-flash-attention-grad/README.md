# SparseFlashAttentionGrad（星辰杯决赛）

> 层级：`competition` · 赛事：中国电信星辰杯·高校 AI 算子开发挑战赛 决赛 · 平台：`6/28, 73.77` · 本机：`not-run`

## 0. 来源登记

| 项 | 值 |
|---|---|
| 官方题面 | [sparseflashattentiongrad.md](../official/descriptions/sparseflashattentiongrad.md) |
| 官方 problem JSON | [problem-6ac8bc7c694b590c3c96728a.json](../official/problem-6ac8bc7c694b590c3c96728a.json) |
| 平台提交 | `sc_final_sparseflashattgrad` |
| 正式源码 | [`source/`](source/) |
| 证据 | [`evidence/`](evidence/)（detail-summary / cases.csv / ranking-summary / sid / source.sha256） |
| 备注 | 源码为平台提交原文（token 压缩形式） |
| 本机状态 | `not-run`（本次未在本机 910B3 重跑） |

## 1. 平台证据

| 项 | 值 |
|---|---|
| contest | `ct_starcup_aiop_final` / 星辰杯决赛 |
| problem_id | `6ac8bc7c694b590c3c96728a` |
| 提交账号 | `feixiangtian16`（uid `6aad2318b0477ec41e57bd53`），团队 `hfutslowpoke 的队伍` |
| SID | `6ae610371dac5d235b1e9b2e` |
| detail ID | 1042875 |
| 排名 | 6 / 28 |
| score | 73.77 |
| 状态 | Pass / valid=true |
| 可见 case | 10 / 15 |
| 可见总时间 | 1847.24 ms |
| 可见平均 t/T | 1.108 |
| 可见平均 case score | 81.30 |
| 最终 score | 73.77（隐藏 5 case 拖低） |

逐 case（仅可见）：

| case | time (ms) | best (ms) | t/T |
|---|---:|---:|---:|
| 1 | 283.10 | 258.20 | 1.096 |
| 2 | 77.88 | 71.22 | 1.094 |
| 3 | 33.00 | 32.38 | 1.019 |
| 4 | 701.82 | 553.51 | 1.268 |
| 5 | 252.08 | 236.00 | 1.068 |
| 6 | 61.03 | 56.87 | 1.073 |
| 7 | 119.72 | 116.52 | 1.027 |
| 8 | 36.43 | 30.44 | 1.197 |
| 9 | 206.66 | 179.89 | 1.149 |
| 10 | 75.52 | 69.19 | 1.091 |

可见 case 平均 81.30，最终 73.77；隐藏 5 个 case 的平均分明显低于可见组。
这与 GroupedMatmulGelu 是同一类现象：**公开/可见 case 接近冠军，隐藏 case 拖后腿**。

## 2. 题目语义

```text
query/key/value: (B,S,N1/D)
dOut/out:        (B,S1,N1,D)
softmaxMax/Sum:  (B,N2,S1,G) FP32
sparseIndices:   每个 query 选中的 KV 位置
sparseBlockSize: 1/8/16/32/64
sparseMode:      0 或 3（rightDownCausal）
D=512, Dr=64, N2=1（MQA），N1∈{1,2,4,8,16,32,64,128}
K∈{1024,2048,3072,4096,5120,6144,7168,8192}
B≤256, S1/S2≤128K
```

三阶段：

```text
1) dP = dO @ V^T
   dV = P^T @ dO
2) P = exp(S - softmaxMax) / softmaxSum
   FlashSoftmaxGrad = sum(dO ⊙ O, 行)
   dS = P ⊙ (dP - FlashSoftmaxGrad)
3) dQ = dS @ K / sqrt(d)
   dK = dS^T @ Q / sqrt(d)
4) scatter dK/dV 回原位置；KV merge 时 dKeyOut = dK + dV
5) RoPE 存在时算 dQueryRope/dKeyRope
```

## 3. 代码来源

- API 详情：``
- `op_host/sparse_flash_attention_grad.cpp`：9 行（主要是 op def）
- `op_kernel/sparse_flash_attention_grad.cpp`：406,057 B / 885 行，
  但单行极长；源码经过 token 替换/宏压缩，变量名如 `J00`、`J01`、`W4B`。
- `tiling_key_sparse_flash_attention_grad.h`：3,391 行，定义 13 个布尔模板参数和
  多个 tiling struct。
- 用户未提供该题 zip；以 API 详情为准。

## 4. 实现结构

### 4.1 tiling key

`ASCENDC_TPL_ARGS_DECL` 暴露 13 个布尔量：

```text
DUP_PATH, HAS_ROPE, HAS_VALUE, CUBE_PATH, FW_PATH,
COUNT_RANK, BATCH_WINDOW, OPT_SCORE, OPT_WINDOW,
BYTE_COUNT, HR_GRAPH_ENTRY, C10_OPT_ENTRY, C10_WB_ENTRY
```

外加 `DT_QUERY`（fp16/bf16）和多个 tiling struct：

```text
SfagLegacyTiling96, SfagCompactTiling8, SfagLegacyTiling100,
SfagGeneralTiling64, ...
```

这说明实现按“是否有 RoPE / 是否有 value / 是否 Cube / 是否 forward /
是否 batch window / 是否 count rank / 是否优化 score”等组合实例化。

### 4.2 主要构件

从符号名可辨认：

- `SfagGeneralVectorShared`：通用 Vector 路径；
- `SfagR1Owner`：R1 owner 路径；
- `SfagC3Submission20261026R1`：C3 提交路径（含 published tiling ABI）；
- `SfagCubeGemm` / `SfagCubePipeline`：Cube GEMM / 流水；
- `SfagStageGemmScore448` / `SfagStageGemmBaseline`：score stage；
- `SfagSparseHalfGemm` / `SfagSparseHalfOwnerProcess`：稀疏半精度路径；
- `SfagSpatialGrad128` / `SfagSpatialAV` / `SfagSpatialIC`：按 head/KV 的空间切分；
- `SfagC5...` / `SfagC7...`：后续版本/精确 stage；
- `SfagC3KvOps` / `SfagC3AheadDqOps`：KV 与 dQ 的跨 stage 流水。

实现特征：

- AIC/AIV `MIX_AIC_1_2`；
- L1 512 KB，按 `q/k/v/dO/out` 分块，双缓冲；
- 稀疏 gather 只读选中的 KV；
- `dK/dV` scatter 回原位置并累加；
- KV merge 时把 dV 合入 dK；
- RoPE 反向融合；
- 按 head / K tile / query tile 分核，跨核 `CrossCoreSetFlag` 同步；
- score stage 的 tile 大小随 `sparseBlockSize`/`K` 变化。

> 注意：源码是 token 替换后的压缩形式，以上是从符号、tiling key、注释和结构推断，
> 不是逐行可读的实现。入库时应保留原文，但分析文档要明确“可读性差”。

## 5. 为什么高效

### 5.1 可见 case

- 10 个可见 case 平均 1.108x best，已经接近冠军；
- 关键收益：
  - 三阶段融合，dP/dV/dS/dQ/dK 不落多次 GM；
  - 复用前向 `softmaxMax/Sum`，不重算 softmax；
  - 稀疏 gather 只处理选中 KV，避免 dense 计算；
  - KV merge 场景把 dV 合入 dK，省一次输出；
  - L1 常驻 + 双缓冲，Cube 与 MTE 重叠；
  - 按 head/K tile 分核，利用多核。

### 5.2 为什么最终只有 73.77

可见 10 case 平均 81.30，最终 73.77。按“总分=全部 case 算术平均”反推，
隐藏 5 case 的平均分约为：

```text
(73.77 * 15 - 81.30 * 10) / 5 = 58.71
```

隐藏组比可见组低约 22.6 分。可能原因：

- 隐藏 case 的 `N1`/`S2`/`sparseBlockSize`/`HAS_ROPE`/`HAS_VALUE` 组合没有命中
  最优 tiling key；
- 某些组合落到 `SfagGeneralVectorShared` 或 legacy tiling，性能明显低于 Cube 路径；
- `DUP_PATH`/`OPT_SCORE`/`OPT_WINDOW` 等布尔组合的启发式选择对隐藏 shape 不一定最优；
- 代码高度压缩，难以做逐 case 调参；某些分支可能是后期补的，未充分优化。

### 5.3 与冠军的差距

| 项 | 本提交 | 第一名 `serendipity_lin` |
|---|---:|---:|
| score | 73.77 | 94.63 |
| 可见总时间 | 1847.24 ms | 1625.21 ms |
| 平均 t/T | 1.108 | 1.0 |

可见 case 只差 13.7% 总时间；最终分差 20.86，说明隐藏 case 的差距更大。
冠军可能在隐藏 shape 上用了更通用的 Cube 路径，而不是依赖精确 tiling key。

## 6. 可复用与风险

**可复用：**

- 三阶段融合 + 前向统计复用；
- 稀疏 gather / scatter + KV merge；
- L1 双缓冲 + AIC/AIV 分工；
- 按 head/K/query tile 分核；
- tiling key 按 `HAS_ROPE/HAS_VALUE/CUBE_PATH` 等组合实例化。

**风险：**

- 源码 406 KB / 885 行 token 替换，几乎不可维护；不建议作为教学样例逐行讲解；
- 13 个布尔模板参数 + 多个 tiling struct，组合爆炸，隐藏 shape 可能选错路径；
- 可见 case 接近冠军，但最终分差 20.86，说明隐藏 case 泛化不足；
- 本次未在本机 910B3 重跑；未看隐藏 case 输入。

## 7. 入库标注

```text
source_match: our-submission (CANNJudge SID 6ae61037...)
platform_evidence: rank 6/28, score 73.77, 10/15 visible
local_status: not-run
specialization: 13 个布尔 tiling key + 多 tiling struct
generalization: 隐藏 5 case 明显慢；可能落到 legacy/general 路径
risk: 源码 token 压缩、不可读；组合爆炸；无法逐行审计
```
