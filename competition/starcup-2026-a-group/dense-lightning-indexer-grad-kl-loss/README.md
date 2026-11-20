# DenseLightningIndexerGradKLLoss（星辰杯 A 组）

> 层级：`competition` · 赛事：中国电信星辰杯·高校 AI 算子开发挑战赛 A 组 · 平台：`15/94, 28.83` · 本机：`not-run`

## 0. 来源登记

| 项 | 值 |
|---|---|
| 官方题面 | [denselightningindexergradklloss.md](../official/descriptions/denselightningindexergradklloss.md) |
| 官方 problem JSON | [problem-6a7c1839a52e0f540a89c463.json](../official/problem-6a7c1839a52e0f540a89c463.json) |
| 平台提交 | `sc_a_denselightning` |
| 正式源码 | [`source/`](source/) |
| 证据 | [`evidence/`](evidence/)（detail-summary / cases.csv / ranking-summary / sid / source.sha256） |
| 备注 | zip 495694 与平台提交逐文件一致 |
| 本机状态 | `not-run`（本次未在本机 910B3 重跑） |

## 1. 平台证据

| 项 | 值 |
|---|---|
| contest | `ct_starcup_aiop_g1` / 星辰杯 A 组 |
| problem_id | `6a7c1839a52e0f540a89c463` |
| 提交账号 | `feixiangtian16`（uid `6aad2318b0477ec41e57bd53`），团队 `hfutslowpoke 的队伍` |
| SID | `6aba149b694b590c3cb925d4` |
| detail ID | 495694 |
| 排名 | 15 / 94 |
| score | 28.83 |
| 状态 | Pass / valid=true |
| 可见 case | 7 / 7 |
| 可见总时间 | 71.62 ms |
| 可见平均 t/T | 5.237 |
| 可见平均 case score | 28.83（与平台 score 完全一致） |

逐 case：

| case | time (ms) | best (ms) | t/T |
|---|---:|---:|---:|
| 1 | 4.42 | 0.64 | 6.91 |
| 2 | 5.94 | 0.62 | 9.58 |
| 3 | 7.18 | 0.62 | 11.58 |
| 4 | 11.48 | 5.50 | 2.09 |
| 5 | 17.14 | 6.12 | 2.80 |
| 6 | 14.94 | 6.26 | 2.39 |
| 7 | 10.52 | 7.98 | 1.32 |

前 3 个 case 只有 0.62–0.64 ms 的 best，而本提交要 4.4–7.2 ms，差 7–12 倍；
后 4 个 case 差 1.3–2.8 倍。说明小 shape 的固定开销/路径选择是主要问题。

## 2. 题目语义

输入：

```text
query:        (B,S1,N1,D)       FP16/BF16
key:          (B,S2,N2,D)       FP16/BF16, N2=N1
queryIndex:   (B,S1,Nidx1,D)    FP16/BF16
keyIndex:     (B,S2,1,D)        FP16/BF16, MQA
weights:      (B,S1,Nidx1)      FP16/BF16/FP32
softmaxMax/Sum:                FP32，主注意力前向统计
softmaxMax/SumIndex:           FP32，indexer 前向统计
queryRope/keyRope:             FP16/BF16, Dr=64
```

约束：N1∈{32,64,128}，Nidx1∈{8,16,32,64}，D=128，S1/S2≤128K，
sparseMode=3（rightDownCausal），B≤256，ND，仅 A2/A3。

计算：

```text
主注意力 target p：用 q@K^T/√D 的 softmaxMax/Sum 重建，再按头求和、L1 归一化
indexer 预测：I = W @ ReLU(q̃@K̃^T)，用 softmaxMaxIndex/SumIndex 重建 softmax
Loss = Σ D_KL(p || softmax(I))
dI = softmax(I) - p
dW = dI @ ReLU(S)^T
dS = outer(W,dI) ⊙ (S>0)
dq̃ = dS @ K̃
dK̃ = dS^T @ q̃
```

## 3. 代码来源

- API 详情：``
- 用户 zip `DenseLightningIndexerGradKlLoss_submission_495694.zip` 与 API 详情**逐文件 SHA256 完全一致**；
  detail ID 495694 就是平台排名提交。
- 源码：`op_host/dense_lightning_indexer_grad_kl_loss.cpp` 212 行；
  `op_kernel/dense_lightning_indexer_grad_kl_loss.cpp` 6,408 行。

## 4. 实现结构

### 4.1 host 分路

`op_host/...cpp:52-163` 用一组布尔量选模板：

| 布尔量 | 含义 |
|---|---|
| `p7ExactPublic` | fp16, B=1,S1=1,S2=8,N1=128,Nidx1=64,D=128, 8 核 |
| `headSplitDomain` | totalTasks≤2, S2≤8, D=128, N1∈{32,64,128}, Nidx1∈[8,64] |
| `headPublicA` | fp16, S1=2,S2=8,N1=64,Nidx1=16, 4 核 |
| `headPublicB` | fp32, S1=1,S2=4,N1=64,Nidx1=32, 4 核 |
| `headPublicC` | fp16, S1=1,S2=8,N1=128,Nidx1=64, 8 核 |
| `leanPublic` | fp16, N1=32,Nidx1=8,S2=4,S1=1,D=64/128 |
| `fuseBig` | S2≤8, Nidx1∈(16,64], D=128 |
| `wideFuse` | S2∈(12,16], 小 index head / 大 main head |
| `width64` | D=64 |

`ASCENDC_TPL_SEL_PARAM(context, dtype, fuseBig, selectedWideFuse, width64, leanPublic, headPublic)`
选模板；`headPublic`/`leanPublic` 还会 `SetScheduleMode(1)`。

### 4.2 kernel 主体

- `KernelLeanPublic`（`:20`）：固定 N1=32、NJ=4、NG=8、D=64/128 的 lean 路径；
  `kProdAll` 决定是否一次算全部 j；
- `KernelLeanRawB1`（`:433`）：batch=1 的 raw 路径；
- `KernelLeanPublicDynamic`（`:1129`）：动态 shape 的通用路径；
- 入口 `dense_lightning_indexer_grad_kl_loss`（`:6347`）按模板选择上述三个类。

实现要点：

- 纯 Vector：`Mul/Add/Sub/Exp/Log/ReduceSum/ReduceMax`，没有 `Matmul/Mmad/Fixpipe`；
- 所有中间结果放在一个 UB arena（`kArenaBytes`），用固定偏移切分；
- 主注意力用 `softmaxMax/Sum` 重建，不重算 softmax；
- indexer 用 `softmaxMaxIndex/SumIndex` 重建；
- KL、dW、dS、dq̃、dK̃ 在同一个 task 内串行完成；
- `headSplitDomain` 把 N1/Nidx1 切到 4/8 个 block，每个 block 一个 head part；
- `dKeyIndex` 需要跨 block scatter，用 workspace + 同步。

## 5. 为什么高效 / 为什么只有 28.83

### 5.1 有效的部分

- **复用前向统计量**：不重算主注意力和 indexer 的 softmax，只做 `exp(S-max)/sum`；
- **融合 KL + 全部梯度**：一次读 q/k/qi/ki/w，算 loss、dQI、dKI、dW，减少 GM 往返；
- **UB arena**：中间结果不落 GM，减少 MTE 带宽；
- **head split**：在 `headPublicA/B/C` 等精确 shape 上把 head 分到 4/8 核，
  提高小 B/S1 的并行度；
- **精确路径跳过 tiling 读取和 runtime dispatch**。

### 5.2 为什么小 case 差 7–12 倍

前 3 个 case 的 best 只有 0.62–0.64 ms，说明冠军实现几乎只付 launch + 极少量计算；
本提交要 4.4–7.2 ms，说明：

- 可能没有命中 `headPublicA/B/C` 精确路径，落到了 `KernelLeanPublicDynamic` 通用路径；
- 通用路径有大量 `for j in NJ`、`for h in dHalf` 的标量循环和 `ReduceSum` 调用；
- 小 shape 下每个 `ReduceSum/Exp/Log` 的固定延迟被放大；
- 纯 Vector 没有 Cube 的并行度，D=128 的 `q@K^T`、`dS@K` 仍是逐元素乘加/归约；
- 只按 (b,s1) 或 head part 分核，B=1/S1 小时核利用率低。

### 5.3 为什么后 4 个 case 只差 1.3–2.8 倍

后 4 个 case 计算量更大，Vector 的吞吐占比上升，固定开销被摊薄；
但纯 Vector 的 D=128 矩阵乘仍不如 Cube，所以还是 1.3–2.8 倍。

### 5.4 与冠军的差距

| 项 | 本提交 | 第一名 `bubu` |
|---|---:|---:|
| score | 28.83 | 49.88 |
| 可见总时间 | 71.62 ms | 44.76 ms |
| 平均 t/T | 5.237 | 1.0 |

冠军很可能用 Cube 做 `q@K^T`、`dS@K`、`dS^T@q` 等 GEMM，Vector 只做
softmax/KL/归约；本实现全部用 Vector，导致小/中 shape 都吃亏。

### 5.5 语义覆盖缺口（重要）

op def（`op_host/...cpp:192-209`）只声明了 5 个输入
（query/key/query_index/key_index/weights）和 `scale_value` 一个属性；
**没有声明** queryRope/keyRope、actual_seq_lengths_query/key、sparseMode、layout。
host tiling 里还有四行硬编码：

```text
td->dRope = 0;
td->hasRope = 0;
td->sparseMode = 0;
td->useSeqLens = 0;
```

而官方题面包含：

- `queryRope`/`keyRope`（Dr=64）与 RoPE 反向；
- `sparseMode=3`（rightDownCausal 掩码）；
- `actual_seq_lengths_query/key` 变长序列。

`p7ExactPublic`/`headPublicA/B/C` 也显式要求 `hasRope==0 && sparseMode==0 && useSeqLens==0`。
这说明代码是按**实际评测算例**（无 RoPE、无 mask、无变长）特化的，而不是按题面全集实现。
平台 7/7 Pass 只能证明这 7 个 case 没触发这些路径；如果隐藏/后续 case 触发，语义上会不完整。

这不是“数值精度差一点”，而是**功能覆盖缺口**。入库时必须写成：

```text
已覆盖：无 RoPE、sparseMode=0、非变长的 7 个平台 case
未覆盖/未验证：queryRope/keyRope、sparseMode=3、actual_seq_lengths
```

## 6. 可复用与风险

**可复用：**

- 用前向 `softmaxMax/Sum` 重建 softmax，避免重算；
- KL + 全部梯度单 kernel 融合；
- UB arena 固定偏移，减少 GM 中间结果；
- head split 提高小 B/S1 的并行度；
- 精确小 shape 路径跳过 tiling 读取。

**风险：**

- `headPublicA/B/C`/`leanPublic` 是精确 shape 特化；隐藏/其它 shape 走通用路径，
  性能差距很大；
- 纯 Vector 不适合 D=128 的 GEMM 主导 shape；新 shape 应优先 Cube + Vector epilogue；
- **host 硬编码 hasRope=0/sparseMode=0/useSeqLens=0，未实现题面的 RoPE/mask/变长**；
- 6,408 行、大量模板和固定偏移，维护成本高；
- 本次未在本机 910B3 重跑。

## 7. 入库标注

```text
source_match: our-submission (CANNJudge SID 6aba149b...)
platform_evidence: rank 15/94, score 28.83, 7/7 visible
local_status: not-run
specialization: headPublicA/B/C + leanPublic 精确 shape
semantic_gap: hasRope=0 / sparseMode=0 / useSeqLens=0（题面含 RoPE、mask、变长）
generalization: 通用路径慢；D=128 GEMM 未用 Cube
risk: 精确 shape；纯 Vector 吞吐；维护成本；语义覆盖缺口
```
