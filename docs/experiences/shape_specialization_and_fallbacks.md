# 形状专用化与回退：把“固定 Shape”变成可证明的路由，而不是代码堆

## 1. 问题

固定 Shape Case 的优化，常见做法是按 shape/dtype/属性选不同 kernel/tiling。它能把资源、并行度、布局都调到最合适，但也会带来：

- Host→tiling key→kernel 入口→设备分支不一致；
- 模板/宏展开导致代码膨胀、编译时间上升；
- 某些 PATH 实际不可达，却被误当成“已覆盖”；
- 换 shape 后没有 fallback，直接错或越界；
- 专用化条件与题目规则/容差脱节。

本轮看到的专用化分三类：

| 类型 | 例子 | 风险 |
|---|---|---|
| Host 条件分路 + 编译期 key | Hansen Fire/Spline/Tree、Mamba-2 | key 漂移、不可达分支、版本耦合 |
| 运行时 `if` 分流 | vLLM-HUST `M<=128` | 分支两侧都要测；大 M 回退 |
| 固定 shape 硬编码 | KernelSwift 三题 | 换 shape 就错；必须配校验/回退 |

## 2. Host 条件分路：先算 shape 族，再选 kernel

### 2.1 Fire：按总行数选 tile

源码：`Hansenwuwuwu/ascend-c-competition-archive`，`FireCandidateDecodeFusion/rank2_submission_1176342/code/op_host/fire_host_plan.h:14-60`。

```text
totalRows = B*N
if totalRows <= 8192:          tile=32/64/128, blocks=min(cores,tiles)
elif totalRows >= cores*1664*3: tile=1640/1664, blocks=B*ceil(N/tile)
elif batchOne:                  tile=512/528/640/896/1024
else:                           workers=..., tile=512/528/640/1024
```

这不是“按 N 分档”，而是按总行数、batch、可用核数、dtype 对齐一起算。目的是让每个 block 的工作量、尾部、UB 占用都落在可控区间。

**复用要点**：

- 分路条件只依赖 shape/dtype/平台，不读张量值；
- 每个分支的 tile/block 都有明确上界；
- 小 shape 走 compact single tile，避免通用路径的固定开销。

**风险**：

- `1664`、`1640`、`8192` 等常数是特定 UB/核数/元素大小推出来的，换 SoC 要重算；
- `totalRows` 的乘法和 `blocks` 对齐要防溢出；
- 需要验证每个分支在实际评测 shape 上可达。

### 2.2 Spline：按 UB 预算和稀疏族选路径

源码：`SplineConv/rank1_submission_1174398/code/op_host/spline_conv.cpp:86-141`。

Host 先算：

```text
baseBytes, cacheWeights, cacheGraph, vectorDot, fusedDot, smallBytes
streaming = 转置/ci=0/baseBytes 超 UB
```

再按 `ci/co/e/n/dim/degree/kernel_size/UB` 选：

- `single` 1–6：极小图/单边/尾块；
- `sparse` 1–8：`e<n`、`e<=64`、`nodesPerBlock<=ci`、`nodesPerBlock*e<=512`、`ci=32/128`、`co=16`、`dim=2`、`degree=1`、`kernel=2`；
- 其它：通用 `streaming`/逐边路径。

**复用要点**：

- 专用化条件同时约束 shape、属性、UB 预算；
- 有明确的通用 fallback；
- `sparse` 只在“高入度、低边数、权重可驻留”时启用。

**风险**：

- 条件很多，容易出现“某个 helper 存在但 Host 不可达”；
- 聚合专用路径主要是 `ci=32/128, co=16`，compact 另有 `ci=8` single 特化；换通道数就回退；
- UB 预算依赖 `elementBytes`、`ciPad/coPad`，换 dtype/对齐要重算。

### 2.3 Tree：按退化维度换并行轴

源码：`TreeLineMinDistance/rank1_submission_1175341/code/op_host/tree_line_min_distance.cpp:446-484`。

```text
if points<=3 && M<=6 or M==1 && points<=32: tiny
elif M==1: 沿点
elif N==1: 沿线段
elif M<=32 && points<=64: short
elif M>=256 && N>=64: 大任务/固定 shape
else: 通用
```

这是“按退化维度改变并行轴”的典型：`M=1` 时删掉 argmin，`N=1` 时删掉点并行，小 shape 直接标量/固定长度。

**复用要点**：

- 先判断哪个维度退化，再决定并行轴；
- 小 shape 不要启动通用 kernel；
- 大 shape 的固定 PATH 要按实际评测 shape 命中，而不是只看模板存在。

**风险**：

- 大量固定 PATH/UB 绝对偏移，代码膨胀；
- `M=1024` 的 half Top-2 是高风险近似（见数值经验）；
- 某些 PATH 可能不可达或只对特定 dtype 编译。

## 3. 编译期 key：把 ABI/布局/tiling 固定下来

### 3.1 Mamba-2 的 tiling key

源码：`mamba2_ssd_chunk_mix.cpp:1293-1339`，`design.md:42-48`。

| Key | T | N | State layout |
|---:|---:|---:|---|
| 2 | 64 | 64 | `[N,P]` |
| 3 | 64 | 128 | legacy `[P,N]` |
| 7 | 128 | 128 | `[N,P]` |
| 9/10/11 | arch35（950） | - | 不同 MIX 配置 |

Kernel 入口用 `TILING_KEY_IS(...)` 分派到不同模板/类；`KERNEL_TASK_TYPE` 还区分 `MIX_AIC_1_1` / `MIX_AIC_1_2`。这意味着：

- key 不只是“tile 参数”，还绑定 state layout、任务类型、workspace 布局；
- Host 必须把 key 与 tiling data 一起写对；
- 950 与 910B 的 key 不同，不能互推。

### 3.2 Fire 的 64-bit 打包 tiling

源码：`fire_candidate_decode_fusion.cpp:104-127`。

Host 把 `batch/n/blocks/tile/units/extra/denseOne` 压进 64-bit word；kernel 解包。好处是 tiling payload 小、缓存友好；代价是位域/打包/解包必须严格一致，任何一个字段溢出都会静默错。

**复用要点**：

- 打包字段要有范围检查；
- Host 和 kernel 共用同一个结构/常量；
- 不要为了省几个字节，把未来要加的字段硬塞进位域。

## 4. 运行时 fail-closed：大 shape 回退

### 4.1 vLLM-HUST Linear+SwiGLU

源码：`vllm_hust_operator_optimizations/optimizations/linear_swiglu_graph.py:34-50`。

```python
if flat_x.shape[0] <= 128:
    output = linear_swiglu(flat_x, gate_up_weight)
else:
    output = npu_swiglu(F.linear(flat_x, gate_up_weight))
```

这是最清晰的 fail-closed：

- 小 M 走融合；
- 大 M 直接回退 native；
- 不在大 M 上冒险（报告说 256 列 tile 超 CBUF、多 tile 出现错误值）。

**复用要点**：

- 回退条件要基于“已验证的安全域”，不是“理论上应该行”；
- 回退路径必须正确、可测；
- 不要为了覆盖率把未验证的 shape 放进融合。

### 4.2 vLLM-HUST 的图结构 fail-closed

`linear_swiglu_graph.py:26-29,92-96`：

- 上游没有 `unquantized_gemm` → 拒绝；
- 上游已经定义 `unquantized_linear_swiglu` → 拒绝；
- `GraphFusionPassManager.configure` 结构不符 → 拒绝；
- 已配置 `LinearSwiGLUFusionPass` → 拒绝。

这比“尽量 patch”更安全：版本不匹配时直接不启用，而不是静默改错图。

## 5. 固定 shape 硬编码：必须配校验

### 5.1 KernelSwift 三题

- GroupedTopk：固定 256 专家、8 组、Top-4 组、Top-8 专家；`forward` 里有 `assert`；
- FusedMoE：固定 8 专家、Top-2、`83/128/64`；`forward` 里硬编码 `83`、`128`、`64`；
- Sinkhorn：固定 `hc_mult=4`、`mix_hc=24`、20 轮、`eps=1e-6`。

这些硬编码在比赛固定 shape 下是合理的，但复用时要：

- 在 Host/Python 入口校验 shape/dtype/属性；
- 不满足时回退通用实现或报错；
- 不要把 `83`、`128`、`64` 带进其它模型。

### 5.2 52Xayah Softmax 的负例

`src/softmax.py:8-41` 固定 `BLOCK_SIZE=1024`、无 mask。它只对 `N=1024` 的规则 shape 正确；换 `N` 就错。这是“没有校验/回退的硬编码”的反面教材。

## 6. 代码膨胀与不可达路径

### 6.1 现象

Hansen 的三份实现都有大量模板/PATH/宏：

- Fire rank1 kernel 8774 行；rank2 也有 1100 行静态表；
- Spline rank1 有 `single` 1–6、`sparse` 1–8 等多个 key；
- Tree rank1 有 20+ PATH、固定 UB 偏移；
- 三份都可能出现“某个 helper/template 存在，但 Host 条件不可达”。

### 6.2 风险

- 读者把不可达路径当成实际优化；
- 编译时间/二进制体积膨胀；
- 某个 key 只在特定 dtype/shape 编译，换 shape 就缺实例；
- Host 条件和 kernel 分支漂移，出现“选了一个没编译的 key”。

### 6.3 检查

- 从 Host 条件 → `ASCENDC_TPL_SEL_PARAM`/`SetTilingKey` → kernel `if constexpr`/`TILING_KEY_IS` → 设备分支，画一张完整路由表；
- 对每个实际评测 shape，确认命中的 key/模板；
- 用编译日志/符号表确认该 key 有实例；
- 对不可达路径，标注“未使用”，不要计入优化证据。

## 7. 负例汇编

1. **专用化条件太多**：Spline/Fire/Tree 的 Host 条件与 kernel 分支必须闭环；只看模板存在不够。
2. **固定常数跨 SoC**：Fire 的 1664/1640、Tree 的 UB 偏移、Mamba-2 的 182.8 KiB，都不能直接搬。
3. **不可达路径**：审计报告已指出 Spline rank1 有不可达分支；本轮未逐条复核，但这是通用风险。
4. **大 M 融合**：vLLM-HUST 明确回退；不要为了“融合率”强行启用。
5. **无 mask 硬编码**：52Xayah Softmax 只对 N=1024 正确。
6. **路径未覆盖**：Mamba-2 950PR 当前 HEAD 的 grouped fast-path 回归说明“同一 key/路径在另一个 SoC 上不一定正确”。

## 8. 可证伪的验证设计

### 8.1 路由覆盖

- 列出评测协议里的所有 shape/dtype/属性组合；
- 对每个组合，记录 Host 条件、key、编译实例、设备分支；
- 如果某个组合没有命中任何专用路径，确认它走通用 fallback 且正确；
- 如果某个专用路径没有任何评测 shape 命中，标记为“未使用”，不当作收益来源。

### 8.2 回退正确性

- 对每条 fail-closed 分支，构造触发输入，确认回退路径正确；
- 对“刚好在边界”的 shape（M=128/129、N=32768/32769、E=8/9）做测试；
- 确认回退不会静默产生不同 dtype/布局。

### 8.3 专用化收益

- 在同一硬件上，对同一 shape 比较“专用路径”和“通用路径”；
- 如果专用路径没有收益，删除它，减少代码/编译负担；
- 如果专用路径有收益但只覆盖一个 shape，评估维护成本。

### 8.4 版本迁移

- 换 CANN/编译器/SoC 后，重新编译所有 key；
- 对每个 key 跑正确性/边界；
- 不要假设“同一个宏/内建在新版本语义不变”。

## 9. 证据边界

- 本文的路由/回退结论主要来自固定 commit 的源码；本机只运行了 KernelSwift 的 GroupedTopk/Sinkhorn（FusedMoE 编译失败），Hansen/Mamba/vLLM-HUST 的专用路径未在本机编译/运行。
- “某个路径不可达”需要逐条静态核对；本轮只对 Hansen 三题的部分路径做了阅读，未做完整可达性证明。
- 固定 shape 的官方成绩不能证明其它 shape 也正确/快；回退路径必须单独验证。
