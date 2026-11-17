# vLLM-HUST/vllm-hust-opset：Linear + SwiGLU Graph 融合深读

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/vLLM-HUST/vllm-hust-opset` |
| 作者/维护者 | Shuhao Zhang (Tony) 等 |
| 场景 | vLLM Ascend 后端的可插拔算子优化集合（生产型，非比赛） |
| 访问日期 | 2026-11-16 |
| 默认分支 | `main` |
| 固定 commit | `58acc93cfb39838b1ef66c42fb4279db6e7040ba` |
| 主要 DSL | Triton + vLLM FX/Inductor pass |
| 算子族 | 线性层+SwiGLU 融合、持久 MatMul 分配优化 |
| 许可证 | MIT |
| 本地缓存 | `repository-research-cache/vLLM-HUST_vllm-hust-opset` |
| 与既有归档关系 | 新来源；算子族与 GroupedMatmulGelu 不同 |
| 本报告范围 | 深读 `linear-swiglu-graph` |

### 1.1 证据链与环境

该仓库把实现和证据分开保存：实现/方法在本仓库，原始测量报告在另一个仓库的固定快照中。`EVIDENCE_SOURCES.json` 记录：

```text
benchmark repo:  vLLM-HUST/vllm-hust-benchmark
benchmark commit: 819b1975f346dca514f58880ca62518850100b2d
source repo:      vLLM-HUST/vllm-hust-opset
source commit:    33d78133905ef432624a786aab24c3e1868c6e1d
snapshot:         archive/mod-git-20261022/vllm-hust-opset/snapshot
```

**注意两个不一致**：

1. 当前仓库 HEAD 是 `58acc93...`，而证据快照的 source commit 是 `33d7813...`；本轮没有证明当前 HEAD 与证据快照逐行一致。
2. 原始报告写 Ascend 910B2、CANN 9.0.0；`compatibility/verified-line.json` 写 CANN 9.1.0-beta.3、torch_npu 2.10.0.post4。两组环境可能对应不同阶段，不能混成同一个实验。

因此本文把“代码机制”和“原始测量”分开标注：

- 代码：当前 HEAD `58acc93`；
- 数字：外部报告 commit `819b197...` 中的原报告，环境以原报告表格为准；
- 本轮未重跑。

## 2. 语义

### 2.1 基线

Qwen2.5-7B 的 MLP 投影：

```text
gate_up = x @ W_gate_up^T          # [M, 2N]
gate, up = gate_up.chunk(2, dim=-1)
y = silu(gate) * up                # [M, N]
```

基线先做一次 `unquantized_gemm` 得到 `[M,2N]`，再启动 `npu_swiglu` 读 `[M,2N]`、写 `[M,N]`。中间张量 `[M,2N]` 要落 GM，并被第二个 kernel 重新读。

### 2.2 融合

候选把 gate/up 两个投影和 SwiGLU 放进一个 persistent Triton kernel：

```text
for k in 0..K step BLOCK_K:
    a = x[:, k:k+BLOCK_K]
    gate_acc += a @ W_gate[k, :]^T
    up_acc   += a @ W_up[k, :]^T
y = gate_acc * sigmoid(gate_acc) * up_acc
store y                            # 只写 [M,N]
```

输入权重是 `gate_up_weight[2N,K]`，gate 在前 N 行、up 在后 N 行；kernel 通过 `gate_ptrs + N*stride_bn` 取 up 权重（`linear_swiglu_kernel.py:53-57`）。

## 3. 固定源码位置

| 内容 | 路径/行 |
|---|---|
| Triton kernel：双 accumulator、共享 A、两次 dot | `src/vllm_hust_operator_optimizations/optimizations/linear_swiglu_kernel.py:35-66` |
| Python 入口、M<=128 限制、tile 选择 | 同上 `:69-107` |
| Graph 自定义 op 注册、M<=128 分流、fake impl | `.../linear_swiglu_graph.py:34-81` |
| FX pattern：`npu_swiglu(unquantized_gemm(x,w,None))` → `unquantized_linear_swiglu` | `.../linear_swiglu_fusion_pass.py:12-31` |
| pass 注册/适用 dtype | 同上 `:34-48` |
| 机制与边界说明 | `docs/optimizations/linear-swiglu-graph.md:3-27` |
| 原始报告（外部固定 commit） | `https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/vllm-hust-opset/snapshot/docs/optimizations/linear-swiglu-graph.md` |

固定 commit 的文件 SHA256：

```text
linear_swiglu_kernel.py       2d49147c0efc91e438e926efe75c94c335206348803fb9eeda2323c474fc5932
linear_swiglu_graph.py        4b82b6da4d1df79fc0ccbaffb67a3cd0c73d4b538bebc346d75faf340940be72
linear_swiglu_fusion_pass.py  fb97ccbf2dee6d4efe9750f634353aadd4ce98e059a1e14002cfcd92549342be
docs/optimizations/linear-swiglu-graph.md  439668ab1aa94fc6ee3789aec478c4f61b61055da76926ae296c4b17ac3838c7
```

## 4. 机制

### 4.1 两个 FP32 accumulator，A 只加载一次

`linear_swiglu_kernel.py:35-59`：

- `gate_acc`、`up_acc` 都是 `[BLOCK_M, BLOCK_N]` 的 FP32；
- 每个 K block 只加载一次 `a_vals`；
- `gate_vals` 和 `up_vals` 分别从 `gate_up_weight` 的前 N 行/后 N 行加载；
- 两次 `tl.dot` 累加到两个 accumulator。

这和“先拼一个 `[M,2N]` 的 GEMM 再切”在数学上等价，但**没有中间的 `[M,2N]` 写回**。GEMM 的 MAC 数不变（仍是 `2*M*N*K`），省的是：

- `[M,2N]` 的 GM 写 + 读；
- 第二个 kernel 的启动；
- `npu_swiglu` 的输入/输出 tensor 生命周期。

### 4.2 只在 M<=128 融合

`linear_swiglu_graph.py:34-50`：

```python
flat_x = x.reshape(-1, input_shape[-1])
if flat_x.shape[0] <= 128:
    output = linear_swiglu(flat_x, gate_up_weight)
else:
    output = torch_npu.npu_swiglu(
        torch.nn.functional.linear(flat_x, gate_up_weight)
    )
```

这是**fail-closed**：大 M 动态 prefill 直接回退到 native GEMM + `npu_swiglu`，不在融合路径上冒险。原因是两个 FP32 accumulator 同时驻留会放大 CBUF 压力；报告明确说 M=256 的 fused tile 超过 CBUF，128 列输出 tile 可以编译，但更大的多 tile shape 出现错误值（`docs/optimizations/linear-swiglu-graph.md:19-27`）。

### 4.3 FX pattern 与自定义 op

`linear_swiglu_fusion_pass.py:12-31` 的 pattern 是：

```text
npu_swiglu(unquantized_gemm(x, gate_up_weight, None))
```

replacement 是：

```text
unquantized_linear_swiglu(x, gate_up_weight)
```

`linear_swiglu_graph.py:34-81` 用 `direct_register_custom_op` 注册自定义 op，并提供 fake impl 供图捕获/编译。`_install_graph_pass()` 还检查上游 `GraphFusionPassManager.configure` 的源码结构，若找不到 `AddRMSNormQuantFusionPass`/`MulsAddFusionPass` 就拒绝安装；若上游已经定义了 `unquantized_linear_swiglu` 也拒绝。这是“未知源码结构 fail closed”的工程做法，避免版本漂移后静默改错图。

## 5. 工作量与代价

### 5.1 工作量

| 项 | 基线 | 融合 |
|---|---|---|
| GEMM MAC | `2*M*N*K` | `2*M*N*K`（不变） |
| 中间 `[M,2N]` 写 | 有 | 无 |
| 中间 `[M,2N]` 读 | 有（swiglu） | 无 |
| kernel 数 | 2（GEMM + swiglu） | 1 |
| 输出写 | `[M,N]` | `[M,N]` |
| accumulator | 1 个 `[M,2N]`（或分块） | 2 个 `[M,N]` FP32 |

对 decode/small batch，`M` 很小，GEMM 的权重读取占主导，但中间 `[M,2N]` 和第二个 kernel 的固定开销仍然可见；报告在 M=1/32/128 上看到 15.93%/17.92%/16.79% 的算子时延下降，说明这个固定开销确实可测。

### 5.2 代价

- 两个 FP32 accumulator 让 CBUF 需求约翻倍；`BLOCK_N=128`、`BLOCK_M=min(next_pow2(M),128)` 时，M=128 的两个 accumulator 是 `2*128*128*4 = 128 KiB`，再加上 A/权重/流水，接近/超过可用 CBUF；
- M 大时不能靠“更多 tile”无脑扩，因为多 tile 的中间值/边界可能出错；
- 首次 JIT 编译成本在冷启动时可能为负；
- 只覆盖 unquantized Graph/Triton 路径，不覆盖 ACLNN/量化；
- 依赖上游模块名和函数名，版本升级可能拒绝安装（这是有意的 fail closed，但也意味着维护成本）。

## 6. 证据

### 6.1 算子级

Qwen2.5-7B MLP projection，`K=3584`、`N=18944`、BF16，10 warmup + 50 synchronized samples per arm：

| M | Baseline mean ± SD | Fused mean ± SD | Latency reduction |
|---:|---:|---:|---:|
| 1 | 0.5001 ± 0.0115 ms | 0.4204 ± 0.0083 ms | 15.93% |
| 32 | 0.4989 ± 0.0126 ms | 0.4095 ± 0.0100 ms | 17.92% |
| 128 | 0.5416 ± 0.0111 ms | 0.4507 ± 0.0068 ms | 16.79% |

报告还记录：完整的 256 列 fused tile 超过 CBUF；128 列输出 tile 能编译，说明**同时驻留两个 FP32 accumulator**是主要约束。

### 6.2 端到端

Qwen2.5-7B-Instruct，BF16，Ascend 910B2，CANN 9.0.0，Graph mode，`VLLM_BATCH_INVARIANT=1`，ShareGPT 64 requests，concurrency 4，output length 128；8-request workload warmup 后 3 轮匹配：

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| TPOT | 19.3481 ± 0.5271 ms | 18.6088 ± 0.1224 ms | 3.82% lower |
| Output token throughput | 187.5615 ± 7.4216 token/s | 197.3891 ± 1.4549 token/s | 5.24% higher |
| Total token throughput | 562.4271 ± 22.2546 token/s | 591.4048 ± 4.3592 token/s | 5.15% higher |
| Request throughput | 1.4979 ± 0.0593 req/s | 1.5744 ± 0.0116 req/s | 5.11% higher |
| TTFT | 219.4263 ± 38.5764 ms | 180.4964 ± 3.2113 ms | 17.74% lower |

正确性：10/10 graph cases + FP16/BF16 tail coverage；6 个正式端到端 run 全部 64/64 请求完成，匹配 prompt 的文本一致。

### 6.3 证据级别

| 项 | 级别 |
|---|---|
| 源码/机制 | `S+L` |
| 算子/端到端数字 | `O+A`（外部固定 commit 报告） |
| 本机复现 | `N` |
| 环境一致性 | 报告 9.0.0 vs 兼容清单 9.1.0-beta.3，未完全对齐 |
| 代码与证据一致性 | 当前 HEAD `58acc93` vs 证据 source `33d7813`，未逐行比对 |

## 7. 适用条件

1. **M<=128**：decode/small batch；M 更大时回退 native。
2. **unquantized、BF16/FP16**：量化/ACLNN 路径不覆盖。
3. **Graph/Triton 路径**：依赖 `unquantized_gemm` 和 `npu_swiglu` 的图结构；未知结构拒绝安装。
4. **CBUF 能容纳两个 FP32 accumulator + 权重/输入流水**：M=128、N tile=128 已接近边界；再大需要更小 tile 或回退。
5. **上游模块结构匹配**：`vllm_ascend.ops.linear` 有 `unquantized_gemm`，`GraphFusionPassManager.configure` 含预期 pass；否则 fail closed。
6. **不是 310P**：`_install_graph_pass()` 里显式 `is_310p()` 检查。

## 8. 负例、退化与未知风险

1. **M>128 融合不可用**：256 列 fused tile 超过 CBUF；大 M 多 tile 出现不正确值；必须回退。
2. **冷启动可能为负**：无 workload warmup 的实验暴露了一次性 Triton JIT 成本，candidate TPOT 为负；报告把 cold start 与 steady-state 分开。
3. **端到端收益远小于算子收益**：算子 15.93–17.92%，端到端 TPOT 只有 3.82%；如果只看算子数字，会高估对总时延的贡献。
4. **TTFT 的方差很大**：baseline 219.4263 ± 38.5764 ms，candidate 180.4964 ± 3.2113 ms；baseline 的方差说明这组端到端数据对系统状态敏感，不能把 17.74% 当成稳定常数。
5. **环境/版本不一致**：CANN 9.0.0 vs 9.1.0-beta.3；source commit 33d7813 vs 当前 58acc93。复现前必须固定其中一条线。
6. **只覆盖 Qwen2.5-7B MLP 的一种形状**：K=3584、N=18944；其它模型/TP 切分/量化路径未覆盖。
7. **不是“融合一定快”**：M 大、CBUF 不够、冷启动、目标 op 占比低时，融合可能不划算。

## 9. 复用建议

### 9.1 优先复用

- **双 accumulator + 共享 A 的 GEMM 融合**：适合 gate/up、QKV、双投影等“同一输入、两个权重、逐元素合并”的结构。
- **只写最终 `[M,N]`**：消除 `[M,2N]` 中间张量，是 decode 场景最直接的工作量/带宽削减。
- **M<=128 的 fail-closed 分流**：小 batch 走融合，大 batch 走 native，不给“为了融合而融合”的机会。
- **FX pattern + 自定义 op + fake impl**：把融合限定在可识别的图结构上，未知结构拒绝安装。

### 9.2 需要验证后再用

- 目标模型的 `K/N`、TP 切分、dtype、CANN 版本是否仍满足 CBUF；
- 端到端收益是否值得维护成本：算子收益 16% 但 TPOT 只有 3.8%，要先确认目标 op 在总时延中的占比；
- 冷启动是否可接受：如果服务需要快速 cold start，JIT 成本可能抵消稳态收益；
- 上游 vLLM/vLLM-Ascend 版本是否仍匹配 pattern。

### 9.3 暂不使用

- 把 M=128 的融合直接扩到 M=256/512；
- 在量化/ACLNN 路径复用这个 kernel；
- 把算子级 16–18% 当作端到端加速比；
- 忽略报告与兼容清单的环境差异。
