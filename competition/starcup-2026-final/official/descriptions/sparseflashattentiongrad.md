# 【决赛算子题】SparseFlashAttentionGrad 算子

---

## 1. 算子说明

SparseFlashAttentionGrad 是 SparseFlashAttention 的**反向梯度算子**（训练场景）：根据稀疏索引对 key/value 重排（gather）后，按 Flash Attention 反向公式，由输出梯度 dOut 计算 query/key/value 的梯度 dQ/dK/dV，供反向传播使用。

### 背景术语（题目内自解释）

- **反向传播与梯度**：神经网络训练时，依据损失对输出的梯度 `dOut` 反推对各输入的梯度 `dQ/dK/dV`。
- **Flash Attention 反向**：前向为 `O = softmax(Q@K^T/√d)@V`；反向分 3 阶段由 `dOut` 计算 `dP/dV → dS → dQ/dK`，避免显式存储大注意力矩阵。
- **FlashSoftmaxGrad**：softmax 反向的核心项，等于 `sum(dO⊙O, 沿行, keepdim)`，用于在 dS 阶段恢复 softmax 的非线性梯度。
- **稀疏 gather / scatter**：根据 `sparseIndices` 从 KV 缓存中 gather 出选中的 key/value 参与计算；反向时将 dK/dV scatter 回原位置，未选中位置梯度为 0。
- **KV merge**：当 `value` 传入空指针时，算子按 `value = key` 处理，不单独输出 dValue，而是将 `dK + dV` 合并到 dKey 输出，减少重复搬运。
- **MQA**：Multi-Query Attention，key/value 头数 `N2=1`，所有 query 头共享一组 KV。
- **RoPE**：旋转位置编码，反向时对应输出 `dQueryRope/dKeyRope`（可选）。
- **维度符号**：`B`=Batch Size，`S1`=query 序列长度，`S2`=key/value 序列长度，`N1`=query 头数，`N2`=key/value 头数（恒为 1），`D`=每个头维度（=512），`Dr`=RoPE 维度（=64），`K`=sparse_size 每个 query token 选取的 key 位置数，`G=N1/N2`。
- **数据排布**：本算子仅支持 `BSND` 排布，即 `(B, S, N, D)`。
- **变长序列**：`actual_seq_lengths` 给出每个 batch 有效 token 数（超出为 padding，不参与计算），可传 None 表示与对应序列长度相同。

### 计算公式

先根据 `sparseIndices` 稀疏 gather：

```
selectedKey   = Gather(key,   sparseIndices[i])
selectedValue = Gather(value, sparseIndices[i])
```

KV merge 场景（value 为空）：`value` 按 `key` 处理，`selectedValue = selectedKey`，且 `dKeyOut = dK + dV`（不单独输出 dValue）。

阶段1（计算 dP、dV）：

```
dP[t,:] = dO[t,:] @ V^T
dV[u]   = P^T[t,:] @ dO[t,:]
```

阶段2（计算 dS）：

```
dS[t,:] = P[t,:] ⊙ ( dP[t,:] - FlashSoftmaxGrad(dO, O) )
FlashSoftmaxGrad(dO, O) = sum(dO ⊙ O, 沿最后一维, keepdim)
```

其中 `P` 为 softmax 权重，由前向的 `softmaxMax/softmaxSum` 重建：`P = exp(S - softmaxMax) / softmaxSum`。

阶段3（计算 dQ、dK）：

```
dQ[t,:] = dS[t,:] @ K / √d_k
dK[u]   = dS[t,:]^T @ Q / √d_k
```

其中 `O=out`（前向输出），`dO=dOut`，`d_k=D`，缩放系数 `scaleValue=1/√d_k`。

---

## 2. 输入输出说明

### 输入规格

| 输入                     | 必选/可选 | 类型 | 形状            | 数据类型         | 含义                                                         |
| ------------------------ | --------- | ---- | --------------- | ---------------- | ------------------------------------------------------------ |
| query                    | 必选      | 张量 | (B, S1, N1, D)  | bfloat16/float16 | 输入 Q。D=512                                                |
| key                      | 必选      | 张量 | (B, S2, N2, D)  | bfloat16/float16 | 输入 K。N2=1                                                 |
| value                    | 可选      | 张量 | (B, S2, N2, D)  | bfloat16/float16 | 输入 V。普通场景须与 key 内容完全一致；传空启用 KV merge（value=key，不输出 dValue） |
| sparseIndices            | 必选      | 张量 | (B, S1, N2, K)  | int32            | 稀疏选取 KV 的索引。K∈{1024,2048,3072,4096,5120,6144,7168,8192} |
| dOut                     | 必选      | 张量 | (B, S1, N1, D)  | bfloat16/float16 | 前向输出 O 的梯度 dO                                         |
| out                      | 必选      | 张量 | (B, S1, N1, D)  | bfloat16/float16 | 前向注意力输出 O                                             |
| softmaxMax               | 必选      | 张量 | (B, N2, S1, G)  | float32          | 前向 softmax 行最大值。G=N1/N2                               |
| softmaxSum               | 必选      | 张量 | (B, N2, S1, G)  | float32          | 前向 softmax 行求和。G=N1/N2                                 |
| actual_seq_lengths_query | 可选      | 张量 | (B,)            | int32            | 每个 batch query 的有效 token 数。传 None 表示与 S1 相同     |
| actual_seq_lengths_kv    | 可选      | 张量 | (B,)            | int32            | 每个 batch key/value 的有效 token 数。传 None 表示与 S2 相同 |
| queryRope                | 可选      | 张量 | (B, S1, N1, Dr) | bfloat16/float16 | query 的 RoPE 信息，Dr=64                                    |
| keyRope                  | 可选      | 张量 | (B, S2, N2, Dr) | bfloat16/float16 | key 的 RoPE 信息，Dr=64                                      |
| scaleValue               | 必选      | 属性 | 标量            | float32          | 注意力缩放系数，对应公式中的 1/√d_k                          |
| sparseBlockSize          | 必选      | 属性 | 标量            | int64            | 稀疏选择的块大小。Atlas A2/A3 系列支持 1/8/16/32/64；Ascend 950PR/950DT 支持 1 |
| sparseMode               | 必选      | 属性 | 标量            | int64            | 掩码模式：0=不屏蔽；3=rightDownCausal（query 右端对齐 key 右端的下三角掩码）。常用取值 0 |
| layout                   | 必选      | 属性 | 标量            | string           | 数据排布格式，默认 "BSND"，本算子仅支持 BSND                 |
| deterministic            | 必选      | 属性 | 标量            | bool             | 确定性计算，与整网确定性参数一致；默认 False（非确定性）     |

> 说明：属性 `pre_tokens`、`next_tokens` 用于稀疏计算的滑窗位置（仅 `sparseMode=4` 时生效，本算子 `sparseMode` 仅支持 0/3 故不生效），本算子仅支持默认最大值 2147483647，参赛者无需修改。

### 输出规格

| 输出类型   | 必选/可选 | 形状            | 数据类型         | 含义                                            |
| ---------- | --------- | --------------- | ---------------- | ----------------------------------------------- |
| dQuery     | 必选      | (B, S1, N1, D)  | bfloat16/float16 | query 的梯度 dQ                                 |
| dKey       | 必选      | (B, S2, N2, D)  | bfloat16/float16 | key 的梯度 dK；KV merge 场景返回 dK+dV 合并梯度 |
| dValue     | 可选      | (B, S2, N2, D)  | bfloat16/float16 | value 的梯度 dV；KV merge 场景传空（合入 dKey） |
| dQueryRope | 可选      | (B, S1, N1, Dr) | bfloat16/float16 | queryRope 的梯度；当 queryRope 存在时输出       |
| dKeyRope   | 可选      | (B, S2, N2, Dr) | bfloat16/float16 | keyRope 的梯度；当 keyRope 存在时输出           |

### 形状约束

- 仅支持训练场景，支持图模式。
- HeadDim `D=512`，RoPE 维度 `Dr=64`。
- query、key、value 数据类型必须一致；当前只支持 value 与 key 完全一致的场景。
- `sparseMode` 仅支持 0（不屏蔽）或 3（rightDownCausal 下三角掩码）。
- query 头数 `N1`：1/2/4/8/16/32/64/128（离散枚举）；Ascend 950PR/950DT 额外支持 48/24/12/6/3。key/value 头数 `N2=1`（MQA）。
- `B` 取值 1~256；`S1、S2` 取值 1~128K，支持不等长。
- `K`（sparse_size）取值 1024/2048/3072/4096/5120/6144/7168/8192（离散）；不建议 `K × sparseBlockSize` 超过 100k（否则可能 OOM）。
- `sparseBlockSize`：Atlas A2/A3 系列支持 1/8/16/32/64；Ascend 950PR/950DT 支持 1。
- 仅支持 ND 数据格式（任意多维非结构化连续格式）。
- 默认非确定性实现，可通过 `aclrtCtxSetSysParamOpt` 开启确定性（`deterministic` 与整网一致）。

### 输出形状计算公式

- `dQuery`：与 query 形状一致 `(B, S1, N1, D)`。
- `dKey` / `dValue`：与 key/value 形状一致 `(B, S2, N2, D)`。
- `dQueryRope` / `dKeyRope`：分别与 queryRope/keyRope 形状一致 `(B, S1, N1, Dr)` / `(B, S2, N2, Dr)`。

---

## 3. 算子逻辑说明

### 3.1 稀疏 gather

对每个 query token，根据 `sparseIndices` 从 KV 缓存中 gather 出选中的 `selectedKey` 与 `selectedValue`。`sparseBlockSize` 决定选择粒度（token-wise 或 block-wise）。未选中位置后续梯度为 0。

### 3.2 阶段1：计算 dP 与 dV

- `dP = dO @ V^T`（输出梯度对注意力得分矩阵的梯度）
- `dV = P^T @ dO`（value 的梯度）

### 3.3 阶段2：计算 dS（softmax 反向）

- 由 `softmaxMax/softmaxSum` 重建 softmax 权重：`P = exp(S - softmaxMax) / softmaxSum`
- `FlashSoftmaxGrad(dO, O) = sum(dO ⊙ O, 沿行, keepdim)`
- `dS = P ⊙ (dP - FlashSoftmaxGrad)`

### 3.4 阶段3：计算 dQ 与 dK

- `dQ = dS @ K / √d_k`（缩放 `scaleValue=1/√d_k`）
- `dK = dS^T @ Q / √d_k`

### 3.5 scatter 与 KV merge

- 将 `dK/dV` 按稀疏索引 scatter 回原位置（累加，因同一 key 位置可能被多个 query token 选中）。
- KV merge 场景（value 传空）：`value=key`，不输出 dValue，`dKeyOut = dK + dV`。
- RoPE 部分（queryRope/keyRope 存在时）按对应反向公式计算 `dQueryRope/dKeyRope`。

---

## 4. 决赛任务要求

- **精度保障**：针对不同 `N1`、变长序列、不同 `sparseMode` 与 `sparseBlockSize`，设计算子逻辑，保证 Flash Attention 反向梯度精度正确（可用数值梯度校验）。
- **性能优化**：优化稀疏 gather/scatter 的离散访存与三阶段矩阵乘的并行度，减少 KV 重复搬运，算子性能更优。
- **切分最优**：探索 query 序列维度、query 头维度与稀疏索引维度在多 batch / 变长 / KV merge 场景下的切分方式，找到不同输入 shape 场景下的最优解。

---

## 功能示例

```python
import numpy as np

# 简易前向：计算 out / softmaxMax / softmaxSum（供反向使用）
def forward(q, k, v, idx, scale):
    Ksel = k[idx]; Vsel = v[idx]
    S = (q @ Ksel.T) * scale
    mx = S.max(-1, keepdims=True); e = np.exp(S - mx)
    P = e / e.sum(-1, keepdims=True)
    O = P @ Vsel
    return O, mx[:, 0], e.sum(-1)

def sparse_flash_attention_grad(query, key, value, sparse_indices, d_out, out,
                                softmax_max, softmax_sum, scale_value):
    # query:(B,S1,N1,D) key/value:(B,S2,N2,D) N2=1 sparse_indices:(B,S1,N2,K)
    # d_out/out:(B,S1,N1,D) softmax_max/sum:(B,N2,S1,G) G=N1/N2
    # 输出 dQuery:(B,S1,N1,D) dKey/dValue:(B,S2,N2,D)
    # 聚焦核心反向公式(3阶段)，省略 RoPE 与 MLA-absorb 融合(见 3.5 节)
    q = query.astype(np.float32); k = key.astype(np.float32); v = value.astype(np.float32)
    do = d_out.astype(np.float32); o = out.astype(np.float32)
    B, S1, N1, D = q.shape; _, S2, N2, _ = k.shape
    dQ = np.zeros_like(q); dK = np.zeros_like(k); dV = np.zeros_like(v)
    for b in range(B):
        Kb = k[b, :, 0, :]; Vb = v[b, :, 0, :]
        for s in range(S1):
            idx = sparse_indices[b, s, 0]
            valid = idx[(idx >= 0) & (idx < S2)]       # 阶段0: 稀疏 gather
            Ksel = Kb[valid]; Vsel = Vb[valid]
            Q = q[b, s]; dO = do[b, s]; O = o[b, s]
            mx = softmax_max[b, 0, s]; sm = softmax_sum[b, 0, s]
            S = (Q @ Ksel.T) * scale_value
            P = np.exp(S - mx[:, None]) / sm[:, None]   # 重建 softmax 权重
            dP = dO @ Vsel.T                            # 阶段1: dP
            dV_sel = P.T @ dO                           # 阶段1: dV
            fsg = (dO * O).sum(-1, keepdims=True)      # FlashSoftmaxGrad(dO,O)
            dS = P * (dP - fsg)                         # 阶段2: dS
            dQ[b, s] = dS @ Ksel * scale_value          # 阶段3: dQ
            dK_sel = dS.T @ Q * scale_value            # 阶段3: dK
            for ii, pos in enumerate(valid):           # scatter 回原位置
                dK[b, pos, 0] += dK_sel[ii]
                dV[b, pos, 0] += dV_sel[ii]
    return dQ.astype(np.float16), dK.astype(np.float16), dV.astype(np.float16)

np.set_printoptions(precision=4, suppress=True)

# 示例1：稀疏 gather 的作用（BSND，B=1, S1=1, N1=1, D=512, S2=4, K=2）
# 为可读性使用较小 K，实际算子要求 K∈{1024,2048,3072,4096,5120,6144,7168,8192}；示例仅为演示逻辑，实际参赛实现需满足 K 的取值约束
D = 512; scale = 1.0 / np.sqrt(D)
query   = np.ones((1, 1, 1, D), dtype=np.float16)          # (B,S1,N1,D)
key     = np.zeros((1, 4, 1, D), dtype=np.float16)
for j in range(4): key[0, j, 0, j] = 1.0                   # key[j]=e_j 单位向量
value   = np.tile(np.arange(1, 5, dtype=np.float16).reshape(1, 4, 1, 1), (1, 1, 1, D))  # value[j]全(j+1)
sparse_indices = np.array([[[[0, 1]]]], dtype=np.int32)    # (1,1,1,2) 只选位置0/1
d_out   = np.ones((1, 1, 1, D), dtype=np.float16)
O, smax, ssum = forward(query[0,0], key[0,:,0,:], value[0,:,0,:], sparse_indices[0,0,0], scale)
out          = O[None]                       # (1,1,1,D)
softmax_max  = smax.reshape(1, 1, 1, -1)     # (1,N2=1,S1=1,G)
softmax_sum  = ssum.reshape(1, 1, 1, -1)
dQ, dK, dV = sparse_flash_attention_grad(query, key, value, sparse_indices, d_out, out,
                                         softmax_max, softmax_sum, scale)
# 输出形状：dQ=(1,1,1,512)  dK/dV=(1,4,1,512)
# 前向 O 每维均值 1.5（softmax 均匀，0.5*(1+2)）
# dQ 前4维 [-5.656, 5.656, 0, 0]（缩放后）
# dK：位置0/1 非零（∓5.656/维），位置2/3 为 0（未选中，体现稀疏 gather）
# dV：位置0/1 各 0.5/维，位置2/3 为 0
print('dQ 前4维:', dQ[0,0,0,:4])
print('dK 位置0/1/2 前1维:', dK[0,0,0,0], dK[0,1,0,0], dK[0,2,0,0])
print('dV 位置0/1/2 均值:', dV[0,0,0].mean(), dV[0,1,0].mean(), dV[0,2,0].mean())

# 示例2：KV merge 场景（value 传空，value=key，dKey=dK+dV，不输出 dValue）
# 复用示例1前向结果，value 按 key 处理，dKeyOut = dK + dV
dK_merge = dK + dV
# dKey 位置0 前1维 = -5.656 + 0.5 = -5.156；位置1 = 5.656 + 0.5 = 6.156；位置2/3 = 0
print('KVmerge dKey 位置0/1 前1维:', dK_merge[0,0,0,0], dK_merge[0,1,0,0])
```
