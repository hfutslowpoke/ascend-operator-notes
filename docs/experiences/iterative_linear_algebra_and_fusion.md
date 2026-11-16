# 迭代线性代数：批量矩阵交错、轮转配对与单 kernel 分解融合

本文从 `Be4ver/svd` 的两份实现提炼可迁移机制，面向 SVD、极分解、特征分解、批量 QR 等迭代线性代数算子。所有源码位置对应固定 commit `b02796511ea62861c647a56b880da15b6a07b3a0`；性能数字区分作者证据与本机 910B3 复测。

## 1. 问题特征

迭代线性代数算子的共同结构：

```text
初始化 -> 若干 sweep/迭代 {
    选择一组旋转/反射/更新
    计算点积、范数、角度
    应用旋转/更新
    检查收敛
} -> 输出因子/奇异值
```

在 Ascend 上常见的瓶颈不是“矩阵乘峰值”，而是：

- 小矩阵的向量指令太短（一行 32/64 个元素），`Mul/Reduce` 的启动/归约开销占比高；
- 每对行/列都要把 `a,b,c` 从向量寄存器读回标量，再算角度，形成串行反馈；
- 多阶段分解（QDWH → SDC → 叶块 Jacobi）每阶段一个 kernel，中间矩阵落 GM，host 往返；
- UB 容量固定，batch/tile 稍大就溢出，稍小就摊不薄固定开销。

下面四种机制分别对应这四类瓶颈。

## 2. 机制 A：跨矩阵交错，合并短向量指令

### 2.1 源码

`paco-jacobi-svd/jacobi_svd_custom.cpp:215-232`：

```text
tile[(i*K + k)*colsW] = A_k 的第 i 行
```

`RunMulti`（`:251-303`）对同一 `(i,j)` 行对：

- 一次 `Mul(prod3_, u, u, n)` 覆盖 K 个矩阵的 `u²`；
- 一次 `WholeReduceSum(red_, prod3_, colsW, K, 1, 1, colsW/8)` 得到 K 个 `|u|²`；
- 同理 `|v|²`、`<u,v>`；
- 角度仍逐矩阵标量算；
- 旋转逐矩阵 `Muls/Axpy`，每对只保留一次 `Adds` 拷回。

K 的选择（`:440-445`）：

```text
kUbBudget = 120 KiB
perMatBytes = (rows + 4) * colsWork * 4
K = min(kUbBudget / perMatBytes, 64)
```

### 2.2 改变了什么

- **减少指令数**：K 个矩阵共享一条 `Mul`/`Reduce`，而不是 K 条短指令；
- **减少归约启动**：`WholeReduceSum(repeat=K)` 一次出 K 个结果；
- **不改变算术复杂度**：仍是每个矩阵 O(n³/sweep) 的 Jacobi；
- **不消除标量角度**：角度仍逐矩阵算，因为作者实测矢量角度链只在 `count=64` 时正确（`paco-jacobi-svd/readme.md` 的“已验证的坑”）。

### 2.3 适用条件

- 小矩阵、批量大：`rows<=128, cols<=256`，`batch` 大到 `perCore>1`；
- UB 能容纳 K 个矩阵的行交错布局；
- 角度可以用标量循环，或能证明矢量角度链在目标 count 下正确；
- 能接受 K 个矩阵一起收敛/一起写回（K 合并路径是“这一块 K 个矩阵全部收敛”才停）。

### 2.4 代价与负例

- K 越大，UB 越紧；作者实测 140 KiB 收益为负，最终取 120 KiB；
- 矩阵按行交错后，单矩阵的连续行不再连续，调试/校验更复杂；
- K 只摊薄“点积/归约”，不摊薄旋转的逐矩阵 `Muls/Axpy`；
- 列 pad 到 64（而不是 32）会让 32 列矩阵带宽翻倍；
- 矢量角度链在 K=1/4/10/13/26 下不可信，不能想当然地把标量循环换成向量链。

### 2.5 证据

- 作者：512×(32×32) = 7.061 ms，512/512 PASS，max σ abs `3.242e-05`；
- 本机 910B3/CANN 9.1.0：6.174 ms，512/512 PASS，max σ abs `3.242e-05`；
- 没有同环境 PyTorch/厂商库 baseline，不能换算相对加速比。

## 3. 机制 B：矩阵内轮转配对，把“每对一次反馈”改成“每轮 32 对”

### 3.1 源码

`paco-jacobi-svd/docs/64x64-optimization-study-2026-11-12.md:62-100`，补丁 `docs/research-64x64/tournament.patch`（未合入主路径）。

64 行共有 `64*63/2=2016` 个无序对。字典序每个 sweep 做 2016 次串行对：

```text
for i<j:
    dot/norm -> scalar a,b,c -> angle -> 4 row updates
```

tournament 把行分成 A/B 两组，每轮 32 对互不共享行，63 轮覆盖全部 2016 对：

```text
round r: (A[k], B[k]) for k=0..31
A_next = [A0, B0, A1..A30]
B_next = [B1..B31, A31]
```

每轮：

- 3 条 `WholeReduceSum(repeat=32)` 得到 32 组 a/b/c；
- 角度链固定 64 lane（前 32 有效），`Compare/Select` 处理无效 lane；
- `Brcb` 把系数广播成 32B 块，`BinaryRepeatParams{1,1,0,8,8,1}` 让一个系数在一行的 8 个数据块上重复；
- 4 条 `Mul/MulAddDst` 更新 32 对；
- 第二个 UB 矩阵 ping-pong，5 段连续 `Adds(...,0)` 完成行重排。

### 3.2 改变了什么

- 每个 sweep 的行更新 API 从 8064 条降到 252 条；
- 角度从“每对一次标量反馈”变成“每轮一次向量链”；
- 算术复杂度仍是 O(n³/sweep)，不是 32× 的算术减少；
- 行重排改变了旋转顺序，浮点结果和所需 sweep 数不保证与字典序相同。

### 3.3 证据

作者同设备、相邻运行、event 中位数：

| 输入 | batch | 原实现 | tournament | 加速 |
|---|---:|---:|---:|---:|
| 标准正态 | 1 | 3.62260 ms | 0.85056 ms | 4.26× |
| 标准正态 | 40 | 3.94784 ms | 0.94690 ms | 4.17× |
| cond≈1e3 | 40 | 4.86842 ms | 1.41022 ms | 3.45× |
| cond≈1e6 | 40 | 6.51702 ms | 1.87254 ms | 3.48× |

### 3.4 代价与负例

- 只常驻 UB、不改配对：3.2–3.4%，说明主要收益来自“轮内 32 对融合”，不是常驻；
- 只把角度链向量化、保留 K≤8 的单对组织：batch=1 14.2394 ms（比 3.62 ms 慢），batch=256 18.3270 ms（比约 16 ms 慢）；固定成本需要更多独立配对来摊薄；
- 先集中算标量系数、再循环更新：batch=256 15.9976→20.7030 ms，更慢；
- 硬截断 sweep：10 sweeps 在 cond=1e6 上小奇异值相对误差明显变差；作者保留 30；
- 该路径仍是研究补丁，未合入当前代码，也没有本机设备复测。

## 4. 机制 C：单 kernel 多阶段融合，中间矩阵不落 GM

### 4.1 源码

`qdwh_sdc_fused_jacobi/fused_qdwh_sdc.hpp:89-188,196-477`、`fused_main.cpp:280-354`：

```text
fused_main: 上传 A 一次 -> launch qdwh_sdc_fused_custom 一次 -> 下载 leaves
```

- `H` 直接由 QDWH 最后一步写进 SDC arena 的 `slot0`（`fused_qdwh_sdc.hpp:103-104,144`）；
- SDC 节点队列、`H±` 搬运、叶块收集都在 device 内（`:196-477`）；
- Phase1 QDWH 与 Phase2 SDC 用 `PhaseBarrier` 在 kernel 内同步（`:733-743`）；
- 计算期间 host 不参与数据传输；结束后的 D2H 只为验证/输出。

### 4.2 改变了什么

- 省掉两次 host main、两个 kernel；
- 省掉 `h.bin/u.bin` 落盘和 H2D/D2H；
- 中间矩阵 H 只被下一阶段消费，适合“生产者-消费者”融合；
- 代价是一个巨大 MIX kernel：flag 协议、arena 布局、padding 清零、跨核 phase barrier 都在一个 kernel 里。

### 4.3 证据

| 形状 | 融合 kernel | 独立 QDWH+SDC + 独立 Jacobi | 节省 |
|---|---:|---:|---:|
| 64×64/batch1 | 4.102 ms | 6.278 ms | 2.176 ms（34.7%） |
| 128×128 | 6.428 ms（回归记录） | — | — |
| 1024×1024 | 46.839 ms | — | — |

本机 910B3/CANN 9.1.0：128/512/1024 = 4.288/20.698/48.475 ms，QDWH/SDC/sigma 全 PASS；与作者 4.392/20.540/46.839 ms 接近。

### 4.4 适用条件与代价

适用：

- 多阶段分解的中间矩阵只被下一阶段读；
- UB/arena 能容纳中间矩阵；
- 能实现跨核 flag/phase barrier；
- 不需要 host 中途检查中间结果。

代价/负例：

- 一个 MIX kernel 里同时有 AIV/Cube/标量/搬运，调试和回归成本高；
- UB 余量可能只有几百字节（QDWH 是 195,840/196,608 B）；
- 任一阶段编译失败会拖垮整包（例如本机 FusedMoE 的 UB 溢出，是另一类但同样是“大融合”的边界）；
- 中间矩阵不落 GM 后，host 无法在失败时检查中间值，需要 device 侧调试缓冲。

## 5. 机制 D：UB 预算驱动的 batch 与冗余工作消除

### 5.1 同层节点 batch QDWH

`qdwh_sdc_fused_jacobi/fused_qdwh_sdc.hpp:386-423,481-500` 把节点队列头最多 `kSdcBatchMax` 个节点打包成一组，做一次 batch QDWH。`kSdcBatchMax` 由 `fused_batch_cfg.hpp:33-83` 的三个上界取最小：

- 固定 UB 占用 + 点积区 + `tQueueDst` 每 batch 256 float；
- `householder_v` 每矩阵 `8*columns + 512` float；
- `QdwhPrepare` 的定长数组 B+1<=16；
- `static_assert` 在编译期兜底。

作者实测 1024² 51.599→46.839 ms，2048² 136.09→134.97 ms；3072² batch 8→15 无收益（401.7→403.5 ms），说明固定开销摊薄已饱和，kernel 计算受限。

### 5.2 跳过显式对称化

`sdc_node.hpp:331-397` 用 `kSymmetrize=false` 跳过 `H←0.5(H+Hᵀ)` 的标量两遍读写。作者记录 160.8→134.3 ms，`eig_err` 2.2e-7→2.0e-7，`sym_max=0`。

条件是：`H±` 由正交相似变换保证对称，GEMM 的舍入不对称在容差内；仍要检查 `sym_max/eig_err`。

### 5.3 λ 的分布式归约

`vec.hpp:397-453`：每个 AIV 只读自己那一段对角线，部分和写 GM 连续数组，`AivBarrier` 后所有核 `WholeReduceSum` 归约。作者 2048² `aiv_scalar` 21007→19917 µs（−5.2%），kernel 132.76→132.29 ms；求和顺序改变会让分裂树 47→46，误差仍 1.5e-06。

这适合“每个核都重复读同一段标量”的场景；如果标量读已命中 cache，收益会很小（README 记录了两类“零收益”的 scalar/同步优化）。

## 6. 选择表

| 瓶颈 | 优先机制 | 预期改变 | 先验证什么 |
|---|---|---|---|
| 小矩阵短向量/归约启动多 | 跨矩阵交错（A） | 指令数/归约次数下降，算术不变 | K 扫描、UB 余量、角度 count 正确性 |
| 每对行标量反馈串行 | 轮内 32 对（B） | 行更新 API 大幅下降 | 63 轮覆盖、条件数、sweep 数、数值 |
| 多阶段中间矩阵落 GM | 单 kernel 融合（C） | kernel 数/传输下降 | 中间矩阵消费者、UB、flag 协议 |
| 逐节点固定开销 | 同层 batch（D1） | 固定搬运/flag 摊薄 | batch 上限、计算受限点 |
| 冗余对称化/标量读 | 跳过对称化/分布式归约（D2/D3） | 标量读写下降 | 对称性/误差、求和顺序 |

## 7. 消融设计

1. **K 扫描**：固定 shape/batch，扫 K=1/2/4/8/16/32/64；记录 event 时间、UB 占用、正确性、角度链是否退化。
2. **轮内配对**：固定 64×64，比较字典序 vs tournament；扫 cond=1e2/1e4/1e6、batch=1/8/40；记录 sweep 数、σ 相对误差、行更新 API 数。
3. **融合边界**：同一 QDWH+SDC 做“两次 launch + H 落盘”vs“单 launch”；记录 kernel 数、H2D/D2H 字节、event 时间、UB、失败模式。
4. **batch 上限**：固定形状，扫 `kSdcBatchMax=1..15`；记录 kernel 时间、分组数、leaves、是否计算受限。
5. **确定性**：同一输入重复 20 次，记录 σ/leaves 是否逐次相同；对 1000×800、seed=42 做专项。
6. **数值**：cond=1e2/1e4/1e6，分别看最大绝对误差和最小奇异值相对误差；不能只用 `atol`。

## 8. 证据边界

- 作者数字来自 910B3，CANN 版本未统一固定；QDWH 文档记录过 8.2.RC1，本机用 9.1.0。
- 本机验证了 PACO 32×32×512、QDWH 128/512/1024，以及 QDWH 1000×800 seed=42 的 3 次重复（sigma 三次一致、全 PASS；作者的 atomic 非确定性未触发）。没有验证 64×64 tournament、2048²、cond=1e6 的完整批量。
- PACO 没有同环境 baseline，不能换算相对加速比；QDWH 的 34.7% 是作者 64×64 的融合对比。
- Be4ver 仓库未声明 LICENSE；本文只做引用和分析，不复制源码。
- 这些机制不保证在 910C/950 或其它 CANN 上成立；UB 预算、atomic 行为、编译器优化都可能变化。
