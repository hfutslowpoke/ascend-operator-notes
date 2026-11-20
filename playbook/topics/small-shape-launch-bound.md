# 小 case / launch-bound 优化

> 一句话：当 case 只有 0.5–2 ms、不同 shape 的耗时几乎一样时，瓶颈通常不是算力，而是
> launch、tiling 读取、TPipe/InitBuffer、同步和通用 dispatch。优化目标是**减少固定开销**，
> 不是换更大的 tile。

## 1. 什么时候适用

同时满足 2 条以上：

- 多个 case 的耗时差 < 20%，且都在 0.5–2 ms 量级；
- 理论 FLOPs 很小，Cube/Vector 利用率都不高；
- best 时间也接近常数，说明冠军也在打固定开销；
- 代码里有 `TPipe`、`InitBuffer`、`GET_TILING_DATA`、通用 `Gather/Scatter` 等入口成本。

反例：shape 一大就线性变慢、Cube 利用率高、耗时 > 5 ms——这时应该看 tiling/带宽，不是本节。

## 2. 做法（按收益排序）

1. **去掉不需要的入口**
   - 精确小 shape 走独立 `PATH`，不读 tiling struct、不构造 `TPipe`、不 `InitBuffer`；
   - 用固定 UB 偏移，不用动态 `TBuf`。
2. **减少 block 数**
   - 一行/一个 task 一个 block，避免跨核同步；
   - 小 case 不要为了“占满核”而切分，切分后的同步比计算还贵。
3. **固定 lane 宽度**
   - 8/16 lane 的 `Brcb + Compare + BlockReduceMax/Sum`，替代通用 `Gather/Scatter`；
   - 所有 buffer 按 32 B 对齐，索引用编译期常量。
4. **lean tiling ABI**
   - 只保留 kernel 真正读的字段；32 B 的 struct 比几百 B 的 struct 更省入口时间；
   - 用 `pad` 字段兼作 dtype/模式标记，避免额外字段。
5. **合并小 kernel**
   - 能在一个 launch 内做的逐元素/归约，不要拆成多个 kernel；
   - 但不要为了合并引入大的 workspace 分配。
6. **复用参数**
   - 小矩阵-向量场景把 `phi/bias/gamma` 一次读入 UB，按 block 复用；
   - 不要把参数放到 GM 每次重读。

## 3. 证据

- **Sparse Softmax（星辰杯 A，2/161，84.36）**：5 个 case 全部 1.40–1.48 ms；
  `PATH6–9` 用 raw UB、单 block、固定 8 lane，不读 tiling/不建 pipe；与第一名总时间相同。
- **mHcPre（星辰杯 A，团队 18/106，65.01；未入库）**：2 个 case 2.44/3.22 ms；
  `n4/d16/k64`、`n8/d16/k128` 精确路径跳过 tiling 读取，纯 Vector 避免 Cube setup。
- **DenseLightning 小 case**：best 0.62–0.64 ms，本提交 4.4–7.2 ms；反例说明
  “没有走 raw/精确路径”时，通用 Vector 的固定开销会直接吃掉 7–12 倍。

## 4. 负例/边界

- 精确 shape 入口过多会变成“按测试用例特化”；至少留一个 shape family 或通用 fallback。
- 小 case 的 raw UB 路径往往只支持单行/固定 dtype；`inner>1`、bf16、长轴要另走通用路径。
- 不要把小 case 的“去 TPipe”直接套到大 case；大 case 需要 pipe/双缓冲/多核。
- 小 case 的绝对时间受频率/平台影响，不要用本地 1.4 ms 推平台排名。

## 5. 检查清单

```text
[ ] 是否真的 launch-bound？（多 shape 时间是否接近常数）
[ ] 是否还在读不需要的 tiling 字段？
[ ] 是否还在构造 TPipe/InitBuffer？
[ ] 是否用了通用 Gather/Scatter 而不是固定 lane 归约？
[ ] 是否为了占满核而做了不划算的切分？
[ ] 是否有 shape family / fallback，而不是只支持一个精确 shape？
[ ] 是否记录了平台逐 case 时间和 best time？
```
