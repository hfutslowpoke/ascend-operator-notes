# 对齐、尾块与 padding

> 一句话：Ascend 的搬运/矩阵乘对 32B/128B 对齐敏感；尾块处理错，轻则慢，重则丢数据或
> 把 padding 0 当成真实值。对齐是正确性的一部分，不只是性能。

## 1. 常见坑

- `DataCopy` 要求元素数 32B 对齐；不对齐时尾块会被静默丢弃；
- `DataCopyPad` 的 `blockLen` 用字节，不是元素；
- max 归约的 padding 必须用 `-inf`，不能用 0；
- 矩阵乘的 K/N 尾块如果直接进 Cube，可能读越界或算错；
- 空段/空专家/零长度 batch 的指针和偏移；
- 多核切分时，尾块可能落在错误 block。

## 2. 做法

1. **搬运**
   - 全局→UB 用 `DataCopyPad`，按字节给 `blockLen`；
   - 行 pitch 按 32B 向上取整，索引时用 pitch 而不是原始 N；
   - 多行搬运用 `DataCopyExtParams` 的 `blockCount/blockLen/srcStride/dstStride`。
2. **max/sum 的 padding**
   - max：`-3.402823466e38F` 或 `-inf`；
   - sum：0；
   - 不要用同一个 padding 值同时做 max 和 sum。
3. **Cube 的 K/N 对齐**
   - K/N 非 32B/128B 对齐时，先 pack 到对齐布局，再进 Cube；
   - pack 的 workspace 和搬运成本要计入总时间；
   - 小矩阵直接 `DataCopyPad` 可能比 pack 便宜，做 A/B。
4. **尾块验证**
   - 至少测：N=1、N=block-1、N=block、N=block+1；
   - K 非 8/16/32 对齐；
   - M/N 非 tile 对齐；
   - 空段/空专家/零长度。
5. **多核边界**
   - 每个 block 的行区间要按实际 group/segment 切，不要假设均分；
   - 尾块归属明确，避免两个 block 写同一行。

## 3. 证据

- **BatchMatmulMaxSum**：`DataCopyPad` 的字节 blockLen；max padding 用 `-3.402823466e38F`；
  非 128 对齐的转置尾块先 pad 再走主路径。
- **GroupedMatmulGelu**：`packB` 把 K/N 非 32B 对齐的 weight 重排到 `bPitch`；
  `padA` 处理 x 的 K 尾块；空专家跳过但保持专家索引不错位。
- **Sparse Softmax**：`DataCopyPad` 单行搬运；`axisLength` 运行时长度；
  长轴用 workspace 分组表，避免整行塞 UB。
- **mHcPre**：`n4/d16/k64`、`n8/d16/k128` 是对齐友好的精确 shape；
  n=6/大 D 走通用路径，尾块要单独验证。

## 4. 负例

- 用 `DataCopy` 搬非 32B 对齐的行，尾块被丢，输出最后一列/几列错；
- max 的 padding 用 0，全负相似度被 max 成 0；
- pack 了 B 但没把 pack 成本计入，结果比不 pack 还慢；
- 多核按 `M/coreNum` 均分，空专家导致某些核做 0 行、某些核做很多行。

## 5. 检查清单

```text
[ ] 所有 GM→UB 尾块是否用 DataCopyPad/字节 blockLen？
[ ] max 的 padding 是否是 -inf，而不是 0？
[ ] sum 的 padding 是否是 0？
[ ] K/N 非对齐时，pack 和 DataCopyPad 做过 A/B 吗？
[ ] 是否测了 N=1/block-1/block/block+1、K 非对齐、空段？
[ ] 多核行区间是否按实际 group/segment 切？
[ ] 尾块是否只被一个 block 写？
```
