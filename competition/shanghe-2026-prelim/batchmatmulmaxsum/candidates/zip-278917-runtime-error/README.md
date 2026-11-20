# 候选：zip 278917（Runtime Error，未采用）

## 身份

| 项 | 值 |
|---|---|
| 来源 | 用户提供 `batchmatmulmaxsum_submission_278917.zip` |
| detail ID | 278917 |
| 提交时间 | 2026-09-12T03:52:42.033Z |
| 平台状态 | **Runtime Error** |
| 与正式版关系 | 更早的纯 Vector 实现；正式版是 `../../source/kernel.asc`（detail 884931，Cube+Vector 混合） |
| zip SHA256 | 见 `../../evidence/zip.sha256` |

## 为什么保留

这是“纯 Vector bmm 在规模上去后不可行”的负例：

- 文件头注释明确写 `Implementation: pure Vector. No Nd2Nz / Mmad / Fixpipe.`；
- `kernel.asc` 102,218 B，约 784 行，含 scalar/tiny/vector 路径；
- 正式版 `kernel.asc` 224,418 B，使用 `MatmulImpl`、`Mmad`、`Fixpipe`、AIC/AIV 流水；
- 两者 diff 约 4,828 行。

## 结论

不要把这个候选当正式提交或性能基线。它只用于说明：

1. 小 shape 的纯 Vector 可以工作；
2. 中大 shape 的 bmm 需要 Cube；
3. 早期版本在平台上是 Runtime Error，不能因为“本地能编译”就提交。

正式源码、平台证据和逐 case 时间见 [`../../README.md`](../../README.md) 与
[`../../evidence/`](../../evidence/)。
