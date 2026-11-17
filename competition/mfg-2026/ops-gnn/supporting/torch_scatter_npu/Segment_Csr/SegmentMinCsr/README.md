# 算子名称
SegmentMinCsr

## 1. 算子功能
`SegmentMinCsr` 接收稠密输入 `src`、CSR 格式的行指针 `indptr`、可选的预分配输出 `optional_out`，在 CSR 语义下对同一 segment 内的向量元素执行逐列最小值聚合：

- **`src`**：源数据张量，形状为 `(E_1, M, K)` 的展平形式，支持最多8维
- **`indptr`**：CSR 行指针张量，形状前缀维度与 `src` 对齐（支持广播），最后一维大小为 `nSegments + 1`，类型为 `int32`
- **`optional_out`**：可选的预分配输出张量。如果提供：
  - 首次更新某个 segment 时，`src` 与 `optional_out` 比较取最小值
  - 空 segment 时，直接从 `optional_out` 复制到 `out`
  - 如果未提供，空 segment 填充该数据类型的最大值
- **`out`**：对每个 segment `s ∈ [0, S)`、列 `j`，计算  
  `out[s, j] = min { src[i, j] | indptr[s] <= i < indptr[s+1] }`
  - 若 segment 为空且提供了 `optional_out`，则 `out[s, j] = optional_out[s, j]`
  - 若 segment 为空且未提供 `optional_out`，则 `out[s, j]` 填充该类型最大值

## 2. 目前数据类型支持
- 支持的数据类型：
  - `fp32`
  - `fp16`
  - `int32`
  - `int16`

## 3. 目前功能支持
- **基础特性**：
  - 支持处理对齐格式向量
  - 支持处理尾块和非对齐数据
  - 使用 Compare + Select 模式实现最小值计算
- **维度支持**：
  - `src` 最多支持8维
  - `indptr` 最多支持8维
  - 支持 `indptr` 广播（`indptr` 的前缀维度可以与 `src` 的前缀维度相同或其中一个为1）
- **可选输入支持**：
  - 支持 `optional_out` 可选输入参数
  - 有 `optional_out` 时，首次更新与 `optional_out` 比较
  - 有 `optional_out` 时，空 segment 从 `optional_out` 复制
  - 无 `optional_out` 时，首次更新直接复制 `src`
  - 无 `optional_out` 时，空 segment 填充该类型最大值
- **空 segment 处理**：
  - 有 `optional_out`：从 `optional_out` 复制到 `out`
  - 无 `optional_out`：`out` 填充该类型最大值

## 4. 工程结构
```
├── SegmentMinCsr                              // SegmentMinCsr算子
│   ├── SegmentMinCsrFrameworkLaunch           // SegmentMinCsr算子FrameworkLaunch形式
│   └── README.md                              // 算子工程README
```


## 5. 开发者信息
- 开发者：史浩睿
- 邮箱：2501060546@qq.com

## 6. 更新日志
| 版本 | 日期       | 更新内容  | 开发者             |
|------|------------|---------|--------------------|
| v1.0 | 2026-3-20 | 初始版本 | 史浩睿 |

