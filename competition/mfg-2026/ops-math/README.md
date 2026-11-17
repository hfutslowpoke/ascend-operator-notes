# ops-math：ComplexAbs / BallQuery / StackBallQuery

> 层级：`competition` · 赛事：2026 制造行业算子专场 · 对应赛题：01 `ComplexAbs`、14 `BallQuery` · 本机 910B3：`not-run`

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | https://gitcode.com/cann/ops-math |
| 维护者 | CANN 开源社区 |
| 固定 commit | `3b2baac63624c32e03781bd9f0290f1a4584f0c7`（2026-11-17） |
| 默认分支 | `master` |
| 许可证 | CANN Open Software License Agreement Version 2.0 |
| 本地缓存 | `repository-research-cache/mfg-2026/ops-math/` |
| 去重 | 与 ops-fft/ops-gnn 不重复；这里看复数逐元素和球查询 |

## 2. 对应赛题与硬件事实

### 2.1 ComplexAbs（01）

- `math/complex_abs/README.md`：**A2 明确 ×**，950/A3 √；
- 代码只有 `op_kernel/arch35/`、`op_host/arch35/`，没有 arch22；
- 公式：`y = sqrt(re² + im²)`；COMPLEX32→FLOAT16、COMPLEX64→FLOAT、COMPLEX128→DOUBLE；
- 结论：只能作为算法参考，**不能直接上 910B**。

### 2.2 BallQuery（14，直接版）

- `conversion/ball_query/README.md` 产品表写 A2 √；
- 但 `op_kernel` 只有 `arch35/ball_query_simt.h`，`op_host` 只有 `arch35/ball_query_tiling.cpp`，`op_host/config` 只有 `ascend350`/`ascend950`；
- **README 与代码/config 冲突**：要么 A2 支持尚未落到本 commit，要么依赖动态编译；不能把 README 的 √ 当成本机可构建。
- 直接版是 SIMT grid-stride：每线程处理多个 query，d² 比较，首命中填充，零命中全 0，无 atomic。

### 2.3 StackBallQuery（14 的 910B 可用形态）

- `conversion/stack_ball_query/README.md`：A2/A3 √；
- 有 `op_host/config/ascend910b/stack_ball_query_binary.json` 与 `simplified_key.ini`；
- 输入是 2D 堆叠布局 + `xyz_batch_cnt`/`center_xyz_batch_cnt`，不是 `[B,3,N]`；
- 语义：按 center 在半径内取前 `sample_num` 个点索引；返回的是半径内前 K 个，不保证最近。

## 3. 源码定位

| 内容 | 路径/行 |
|---|---|
| ComplexAbs 语义/约束/dtype | `math/complex_abs/README.md:1-60` |
| ComplexAbs DAG：平方和/sqrt/缩放 | `math/complex_abs/op_kernel/arch35/complex_abs_dag.h:23-178`（关键 `:105-152`） |
| ComplexAbs tiling（arch35） | `math/complex_abs/op_host/arch35/complex_abs_tiling_arch35.cpp` |
| BallQuery 语义/约束 | `conversion/ball_query/README.md:1-60` |
| BallQuery SIMT | `conversion/ball_query/op_kernel/arch35/ball_query_simt.h:62-160` |
| BallQuery tiling（arch35） | `conversion/ball_query/op_host/arch35/ball_query_tiling.cpp` |
| StackBallQuery 语义/约束 | `conversion/stack_ball_query/README.md:1-60` |
| StackBallQuery kernel 类/流程 | `conversion/stack_ball_query/op_kernel/stack_ball_query.h:28-140,281-397` |
| StackBallQuery 入口/dtype 分派 | `conversion/stack_ball_query/op_kernel/stack_ball_query.cpp:17-40` |
| StackBallQuery 910B 配置 | `conversion/stack_ball_query/op_host/config/ascend910b/` |
| ComplexV2（实虚交错基础件） | `experimental/math/complex_v2/op_kernel/complex_v2.h:42-150` |

机器索引：[source.json](source.json)。

本地源码副本：[`source/`](source/)（按上游路径保存，字节与固定 commit 一致；`source.json` 的 `local_path` 指向对应文件。上游原文的相对链接假设整仓，本地不保证可解析）。

## 4. 机制

### 4.1 ComplexAbs

- 对每个复数，取实部/虚部，平方相加，开方；
- 对超大值先按 `SCALE_UP` 缩放、再 `SCALE_DOWN`，避免平方溢出；
- `Reg::Compare/Select` 处理 |re|/|im| 的大小关系，避免直接平方溢出；
- 输出按 dtype pack（complex32→half、complex64→float、complex128→double）。

### 4.2 BallQuery（直接版）

- 每个线程处理多个 query（grid-stride）；
- 用 `UintDiv` magic 把 `qIdx` 分解成 `(m,b)`；
- 对每个 query 扫描 N 个点，`d² = (cx-x)² + ...`；命中条件 `d²==0 || minR² <= d² < maxR²`；
- 命中按 k 顺序写入 `idx`；不足 `sample_num` 用首命中填充；零命中写 0；
- 不涉及 atomic，输出确定。

### 4.3 StackBallQuery（910B）

- 输入 `xyz` 是 `(3, N_total)`，`center_xyz` 是 `(M, 3)`；`*_batch_cnt` 给每 batch 的点数；
- 按 center 分核/分段；每次把一个 batch 的 xyz 片段搬进 UB，计算到 center 的距离；
- 距离平方后与 `maxRadius²` 比较；命中写结果，达到 `sample_num` 提前停；
- 结果按段 `SendResultToGm`，未命中/不足用首索引补齐（`:384`）；
- FP16 路径把 `maxRadius²` 先转 half 再平方，和 FP32 路径有精度差异（`:43-49`）。

### 4.4 ComplexV2（基础件）

- 输入 real/imag 两个平面，输出实虚交错 complex；
- `Compute` 用 `SetValue(2i, real)`/`SetValue(2i+1, imag)`；
- 有 `ascend910b` 配置，可作为 ComplexAbs/Div/Exp 的输出布局基础。

## 5. 910B 适用性与证据

- **StackBallQuery**：有 A2 配置 + 通用 kernel，最可能在 910B 构建；优先验证。
- **ComplexAbs/BallQuery 直接版**：代码 arch35-only，README 与配置冲突；不作为 910B 可运行结论。
- **ComplexV2**：有 910B 配置，可作为复数布局基础，但不是 01/02/03 的完整实现。
- 本批未构建/未运行：`local_status=not-run`。

## 6. 负例/风险

- **README/代码冲突**：`ball_query` 的 A2 √ 与 arch35-only 代码/config 不一致；必须以本机构建为准。
- **ComplexAbs A2 ×**：不要把 950 的 DAG 直接搬 910B；910B 可能需要 Vector/RegBase 的另一套实现。
- **StackBallQuery 语义**：输入是堆叠 2D + batch counts，不是原始 `[B,3,N]`；需要适配层。
- **零命中**：输出全 0 与“命中第 0 点”不可区分，题面若要求区分需另加 mask。
- **FP16 半径**：`maxRadius²` 先转 half 可能改变边界命中；要和题面精度对齐。
- **ComplexV2**：只是布局构造，不含 abs/div/exp 数学。

## 7. 复用判断

**优先复用**：StackBallQuery 的堆叠布局/分段流式/提前停；ComplexAbs 的缩放+平方和；ComplexV2 的实虚交错布局。

**需要验证**：910B3 构建 `stack_ball_query`；直接 `ball_query` 是否能编 A2；ComplexAbs 的 910B 替代实现。

**暂不使用**：把 arch35 的 ComplexAbs/BallQuery 直接当 910B 模板；把 ComplexV2 当完整复数算子。

## 8. 下一步

1. 构建 `stack_ball_query`（`ascend910b` 配置），跑官方 GEIR 用例；
2. 用 CPU 参考对照索引/零命中/不足填充；
3. 试编 `ball_query` 的 A2，记录 README/config 冲突的实际结果；
4. 对 ComplexAbs，先确认题面 dtype，再决定是否写 910B Vector 版本。
