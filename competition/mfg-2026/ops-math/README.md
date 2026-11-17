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
| 本地源码 | `competition/mfg-2026/ops-math/source/` |
| 去重 | 与 ops-fft/ops-gnn 不重复；这里看复数逐元素和球查询 |

## 2. 对应赛题与硬件事实

### 2.0 官方题面校正（2026-11-17）

| 赛题 | 官方规格 | 源码差异 |
|---|---|---|
| 01 `ComplexAbs` | `src [N,M] complex64`→`out float32`，`N,M≤65536`，分量绝对值≤10，atol=rtol=1e-4 | `reference`：源码 arch35 支持 complex32/64/128，A2 不支持 |
| 14 `BallQuery` | 单 `radius`、严格 `d²<radius²`、扫描序 first-K、零命中全 0、fp16→fp32 阈值 | `adapt`：直接版有 min/max shell；stack 版是堆叠布局且 fp16 把 `radius²` 转 half |


### 2.1 ComplexAbs（01）

- `math/complex_abs/README.md`：**A2 明确 ×**，950/A3 √；
- 代码只有 `op_kernel/arch35/`、`op_host/arch35/`，没有 arch22；
- 公式：`y = sqrt(re² + im²)`；COMPLEX32→FLOAT16、COMPLEX64→FLOAT、COMPLEX128→DOUBLE；
- 结论：只能作为算法参考，**不能直接上 910B**。

### 2.2 BallQuery（14，直接版）

- `conversion/ball_query/README.md` 产品表写 A2 √；
- 但 `op_kernel` 只有 `arch35/ball_query_simt.h`，`op_host` 只有 `arch35/ball_query_tiling.cpp`，`op_host/config` 只有 `ascend350`/`ascend950`；
- **README 与代码/config 冲突**：要么 A2 支持尚未落到本 commit，要么依赖动态编译；不能把 README 的 √ 当成本机可构建。
- 直接版是 SIMT grid-stride：每线程处理多个 query，d² 比较，首命中填充，零命中全 0，无 atomic；
- **官方 BallQuery 只用单 `radius`，严格 `d² < radius²`**；直接版的 `min_radius/max_radius` 与 `d²==0` 命中语义不是官方契约，需要裁剪/重写。

### 2.3 StackBallQuery（14 的 910B 可用形态）

- `conversion/stack_ball_query/README.md`：A2/A3 √；
- 有 `op_host/config/ascend910b/stack_ball_query_binary.json` 与 `simplified_key.ini`；
- 输入是 2D 堆叠布局 + `xyz_batch_cnt`/`center_xyz_batch_cnt`，不是 `[B,3,N]`；
- 语义：按 center 在半径内取前 `sample_num` 个点索引；返回的是半径内前 K 个，不保证最近。

### 2.4 910B 补充基础件：ComplexExp 的组合路径

官方 `ComplexExp` 没有公开单 kernel 实现，但 `ops-math` 同一 commit 有可组合的 910B 基础件：

| 基础件 | 能力 | 路径 |
|---|---|---|
| `experimental/math/exp` | 910B arch22，float16/float32/bfloat16 指数 | `source/experimental/math/exp/README.md` |
| `experimental/math/sin` | 910B，float16/float32/bfloat16 正弦 | `source/experimental/math/sin/README.md` |
| `experimental/math/cos` | 910B，float16/float32/bfloat16 余弦 | `source/experimental/math/cos/README.md` |
| `experimental/math/complex_v3` | 910B arch22，real/imag float16/float32 → complex32/64，支持广播 | `source/experimental/math/complex_v3/README.md`、`op_kernel/complex_v3.h` |

组合方式：`real = exp(a)·cos(b)`、`imag = exp(a)·sin(b)`，再用 `ComplexV3` 拼成复数为输出。代价是 4 个算子、额外的 GM 读写，且不是单 kernel 融合；官方禁止“现成复数 exp 接口”，但不禁止用实数 exp/sin/cos + 复数布局组合。该路径需本机验证 fp16 极值/精度与 `atol=rtol=1e-3`。

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
| ComplexV3（910B real/imag→complex，支持广播） | `source/experimental/math/complex_v3/op_kernel/complex_v3.h:1-535`、`op_host/complex_v3_def.cpp:1-59` |
| Exp/Sin/Cos（910B 基础件） | `source/experimental/math/exp/README.md`、`sin/README.md`、`cos/README.md` |

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
- 对每个 query 扫描 N 个点，`d² = (cx-x)² + ...`；源码命中条件 `d²==0 || minR² <= d² < maxR²`；**官方只有 `d² < radius²`**，两者不同；
- 上游核对：AnyGrasp SDK `ball_query_gpu.cu`（commit `b8eaafc`）使用严格 `d² < radius²`、k 升序 first-K、不足重复首个命中、零命中全 0；官方单半径语义与它一致。详见 [算法参考核对](../../../docs/research/MFG2026_ALGORITHM_REFERENCES_20261117.md#3-ballquery严格--与-first-k-已确认)。
- 命中按 k 顺序写入 `idx`；不足 `sample_num` 用首命中填充；零命中写 0；
- 不涉及 atomic，输出确定。

### 4.3 StackBallQuery（910B）

- 输入 `xyz` 是 `(3, N_total)`，`center_xyz` 是 `(M, 3)`；`*_batch_cnt` 给每 batch 的点数；
- 按 center 分核/分段；每次把一个 batch 的 xyz 片段搬进 UB，计算到 center 的距离；
- 距离平方后与 `maxRadius²` 比较；命中写结果，达到 `sample_num` 提前停；
- 结果按段 `SendResultToGm`，未命中/不足用首索引补齐（`:384`）；
- FP16 路径把 `maxRadius²` 先转 half 再平方，和 FP32 路径有精度差异（`:43-49`）。官方要求 float16 输入先转 float32 再算距离/阈值，因此这是语义差异。

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

- **README/代码冲突**：`ball_query` 的 A2 √ 与 arch35-only 代码/config 不一致；必须以本机构建为准。官方要求单半径/严格 `<`/fp32 阈值，直接版的 min/max shell 与 `d²==0` 语义需要重写。
- **ComplexAbs A2 ×**：不要把 950 的 DAG 直接搬 910B；910B 可能需要 Vector/RegBase 的另一套实现。
- **StackBallQuery 语义**：输入是堆叠 2D + batch counts，不是原始 `[B,3,N]`；需要适配层。
- **零命中**：输出全 0 与“命中第 0 点”不可区分，题面若要求区分需另加 mask。
- **FP16 半径**：`maxRadius²` 先转 half 可能改变边界命中；官方明确要求 float32 阈值语义，必须改。
- **ComplexV2**：只是布局构造，不含 abs/div/exp 数学。
- **ComplexExp 组合路径**：exp/sin/cos + ComplexV3 是 4 个算子的组合，不是单 kernel；额外 GM 往返和 fp16 中间精度会改变性能/误差。

## 7. 复用判断

**优先复用**：StackBallQuery 的堆叠布局/分段流式/提前停；ComplexAbs 的缩放+平方和；ComplexV2 的实虚交错布局；ComplexExp 的 `exp/sin/cos + ComplexV3` 组合路径。

**需要验证**：910B3 构建 `stack_ball_query`；直接 `ball_query` 是否能编 A2；ComplexAbs 的 910B 替代实现。

**暂不使用**：把 arch35 的 ComplexAbs/BallQuery 直接当 910B 模板；把 ComplexV2 当完整复数算子。

## 8. 下一步

1. 构建 `stack_ball_query`（`ascend910b` 配置），跑官方 GEIR 用例；
2. 用 CPU 参考对照索引/零命中/不足填充；
3. 试编 `ball_query` 的 A2，记录 README/config 冲突的实际结果；
4. ComplexAbs 官方 dtype 已确认是 complex64→float32；910B 需另写 Vector 版本，不能直接复用 arch35。
5. ComplexExp 先用 `exp/sin/cos + ComplexV3` 组合跑通 float32/complex64，再评估 fp16 和单 kernel 融合。
