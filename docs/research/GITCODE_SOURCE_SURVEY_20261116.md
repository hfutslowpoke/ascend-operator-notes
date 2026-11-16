# GitCode 来源补充调研（2026-11-16）

状态：GitCode 直连可用；已固定 2 个官方仓、深读 3 份新实现（GDN `recompute_w_u`、S9 `Concat`、批量 `LinalgLstsq`），并在本机 910B3 上完成 1 份正向验证、1 份编译/运行负例、1 份离线布局验证。未向比赛平台提交，未修改 ACW 框架或当前算子工程。

## 1. 为什么补这一轮

第一批调研记录 GitCode `git clone` 返回 `gnutls_handshake() failed`，因此只用了 GitHub 来源。任务书明确要求核对：

- `https://gitcode.com/cann/cann-ops-competitions` 及现存索引引用的公开 PR；
- “搜索引擎、榜单与文章用于发现，技术结论回到原始源码”。

本轮定位到失败原因是环境代理：`https_proxy=http://127.0.0.1:29670` 能访问 GitHub，但访问 GitCode 会超时；**去掉代理直连 GitCode 正常**。

```text
curl --noproxy '*' https://gitcode.com/cann/cann-ops-competitions        # 200
env -u https_proxy ... git ls-remote https://gitcode.com/cann/cann-ops-competitions.git
5126fead3327c7544ed4becff93775505071334c  HEAD
```

因此上一轮“GitCode 没有可抓取内容”的结论不成立；按目标要求，不能直接收尾，必须把 GitCode 合格来源纳入知识库。

## 2. 搜索范围与固定 commit

| 来源 | 类型 | 固定 commit | 树文件数 | 本轮决定 |
|---|---|---|---:|---|
| [`cann/cann-ops-competitions`](https://gitcode.com/cann/cann-ops-competitions) | CANN 官方竞赛仓：2026 天梯赛、S8/S9 算子挑战赛、高校/行业赛 | `5126fead3327c7544ed4becff93775505071334c` | 10,796 | **深读 2 份**：`recompute_w_u`、S9 `Concat` |
| [`cann/elec-ops-inspection`](https://gitcode.com/cann/elec-ops-inspection) | 2026 电力行业算子专场官方仓 | `dfe55921db351f71cfb8c96f795fd4472bd6bd47` | 1,341 | **深读 1 份**：`LinalgLstsq` |
| [`cann-competition`](https://gitcode.com/cann-competition) 组织 | 赛事报名/题面/赛区页 | 各仓独立 | — | 只用于发现；不计实现 |
| S9 选手个人仓（如 `yuhongming-2026/ascend-s9-greater`） | 选手自建仓 | 未固定 | — | 候选；本轮未深读 |
| S9 `IndexAdd`（VibeThinking） | 官方仓内提交 | 同上 | — | 候选；只读 README，无性能数字 |

搜索词（实际使用或组合）：

```text
site:gitcode.com 昇腾 算子 比赛 CANN 2026
site:gitcode.com "cann-ops-ladder-2026" OR "operator-challenge" Ascend C 算子
gitcode CANN 算子挑战赛 2026 参赛 源码 Ascend C 仓库
gitcode recompute_w_u / Concat / IndexAdd / LinalgLstsq
```

GitCode 的网页搜索/目录页可读，但代码固定 commit 以 `git ls-remote` + `raw.gitcode.com` 为准；没有使用登录态、token 或私有接口。

## 3. 筛选结果

### 3.1 深读

| 实现 | 赛事/目录 | 算子族 | 证据 | 本机 |
|---|---|---|---|---|
| `recompute_w_u`（Delicate02） | 2026 天梯赛 7 月 | GDN 注意力重算 / GQA GEMM | CANNJudge 22/22、第 1 名、逐 case µs（`O+A`） | 编译成功；运行 `507057` |
| `Concat`（BatchLions） | S9 算子挑战赛 | 多输入拼接 / 布局搬运 | 最终第十名、五档 348.33 µs（`O+A`）；离线测试 | 离线 215,242+32,681+8,247 项通过；未上 NPU |
| `LinalgLstsq`（XDU_XDU-BTD） | 2026 电力行业预赛 | 批量最小二乘 / QR+LU | 作者说明无性能数字（`S+L+A`） | 2 行兼容 patch 后编译成功；16/50/64 单 batch PASS；多 batch/矩形/200/256 失败 |

### 3.2 候选/未深读

- S9 `IndexAdd`（VibeThinking）：5 种 dtype、重复索引、行缓存/分组累加；只有实现说明，没有性能数字。
- S9 `Greater`、`Transpose`、`SquareSumV1`：机制简单/偏搬运，暂不作为高性能主来源。
- 天梯赛其它月份（June/August/September）：部分只有题面/`.gitkeep`；需要逐月筛“有提交+有日志”的目录。
- 电力赛其它算子：`pinv`、`eigh`、`det`、`solve`、`swiglu` 等；与既有线性代数/融合主题重叠，留作下一轮。

### 3.3 不计入

- `cann/cann-ops-competitions` 中既有 26 份 GroupedMatmulGelu PR：已在私仓归档，本轮不重复计数。
- `cann/elec-ops-inspection` 与 `Hansenwuwuwu/ascend-c-competition-archive`：同一赛事的官方仓/归档关系，不重复计为独立算法来源。
- fork、镜像、二次搬运、只有题面没有源码/证据的目录：不计。

## 4. 计数与去重

本轮之后：

| 口径 | 数量 |
|---|---:|
| 登记仓库记录 | **14**（原 12 + GitCode 2） |
| 独立算法来源 | **12**（GitCode 2 个记录分别与既有官方基线/Hansen 上游重叠，不新增独立来源） |
| 深读仓库记录 | **9**（原 7 + GitCode 2），其中独立深读来源仍为 7 |
| 深读实现 | **18**（原 15 + 3） |
| 本机运行/失败/构建记录 | 见 `our_work/evidence/local-910b-validation-20261116.json`（新增 3 个 PASS、2 类失败、2 个构建） |

这样既记录了 GitCode 的实际内容，也避免把同一官方仓/同一赛事归档重复算成“新独立来源”。

## 5. 本机验证摘要

设备：Ascend 910B3（容器内 logical device 0，物理 NPU 2）；CANN 9.1.0；gcc 11.4.0；cmake 3.22.1；所有设备运行经 `bin/npu-lock`。

### 5.1 正向

`LinalgLstsq`（XDU_XDU-BTD）在本地副本加 2 处 `template` 兼容性关键字后编译成功：

| shape | 结果 | best | max abs |
|---|---|---:|---:|
| 16×16, K=1, b=1 | PASS | 0.0265 ms | 1.192e-07 |
| 50×50, K=2, b=1 | PASS | 0.0301 ms | 3.576e-07 |
| 64×64, K=2, b=1 | PASS | 0.0395 ms | 3.576e-07 |

这是 GitCode 来源在本机 NPU 上的**正向**验证；时延是“launch + sync”单次最好值，不是作者成绩，也不是稳定中位数。

### 5.2 负例/边界

- `LinalgLstsq`：16×16 b=2、50×50 b=4、64×32、64×50、100×64 报 `507035`（vector core exception / invalid GM address）；200×200、256×256 报 `507057`（aicore exception / MTE invalid GM address）。
- `recompute_w_u`：CANN 9.1.0 编译成功，但 5 个 shape（T=256/10016，H=8，Hg=2/8，D=128/256）运行均报 `507057`；显式给 10 个 tiling key 加 `KERNEL_TASK_TYPE` 后仍失败。
- `Concat`：未构建 NPU 包；离线测试 215,242 + 32,681 + 8,247 项通过，只证明地址/布局/UB 账，不证明设备精度/性能。

### 5.3 结论

- GitCode 确有合格来源，且至少一份实现（Lstsq 小尺寸单 batch）可在本机 910B3 上跑通；
- **编译成功 ≠ 可运行**：recompute_w_u 是本轮最明显的反例；
- **小 shape PASS ≠ 全 shape PASS**：Lstsq 的多 batch/矩形/200/256 在 CANN 9.1 上崩溃；
- 不能把作者的 CANNJudge/榜单数字当成本机复现，也不能把本机失败直接归因于作者实现错误；CANN 版本、shape 布局、workspace/残差约定都可能是原因。

## 6. 对知识库的增量

新增文档：

- [GitCode `cann/cann-ops-competitions` 报告](../repositories/GitCode__cann-ops-competitions.md)
- [GitCode `cann/elec-ops-inspection` 报告](../repositories/GitCode__elec-ops-inspection.md)
- [GitCode 机制经验](../experiences/gitcode_operator_mechanisms.md)

新增机器登记：`catalog/external-repositories.json` 增加 2 个 repository 记录、3 个 implementation 记录、对应性能/本机验证字段。

## 7. 仍未验证 / 下一轮

1. `Concat` 的 NPU 构建/运行：需要动态张量列表 ACLNN 驱动和官方 case；离线测试不能替代。
2. `recompute_w_u` 在 CANN 8.5.0 的复跑：本机只有 9.1.0；需要匹配环境或定位 9.1 的 ABI/同步差异。
3. `LinalgLstsq` 多 batch/矩形/200/256：需要确认 `residuals` 布局、workspace 和 CANN 9.1 的 shared-QR 路径；当前只验证了 16/50/64 单 batch。
4. S9 `IndexAdd`、`Greater`、`Transpose`、`SquareSumV1`：按“有源码+有测量”筛，不按目录数量凑数。
5. 电力赛 `pinv/eigh/det/solve`：与既有 SVD/线性代数主题合并看，优先找有原始日志的实现。
6. 910C / profiler/PMU：仍无数据。
