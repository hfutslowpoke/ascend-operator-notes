# 知识库架构（v2，2026-11-17）

本库是 ACW 的外部算子知识层，不再按“第几批 / 来自哪个网站”平铺，而是分成两个一级知识库：

- `competition/`：竞赛中的高性能 PR/实现，按赛事组织；
- `general/`：CANNBot 及其它非竞赛来源，按算子族组织。

两个库共享同一套条目契约、证据等级、本机验证协议和机器索引。原先的 `third_party/`、`our_work/`、`catalog/`、`docs/repositories/` 已迁移；旧路径与新路径的对应见 [迁移记录](MIGRATION_20261117.md)。

## 1. 为什么分两层

竞赛实现和生产/通用实现的可复用性不同：

| 维度 | 竞赛知识库 | 通用算子库 |
|---|---|---|
| 主要价值 | 特定题面/隐藏用例下的极致机制、分路、tiling 经验 | 长期稳定的语义、接口、数值和跨模型复用 |
| 约束 | 赛题规则、固定 shape 族、平台评分口径 | 框架接口、dtype/layout、生产可维护性 |
| 典型风险 | 过拟合测试点、规则变化、只对某代 NPU 有效 | 上游版本漂移、依赖链重、缺少逐点证据 |
| 复用方式 | 提炼机制，不直接提交竞赛代码 | 直接参考接口/算法，或移植到 ACW 算子 |

把两者混在一起会导致：竞赛 PR 的“榜单第十/第一名”被误当成通用性能证明；生产库的稳定接口又被竞赛特化代码淹没。分两层后，索引可以按层过滤，ACW skill 也能先判断“我要找的是竞赛机制还是通用实现”。

## 2. 目录结构

```text
.
├── README.md                     # 全库入口
├── THIRD_PARTY.md                # 来源、许可、复制边界
├── MANIFEST-SHA256.json          # 全库文件哈希
├── competition/                  # 竞赛知识库
│   ├── README.md                 # 竞赛库入口、准入规则
│   ├── catalog.json              # 竞赛层机器索引（生成）
│   ├── excluded.md               # 竞赛层淘汰记录
│   ├── grouped-matmul-gelu/      # 2026 CANN 上合决赛基线
│   │   ├── README.md
│   │   ├── prs/                  # 26 份公开 PR 快照
│   │   ├── official/             # 题面、规则、LICENSE
│   │   ├── our-submission/       # 我们的 kernel.asc 与驱动
│   │   ├── components/           # 历史候选组件
│   │   ├── analysis/             # 公开源码分析、复盘、平台证据
│   │   ├── catalog/              # 26 PR 的机器索引
│   │   └── evidence/             # PMU、负实验、平台快照
│   ├── cann-ops-competitions/    # GitCode CANN 竞赛仓
│   ├── elec-ops-inspection/      # GitCode 电力行业仓
│   ├── hansen-2026-power/        # 电力行业赛公开归档
│   ├── kernelswift-2026/         # KernelSwift 2026
│   └── mfg-2026/                 # 制造行业专场（15 题源码地图）
├── general/                      # 通用算子库
│   ├── README.md
│   ├── catalog.json              # 通用层机器索引（生成）
│   ├── excluded.md
│   ├── linear-algebra/           # be4ver-svd、ascend-eigh
│   ├── attention/                # mamba-ascendc
│   ├── transformer/              # vllm-hust-opset
│   ├── equivariant/              # parity-ictc-o3
│   ├── elementwise/              # 52xayah-triton-kernels
│   ├── benchmarks/               # TritonAscendBench、triton-operator-benchmarks
│   └── cannbot/                  # CANNBot 接入位与准入规则
├── index/                        # 统一索引
│   ├── README.md
│   ├── catalog.json              # 唯一机器可读主索引
│   ├── entry-schema.json         # 条目契约
│   ├── by-family.md              # 按算子族（生成）
│   └── by-evidence.md            # 按本机证据（生成）
├── skills/                       # 给 ACW/agent 的工作流
│   ├── README.md
│   ├── operator-pattern-lookup/
│   ├── competition-repo-intake/
│   └── ascend-910b-local-validation/
├── docs/                         # 方法论、经验、验证记录
│   ├── ARCHITECTURE.md
│   ├── MIGRATION_20261117.md
│   ├── PATTERN_COOKBOOK.md
│   ├── TOOLS_AND_REPRODUCIBILITY.md
│   ├── PRIVATE_REPO_HANDOVER.md
│   ├── GITHUB_QUICKSTART.md
│   ├── experiences/
│   ├── research/
│   └── validation/
└── tools/
    ├── build_index.py            # 生成层索引、source.json、by-*.md
    └── verify_archive.py         # 哈希、JSON、Python、链接、敏感模式校验
```

## 3. 条目契约

每个准入条目必须是一个目录，至少包含：

```text
<entry>/
├── README.md      # 人读：语义、环境、源码定位、机制、证据、负例、复用判断
├── source.json    # 机器读：固定 commit、许可、实现、精确源码路径/行号/SHA256
└── evidence/      # 原始日志、结果 JSON、构建元数据；没有则为空并说明
```

### 3.1 README.md 的固定章节

深读条目建议按以下顺序写；候选条目可以只写已有信息，但必须标明“未深读/未验证”：

1. 来源登记：URL、作者/团队、赛事或上下文、访问日期、固定 commit、默认分支、许可证、本地缓存、去重关系。
2. 语义与约束：数学式、shape/dtype/layout、属性、边界、测试范围；未知项写“未知”。
3. 源码定位：文件路径、符号、固定 commit 的行号、Host 分路、kernel 入口。
4. 机制：任务切分、布局、搬运、片上驻留、缓冲生命周期、同步、融合、数值策略；减少了什么工作/等待，增加了什么代价。
5. 作者证据：baseline、计时边界、单位、warmup/重复/统计、输入、硬件、原始位置；未披露的字段不补造。
6. 本机验证：**只写 910B3** 的结果；命令、日志、shape、精度、时延、失败码。
7. 适用条件与负例：容量/工作量门槛、边界输入、退化、版本敏感、不可外推项。
8. 复用判断：优先复用、需要验证、暂不使用；对应 ACW 的哪个阶段。

### 3.2 source.json

由 `tools/build_index.py` 从 `index/catalog.json` 生成，避免手抄漂移。字段见 [index/entry-schema.json](../index/entry-schema.json)。核心字段：

- `entry`、`tier`、`repository`、`commit`、`license`；
- `implementations[]`：算子、族、源码文件（路径、行号、SHA256、原始 URL）；
- `local_validation`：本机状态、环境、日志路径、结论；
- `generated_from`：固定指向 `index/catalog.json`。

没有外部仓库记录的 `competition/grouped-matmul-gelu` 使用手写 `source.json`，指向 `catalog/public-final-sources.json` 和 `catalog/our-work.json`。

### 3.3 evidence/

- 原始日志不改写；文件名带来源/日期/形状，例如 `gitcode_lstsq_M64_N64_K2_B1_f.log`。
- 结果 JSON 记录设备、CANN、命令、shape、计时口径、best/median、失败码。
- 不保存账号、token、私钥、完整 PROF、NPU 二进制或大体积构建目录。
- 失败也是证据；`507035`/`507057` 这类负例必须保留。

## 4. 两层准入规则

### 4.1 competition/

准入：

- 公开竞赛/榜单/官方归档中的高性能 PR 或实现；
- 有固定 commit、可定位源码、作者/平台证据或本机复现；
- 与既有来源不是 fork/镜像/重复搬运；
- 能提炼出可迁移机制，而不是只有测试脚手架。

典型拒绝：只有 aclnn 测试代码、没有 kernel/时延的“参赛模板”；只有题面、没有提交的目录；无法固定 commit 的 fork。

### 4.2 general/

准入：

- 生产库、框架算子、CANNBot 知识、独立算子仓库；
- 有算子语义/实现/接口/数值策略，且能定位源码或权威文档；
- 有作者性能数据或本机复现；两者都没有时只能列为候选；
- CANNBot 的工具/API 页面只作参考，不计入“高性能算子实现”数量。

### 4.3 excluded

淘汰记录不建条目目录，只在 `competition/excluded.md` 或 `general/excluded.md` 写：URL、固定 commit、淘汰原因、是否保留为负例。机器记录仍在 `index/catalog.json` 的 `repositories` 中，`status=excluded`。

## 5. 证据等级

| 级别 | 含义 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| `S` | 固定源码路径/commit | 代码确实存在、位置可追溯 | 正确性、性能 |
| `L` | 精确行号/符号 | 机制定位可复查 | 作者/平台结论 |
| `A` | 作者文档/README/报告 | 作者声称的语义、环境、数字 | 本机可复现 |
| `O` | 官方榜单/平台记录/公示 | 平台侧成绩或状态 | 通用性、规则合规 |
| `V` | 本机 910B3 构建/运行 | 当前环境下的可执行性/精度/时延 | 910C、其它 CANN、生产稳定性 |
| `N` | 未验证 | 明确缺口 | 任何性能/正确性结论 |

组合写法：`S+L+A` 表示源码、行号、作者说明都有；`O+A` 表示平台成绩 + 作者说明；`V` 表示本机结果。**`O+A` 不是 `V`，`V` 也不自动升级为 `O`。**

## 6. 本机验证协议（910B3）

本环境只有 Ascend 910B3，没有 910C。所有设备运行必须经 `bin/npu-lock`：

```bash
cd /workspace/acw
bin/npu-lock bash -lc '<命令>'
```

记录：

- 物理设备与 ACL logical device（本机：物理 NPU 2，容器内 logical 0）；
- CANN、驱动、gcc、cmake、torch/torch_npu/Triton-Ascend 版本；
- 构建命令、产物路径、是否有本地兼容 patch（patch 只放工作区外副本，不写入第三方仓库）；
- 输入 shape/dtype/布局、参考实现、误差阈值；
- 计时口径：ACLNN launch+sync、device event、msprof 或平台 Task Duration，不能混用；
- warmup/重复/统计：best-of-N 与 median 必须写清；首次运行含懒加载时不能当稳定值；
- 失败码、fault kernel、是否尝试过兼容性 patch。

硬规则：

1. **编译成功 ≠ 可运行**。`recompute_w_u` 在 CANN 9.1.0 编译成功但 5 个 shape 全部 `507057`。
2. **小 shape PASS ≠ 全 shape PASS**。`LinalgLstsq` 16/50/64 单 batch 通过，多 batch/矩形/200/256 失败。
3. **910B3 结果不外推 910C**。没有 910C 数据就写 `N`。
4. **作者数字与本机数字分开写**。不同设备/版本/参考实现不能直接算加速比。
5. **负例必须保留**。失败日志、错误码、尝试过的 patch 都是后续判断的依据。

## 7. 索引与生成

`index/catalog.json` 是唯一机器可读主索引，包含：

- `repositories[]`：20 个仓库记录，带 `tier`、`entry_path`、固定 commit、许可、状态、去重关系；
- `implementations[]`：33 份实现，带算子族、源码文件/行号/哈希、作者证据、适用条件、负例、复用判断、本机验证；
- `excluded_candidates[]`：未计入的候选及原因。

运行：

```bash
python3 tools/build_index.py
python3 tools/verify_archive.py
```

`build_index.py` 生成：

- `competition/catalog.json`、`general/catalog.json`：按层过滤；
- 每个条目的 `source.json`；
- `index/by-family.md`、`index/by-evidence.md`；
- `index/cards.json`、`index/cards.md`：每份实现的阅读卡（一句话机制、源码入口、本机状态、边界）。

人类/agent 的学习顺序见 [AGENT_READING_GUIDE.md](AGENT_READING_GUIDE.md)：先读 cards 决定要不要深入，再读条目 README 的机制/负例，最后核对 `source.json` 和 `evidence/`。

不要手改生成文件；要改就改 `index/catalog.json`，再重新生成。

## 8. 给 ACW 的 skills

`skills/` 下是可直接加载的工作流，不是算子源码：

- `operator-pattern-lookup`：按瓶颈/算子族查机制，先看证据等级和负例；
- `competition-repo-intake`：竞赛仓库筛选、固定 commit、去重、深读、建条目；
- `ascend-910b-local-validation`：`bin/npu-lock`、环境记录、计时口径、负例保留。

Skill 的输出应引用条目 `README.md`、`source.json` 和 `evidence/`，不要复制大段第三方源码，也不要自动向竞赛平台提交。

## 9. 来源与许可

- 优先“引用 + 固定 commit + 路径/行号 + 哈希”，不复制第三方源码。
- `competition/grouped-matmul-gelu/prs/` 是例外：26 份公开 PR 快照已按固定 commit 归档，保留原始 LICENSE/PROVENANCE；这是历史基线，不因重构而重新分发。
- 未声明许可证的仓库（Hansen、KernelSwift、52Xayah、Be4ver）只做引用、行号定位和分析，源码缓存留在工作区外。
- 本地兼容 patch 只放工作区外副本，不写入第三方仓库，也不写入本库。
- 作者成绩、官方公示分数、本机分析都不是规则合规或通用正确性证明。

## 10. 后续演进

1. **CANNBot 接入**：先确认访问方式、版本、许可和可引用粒度；只把“算子语义/实现/性能”内容建成条目，工具/API 页面放参考区。
2. **910C**：新增独立验证记录，不覆盖 910B3 结论；没有设备就保持 `N`。
3. **合并到 main**：当前重构在 `architecture/two-tier-knowledge-base-20261117` 分支，验证通过后再决定是否合并；`main` 保留历史 GroupedMatmulGelu 基线。
4. **更多竞赛**：按 `competition/<赛事>/<条目>` 扩展，不把新赛事塞进 GroupedMatmulGelu 目录。
5. **更多通用算子**：按 `general/<算子族>/<条目>` 扩展；跨家族机制写进 `docs/experiences/`，不要重复写在每个条目里。
