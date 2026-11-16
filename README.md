# 昇腾算子比赛源码与经验库

最初整理 2026 CANN 挑战赛上合赛区决赛 GroupedMatmulGelu。整理日期：2026-11-16。用于以后算子开发、设计对照和实验复现；后续已扩展到公开比赛与高性能算子仓库调研。

本库保存 **26 份公开最终提交 PR、248 个原始文件**，每个文件有 SHA256、Git blob ID 和固定提交来源。公开 PR 下载时仍为 open；公开提交、平台 Pass 与人工核验是不同证据，本文不把它们等同。

我们的实际榜单代表源码是 [`our_work/final_submission/kernel.asc`](our_work/final_submission/kernel.asc)，SHA256 `161ac0ff2ab1920ca3de3e4f5d8b1b4d19a9a720c2781dcbc5c065f989b0b887`，SID `6af7d84c6bd0f9be3716f4c6`。11 月 16 日只读查询为 **58.54 分、第 11，整包 Pass**。这份代码与部分较早交接文件所指的“最新母包”不同。

## 先看什么

换环境或交接给新开发会话时，先读 [私仓迁移与继续提交说明](docs/PRIVATE_REPO_HANDOVER.md)，其中包含现有内容、授权克隆、后续归档与 Git 提交流程。

1. [公开源码技术分析](docs/OPEN_SOURCE_REVIEW.md)：领先方案的运行时有限值域查表、窄输出归约、实际分组调度、生产消费流水。
2. [比赛复盘](docs/RETROSPECTIVE.md)：取得收益的结构、没有兑现的方向、时间和证据管理教训。
3. [可迁移设计手册](docs/PATTERN_COOKBOOK.md)：如何从语义、代价、存储层次和最后消费者推导设计。
4. [工具与复现](docs/TOOLS_AND_REPRODUCIBILITY.md)：构建、测试、PMU、提交、协作工具及其适用边界。
5. [平台成绩和源码对应关系](docs/PLATFORM_EVIDENCE.md)：当前快照、缺失字段、历史最低值与当前集成的区别。
6. [数学顺序与赛事约束](docs/RULES_AND_MATH.md)：保留完整计算、单 kernel 和 Host/Device 边界。
7. [GitHub 使用说明](docs/GITHUB_QUICKSTART.md)：浏览、下载、以后追加资料的方法。

## 2026-11-16 新增：公开算子仓库调研

- [总览：范围、方法、12 个来源、深读 10 份实现、5 项待验证机制](docs/research/OPEN_SOURCE_REPOSITORY_SURVEY_20261116.md)
- [本机 Ascend 910B3 复测记录](docs/research/LOCAL_910B_VALIDATION_20261116.md)：GroupedTopk/Sinkhorn 通过，FusedMoE 编译失败，52Xayah 负例复现
- [机器可读来源登记](catalog/external-repositories.json)：固定 commit、源码路径/行号、环境、证据、负例、复用判断
- 深读报告：
  - [Hansen 2026 电力行业算子归档](docs/repositories/Hansenwuwuwu__ascend-c-competition-archive.md)
  - [KernelSwift 2026 Ascend 910B](docs/repositories/mantuoluozk__kernelswift-2026-triton-kernels.md)
  - [Mamba-2 AscendC](docs/repositories/So-cean__mamba-ascendc.md)
  - [vLLM-HUST Linear+SwiGLU](docs/repositories/vLLM-HUST__vllm-hust-opset.md)
  - [52Xayah Triton 负例](docs/repositories/52Xayah__triton-ascend-kernels.md)
- 跨实现经验：
  - [工作量与固定开销](docs/experiences/work_reduction_vs_fixed_overhead.md)
  - [布局与缓冲生命周期](docs/experiences/layout_and_buffer_lifecycle.md)
  - [数值与选择性精度](docs/experiences/numerical_selective_precision.md)
  - [形状专用化与回退](docs/experiences/shape_specialization_and_fallbacks.md)
- 纯 Triton 任务已在本机 910B3 复测；未在本机编译 Ascend C 自定义算子，未提交比赛平台。作者数字与本机数字分别标注，不能横比。

## 文件布局

| 目录 | 内容 |
|---|---|
| `third_party/shanghe_final/pr-*` | 公开参赛队提交的源码、README、附带文档及逐文件来源 |
| `third_party/official` | 固定版本的题目、提交规则、原始 LICENSE |
| `catalog` | 全部 PR 索引、文件哈希、我们最终版及候选的来源；`external-repositories.json` 为外部调研登记 |
| `docs/research` | 公开仓库调研总览、方法、证据局限与下一批建议 |
| `docs/repositories` | 逐个外部仓库的源码深读报告 |
| `docs/experiences` | 跨实现的可迁移机制、适用条件与负例 |
| `our_work/final_submission` | 我们实际最终提交代码及当时的构建/验证驱动 |
| `our_work/components` | 选中组件与未晋升候选，状态详见索引和分析 |
| `our_work/evidence` | 精选 PMU、对照结果、负实验、最终提交凭据 |
| `tools/competition` | 当时的提交、排名快照、日志协调、NPU 串行锁工具 |
| `tools/analysis` | 性能、数值、ELF、组件提取、分层模型工具 |

## 验证与使用

先运行 `python3 tools/verify_archive.py` 检查来源和完整性。源码收集阶段没有重新运行第三方代码；工具为历史快照，部分路径需要迁移。完整比赛环境、SDK/驱动、输入数据与二进制未打包，不能保证在任意环境直接重现平台耗时。

本库不包含账号密码、访问令牌、浏览器 Cookie、旧输出缓存、NPU 构建产物或大体积 profiler 原始目录。第三方源码保留原作者和提交信息，使用前查看 [第三方来源与许可说明](THIRD_PARTY.md)。

