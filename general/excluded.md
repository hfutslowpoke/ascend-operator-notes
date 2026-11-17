# 通用库淘汰/未计入记录

本页记录“非竞赛来源、但不足以形成高性能算子条目”的仓库。机器记录仍在 [../index/catalog.json](../index/catalog.json)。

## asfsdsf/S6（Fmax）

| 项 | 值 |
|---|---|
| URL | https://github.com/asfsdsf/S6 |
| 固定 commit | `0454168359b4749058b61432bf15b3945b44efa9` |
| 上下文 | 单算子 Fmax（比赛来源不明） |
| 许可 | MIT |
| 内容 | 6 个 Fmax 源文件和 LICENSE |

淘汰原因：

- 没有 README、测试、输入/边界说明；
- 没有性能数据，也没有可定位的 tiling/优化说明；
- 比赛来源不明确，不能归入竞赛库；
- 单算子、代码量小，无法支撑“高性能”判断。

保留价值：仅作为 Fmax/广播逐元素实现的代码样本；不计入实现数量。

## 明确未计入的候选

| 仓库 | 原因 |
|---|---|
| [WildPickledPepper/ascend-rowcol-int8-quant](https://github.com/WildPickledPepper/ascend-rowcol-int8-quant) | 只有任务书、benchmark/golden 脚本和配置，未看到 kernel 源文件 |
| [m-rui001/ascend-c-operator-dojo](https://github.com/m-rui001/ascend-c-operator-dojo) | 59 轮学习/对照笔记与检查表，偏训练材料，不是独立高性能实现 |
| [CosmosMind-ai/PyPTO-Ops](https://github.com/CosmosMind-ai/PyPTO-Ops) | PyPTO DSL 的 FlagGems 风格算子库；与当前 Ascend C/Triton-Ascend 目标差异大，留作后续专题 |
| [Luoshu-Intelligent-Computing/ascend-310p-kernels](https://github.com/Luoshu-Intelligent-Computing/ascend-310p-kernels) | 面向 310P，不是 910B/910C 目标 |

这些不建条目，也不计入 14 个仓库记录；如果后续出现固定源码、文档和可复现证据，再重新评估。
