# CANNBot 知识接入位（待接入）

`general/cannbot/` 预留给 CANNBot Knowledge 及其它官方/社区知识源。当前**没有导入任何条目**，因此不计入 39 份实现，也不提供性能结论。

## 为什么单独放

CANNBot 的内容混合了三类信息：

1. **算子级知识**：语义、dtype/layout、tiling、Ascend C API 约束、性能建议；
2. **工具/API 参考**：环境配置、编译命令、调试接口；
3. **泛化经验**：常见错误、版本差异、调优 checklist。

只有第 1 类可以形成算子条目；第 2 类放 [../../docs/TOOLS_AND_REPRODUCIBILITY.md](../../docs/TOOLS_AND_REPRODUCIBILITY.md) 或参考区；第 3 类提炼进 [../../docs/experiences/](../../docs/experiences/)，不重复建条目。

## 接入状态

| 项 | 状态 |
|---|---|
| 访问方式 | 待确认（API/导出/人工整理） |
| 版本/日期 | 待固定 |
| 许可/再分发 | 待确认；未确认前只做引用，不复制原文 |
| 已导入条目 | 0 |
| 本机验证 | 无 |

## 下一步

按 [INGESTION.md](INGESTION.md) 的清单做一次小批量试接入：选 1 个有明确算子语义、能定位 API/实现、且允许引用的主题，建立 `general/<family>/<entry>/`，跑 `tools/build_index.py` 和 `tools/verify_archive.py`，再决定是否扩大。

## 不要做什么

- 不把通用 API 页面伪装成“高性能算子实现”；
- 不复制未确认许可的内部/付费内容；
- 不把 CANNBot 的建议直接写成“已验证性能”；
- 不在没有 910B3 结果时把 `local_status` 写成 `pass`。
