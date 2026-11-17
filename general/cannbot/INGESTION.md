# CANNBot 接入清单

本清单用于把 CANNBot 内容转成通用库条目。先小批量试接入，不要一次性导入全部页面。

## 1. 来源固定

- [ ] 记录 CANNBot 的访问方式（API/导出/页面）、版本、日期、维护者。
- [ ] 记录许可/使用条款；不确定时只做引用和链接，不复制原文。
- [ ] 如果是代码片段，固定 commit/版本，记录路径/行号/SHA256。
- [ ] 如果是纯文档，记录页面 ID/标题/版本/日期，避免“无版本引用”。

## 2. 分类

| 内容 | 去向 |
|---|---|
| 算子语义、shape/dtype/layout、tiling、实现机制 | `general/<family>/<entry>/README.md` |
| Ascend C API 约束、编译/调试命令 | `docs/TOOLS_AND_REPRODUCIBILITY.md` 或 `docs/` 参考页 |
| 跨算子调优经验 | `docs/experiences/<topic>.md` |
| 无算子内容的问答/索引页 | 不建条目，最多放 `docs/` 链接 |

## 3. 条目字段

- [ ] 语义：数学式、输入/输出、属性、边界。
- [ ] 环境：SoC、CANN、编译器、Python/torch 版本。
- [ ] 源码/API 位置：文件/符号/行号，或页面锚点。
- [ ] 机制：任务切分、布局、搬运、同步、数值策略；减少/增加了什么。
- [ ] 证据：作者/官方数字、计时口径、输入、硬件；没有就写“未披露”。
- [ ] 本机 910B3：构建/运行/精度/时延/失败码；没有就 `not-run`。
- [ ] 适用条件与负例。
- [ ] 复用判断：直接参考、需要移植、暂不使用。

## 4. 机器登记

1. 在 `index/catalog.json` 增加 `repositories` 或复用已有官方来源记录；
2. 增加 `implementations`，填 `operator_family`、`source_files`、`conditions`、`negative_cases`、`reuse_decision`、`local_status`；
3. 在 `general/<family>/<entry>/` 建 `README.md` 和 `evidence/`；
4. 运行：

```bash
python3 tools/build_index.py
python3 tools/verify_archive.py
git diff --check
```

## 5. 质量门槛

- 没有固定版本/来源的页面：不建条目。
- 只有 API 列表、没有算子语义：放参考区。
- 只有“建议”、没有实现/测量：标 `candidate`，不写“高性能”。
- 未确认许可：不复制原文，只引用链接和短摘要。
- 本机未验证：`local_status=not-run`，不要用作者数字替代。

## 6. 建议的首批试点

优先选与现有条目互补、且能本机 910B3 验证的主题：

1. 一个线性代数算子（例如 `eigh` / `solve`）的 API 约束与 tiling；
2. 一个 Transformer 融合算子（例如 SwiGLU/MatMul+激活）的 dtype/layout 约束；
3. 一个搬运/布局算子（例如 Concat/Transpose）的 UB 预算与对齐规则。

每批不超过 3 个条目；先证明“来源可固定、语义可写清、本机能跑或能明确失败”，再扩大。
