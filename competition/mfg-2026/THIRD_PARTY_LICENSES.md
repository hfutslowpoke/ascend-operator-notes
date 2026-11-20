# 制造专场第三方许可证映射

本目录的 `source/` 和 `supporting/` 是上游固定 commit 的**片段快照**，不是整仓再分发。每个片段保留上游版权头、路径、commit 和 SHA256；本页只做许可证汇总，不替代各上游 LICENSE。

## 1. 源码快照

| 本地目录 | 上游 | 固定 commit | 许可证 | 许可证全文 |
|---|---|---|---|---|
| `ops-fft/source/` | <https://gitcode.com/cann/ops-fft> | `65f5ff1dee32802562a9412789b0c7719ef99269` | CANN Open Software License 2.0 | [`LICENSE-CANN-Open-Software-2.0.txt`](LICENSE-CANN-Open-Software-2.0.txt) |
| `ops-gnn/source/` | <https://gitcode.com/cann/ops-gnn> | `ea0b18a1d3e97f1ae2315df69338bb51c9f96466` | CANN Open Software License 2.0 | 同上 |
| `ops-math/source/` | <https://gitcode.com/cann/ops-math> | `3b2baac63624c32e03781bd9f0290f1a4584f0c7` | CANN Open Software License 2.0 | 同上 |
| `ops-solver/source/` | <https://gitcode.com/cann/ops-solver> | `739775a145d7789693f97758f6771bb1b3eb6a8f` | CANN Open Software License 2.0 | 同上 |
| `ops-sparse/source/` | <https://gitcode.com/cann/ops-sparse> | `8fae00b8832bf31f1ce26059680bce67c6026c4f` | CANN Open Software License 2.0 | 同上 |
| `driving-sdk/source/` | <https://gitcode.com/Ascend/DrivingSDK> | `d10b13c95ec7f25a54a4aa9677c436cde30ff6e2` | Apache-2.0 | [`LICENSE-APACHE-2.0.txt`](LICENSE-APACHE-2.0.txt) |

## 2. supporting 快照

| 本地目录 | 上游 | 许可证 | 本地许可证 |
|---|---|---|---|
| `ops-gnn/supporting/torch_scatter_npu/` | `torch_scatter_npu` 上游 | LGPL-3.0（根目录） | `ops-gnn/supporting/torch_scatter_npu/LICENSE` |
| `ops-gnn/supporting/torch_scatter_npu/Segment_Coo/` | 同上 | CANN Open Software License 2.0 | `.../Segment_Coo/LICENSE` |
| `driving-sdk/supporting/ops-gnn-fps-950/` | `ops-gnn` FPS 950 | CANN Open Software License 2.0 | `.../ops-gnn-fps-950/LICENSE` |

## 3. 使用边界

- 本库没有为整个集合设置统一许可证；各快照按其原始许可证处理。
- `source.json` 的 `license` 字段和 `local_path` 是逐文件定位入口；`PROVENANCE.md` 记录 supporting 快照的来源。
- 未声明许可证的仓库（Hansen、KernelSwift、52Xayah、Be4ver）没有源码进入本库，只做引用/行号分析。
- 如果你要再分发某个片段，请同时保留上游版权头、本页映射和对应许可证全文；不要只引用本库的中文分析。
