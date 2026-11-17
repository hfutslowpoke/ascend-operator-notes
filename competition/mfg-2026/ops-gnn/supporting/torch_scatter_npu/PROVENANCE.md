# Provenance: HIT1920/torch_scatter_npu

- 上游：https://gitcode.com/HIT1920/torch_scatter_npu
- 固定 commit：`0199ab0d836ea07e26d2914179a22bf3c793b62e`（2026-09-09）
- 许可证：根目录 LGPL-3.0；`Segment_Coo/` 为 CANN Open Software License Agreement 2.0。
- 选中原因：`Segment_Csr/SegmentMinCsr` 是一份独立的 A2/A3（910B/910C）msOpGen 工程，含 host tiling、kernel、AclNN 调用示例；`Segment_Coo` 的 arch2201 kernel 提供 segment min/max + arg 的另一套实现。
- 只作补充参考，不是官方题解：
  - `indptr` 为 int32，官方要求 int64；
  - 只输出 `out`，没有官方要求的 `arg_out`；
  - 空段填 dtype 最大值，官方要求 `out=0`、`arg_out=E`；
  - `Segment_Coo` 相等极值保留最后一次出现，官方要求首遇 tie-break。
- 本地副本共 15 个文件，路径与上游一致；逐文件 SHA256、URL、行号在 `index/catalog.json` 的 `MFG2026-GNN-SEGMENT-MIN-CSR.supporting_sources` 与 `ops-gnn/source.json` 中。
- 未在本机 910B3 构建/运行；不能把上游 README 的 A2/A3 支持当作本机通过。
