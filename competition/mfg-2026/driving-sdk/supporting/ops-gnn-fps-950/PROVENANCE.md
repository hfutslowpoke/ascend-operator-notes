# Provenance: gcw_i0yFZkCV/ops-gnn-fps-950

- 上游：https://gitcode.com/gcw_i0yFZkCV/ops-gnn-fps-950
- 固定 commit：`9d8da03b18d0b7bc70bfbab47f10c9361c942b91`（2026-09-23）
- 许可证：CANN Open Software License Agreement Version 2.0。
- 选中原因：该仓的 `test/fps/README.md` 与 `compatibility/fps/` 记录了 torch_cluster FPS 的 FP16 逐维舍入、mod-256 lane tie-break、ratio/count 边界差异，可作为官方制造题 FPS 的精度/tie 对照材料。
- 重要边界：
  - 该仓是 **Ascend 950（arch35）** 社区任务，不是 910B 实现；
  - 接口是 `torch_cluster.fps(src, ratio, random_start, ptr)`，不是官方制造的 `furthest_point_sampling(points, npoint)`；
  - 官方制造题使用 `npoint`、首点固定 0、距离并列取最小输入索引；不能照搬 ratio/ceil 或 torch_cluster 的 mod-256 lane 规则。
- 本地副本共 5 个文件（LICENSE、两份 README、golden、frozen oracle test）；逐文件 SHA256/URL/行号在 `index/catalog.json` 的 `MFG2026-FPS.supporting_sources` 与 `driving-sdk/source.json` 中。
- 未在本机 910B3 构建/运行；本机也没有对应的 950 环境。
