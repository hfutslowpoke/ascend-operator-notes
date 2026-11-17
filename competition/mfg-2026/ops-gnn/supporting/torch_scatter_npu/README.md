# torch_scatter_npu

本仓库用于交付昇腾平台上的图算子源码与必要使用说明。

## 目录

- `Segment_Coo/`：本轮维护内容，提供 Segment COO 的 sum/add/mean/min/max PyTorch API、NPU host/kernel 源码、构建脚本和基础验收测试。
- `Segment_Csr/`：保留原目录，本轮不修改。
- `Model_Adaptation/`：独立的模型适配目录，本轮不修改。

Segment COO 当前面向 Ascend 910B/910B3（`dav-2201`）和匹配的 CANN、PyTorch、torch_npu 环境。Ascend 950 尚未完成本算子的构建与硬件验收，因此当前版本不声明支持 950。

构建、安装和调用方式见 [`Segment_Coo/README.md`](Segment_Coo/README.md)。详细设计文档、benchmark 脚本和性能结果属于本地验收材料，不纳入该代码仓库。

## 分支

- `original`：仓库改造前的原始存档。
- `main`：当前交付版本。

## License

仓库根目录原有内容沿用根目录 `LICENSE`。`Segment_Coo/` 中带 CANN Open Software License Agreement Version 2.0 版权头的代码按 [`Segment_Coo/LICENSE`](Segment_Coo/LICENSE) 授权。
