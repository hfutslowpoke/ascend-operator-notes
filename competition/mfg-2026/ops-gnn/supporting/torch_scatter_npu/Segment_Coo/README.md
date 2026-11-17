# Segment COO for Ascend NPU

该目录是可独立构建和打包的 Segment COO 交付单元，提供与 `torch_scatter` 风格一致的 PyTorch API：

```python
from torch_scatter_npu import (
    segment_coo,
    segment_sum_coo,
    segment_add_coo,
    segment_mean_coo,
    segment_min_coo,
    segment_max_coo,
)
```

`segment_min_coo` 和 `segment_max_coo` 返回 `(value, arg)`；通用 `segment_coo` 对 min/max 仅返回 value。`index` 必须为非降序的 int32/int64 张量。

## API 语义

- segment 维为 `index.dim() - 1`；`index` 的前缀维允许以大小 1 向 `src` 广播。
- 未传 `dim_size` 时由 `index.max() + 1` 推断；空 `index` 推断为 0。
- sum/add 的空段为 0；mean 的空段为 0。
- min/max 的空段 value 为 0，arg 为 `src.size(segment_dim)`；相等极值保留最后一次出现的位置。
- 传入 `out` 时原位更新并返回同一对象。sum/add 在初值上累加，mean 对初值与源值之和除以源元素数，min/max 将 `out` 作为初始候选。

示例：

```python
import torch
from torch_scatter_npu import segment_sum_coo

src = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
index = torch.tensor([0, 0], dtype=torch.int64, device="npu")
result = segment_sum_coo(src, index, dim_size=2)
# tensor([[4., 6.], [0., 0.]], device="npu:0")
```

## 支持范围

- 硬件：Ascend 910B / 910B3
- 架构：`dav-2201`
- Python：3.8 及以上
- 依赖：匹配版本的 CANN、PyTorch、torch_npu、CMake 3.18+
- 原生 dtype：bfloat16、float16、float32、int8、uint8、int16、int32
- 兼容 dtype：float64、int64 通过显式 CPU fallback

Ascend 950 尚未完成本算子的构建和硬件验收，当前版本不声明支持。

## 构建

激活安装了 PyTorch 与 torch_npu 的 Python 环境，然后执行：

```bash
cd Segment_Coo
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
PYTHON=/path/to/python bash scripts/build.sh
```

构建产物位于 `python/torch_scatter_npu/_pybind.so`。

## 安装与 wheel

```bash
PYTHON=/path/to/python bash scripts/install.sh
PYTHON=/path/to/python bash scripts/build_wheel.sh
```

wheel 输出到 `output/wheel/`。

## 验收测试

CPU API 回归：

```bash
PYTHON=/path/to/python bash tests/segment_coo/run_acceptance.sh
```

启用 NPU 用例：

```bash
SEGMENT_COO_TEST_NPU=1 \
SEGMENT_COO_TEST_NPU_DEVICE=7 \
PYTHON=/path/to/python \
bash tests/segment_coo/run_acceptance.sh
```

## 目录说明

- `csrc/npu/host/`：PyTorch/torch_npu host 接口和 tiling/launch 逻辑。
- `csrc/npu/kernel/segment/arch2201/`：sum、mean、min、max 四个独立 Ascend C kernel 入口。
- `csrc/npu/kernel/segment/common/`：共享归约实现与 tiling 数据结构。
- `python/torch_scatter_npu/`：公开 Python API。
- `scripts/`：构建、安装、wheel 打包脚本。
- `tests/segment_coo/`：可提交的 API 与功能验收测试。

详细设计、完整 benchmark 和性能结果仅作为本地验收材料保存，不包含在提交中。
