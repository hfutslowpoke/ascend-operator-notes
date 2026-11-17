## 概述

本样例基于 SegmentMinCsr 自定义算子工程，演示如何通过 `msOpGen` 生成简易算子工程、补充 host/kernel 实现，并使用 AclNN 两段式接口调用算子，完成数据生成、执行与结果校验的完整流程。算子规格如下表所示。

## 目录结构介绍

```
├── SegmentMinCsrFrameworkLaunch
│   ├── AclNNInvocationNaive      // 通过aclnn调用的方式调用SegmentMinCsr算子, 简化了编译脚本
│   ├── SegmentMinCsr             // SegmentMinCsr算子工程
│   ├── SegmentMinCsr.json        // SegmentMinCsr算子的原型定义json文件
│   └── install.sh                // 脚本，调用msOpGen生成简易自定义算子工程，并编译
```

## 算子描述
`SegmentMinCsr` 在每个 segment 上沿列维执行最小值归约，输出该最小值结果：

数值输出 `out`：
$$
\mathrm{out}[s, j] = \min_{\,i \in \{\mathrm{indptr}[s],\dots,\mathrm{indptr}[s+1]-1\}} X_{i, j}
$$

若 segment 为空且提供了 `optional_out`，则 `out[s, j] = optional_out[s, j]`；若未提供 `optional_out`，则 `out[s, j]` 填充该类型最大值。

## 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">SegmentMinCsr</td></tr>
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">src</td><td align="center">8 * 2048</td><td align="center">float32</td><td align="center">ND</td></tr>
<tr><td align="center">indptr</td><td align="center">5</td><td align="center">int32</td><td align="center">ND</td></tr>
<tr><td align="center">optional_out</td><td align="center">nullptr</td><td align="center">float32</td><td align="center">ND</td></tr>

<tr><td rowspan="2" align="center">算子输出</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">out</td><td align="center">4 * 2048</td><td align="center">float32</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">segment_min_csr</td></tr>
</table>

## 支持的产品型号

本样例支持如下产品型号：

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品 / Atlas 800I A2 推理产品


## 算子工程介绍

`SegmentMinCsr` 自定义算子工程目录结构如下：

```
├── SegmentMinCsr            // 算子源码
│   ├── op_host              // host侧实现文件
│   └── op_kernel            // kernel侧实现文件
```

`msOpGen` 可依据 `SegmentMinCsr.json` 自动生成 `CustomOp` 骨架工程。生成后，将现成的 host/kernel 实现复制到 `CustomOp` 对应目录，再执行编译即可得到 `libcust_opapi.so` 及 `aclnnSegmentMinCsr` 头文件。本样例的 `install.sh` 已封装上述流程，会自动完成工程生成、文件拷贝与算子编译。

## 编译运行样例算子
针对简易自定义算子工程，编译运行包含如下步骤：
- 调用msOpGen工具生成简易自定义算子工程；
- 完成算子host和kernel实现；
- 编译简易自定义算子工程；
- 调用执行自定义算子；

详细操作如下所示。

### 1. 生成简易自定义算子工程，复制host和kernel实现并编译算子<a name="operatorcompile"></a>
  - 切换到msOpGen脚本install.sh所在目录
    ```bash
    # 若开发者以git命令行方式clone了master分支代码，并切换目录
    cd ${git_clone_path}/AscendBasicOps/AscendTensorOps/SegmentMinCsr/SegmentMinCsrFrameworkLaunch
    ```

  - 调用脚本，生成简易自定义算子工程，复制host和kernel实现并编译算子
    - 方式一：配置环境变量运行脚本   
      请根据当前环境上CANN开发套件包的[安装方式](https://hiascend.com/document/redirect/CannCommunityInstSoftware)，选择对应配置环境变量命令。
      - 默认路径，root用户安装CANN软件包
        ```bash
        export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
        ```
      - 默认路径，非root用户安装CANN软件包
        ```bash
        export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
        ```
      - 指定路径install_path，安装CANN软件包
        ```bash
        export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
        ```
        运行install.sh脚本
        ```bash
        bash install.sh -v [SOC_VERSION]
        ```
    - 方式二：指定命令行安装路径来运行脚本
      ```bash
      bash install.sh -v [SOC_VERSION] -i [ASCEND_INSTALL_PATH]
      ```
    参数说明：
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的"Name"前增加Ascend信息，例如"Name"对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
        - Atlas A3 训练系列产品/Atlas A3 推理系列产品
        - Atlas A2 训练系列产品 / Atlas 800I A2 推理产品
    - ASCEND_INSTALL_PATH：CANN软件包安装路径

    脚本运行成功后，会在当前目录下创建CustomOp目录，编译完成后，会在CustomOp/build_out/op_api/lib目录下生成自定义算子库文件libcust_opapi.so，在CustomOp/build_out/op_api/include目录下生成aclnn接口的头文件。


### 3. 调用执行算子工程
- 参考AclNNInvocationNaive中的README

### 4. 更新日志
| 版本 | 日期       | 更新内容  | 开发者             |
|------|------------|---------|--------------------|
| v1.0 | 2026-3-20 | 初始版本 | 史浩睿 |

