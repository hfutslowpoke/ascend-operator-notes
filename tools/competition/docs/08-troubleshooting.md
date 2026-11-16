# 常见报错与排查

## 1. 平台/提交

### 401 Unauthorized

- Cookie 过期或不是目标账号；
- 复制时带了 `cannjudge_auth=` 前缀或整段浏览器 Cookie；
- 用了另一个账号的 UID；
- 服务端会话失效（声明未过期也可能失效）。

处理：重新登录 → 重新复制 `cannjudge_auth` → `set-cookie` → `check-auth`。

### 403 / 无权限

- 账号未报名/未绑定 GitCode/华为账号；
- 题目属于其它赛事/队伍；
- 提交详情不是自己的 SID。

### 429 Too Many Requests

- 按 `retry_after_seconds` 退避；
- 查该账号当天额度；
- 不要换账号绕过、不要改包名重传。

### `tag already exists` / duplicate digest

- 工具按源码 digest 去重；
- 确实要复投用 `--allow-duplicate`；
- 但同一源码复投通常没有收益。

### `template editable mismatch`

- 题目 ID 错了；
- 账号/模板版本变了；
- 模板可编辑文件为空（用 `--all-files` 仅本地检查）；
- 重新拉模板，按最新 `path` 调整本地目录。

### `payload too large`

- 平台对 JSON 大小有限制（观察到约 610KB）；
- 检查是否把 ZIP/二进制/大日志塞进 `files`；
- 只提交可编辑源码文本。

### POST 成功但没拿到 SID

- 工具标记 `ambiguous_do_not_retry`；
- 先查该账号历史，按源码 SHA/时间/题目匹配；
- 已受理就查详情，不要重发。

### Compile Error

- 模板路径/文件名不对；
- 用了目标 CANN 不支持的 API/宏；
- 缺少头文件或 namespace；
- host/kernel tiling header 不一致；
- 兼容字段指向了错误文件。

### Runtime Error

- `aclrtSetDevice` 失败：设备被占用/权限/驱动；
- workspace 太小或没分配；
- 越界访问、未对齐；
- 同步死锁、`Stream_Synchronize_Timeout`；
- kernel 实际没启动或 block dim 错；
- 输出 buffer 未初始化。

### Wrong Answer

- 精度表达/累加顺序；
- 尾块 mask；
- 重复索引/非法索引语义；
- RoPE/KV merge/mask；
- 输入被改写；
- 输出漏写/被覆盖；
- 实际走了 fallback 或错误 tiling key。

### Time Limit Exceeded

- 死循环/轮询不退出；
- 跨核等待方向错；
- 任务量爆炸（重复索引没处理）；
- 同步粒度过细；
- 单核串行太长。

### System Error

- 平台/驱动/环境问题；
- 保留 SID、detail、时间；
- 稍后重查；必要时联系赛方；
- 不要把它当成代码失败反复重提。

## 2. 本地编译

### `bisheng: command not found`

```bash
source "$CANN_HOME/set_env.sh"
# 或
export PATH="$CANN_HOME/bin:$CANN_HOME/compiler/ccec_compiler/bin:$PATH"
```

以实际 CANN 安装为准。

### CMake 找不到 ASC

```bash
cmake -S . -B build \
  -DASCEND_COMPUTE_UNIT=ascend910b \
  -DASC_DIR="$CANN_HOME/lib64/cmake"
```

检查 `ASC_DIR` 下是否有 `AscendC`/`ascendc` 的 cmake 配置。

### 链接找不到 `libascendc` / `libcust_opapi`

- `LD_LIBRARY_PATH` 没包含 CANN lib64；
- `ASCEND_CUSTOM_OPP_PATH` 没指向 build 的 `tmp/vendors/custom`；
- 构建产物没生成完整。

### 编译过了但运行加载旧算子

- `ASCEND_CUSTOM_OPP_PATH` 指向旧目录；
- 动态库缓存；
- 安装路径/算子注册路径不对；
- 实际 kernel 名不是新版本。

记录 `.so` SHA 和 `ASCEND_CUSTOM_OPP_PATH`，不要只看源码时间。

## 3. 本地运行

### `aclrtSetDevice` 失败

- 设备被其他进程占用；
- 多 agent 并发；
- 容器/驱动权限；
- 设备号不对（`ASCEND_VISIBLE_DEVICES`）。

先 `npu-smi info`，再用 `npu_lock.sh` 串行。

### `Stream_Synchronize_Timeout`

- 死锁：跨核 flag 等不到；
- 事件 ID/pipe 不匹配；
- 某个核提前 return，另一个核还在等；
- 任务划分导致某核没执行；
- 循环次数与 tiling 不一致。

先做小 shape、单核、无跨核同步的版本，逐步加回。

### 结果随机/偶发错误

- 缺少同步；
- 多缓冲复用冲突；
- 原子累加顺序/初始化；
- 输入/输出别名；
- 越界写到相邻 buffer；
- 频率/调度导致时序变化。

用 `compute-sanitizer` 类工具（若可用）、边界检查、单核复现、固定 seed。

### NaN/Inf

- 除零、log(0)、sqrt(负数)；
- FP16 溢出；
- 未初始化 workspace；
- 非法索引读到脏数据；
- mask 没加。

## 4. 性能

### 本地比平台快很多

- 频率不同（1800 vs 1650/1500）；
- L2 容量不同；
- 平台输入分布不同；
- 平台有额外启动/同步/采样开销；
- 本地 workspace 复用/热缓存。

不要用统一倍率换算；看相对排序和机制。

### 优化后反而慢

- 新增同步/搬运/原子/FIX；
- 任务切碎导致 underfill；
- 缓存/地址跨度变差；
- 编译器布局变化；
- 只优化了非关键路径。

做 ON/OFF 对照，确认实际执行路径。

### 计时波动大

- NPU 被其他进程占用；
- 频率爬升；
- 输入分布/缓存状态；
- 样本太少；
- 冷/热启动混在一起。

用锁、固定预热、ABBA、多窗口、记录频率。

## 5. 排查顺序

1. 题目/模板/路径对不对；
2. 账号/权限/额度；
3. 编译是否真的成功；
4. 跑的是不是新产物；
5. 实际 tiling key / kernel 名 / block dim；
6. 单核/小 shape 是否正确；
7. 加同步/多核/大 shape；
8. 性能是否在完整 Task 上成立；
9. 平台提交是否 Pass、valid；
10. 保存证据，再决定下一步。
