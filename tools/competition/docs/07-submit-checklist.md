# 提交前检查清单

## 1. 题目与模板

- [ ] 已读完整题面、公式、输入输出、dtype、layout、shape 范围
- [ ] 已确认 precision 门槛（绝对/相对/worst/比例）
- [ ] 已确认特殊语义：重复索引、非法索引、空块、尾块、RoPE、KV merge、mask
- [ ] 已确认评测硬件、CANN 版本、launch 限制
- [ ] 已用 `cann_client.py template` 拉取模板
- [ ] 已核对 `editable: true` 的文件和相对路径
- [ ] 没有把 ZIP 当提交物、没有用 `code/` 前缀污染 path
- [ ] 旧 host/kernel 模板已确认是否需要兼容字段

## 2. 正确性

- [ ] 官方样例 / 模板用例通过
- [ ] 最小 shape、最大 shape、非对齐尾块通过
- [ ] 单行/单列、空输入、全重复、全非法、极端重复度通过
- [ ] 多个随机 seed 与 CPU oracle 对比通过
- [ ] 最大绝对/相对误差在门槛内，nonfinite=0
- [ ] 输入运行前后逐字节一致（只读）
- [ ] 输出完整写回 GM，没有被覆盖/漏写
- [ ] 冷启动、热启动、连续调用都验证
- [ ] 多 batch / 多 head / 多 block 边界正确

## 3. 合规

- [ ] 无 hash(shape)、指纹、白名单、测试用例编号
- [ ] 无 host 数值计算参与输出
- [ ] 无 golden/历史输出复用、无结果重放
- [ ] 无空 kernel、纯搬运、entry 直接 return
- [ ] 特化分支域内任意输入正确
- [ ] fallback 真实可用且数值正确
- [ ] 最终版无“针对某 case”注释/调试代码
- [ ] CPU 参考实现不在评分路径

## 4. 构建与绑定

- [ ] 源码文件 SHA-256 已记录
- [ ] 编译命令、CANN/编译器版本已记录
- [ ] 产物 `.so` / kernel binary SHA 已记录
- [ ] `ASCEND_CUSTOM_OPP_PATH` 指向正确 build
- [ ] 实际 tiling key / kernel 名 / block dim 已确认
- [ ] workspace 大小和复用策略已记录
- [ ] 运行的是新构建，不是旧包
- [ ] 插桩/调试代码已关闭，ON/OFF 对照已做

## 5. 性能

- [ ] 使用完整 Task 窗口，不只看局部 busy
- [ ] ABBA / 多窗口，预热/采样策略一致
- [ ] NPU 全程串行锁
- [ ] 记录设备型号、核数、频率
- [ ] 记录输入分布、seed、有效边数/重复度
- [ ] 报告全部样本、中位数、均值、成对差、方向
- [ ] 新增成本（搬运/同步/原子/FIX/workspace）已计入
- [ ] 没有用不同日期/不同上下文的绝对时间直接相减

## 6. 提交

- [ ] 目标 problemId 是该赛事的题目，不是同名题目
- [ ] 账号 UID 与 Cookie 一致
- [ ] 本地源码 SHA 与要提交的源码一致
- [ ] 该源码没有提交过（全局 ledger 去重，本地策略）
- [ ] 账号在当前 12:00→12:00 窗口、全部题目的额度足够
- [ ] dry-run 输出已检查：文件、字节数、digest、payload 大小
- [ ] `--submit` 后拿到 SID，并已落盘
- [ ] 没有在 POST 后重发
- [ ] 最终 detail 已保存
- [ ] 下载提交 ZIP，回读源码 SHA 一致
- [ ] 确认 `status`、`valid`、逐用例结果
- [ ] 知道本次提交是否覆盖当前队伍成绩

## 7. 失败时

- [ ] 保留 SID、完整 detail、日志、源码 SHA
- [ ] 区分 Compile Error / Runtime Error / WA / TLE / System Error
- [ ] 从第一个非 0 ACL 返回值或第一条 error 日志开始查
- [ ] 不覆盖旧失败记录
- [ ] 修复后新 revision，不复用旧 tag
