# 来源、修改与许可

本目录不是从零写的。提交/额度工具来自 ACW 工作区的 `tools/cannjudge/`，在 `cannjudge` 分支上做了独立化、文档化和测试补充。

## 1. 来源

| 本目录文件 | 来源 | 同步状态 |
|---|---|---|
| `tools/cann_client.py` | `/workspace/acw/tools/cannjudge/cann_client.py` | 当前与 ACW 一致（SHA256 见下） |
| `tools/cann_quota.py` | `/workspace/acw/tools/cannjudge/cann_quota.py` | 字节一致 |
| `tools/cann_submit.py` | `/workspace/acw/tools/cannjudge/cann_submit.py` | 同步全局锁与原子台账，适配独立环境和账户额度 |
| `tools/cann_package.py` / `tools/check_env.sh` | 本机 ACW 实用工具 | 复用提交选源和摘要，适配目录 |
| `tools/cann_eval.py` | ACW `tools/local_npu/run_phase.py` | 可独立读取绑定和运行设备阶段 |
| `tools/acl_probe.py` | 本地 NPU 初始化诊断 | 可独立运行，支持选择设备 |
| `tools/npu_lock.sh` | `/workspace/acw/bin/npu-lock` | 独立化改写（路径/环境变量适配） |
| `RULES.md` / `TOOLS.md` | ACW `docs/SOLO_TRIAL_20261118.md`、`docs/ACW_HANDOFF_20261119.md`、工具行为 | 重新整理 |
| `tools/cann_score.py` | 本分支新增 | 独立实现，参考 ACW `acw/solo.py` 的公式 |
| `tests/` | 本分支新增 | 离线测试 |

固定来源：

```text
ACW 版本: 0.30.2
ACW tag:  v0.30.2-solo
ACW commit: 6f33e766fd6a7245d2edfb66348ca66c6860c98a
```

`cannjudge` 分支的早期提交：

```text
76ad2e3 feat(cannjudge): add 12:00 quota rules and submission toolkit
3afb2a0 fix(cannjudge): require explicit ACW_ALLOW_SUBMIT in standalone submit
```

## 2. 本分支的修改

与 ACW 版本相比：

- 独立运行，不 import ACW 包；
- 提交默认 dry-run，POST 需要 `--submit` + `ACW_ALLOW_SUBMIT=1`；
- 额度按北京时间 12:00 窗口、账号全部题目统计；
- 429 原文 body/headers 保存到账本；
- 增加 `cann_score.py`（逐 case 分数/headroom）；
- 增加离线测试、开发规范、载荷契约、runbook、安全边界；
- `npu_lock.sh` 改为可独立调用的薄封装。

## 3. 哈希（同步时）

```text
ae8354e7d39db0d6f799352aeb88a126c9f2b27a4badb66db2bbc9bbf9ce5667  cann_client.py
9ba679c4c275b56753d92f5cbb9011dd8f21834d73b3640ead64f21272e6f31e  cann_quota.py
c01fa96371c4064b3f6ffd07e2992ded8b92dc6e1506bfcb9e46a8d152e22590  cann_submit.py
b8552ab122e0c14152111467d04283e1e38df86c1654ffa19430efea95df981d  cann_package.py
160dd46a062f93011f94356ae38333bf4d052f15b1601f2e8e33cd282eec9fb2  cann_eval.py
d506bd1d0a71d67a04acd608a224ab3bf999f3eadf3f7615776b189bc1ec2a11  acl_probe.py
65ba2b717ac3f0b18a5194d360399c93c896dc5a39ce31f5d7bc26425d924322  npu_lock.sh
```

如果 ACW 侧再次修改这些文件，先 diff，再决定是否同步；不要直接覆盖本分支的独立化改动。

## 4. 许可

- ACW 工作区当前没有 `LICENSE` 文件；
- 本仓库也没有为整个集合设置统一的新许可证；
- 因此本目录**不声称**对这些代码授予新的开源许可；
- 默认按内部/私密研究使用；
- 如果未来 ACW 或本仓库补充许可证，以补充的许可证为准；
- 第三方平台规则、API 和页面内容的权利属于原权利人。

## 5. 维护

1. 先读 [DEVELOPMENT.md](DEVELOPMENT.md) 的测试/安全要求；
2. 修改工具后更新本页的哈希/来源；
3. 不在工具里写死账号、UID、SID、题目 ID；
4. 不把 Cookie、账号文件、平台原始响应提交进仓库；
5. 运行：

```bash
python3 -m unittest discover -s cannjudge/tests -v
python3 tools/verify_archive.py
git diff --check
```
