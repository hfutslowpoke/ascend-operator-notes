# 工具库入口

工具库回答：**怎么生成索引、怎么校验归档、怎么在本机验证、怎么打包/提交/回读。**

## 1. 开发与维护

| 工具 | 用途 | 命令 |
|---|---|---|
| `tools/build_index.py` | 生成 catalog、cards、source.json、by-family/by-evidence/by-source-match | `python3 tools/build_index.py` |
| `tools/build_mfg_official.py` | 生成制造专场 specs.md/problems.md/by-cannjudge.md | `python3 tools/build_mfg_official.py` |
| `tools/verify_archive.py` | 哈希、JSON、Python、链接、敏感模式校验 | `python3 tools/verify_archive.py` |
| `skills/competition-repo-intake/` | 新比赛/新仓库入库流程 | agent skill |
| `skills/ascend-910b-local-validation/` | 910B3 本机验证协议 | agent skill |

修改后固定跑：

```bash
python3 tools/build_index.py --check
python3 tools/build_mfg_official.py --check
python3 tools/verify_archive.py
git diff --check
```

## 2. CANNJudge 便携工具

仓库内 `tools/competition/` 是可带走的起步包：

```text
tools/competition/tools/cann_client.py    查询赛事/题目/模板/历史/详情/榜单
tools/competition/tools/cann_package.py   按模板生成 zip/payload，不 POST
tools/competition/tools/cann_submit.py    dry-run 默认，--submit 才 POST
tools/competition/tools/check_env.sh      CANN/NPU/编译器/profiler 检查
tools/competition/tools/npu_lock.sh       NPU 串行锁
```

常用：

```bash
python3 tools/competition/tools/cann_client.py resolve gcc_2026_mfg
python3 tools/competition/tools/cann_client.py problems gcc_2026_mfg
```

## 3. ACW 环境包装入口

ACW 根目录有同语义包装：

```bash
bin/cann-client    # 查询
bin/cann-package   # 打包，不 POST
bin/cann-submit    # dry-run 默认；POST 需 ACW_ALLOW_SUBMIT=1
```

```bash
bin/cann-client resolve gcc_2026_mfg
bin/cann-client problems gcc_2026_mfg

bin/cann-package --problem-id <PID> --account A \
  --source-dir ./my_submission --out submission.zip

bin/cann-submit --problem-id <PID> --account A \
  --source-dir ./my_submission --tag mfg-r1

# 只有用户明确授权后
ACW_ALLOW_SUBMIT=1 bin/cann-submit ... --submit
```

## 4. 本机验证

- 所有设备运行经 `bin/npu-lock`；
- 记录设备、CANN、gcc/cmake、源码/产物 SHA、shape/dtype/seed、误差、时延；
- 编译成功 ≠ 可运行；小 shape PASS ≠ 全 shape PASS；
- 910B3 结果不外推 910C/平台；
- 详细协议见 `skills/ascend-910b-local-validation/SKILL.md`。

## 5. 工具与规则的分工

| 问题 | 去 |
|---|---|
| 怎么构造输入、平台精度怎么判 | `playbook/rules/` |
| 某题源码/官方题面在本机哪里 | `knowledge/` |
| 怎么打包/提交/回读 | 本页 + `tools/competition/` |
| 提交结果怎么回填 | `playbook/rules/submission-and-readback.md` |

## 6. 安全边界

- 工具不在仓库保存 Cookie/Token/账号私有配置；
- `cann_submit` 默认 dry-run；POST 必须显式授权；
- 不绕过隐藏测试数据、额度或重复限制；
- 不自动 POST；不伪造 SID/榜单；
- `cann-package` 只生成包，不联网提交。
