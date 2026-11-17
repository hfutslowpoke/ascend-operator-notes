# ACW Skills

`skills/` 存放给 ACW/agent 加载的工作流，不存放算子源码。每个 skill 是一个目录，包含 `SKILL.md`，按“何时用 → 输入 → 步骤 → 输出 → 边界”组织。

## 可用 skills

| Skill | 用途 | 主要输入 | 主要输出 |
|---|---|---|---|
| [operator-pattern-lookup](operator-pattern-lookup/SKILL.md) | 按瓶颈/算子族查可复用机制 | 算子族、瓶颈、shape/dtype、目标 SoC | 候选机制、证据等级、适用条件、负例、验证计划 |
| [competition-repo-intake](competition-repo-intake/SKILL.md) | 竞赛仓库筛选、固定 commit、深读、建条目 | 仓库 URL、赛事、候选实现 | `competition/<event>/<entry>/` + catalog 更新 |
| [ascend-910b-local-validation](ascend-910b-local-validation/SKILL.md) | 910B3 构建/运行/计时/负例记录 | 源码、shape、命令、参考实现 | `evidence/` 日志、`local_status`、验证章节 |

## 使用方式

1. 先读 [../index/catalog.json](../index/catalog.json)、[../index/cards.md](../index/cards.md) 或 [../index/README.md](../index/README.md)，确定要查/要加的条目；学习路径见 [../docs/AGENT_READING_GUIDE.md](../docs/AGENT_READING_GUIDE.md)。
2. 加载对应 `SKILL.md`，按步骤执行；所有路径以仓库根为基准。
3. 任何 NPU 运行必须经 `/workspace/acw/bin/npu-lock`。
4. 新增/修改后运行：

```bash
python3 tools/build_index.py
python3 tools/verify_archive.py
git diff --check
```

## 共同边界

- 不自动向竞赛平台提交，不修改 ACW 框架或当前算子工程。
- 不复制未确认许可的第三方源码；优先引用固定 commit + 路径/行号 + 哈希。
- `O+A`（作者/平台）不是 `V`（本机）；910B3 结果不外推 910C。
- 编译成功不等于可运行；小 shape 通过不等于全 shape 通过。
- 失败/负例必须写入 `evidence/` 和条目 README，不能只留成功记录。

## 新增 skill

1. 建 `skills/<name>/SKILL.md`；
2. 写清触发条件、输入、步骤、输出模板、失败处理；
3. 在本页表格加一行；
4. 运行 `python3 tools/verify_archive.py` 检查链接/JSON。

Skill 应引用本库条目和 `source.json`，不要在 skill 里复制大段第三方代码或一次性实验数据。
