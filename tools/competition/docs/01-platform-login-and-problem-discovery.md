# 登录、找比赛、找题目

## 1. 平台地址

以 CannJudge 为例（实际域名以赛事通知为准）：

| 用途 | 地址 |
|---|---|
| 登录 | `https://cannjudge.cn/auth/login` |
| 开放题库 | `https://cannjudge.cn/problems` |
| 赛事首页 | `https://cannjudge.cn/public/<contest-slug>` |
| 题目说明 | `https://cannjudge.cn/public/<contest-slug>/<problem-slug>` |
| 在线编辑/提交 | `https://cannjudge.cn/public/<contest-slug>/<problem-slug>/submit` |
| 题目榜单 | `https://cannjudge.cn/public/<contest-slug>/<problem-slug>/ranking` |
| 提交详情 | `https://cannjudge.cn/public/<contest-slug>/<problem-slug>/submission/<SID>` |
| API 基址 | `https://cannjudge.cn/api` |

**关键区别：**

- 页面路由用 **slug**；
- API 里的 `contestId` / `problemId` 是 24 位数据库 ObjectId；
- 页面显示的数字 ID 通常不是 ObjectId；
- 不同赛事可以有同名题目，必须限定 `contestId` 再查。

## 2. 登录方式

平台通常支持三类入口，具体以赛事配置为准：

1. **账号密码 + 图形验证码**
2. **GitCode OAuth**
3. **华为账号 OAuth**

团队赛还需要：

- 账号已绑定 GitCode / 华为账号；
- 在外部赛事页面完成报名；
- 队伍归属正确（否则提交可能记到错误队伍）；
- 确认“采用最近一次提交”还是“历史最高”，不同赛事规则不同。

### 2.1 浏览器登录

1. 用独立浏览器 profile 登录，避免多个账号 Cookie 串用。
2. 打开登录页，按赛事要求选择账号密码 / GitCode / 华为账号。
3. 如果走 OAuth，完成外部授权后回到平台回调页。
4. 确认页面右上角显示的是目标账号。

### 2.2 取出 `cannjudge_auth`

登录后：

1. 打开开发者工具（F12）。
2. Application / Storage → Cookies → `https://cannjudge.cn`。
3. 找到 `cannjudge_auth`，复制它的值。
4. 如果该 Cookie 是 HttpOnly，`document.cookie` 读不到，必须用开发者工具。

不要复制整段浏览器 Cookie 串；只取 `cannjudge_auth` 的值。不要把 `cannjudge_auth=` 前缀重复加一遍。

### 2.3 保存到私有账号文件

```bash
python3 tools/cann_client.py set-cookie A
```

输入是隐藏的，不会进入 shell 历史。默认写入：

```text
~/.config/cannjudge/accounts.json
```

权限 `0600`。格式见 `examples/accounts.example.json`。也可以放到自定义位置：

```bash
export CANNJUDGE_ACCOUNT_FILE=/secure/path/accounts.json
```


可选环境变量：

```bash
# 自定义 API 基址（默认 https://cannjudge.cn/api）
export CANNJUDGE_API=https://cannjudge.cn/api

# 默认直连；如环境必须走代理，设为 1
export CANNJUDGE_USE_PROXY=1

# 临时覆盖某个账号 Cookie（不要写进文件/命令历史）
export CANNJUDGE_COOKIE_A='<cookie value>'
```

不要把 Cookie 写进脚本、README、命令参数、截图或提交包。

### 2.4 验证登录

公开资料页返回 200 **不代表**登录有效。用“本账号题目提交历史”这种受限接口验证：

```bash
python3 tools/cann_client.py accounts
python3 tools/cann_client.py check-auth --problem <PROBLEM_ID>
```

`accounts` 只显示标签、UID、到期时间，不显示 Cookie。`check-auth` 对每个账号请求受限历史，返回 `authenticated_history_http: 200` 才算通过。

### 2.5 Cookie 过期 / 401

1. 重新在浏览器登录。
2. 重新复制 `cannjudge_auth`。
3. 再执行 `set-cookie`。
4. 再 `check-auth`。

Cookie 载荷里的 `exp` 只是声明到期时间，不能证明服务端会话仍有效；以受限接口结果为准。

## 3. 发现赛事与题目

### 3.1 列出公开赛事

```bash
python3 tools/cann_client.py contests
```

输出包含 `_id`、`name`、`title`、`organization_mode`、`ongoing`、`submit_enabled` 等。`name` 是页面 slug，`_id` 是 API 的 contestId。

### 3.2 用 slug 解析 ID

```bash
# 只解析赛事
python3 tools/cann_client.py resolve <contest-slug>

# 同时解析题目
python3 tools/cann_client.py resolve <contest-slug> <problem-slug>
```

示例输出会给出：

```json
{
  "contest": {"_id": "...", "name": "...", "title": "..."},
  "problem": {"_id": "...", "name": "...", "title": "...", "code_template": "..."}
}
```

### 3.3 列出某赛事全部题目

```bash
python3 tools/cann_client.py problems <contest-slug>
# 或
python3 tools/cann_client.py problems <CONTEST_ID>
```

重点字段：

- `_id`：problemId，提交时使用；
- `name`：题目 slug；
- `title`：展示名；
- `code_template`：模板类型，决定可编辑文件；
- `editable`：模板返回后确认，不能只看 `code_template`。

### 3.4 读取题目详情

```bash
python3 tools/cann_client.py problem <PROBLEM_ID>
```

题目详情一般包含：

- 题面、公式、输入输出；
- dtype、layout、shape 范围；
- 精度要求、特殊语义；
- 开发形式（单 kernel / 多文件 / host+kernel）；
- 评测硬件、launch 限制；
- 公开样例和隐藏点说明。

**先把题面读完，再下载模板。** 不要用另一道题的 shape、dtype、layout 或语义做假设。

## 4. 下载模板与确认可编辑文件

### 4.1 查看模板文件列表

```bash
python3 tools/cann_client.py template <PROBLEM_ID> --account A
```

输出每个文件的：

- `path`：提交时使用的相对路径；
- `editable`：是否可改；
- `content` 长度 / SHA（工具默认不打印全文）。

### 4.2 下载模板 ZIP

```bash
python3 tools/cann_client.py download-template <PROBLEM_ID> template.zip --account A
unzip -l template.zip
```

注意：模板 ZIP 里的文件可能放在 `code/` 目录下，但提交 JSON 的路径可能是去掉 `code/` 的相对路径。**以 `template` 返回的 `path` 为准，不要以 ZIP 解压后的目录结构猜。**

### 4.3 判断模板类型

常见两类：

| 类型 | 特征 | 提交 |
|---|---|---|
| `npu_kernel_dev` | 通常只有 `kernel.asc` 可编辑；`CMakeLists.txt`、`main.asc`、`run.sh` 等不可编辑 | `files: [{"path":"kernel.asc","content":"..."}]` |
| `custom_template` | 多文件，可能是 `op_host/*.cpp`、`op_kernel/*.cpp`、`*_tiling.h` 等 | 按模板 `path` 提交；旧 host/kernel 题可能还需要兼容字段 |

如果 `editable` 为空或与预期不符：

1. 重新拉模板，确认账号/题目 ID 正确；
2. 检查是否下载了另一道题的模板；
3. 不要擅自把所有文件都提交；
4. 必要时用 `--all-files` 仅做本地检查，并在提交前与官方模板核对。

## 5. 账号与队伍注意事项

- 团队赛先确认账号已绑定 GitCode/华为账号，并已报名；
- 多账号属于同一队伍时，确认哪个账号提交、是否共享额度；
- 有些赛事“最近一次提交覆盖当前成绩”，差包会覆盖好包；
- 有些赛事按历史最高，但违规/非 Pass 的成绩会作废；
- 提交前查看该账号在该题的历史和当前榜单，避免误覆盖；
- 多 agent 共用账号时，提交顺序要串行协调。

## 6. 平台 API 速查

以下路径不含 `/api` 前缀：

| 用途 | 请求 |
|---|---|
| 公共小组 | `GET /groups/public` |
| 赛事列表 | `GET /contests/group/<GID>` |
| 赛事详情（slug） | `GET /contests/name/<slug>?groupId=<GID>` |
| 赛事详情（ID） | `GET /contests/<CID>` |
| 赛事题目 | `GET /problems/contest/<CID>` |
| 题目详情 | `GET /problems/<PID>` |
| 模板 | `GET /problems/<PID>/template?userId=<UID>` |
| 空工程 ZIP | `GET /problems/<PID>/package?userId=<UID>` |
| 提交 | `POST /submissions/submit` |
| 本账号历史 | `GET /submissions/user/<UID>/problem/<PID>` |
| 提交详情 | `GET /submissions/<SID>?userId=<UID>` |
| 提交工程 ZIP | `GET /submissions/<SID>/package` |
| 单题榜 | `GET /problems/<PID>/ranking?page=1&size=100&q=<name>` |
| 赛事统计 | `GET /submissions/contest/<CID>/stats` |

部分接口返回裸数组/对象，部分返回 `{code,msg,data}`；工具会兼容 `data` 包装。

## 7. 最小命令序列

```bash
export CANNJUDGE_ACCOUNT_FILE="$HOME/.config/cannjudge/accounts.json"

python3 tools/cann_client.py set-cookie A
python3 tools/cann_client.py accounts
python3 tools/cann_client.py contests
python3 tools/cann_client.py resolve <contest-slug> <problem-slug>
python3 tools/cann_client.py template <PROBLEM_ID> --account A
python3 tools/cann_client.py download-template <PROBLEM_ID> template.zip --account A
```
