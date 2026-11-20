# GitHub 使用说明

仓库：https://github.com/hfutslowpoke/ascend-operator-notes 。仓库已设为 **Public**：匿名可读、可 clone；修改/推送仍需 GitHub 授权（collaborator 权限或 token）。

## 直接在网页使用

不需要登录即可打开仓库、浏览源码和 Markdown。想下载整个库，点绿色 Code → Download ZIP。想看以前版本，打开 Commits。匿名访问 GitHub API 有速率限制（约 60 次/小时），大量读取建议 clone 到本地。

以后可以点 Add file → Upload files 追加少量文件，写一句此次修改说明后 Commit changes。已有文件页面右上角的铅笔按钮可以编辑文字。大批量源码和连续开发更适合 Git。

## 换电脑后通过命令行使用

只读访问（不需要登录）：

```bash
git clone https://github.com/hfutslowpoke/ascend-operator-notes.git
cd ascend-operator-notes
```

需要推送时，再登录有 collaborator 权限的 GitHub 账号：

```bash
gh auth login
gh auth setup-git
```

账号密码不发给协作者或聊天工具。授权凭据留在本机的 GitHub CLI 配置/凭据存储中，不放进仓库。

以后每次更新：

```bash
git pull --ff-only
# 在文件夹中新增、修改资料
git status
git add README.md competition general index skills docs tools THIRD_PARTY.md MANIFEST-SHA256.json
git commit -m "补充新的算子实验记录"
git push
```

`git status` 查看待提交内容；`git commit` 保存本地版本；`git push` 上传到 GitHub。提交前确认没有账号文件、访问令牌和大体积构建产物。加入新第三方代码时，同时记录原网址、固定 commit、许可和哈希。

若服务器以后交给他人或不再使用，可在 GitHub Settings → Applications 中撤销对应 GitHub CLI 授权，本机也可用 `gh auth logout` 清除登录。

官方说明：[创建仓库](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)、[GitHub CLI 登录](https://cli.github.com/manual/gh_auth_login)。

