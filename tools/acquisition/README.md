# 公开源码重新获取

`download_final_prs.py` 是本次实际下载器的可迁移版本，仅匿名 GET 和公开 Git。需要 Python requests、Git。固定 PR head 规格见 public-pr-heads.json，不含用户账号、邮件或凭据。

运行后写入当前目录的 `public-source-download/`，不覆盖本库。可以用环境变量 `CANN_ARCHIVE_DOWNLOAD_ROOT` 指定独立下载工作区。网页 raw 不支持预览时使用正常公开 Git 提取同一个固定 commit，再验证 blob ID。受保护接口的 401/403 不会尝试规避授权。

```bash
python3 tools/acquisition/download_final_prs.py
```

API 和仓库在以后可能变化，此工具不保证长期服务可用。本库已有完整固定版本及 SHA256，可离线验证。
