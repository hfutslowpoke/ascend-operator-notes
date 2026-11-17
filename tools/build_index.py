#!/usr/bin/env python3
"""Generate tier catalogs, per-entry source.json, and index views.

The single source of truth is index/catalog.json. Do not edit generated files.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "index/catalog.json"
STATUS_ORDER = [
    "pass",
    "pass-negative",
    "partial",
    "fail",
    "offline-only",
    "platform-only",
    "not-run",
    "pending",
]
SOURCE_MATCH_ORDER = [
    "direct",
    "adapt",
    "reference",
    "none",
    "building-block",
    "not-applicable",
]


def load() -> dict:
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    for impl in data["implementations"]:
        impl.setdefault("local_status", "not-run")
    for repo in data["repositories"]:
        repo.setdefault("local_status", "not-run")
    return data


def dump(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def write_or_check(path: Path, text: str, check: bool) -> bool:
    if check:
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            print(f"out-of-date: {path.relative_to(ROOT)}", file=sys.stderr)
            return False
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def repo_view(repo: dict) -> dict:
    keys = [
        "id", "url", "owner", "author", "year", "name", "default_branch",
        "commit", "commit_date", "accessed", "license", "license_note",
        "primary_dsl", "operator_families", "competition_or_context",
        "local_cache", "relationship_to_existing", "status", "tier",
        "entry_path", "local_status", "notes", "report",
    ]
    return {k: repo[k] for k in keys if k in repo}


def impl_view(impl: dict) -> dict:
    keys = [
        "id", "repository_id", "source_commit", "operator", "operator_family",
        "semantics", "shape_dtype", "environment", "source_files", "symbol",
        "mechanism", "work_model", "evidence", "conditions", "negative_cases",
        "reuse_decision", "report", "verification", "tier", "entry_path",
        "local_status", "local_validation", "source_match", "official_problem",
        "official_semantics", "official_shape_dtype", "official_conditions",
        "official_negative_cases", "supporting_sources",
    ]
    return {k: impl[k] for k in keys if k in impl}


def grouped_matmul_source() -> dict:
    entry = ROOT / "competition/grouped-matmul-gelu"
    pub = json.loads((entry / "catalog/public-final-sources.json").read_text(encoding="utf-8"))
    own = json.loads((entry / "catalog/our-work.json").read_text(encoding="utf-8"))
    final = own["actual_final_submission"]
    return {
        "schema_version": 2,
        "entry": "competition/grouped-matmul-gelu",
        "tier": "competition",
        "repository": {
            "id": "grouped-matmul-gelu-archive",
            "url": "https://gitcode.com/cann/cann-ops-competitions",
            "commit": pub["upstream_main_commit"],
            "default_branch": "master",
            "license": "Apache-2.0",
            "status": "deep-read",
            "local_cache": "competition/grouped-matmul-gelu/prs",
        },
        "implementations": [
            {
                "id": "GROUPED-MATMUL-GELU-26PR",
                "operator": "GroupedMatmulGelu (26 public final PRs)",
                "operator_family": "grouped-gemm",
                "source_files": [],
                "archive_catalog": "catalog/public-final-sources.json",
                "pr_count": pub["downloaded_PR_count"],
                "file_count": pub["file_count"],
                "local_status": "platform-only",
            },
            {
                "id": "GROUPED-MATMUL-GELU-OURS",
                "operator": "GroupedMatmulGelu (our final submission)",
                "operator_family": "grouped-gemm",
                "source_files": [
                    {
                        "path": "our-submission/kernel.asc",
                        "lines": "whole-file",
                        "sha256": final["sha256"],
                    }
                ],
                "local_status": "platform-only",
                "platform": {
                    "sid": final["sid"],
                    "status": final["platform_status"],
                    "score_observed_20261116": final["score_observed_20261116"],
                    "rank_observed_20261116": final["rank_observed_20261116"],
                },
            },
        ],
        "generated_from": "competition/grouped-matmul-gelu/catalog/public-final-sources.json",
    }


def build_tier_catalog(data: dict, tier: str) -> dict:
    repos = [repo_view(r) for r in data["repositories"] if r.get("tier") == tier]
    impls = [impl_view(i) for i in data["implementations"] if i.get("tier") == tier]
    return {
        "schema_version": 2,
        "generated_from": "index/catalog.json",
        "tier": tier,
        "counts": {
            "repositories": len(repos),
            "deep_read": sum(1 for r in repos if r.get("status") in ("deep-read", "deep-read-negative")),
            "implementations": len(impls),
            "excluded": sum(1 for r in repos if r.get("status") == "excluded"),
            "by_local_status": dict(Counter(i.get("local_status", "not-run") for i in impls)),
        },
        "repositories": repos,
        "implementations": impls,
    }


def rel_link(from_path: Path, target: Path) -> str:
    return os.path.relpath(target, from_path.parent).replace(os.sep, "/")


def build_by_family(data: dict) -> str:
    rows = []
    for impl in data["implementations"]:
        entry = ROOT / impl["entry_path"]
        rows.append(
            (
                impl["operator_family"],
                impl["id"],
                impl["tier"],
                rel_link(ROOT / "index/by-family.md", entry / "README.md"),
                impl.get("local_status", "not-run"),
            )
        )
    rows.sort(key=lambda r: (r[0], r[1]))
    lines = [
        "# 按算子族",
        "",
        "由 `tools/build_index.py` 从 `index/catalog.json` 生成；不要手改。",
        "",
        "| 算子族 | 实现 | 层 | 条目 | 本机 910B3 |",
        "|---|---|---|---|---|",
    ]
    for family, impl_id, tier, link, status in rows:
        lines.append(f"| `{family}` | `{impl_id}` | `{tier}` | [README]({link}) | `{status}` |")
    return "\n".join(lines) + "\n"


def build_by_evidence(data: dict) -> str:
    grouped: dict[str, list[dict]] = {s: [] for s in STATUS_ORDER}
    for impl in data["implementations"]:
        grouped.setdefault(impl.get("local_status", "not-run"), []).append(impl)
    lines = [
        "# 按本机证据",
        "",
        "由 `tools/build_index.py` 从 `index/catalog.json` 生成；不要手改。",
        "",
        "`pass` 只表示本机 910B3 构建/运行通过，不外推到 910C。",
        "",
    ]
    for status in STATUS_ORDER:
        items = grouped.get(status, [])
        if not items:
            continue
        lines += [
            f"## `{status}`",
            "",
            "| 实现 | 层 | 条目 | 说明 |",
            "|---|---|---|---|",
        ]
        for impl in sorted(items, key=lambda i: i["id"]):
            entry = ROOT / impl["entry_path"]
            link = rel_link(ROOT / "index/by-evidence.md", entry / "README.md")
            note = str(impl.get("verification", "")).replace("|", "/")
            lines.append(f"| `{impl['id']}` | `{impl['tier']}` | [README]({link}) | {note} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_by_source_match(data: dict) -> str:
    grouped: dict[str, list[dict]] = {s: [] for s in SOURCE_MATCH_ORDER}
    for impl in data["implementations"]:
        match = impl.get("source_match")
        if match:
            grouped.setdefault(match, []).append(impl)
    lines = [
        "# 按官方源码匹配级别",
        "",
        "由 `tools/build_index.py` 从 `index/catalog.json` 生成；不要手改。",
        "",
        "只列出制造专场的 15 题映射与 2 个基础件；级别定义见 `competition/mfg-2026/README.md`。",
        "",
        "> 同一题可能有多个候选实现（例如 SpmmSum 的 ops-gnn/ops-sparse）。`pass` 只表示本机测试通过，不一定是官方全 shape。",
        "",
    ]
    for match in SOURCE_MATCH_ORDER:
        items = grouped.get(match, [])
        if not items:
            continue
        lines += [
            f"## `{match}`",
            "",
            "| 实现 | 官方题 | 条目 | 本机 |",
            "|---|---|---|---|",
        ]
        for impl in sorted(items, key=lambda i: i["id"]):
            entry = ROOT / impl["entry_path"]
            link = rel_link(ROOT / "index/by-source-match.md", entry / "README.md")
            problem = impl.get("official_problem", {}) or {}
            title = str(problem.get("title", "")).replace("|", "/")
            lines.append(
                f"| `{impl['id']}` | {title or '—'} | [README]({link}) | `{impl.get('local_status', 'not-run')}` |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def card_view(impl: dict) -> dict:
    mechanism = re.sub(r"\s+", " ", str(impl.get("mechanism", ""))).strip()
    one_line = mechanism.split("。", 1)[0][:260]
    source_files = []
    for item in impl.get("source_files", [])[:6]:
        source_files.append(f"{item.get('local_path') or item.get('path')}:{item.get('lines')}")
    supporting = []
    for item in impl.get("supporting_sources", [])[:6]:
        supporting.append(f"{item.get('local_path') or item.get('path')}:{item.get('lines')}")
    logs = []
    lv = impl.get("local_validation", {})
    if isinstance(lv, dict):
        for key in ("log", "build_log", "run_log"):
            if lv.get(key):
                logs.append(lv[key])
        for run in lv.get("runs", []):
            if isinstance(run, dict) and run.get("log"):
                logs.append(run["log"])
        for run in lv.get("passes", []):
            if isinstance(run, dict) and run.get("log"):
                logs.append(run["log"])
    logs = list(dict.fromkeys(logs))
    return {
        "id": impl["id"],
        "title": impl["operator"],
        "tier": impl["tier"],
        "entry": impl["entry_path"],
        "family": impl["operator_family"],
        "readme": f"{impl['entry_path']}/README.md",
        "source_json": f"{impl['entry_path']}/source.json",
        "one_line": one_line,
        "local_status": impl.get("local_status", "not-run"),
        "source_match": impl.get("source_match", ""),
        "official_problem": impl.get("official_problem", {}),
        "verification": impl.get("verification", ""),
        "conditions": impl.get("conditions", []),
        "negative_cases": impl.get("negative_cases", []),
        "reuse_decision": impl.get("reuse_decision", ""),
        "source_files": source_files,
        "supporting_sources": supporting,
        "evidence": logs,
    }


def build_cards(data: dict) -> dict:
    return {
        "schema_version": 2,
        "generated_from": "index/catalog.json",
        "cards": [card_view(i) for i in data["implementations"]],
    }


def build_cards_md(data: dict) -> str:
    cards = [card_view(i) for i in data["implementations"]]
    lines = [
        "# 阅读卡",
        "",
        "由 `tools/build_index.py` 从 `index/catalog.json` 生成；不要手改。",
        "每张卡给出“一句话机制 + 先读文件 + 本机状态 + 边界”，用于 agent 快速定位，不替代条目 README。",
        "",
    ]
    for tier in ("competition", "general"):
        subset = [c for c in cards if c["tier"] == tier]
        if not subset:
            continue
        lines += [f"## {tier}", ""]
        for c in sorted(subset, key=lambda x: (x["family"], x["id"])):
            entry = ROOT / c["entry"]
            readme = rel_link(ROOT / "index/cards.md", entry / "README.md")
            source = rel_link(ROOT / "index/cards.md", entry / "source.json")
            lines += [
                f"### `{c['id']}`",
                "",
                f"- **{c['title']}** · `{c['family']}` · 本机 `{c['local_status']}`" + (f" · 官方匹配 `{c['source_match']}`" if c.get('source_match') else ""),
                f"- 一句话：{c['one_line'] or '见条目 README'}",
                f"- 先读：[README]({readme}) · [source.json]({source})",
            ]
            if c["source_files"]:
                lines.append("- 源码入口：" + "；".join(f"`{x}`" for x in c["source_files"][:3]))
            if c.get("supporting_sources"):
                lines.append("- 补充源码：" + "；".join(f"`{x}`" for x in c["supporting_sources"][:3]))
            if c["conditions"]:
                lines.append("- 适用条件：" + "；".join(str(x) for x in c["conditions"][:4]))
            if c["negative_cases"]:
                lines.append("- 负例/风险：" + "；".join(str(x) for x in c["negative_cases"][:3]))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated files are out of date")
    args = parser.parse_args()
    data = load()
    ok = True

    # Per-entry source.json (one repository per entry in the current catalog)
    by_entry: dict[str, list[dict]] = {}
    for repo in data["repositories"]:
        if repo.get("status") == "excluded":
            continue
        entry = repo.get("entry_path")
        if not entry or not (ROOT / entry).is_dir():
            print(f"skip missing entry: {repo['id']} -> {entry}", file=sys.stderr)
            continue
        by_entry.setdefault(entry, []).append(repo)
    for entry, repos in sorted(by_entry.items()):
        if len(repos) != 1:
            print(f"entry has {len(repos)} repositories, expected 1: {entry}", file=sys.stderr)
            ok = False
            continue
        repo = repos[0]
        impls = [i for i in data["implementations"] if i["repository_id"] == repo["id"]]
        source = {
            "schema_version": 2,
            "entry": entry,
            "tier": repo["tier"],
            "repository": {
                k: repo.get(k)
                for k in ("id", "url", "commit", "default_branch", "license", "status", "local_cache")
            },
            "implementations": [
                {
                    "id": i["id"],
                    "operator": i["operator"],
                    "operator_family": i["operator_family"],
                    "source_files": i.get("source_files", []),
                    **({"supporting_sources": i["supporting_sources"]} if i.get("supporting_sources") else {}),
                    "local_status": i.get("local_status", "not-run"),
                    **({"source_match": i["source_match"]} if i.get("source_match") else {}),
                    **({"official_problem": i["official_problem"]} if i.get("official_problem") else {}),
                    "local_validation": i.get("local_validation", {}),
                }
                for i in impls
            ],
            "local_validation": {"status": repo.get("local_status", "not-run"), "notes": repo.get("notes", "")},
            "generated_from": "index/catalog.json",
        }
        ok &= write_or_check(ROOT / entry / "source.json", dump(source), args.check)

    ok &= write_or_check(
        ROOT / "competition/grouped-matmul-gelu/source.json",
        dump(grouped_matmul_source()),
        args.check,
    )

    for tier in ("competition", "general"):
        ok &= write_or_check(
            ROOT / tier / "catalog.json",
            dump(build_tier_catalog(data, tier)),
            args.check,
        )

    ok &= write_or_check(ROOT / "index/by-family.md", build_by_family(data), args.check)
    ok &= write_or_check(ROOT / "index/by-evidence.md", build_by_evidence(data), args.check)
    ok &= write_or_check(ROOT / "index/by-source-match.md", build_by_source_match(data), args.check)
    ok &= write_or_check(ROOT / "index/cards.json", dump(build_cards(data)), args.check)
    ok &= write_or_check(ROOT / "index/cards.md", build_cards_md(data), args.check)

    counts = Counter(i.get("local_status", "not-run") for i in data["implementations"])
    print(
        f"generated: repos={len(data['repositories'])} impls={len(data['implementations'])} "
        f"entries={len(by_entry) + 1} local={dict(counts)} check={args.check}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
