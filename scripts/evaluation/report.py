"""Markdown 报告片段生成。

只负责把结构化结果变成可读的表格，不做任何统计——
统计必须由 metrics 或脚本自己算好传进来，避免两处口径不一致。
"""

from __future__ import annotations

from typing import Any, Sequence


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        cells = ["" if c is None else str(c) for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def metric_table(metrics: dict[str, Any], title: str = "指标") -> str:
    """把扁平指标字典渲染成两列表格。嵌套字典只展开一层。"""
    rows: list[tuple[str, Any]] = []
    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float, str)):
                    rows.append((f"{key}.{sub_key}", sub_value))
        else:
            rows.append((key, value))
    return f"**{title}**\n\n" + table(["指标", "值"], rows)


def failure_table(
    case_results: Sequence[dict[str, Any]],
    limit: int = 30,
) -> str:
    """失败样本表：只展示前 limit 条，避免报告过长。"""
    fails = [c for c in case_results if not c.get("passed", False)]
    if not fails:
        return "无失败样本。"

    rows = []
    for c in fails[:limit]:
        rows.append(
            (
                c.get("id", "-"),
                c.get("category", "-"),
                _clip(c.get("input", ""), 40),
                _clip(c.get("reason", ""), 80),
            )
        )
    body = table(["ID", "类别", "输入", "失败原因"], rows)
    if len(fails) > limit:
        body += f"\n\n（共 {len(fails)} 条失败，此处显示前 {limit} 条）"
    return body


def _clip(text: Any, width: int) -> str:
    s = str(text).replace("\n", " ").replace("|", "／")
    return s if len(s) <= width else s[: width - 1] + "…"


def section(title: str, body: str, level: int = 2) -> str:
    return f"{'#' * level} {title}\n\n{body}\n"


def limitations_block(items: Sequence[str]) -> str:
    if not items:
        return ""
    return "**评测限制**\n\n" + "\n".join(f"- {x}" for x in items)
