"""数据集加载与版本管理。

每条样本是 JSONL 的一行，字段约定：
    id          唯一标识（字符串）
    category    测试类别（用于分类统计）
    input       输入内容（问题 / 用户消息，或消息列表）
    expected    预期结果（结构随评测类型而定）
    context     可选，上下文（多轮对话、待澄清状态等）
    note        可选，评测说明

数据集版本 = 文件内容 sha256 的前 12 位。
改了任何一条样本，版本号就会变，历史结果不可直接比较——
这是有意设计，防止「偷偷改样本让指标变好看」。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DATASET_DIR = Path(__file__).resolve().parent / "datasets"


def dataset_path(name: str) -> Path:
    return DATASET_DIR / f"{name}.jsonl"


def dataset_version(name: str) -> str:
    """数据集内容哈希，取前 12 位。"""
    path = dataset_path(name)
    if not path.exists():
        return "missing"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:12]


def load_dataset(name: str) -> list[dict[str, Any]]:
    """读取 JSONL 数据集。

    空行与以 # 开头的注释行会被跳过，方便给数据集写说明。
    """
    path = dataset_path(name)
    if not path.exists():
        raise FileNotFoundError(f"数据集不存在：{path}")

    cases: list[dict[str, Any]] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} 第 {lineno} 行不是合法 JSON：{exc}") from exc

        # 兼容三种输入字段名：input（通用）/ probe（安全探测）/ question（容错场景）
        if "input" not in item:
            for alias in ("probe", "question"):
                if alias in item:
                    item["input"] = item[alias]
                    break

        if "id" not in item or "input" not in item:
            raise ValueError(f"{path.name} 第 {lineno} 行缺少 id 或输入字段")
        cases.append(item)

    if not cases:
        raise ValueError(f"数据集为空：{path}")
    return cases


def summarize(cases: list[dict[str, Any]]) -> dict[str, int]:
    """按 category 统计样本数。"""
    out: dict[str, int] = {}
    for c in cases:
        cat = str(c.get("category", "未分类"))
        out[cat] = out.get(cat, 0) + 1
    return out
