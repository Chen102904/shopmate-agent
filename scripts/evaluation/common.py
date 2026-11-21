"""评测公共工具：运行标识、环境快照、结果落盘。

设计原则：
1. 每次运行有唯一 run_id，结果独立目录，互不覆盖。
2. 环境快照记录「这次结果是在什么条件下产生的」，保证可追溯。
3. 只用标准库，避免评测本身引入新依赖。
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# 项目根目录：.../shopmate-agent
ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results"


def new_run_id() -> str:
    """生成运行 ID，格式 20261120T113825-a1b2c3。"""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _pkg_version(name: str) -> str:
    """读取已安装包版本，失败返回 unknown。"""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def environment_snapshot() -> dict[str, Any]:
    """采集本次运行的环境信息。

    注意：模型服务名（如 deepseek-flash）是服务端别名，
    不是可固定的版本号，报告中必须如实说明这一点。
    """
    snap: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: _pkg_version(name)
            for name in (
                "fastapi",
                "openai",
                "sentence-transformers",
                "numpy",
                "pytest",
            )
        },
    }

    # 模型与 RAG 参数：能读到就记，读不到就标 unknown
    try:
        from app.core.llm_client import DEEPSEEK_MODEL, DEEPSEEK_BASE_URL

        snap["llm_model"] = DEEPSEEK_MODEL
        snap["llm_base_url"] = DEEPSEEK_BASE_URL
    except Exception:
        snap["llm_model"] = "unknown"

    try:
        from app.rag.retriever import DEFAULT_SCORE_THRESHOLD

        snap["rag_score_threshold"] = DEFAULT_SCORE_THRESHOLD
    except Exception:
        snap["rag_score_threshold"] = "unknown"

    snap["embedding_model"] = "bge-small-zh-v1.5 (local)"
    snap["temperature"] = 0.0  # 项目未显式设置，按默认记录

    # 知识库版本。
    #
    # 只记录数据集版本是不够的：
    #     「这次 Hit@3 掉了 4 个点」到底是模型波动，
    #     还是有人改了知识库？没有这一项就永远说不清。
    #     记上之后，两次运行的知识库版本不同，比较指标时
    #     就必须先说明这一点，不能直接当成「改代码导致的效果」。
    try:
        from app.rag.knowledge_version import compute_knowledge_version

        snap["knowledge_version"] = compute_knowledge_version()
    except Exception:
        snap["knowledge_version"] = "unknown"

    return snap


def save_result(
    evaluation_name: str,
    run_id: str,
    payload: dict[str, Any],
) -> Path:
    """把一次评测结果写成 JSON，返回文件路径。"""
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{evaluation_name}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def build_payload(
    *,
    evaluation_name: str,
    run_id: str,
    dataset_version: str,
    total_cases: int,
    passed_cases: int,
    failed_cases: int,
    metrics: dict[str, Any],
    case_results: list[dict[str, Any]],
    error_details: list[dict[str, Any]] | None = None,
    limitations: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """组装统一的结果结构。

    字段名与任务书约定一致，便于跨模块汇总。
    """
    payload = {
        "run_id": run_id,
        # 批次号：由 run_all_evaluations 下发给子模块。
        # 每个模块自己会生成 run_id（时间戳不同），所以单看 run_id
        # 无法区分「同一次全量运行的十个模块」和「后来单独重跑的某个模块」。
        # 批次号只回答一个问题：这些数字是不是同一次跑出来的。
        "batch_id": os.getenv("SHOPMATE_EVAL_BATCH") or run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "evaluation_name": evaluation_name,
        "dataset_version": dataset_version,
        "environment": environment_snapshot(),
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "metrics": metrics,
        "case_results": case_results,
        "error_details": error_details or [],
        "limitations": limitations or [],
    }
    if extra:
        payload.update(extra)
    return payload


class Stopwatch:
    """简单计时器，用于记录单条样本耗时。"""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 1)
