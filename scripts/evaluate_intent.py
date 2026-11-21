"""意图识别评测。

改动要点（相对旧脚本）：
1. 样本从 15 条扩到 40 条，按 10 类分布。
2. 指标从单一 Accuracy 扩展为 Accuracy + 各类别 P/R/F1 + Macro 平均。
3. 「服务异常」与「分类错误」分开统计，不再混为一个失败。
4. 结果落盘 JSON，包含每条样本的预期、实际、置信度、耗时。

运行：
    python scripts/evaluate_intent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from app.agent.intent_classifier import VALID_INTENTS, classify_intent  # noqa: E402
from evaluation.common import (  # noqa: E402
    Stopwatch,
    build_payload,
    new_run_id,
    save_result,
)
from evaluation.dataset import dataset_version, load_dataset, summarize  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    accuracy,
    macro_average,
    per_class_prf,
)
from evaluation.report import failure_table, metric_table  # noqa: E402

NAME = "intent"


def run() -> dict:
    cases = load_dataset(NAME)
    version = dataset_version(NAME)
    run_id = new_run_id()

    print("=" * 68)
    print("ShopMate 意图识别评测")
    print(f"run_id={run_id}  数据集版本={version}  样本数={len(cases)}")
    print("类别分布：", summarize(cases))
    print("=" * 68)

    y_true: list[str] = []
    y_pred: list[str] = []
    case_results: list[dict] = []
    infra_errors = 0

    for case in cases:
        ctx = case.get("context")
        history = case.get("history")

        # 数据集字段类型写错时直接报出是哪一条，而不是让整个评测中途崩掉。
        # 这个坑真踩过：把 history 写成了 context（list），
        # 结果 classify_intent 里 pending_clarification.get() 抛
        # AttributeError，47 条已经跑完的结果全丢。
        if ctx is not None and not isinstance(ctx, dict):
            raise ValueError(
                f"样本 {case.get('id')} 的 context 必须是对象（dict），"
                f"实际是 {type(ctx).__name__}。"
                "多轮对话历史请用 history 字段。"
            )

        if history is not None and not isinstance(history, list):
            raise ValueError(
                f"样本 {case.get('id')} 的 history 必须是数组（list），"
                f"实际是 {type(history).__name__}"
            )

        sw = Stopwatch()
        result = classify_intent(
            case["input"],
            pending_clarification=ctx,
            history=history,
        )
        elapsed = sw.ms()

        expected = case["expected"]["intent"]
        actual = result.intent
        is_infra_error = result.error is not None
        if is_infra_error:
            infra_errors += 1

        passed = (actual == expected) and not is_infra_error

        y_true.append(expected)
        y_pred.append(actual)

        case_results.append(
            {
                "id": case["id"],
                "category": case.get("category", "-"),
                "input": case["input"],
                "multi_turn": bool(case.get("multi_turn")),
                "expected": expected,
                "actual": actual,
                "confidence": round(float(result.confidence), 4),
                "infra_error": is_infra_error,
                "error": result.error,
                "elapsed_ms": elapsed,
                "passed": passed,
                "reason": (
                    "服务异常" if is_infra_error
                    else ("" if passed else f"预期 {expected}，实际 {actual}")
                ),
            }
        )

        flag = "PASS" if passed else "FAIL"
        print(
            f"[{flag}] {case['id']} 预期={expected} 实际={actual} "
            f"conf={result.confidence:.2f} {elapsed:.0f}ms"
        )

    per_class = per_class_prf(y_true, y_pred, sorted(VALID_INTENTS))
    macro = macro_average(per_class)

    metrics = {
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "infra_error_rate": round(infra_errors / len(cases), 4),
        "multi_turn_accuracy": round(
            sum(1 for c in case_results if c["multi_turn"] and c["passed"])
            / max(1, sum(1 for c in case_results if c["multi_turn"])),
            4,
        ),
        "per_class": {
            label: {
                "precision": v["precision"],
                "recall": v["recall"],
                "f1": v["f1"],
                "support": v["support"],
            }
            for label, v in per_class.items()
            if v["support"] > 0
        },
    }

    passed_cases = sum(1 for c in case_results if c["passed"])
    payload = build_payload(
        evaluation_name=NAME,
        run_id=run_id,
        dataset_version=version,
        total_cases=len(cases),
        passed_cases=passed_cases,
        failed_cases=len(cases) - passed_cases,
        metrics=metrics,
        case_results=case_results,
        limitations=[
            "模型返回的 confidence 未经概率校准，不能当作真实置信度使用。",
            "单条样本一次调用，未做重复采样，因此不反映模型稳定性。",
            f"模型为服务端别名 {payload_model()}，无法固定具体版本。",
            f"{len(cases)} 条样本中部分类别仅 1-3 条，类别级指标波动较大。",
        ],
    )
    path = save_result(NAME, run_id, payload)

    print("\n" + "=" * 68)
    print(metric_table(metrics, "意图识别指标"))
    print()
    print(failure_table(case_results))
    print(f"\n结果已保存：{path}")
    return payload


def payload_model() -> str:
    from evaluation.common import environment_snapshot

    return str(environment_snapshot().get("llm_model", "unknown"))


if __name__ == "__main__":
    run()
