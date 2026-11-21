"""工具选择评测。

改动要点（相对旧脚本）：
1. 样本从 5 条扩到 30 条，含 4 条负样本（不该调工具）。
2. 保留完整调用轨迹（list，不转 set），可识别重复调用与顺序错误。
3. 指标：Accuracy / Precision / Recall / F1 + 不必要调用率 + 重复调用率 + 顺序正确率。
4. 负样本单独统计，不混进主指标掩盖问题。

运行：
    python scripts/evaluate_tool_selection.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from app.agent.agent import run_agent  # noqa: E402
from app.agent.conversation_store import clear_conversation  # noqa: E402
from evaluation.common import (  # noqa: E402
    Stopwatch,
    build_payload,
    environment_snapshot,
    new_run_id,
    save_result,
)
from evaluation.dataset import dataset_version, load_dataset, summarize  # noqa: E402
from evaluation.metrics import ratio, sequence_match, set_prf  # noqa: E402
from evaluation.report import failure_table, metric_table  # noqa: E402

NAME = "tool_selection"


def _fake_tool(payload: dict, trace: list[str]):
    """构造工具桩：只记录名称，不执行真实业务。"""

    # conversation_id 必须接收并忽略：
    # 主链路现在会给工具多传一个会话上下文（写操作要用），
    # 桩函数签名不跟上，patch 之后每次工具调用都会 TypeError。
    def fake_execute_tool(
        tool_name: str,
        arguments: dict,
        conversation_id: str | None = None,
    ) -> str:
        trace.append(tool_name)
        if tool_name == "search_after_sales_policy":
            return json.dumps(
                {
                    "query": arguments.get("query", ""),
                    "answer": {"answer": "模拟政策检索结果", "sources": []},
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {"mock": True, "tool": tool_name, "arguments": arguments},
            ensure_ascii=False,
        )

    return fake_execute_tool


def run() -> dict:
    cases = load_dataset(NAME)
    version = dataset_version(NAME)
    run_id = new_run_id()

    print("=" * 68)
    print("ShopMate 工具选择评测")
    print(f"run_id={run_id}  数据集版本={version}  样本数={len(cases)}")
    print("类别分布：", summarize(cases))
    print("=" * 68)

    case_results: list[dict] = []
    # 集合级累计（用于 micro 平均）
    sum_tp = sum_fp = sum_fn = 0
    seq_correct = seq_total = 0
    unnecessary_calls = total_actual_calls = 0
    duplicate_cases = 0
    negative_total = negative_passed = 0

    for case in cases:
        conv = f"eval_tool_sel_{case['id']}"
        clear_conversation(conv)
        trace: list[str] = []
        expected = case["expected"]["tools"]
        order_sensitive = case["expected"].get("order_sensitive", False)

        sw = Stopwatch()
        error = None
        try:
            with patch(
                "app.agent.agent.execute_tool",
                side_effect=_fake_tool(case, trace),
            ):
                run_agent(user_message=case["input"], conversation_id=conv)
        except Exception as exc:  # 记录但不中断
            error = f"{type(exc).__name__}: {exc}"
        elapsed = sw.ms()
        clear_conversation(conv)

        exp_set, act_set = set(expected), set(trace)
        set_ok = exp_set == act_set
        seq_ok = sequence_match(expected, trace)
        passed = set_ok and (seq_ok if order_sensitive else True) and error is None

        # 集合级 P/R/F1 素材
        sum_tp += len(exp_set & act_set)
        sum_fp += len(act_set - exp_set)
        sum_fn += len(exp_set - act_set)

        if order_sensitive:
            seq_total += 1
            if seq_ok:
                seq_correct += 1

        total_actual_calls += len(trace)
        unnecessary_calls += len(act_set - exp_set)

        has_dup = len(trace) != len(act_set)
        if has_dup:
            duplicate_cases += 1

        if not expected:
            negative_total += 1
            if passed:
                negative_passed += 1

        if error:
            reason = f"执行异常：{error}"
        elif not set_ok:
            reason = f"集合不符：预期 {sorted(exp_set)}，实际 {trace}"
        elif order_sensitive and not seq_ok:
            reason = f"顺序不符：预期 {expected}，实际 {trace}"
        elif has_dup:
            reason = f"存在重复调用：{trace}"
        else:
            reason = ""

        case_results.append(
            {
                "id": case["id"],
                "category": case.get("category", "-"),
                "input": case["input"],
                "expected_tools": expected,
                "actual_tools": trace,
                "order_sensitive": order_sensitive,
                "set_match": set_ok,
                "sequence_match": seq_ok,
                "duplicate": has_dup,
                "unnecessary": sorted(act_set - exp_set),
                "missing": sorted(exp_set - act_set),
                "elapsed_ms": elapsed,
                "passed": passed,
                "reason": reason,
            }
        )

        print(
            f"[{'PASS' if passed else 'FAIL'}] {case['id']} "
            f"预期={expected} 实际={trace} {elapsed:.0f}ms"
        )

    prf = set_prf(
        [t for c in case_results for t in c["expected_tools"]],
        [t for c in case_results for t in c["actual_tools"]],
    )
    main_cases = [c for c in case_results if c["expected_tools"]]
    main_passed = sum(1 for c in main_cases if c["passed"])

    metrics = {
        "tool_selection_accuracy": ratio(
            sum(1 for c in case_results if c["passed"]), len(case_results)
        ),
        "positive_case_accuracy": ratio(main_passed, len(main_cases)),
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "unnecessary_call_rate": ratio(unnecessary_calls, total_actual_calls),
        "duplicate_call_rate": ratio(duplicate_cases, len(case_results)),
        "sequence_accuracy": ratio(seq_correct, seq_total),
        "negative_case_accuracy": ratio(negative_passed, max(1, negative_total)),
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
            f"模型为服务端别名 {environment_snapshot().get('llm_model')}，无法固定版本。",
            "工具被替换为桩函数，只验证「选了什么工具」，不验证工具内部是否正确。",
            "sequence_accuracy 仅在 order_sensitive=true 的样本上计算，样本数较少。",
            "单次采样，模型输出的随机性会带来波动。",
        ],
    )
    path = save_result(NAME, run_id, payload)

    print("\n" + "=" * 68)
    print(metric_table(metrics, "工具选择指标"))
    print()
    print(failure_table(case_results))
    print(f"\n结果已保存：{path}")
    return payload


if __name__ == "__main__":
    run()
