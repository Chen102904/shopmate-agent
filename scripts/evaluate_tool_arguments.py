"""工具参数评测（本次改造重点）。

相对旧脚本的改动：
1. 样本 5 → 30 条。
2. 不再「整体字典比较」给单一 0/1，改为三层计分，单位不混用：
   - 参数级：单个 (tool, key, value) 三元组
   - 调用级：单次 tool call 的全部参数是否完全匹配
   - 样本级：整条样本的全部调用序列是否完全匹配
3. 新增类型校验（types 声明）、多余参数检测、缺失参数检测。
4. query 参数允许改写：只校验非空，不要求与预期字符串逐字相等（RAG 的 query 属于自然语言）。
5. 结果落盘 JSON，含每条样本的三层明细。

统计口径说明：
- 旧版 TEST_REPORT.md 记的「工具参数 9/10」在旧代码（5 条样本）下无法复现，
  已标记为「历史结果，口径不明」，本脚本输出的数字为唯一有效口径。

运行：
    python scripts/evaluate_tool_arguments.py
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
from evaluation.metrics import ratio  # noqa: E402
from evaluation.report import failure_table, metric_table  # noqa: E402

NAME = "tool_arguments"

TYPE_MAP = {"str": str, "int": int, "float": (int, float), "bool": bool}


def _fake_tool(trace: list[dict]):
    # conversation_id 必须接收并忽略：
    # 主链路现在会给工具多传一个会话上下文（写操作要用），
    # 桩函数签名不跟上，patch 之后每次工具调用都会 TypeError，
    # 整个评测会变成「全挂」而不是「测出问题」。
    def fake_execute_tool(
        tool_name: str,
        arguments: dict,
        conversation_id: str | None = None,
    ) -> str:
        trace.append({"tool": tool_name, "arguments": arguments})
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


def _compare_call(exp_call: dict, act_call: dict | None, flexible_query: bool) -> dict:
    """比较单次调用，返回各维度结果。"""
    exp_args = exp_call.get("arguments", {})
    exp_types = exp_call.get("types", {})

    if act_call is None:
        return {
            "tool_ok": False,
            "arg_total": len(exp_args),
            "arg_value_correct": 0,
            "arg_key_present": 0,
            "key_set_equal": False,
            "extra_keys": [],
            "missing_keys": sorted(exp_args),
            "type_checked": 0,
            "type_correct": 0,
            "exact": False,
            "flexible_used": False,
            "note": "未发生该次调用",
        }

    act_args = act_call.get("arguments") or {}
    tool_ok = exp_call["tool"] == act_call.get("tool")

    value_correct = 0
    key_present = 0
    flexible_used = False

    for key, exp_value in exp_args.items():
        if key not in act_args:
            continue
        key_present += 1
        act_value = act_args[key]
        if flexible_query and key == "query":
            if isinstance(act_value, str) and act_value.strip():
                value_correct += 1
                flexible_used = True
        elif act_value == exp_value:
            value_correct += 1

    type_checked = 0
    type_correct = 0
    for key, type_name in exp_types.items():
        if key in act_args:
            type_checked += 1
            expected_py = TYPE_MAP.get(type_name)
            if expected_py and isinstance(act_args[key], expected_py) and not isinstance(
                act_args[key], bool
            ):
                type_correct += 1

    extra_keys = sorted(set(act_args) - set(exp_args))
    missing_keys = sorted(set(exp_args) - set(act_args))
    key_set_equal = not extra_keys and not missing_keys

    exact = (
        tool_ok
        and key_set_equal
        and value_correct == len(exp_args)
    )

    return {
        "tool_ok": tool_ok,
        "arg_total": len(exp_args),
        "arg_value_correct": value_correct,
        "arg_key_present": key_present,
        "key_set_equal": key_set_equal,
        "extra_keys": extra_keys,
        "missing_keys": missing_keys,
        "type_checked": type_checked,
        "type_correct": type_correct,
        "exact": exact,
        "flexible_used": flexible_used,
        "note": "",
    }


def run() -> dict:
    cases = load_dataset(NAME)
    version = dataset_version(NAME)
    run_id = new_run_id()

    print("=" * 68)
    print("ShopMate 工具参数评测（三层计分）")
    print(f"run_id={run_id}  数据集版本={version}  样本数={len(cases)}")
    print("类别分布：", summarize(cases))
    print("=" * 68)

    case_results: list[dict] = []

    # 累计量
    tool_name_correct = tool_name_total = 0
    arg_present = arg_total = 0
    arg_value_correct = 0
    key_set_equal_calls = 0
    call_total = 0
    call_exact = 0
    type_checked = type_correct = 0
    case_exact = 0
    multi_total = multi_exact = 0

    for case in cases:
        conv = f"eval_tool_arg_{case['id']}"
        clear_conversation(conv)
        trace: list[dict] = []
        expected = case["expected"]
        exp_calls = expected.get("calls", [])
        flexible = bool(expected.get("query_flexible"))

        sw = Stopwatch()
        error = None
        try:
            with patch(
                "app.agent.agent.execute_tool",
                side_effect=_fake_tool(trace),
            ):
                run_agent(user_message=case["input"], conversation_id=conv)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = sw.ms()
        clear_conversation(conv)

        # 逐调用比较（按 index 对齐，以便发现顺序/数量问题）
        call_details = []
        for idx, exp_call in enumerate(exp_calls):
            act_call = trace[idx] if idx < len(trace) else None
            detail = _compare_call(exp_call, act_call, flexible)
            call_details.append({"index": idx, **detail})

            tool_name_total += 1
            tool_name_correct += int(detail["tool_ok"])
            arg_total += detail["arg_total"]
            arg_present += detail["arg_key_present"]
            arg_value_correct += detail["arg_value_correct"]
            type_checked += detail["type_checked"]
            type_correct += detail["type_correct"]
            call_total += 1
            call_exact += int(detail["exact"])
            key_set_equal_calls += int(detail["key_set_equal"])

        extra_calls = trace[len(exp_calls):]

        sample_exact = (
            error is None
            and len(trace) == len(exp_calls)
            and all(d["exact"] for d in call_details)
        )
        case_exact += int(sample_exact)

        if len(exp_calls) > 1:
            multi_total += 1
            multi_exact += int(sample_exact)

        # 失败原因
        if error:
            reason = f"执行异常：{error}"
        elif extra_calls:
            reason = f"多出 {len(extra_calls)} 次调用：{[c['tool'] for c in extra_calls]}"
        elif len(trace) < len(exp_calls):
            reason = f"少调用：预期 {len(exp_calls)} 次，实际 {len(trace)} 次"
        elif not sample_exact:
            bad = [
                d for d in call_details if not d["exact"]
            ]
            parts = []
            for d in bad:
                if d["missing_keys"]:
                    parts.append(f"缺参数 {d['missing_keys']}")
                if d["extra_keys"]:
                    parts.append(f"多参数 {d['extra_keys']}")
                if d["arg_value_correct"] < d["arg_total"]:
                    parts.append(
                        f"值不符（{d['arg_value_correct']}/{d['arg_total']} 正确）"
                    )
                if not d["tool_ok"]:
                    parts.append("工具名不符")
            reason = "；".join(parts) or "未完全匹配"
        else:
            reason = ""

        case_results.append(
            {
                "id": case["id"],
                "category": case.get("category", "-"),
                "input": case["input"],
                "expected_calls": exp_calls,
                "actual_calls": trace,
                "call_details": call_details,
                "extra_calls": extra_calls,
                "flexible_query": flexible,
                "sample_exact": sample_exact,
                "elapsed_ms": elapsed,
                "passed": sample_exact,
                "reason": reason,
            }
        )

        print(
            f"[{'PASS' if sample_exact else 'FAIL'}] {case['id']} "
            f"预期 {len(exp_calls)} 次 / 实际 {len(trace)} 次 {elapsed:.0f}ms"
            + ("" if sample_exact else f"  ← {reason}")
        )

    metrics = {
        "tool_name_accuracy": ratio(tool_name_correct, tool_name_total),
        "required_argument_completeness": ratio(arg_present, arg_total),
        "argument_key_accuracy": ratio(key_set_equal_calls, call_total),
        "argument_value_accuracy": ratio(arg_value_correct, arg_total),
        "argument_type_accuracy": ratio(type_correct, type_checked),
        "call_level_exact_match": ratio(call_exact, call_total),
        "sample_level_exact_match": ratio(case_exact, len(cases)),
        "multi_call_complete_match": ratio(multi_exact, multi_total),
        "_raw": {
            "tool_name_correct": tool_name_correct,
            "tool_name_total": tool_name_total,
            "arg_present": arg_present,
            "arg_total": arg_total,
            "arg_value_correct": arg_value_correct,
            "call_exact": call_exact,
            "call_total": call_total,
            "case_exact": case_exact,
            "case_total": len(cases),
            "type_checked": type_checked,
            "type_correct": type_correct,
        },
    }

    payload = build_payload(
        evaluation_name=NAME,
        run_id=run_id,
        dataset_version=version,
        total_cases=len(cases),
        passed_cases=case_exact,
        failed_cases=len(cases) - case_exact,
        metrics=metrics,
        case_results=case_results,
        limitations=[
            "query 参数属自然语言，按「非空且为字符串」放宽校验，未与预期逐字比对。",
            f"模型为服务端别名 {environment_snapshot().get('llm_model')}，无法固定版本。",
            "工具被桩函数替换，参数只校验模型输出，不校验工具内部使用是否正确。",
            "旧版 TEST_REPORT.md 的「工具参数 9/10」在 5 条样本的旧代码下无法复现，已标记为口径不明。",
            "单次采样，同一输入重跑结果可能有差异。",
        ],
        extra={"legacy_metric_note": "历史 9/10 口径不明，不作为有效成绩"},
    )
    path = save_result(NAME, run_id, payload)

    print("\n" + "=" * 68)
    print(metric_table(metrics, "工具参数指标（三层计分）"))
    print()
    print(failure_table(case_results))
    print(f"\n结果已保存：{path}")
    return payload


if __name__ == "__main__":
    run()
