"""Agent 端到端评测（分环节归因）。

改动要点：
1. 样本 6 → 30 条，覆盖 12 类场景（含多轮、缺参、越界、无知识、工具故障）。
2. 不再只看最终回答：用 spy 同时采集
   - 链路内真实意图（patch classify_intent 记录返回值）
   - 完整工具调用轨迹与参数（patch execute_tool）
   - 最终回答
3. 每个环节单独判定，失败可归因到：意图 / 工具 / 参数 / 业务结果 / 回答。
4. 每条样本输出 11 个字段（任务书要求），便于人工复核。

运行：
    python scripts/evaluate_agent_e2e.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from app.agent import agent as agent_module  # noqa: E402
from app.agent.agent import run_agent  # noqa: E402
from app.agent.conversation_store import clear_conversation  # noqa: E402
from app.agent.tool_executor import execute_tool as real_execute_tool  # noqa: E402
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

NAME = "agent_e2e"
DATASET = "e2e"


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _contains_any(answer: str, groups: list[list[str]]) -> tuple[bool, list[list[str]]]:
    a = _norm(answer)
    failed = [g for g in groups if not any(_norm(k) in a for k in g)]
    return (not failed), failed


def run() -> dict:
    cases = load_dataset(DATASET)
    version = dataset_version(DATASET)
    run_id = new_run_id()

    print("=" * 68)
    print("ShopMate Agent 端到端评测")
    print(f"run_id={run_id}  数据集版本={version}  样本数={len(cases)}")
    print("类别分布：", summarize(cases))
    print("=" * 68)

    case_results: list[dict] = []
    intent_ok_n = tool_ok_n = param_ok_n = answer_ok_n = 0
    intent_total = tool_total = param_total = answer_total = 0
    multi_total = multi_pass = 0
    exception_total = exception_pass = 0

    for case in cases:
        input_data = case["input"]
        turns = input_data if isinstance(input_data, list) else [input_data]
        conv = f"e2e_{case['id']}"
        clear_conversation(conv)

        trace: list[dict] = []
        intents: list[str] = []
        answer = ""
        error = None
        mock_cfg = case.get("mock", {})

        real_classify = agent_module.classify_intent

        def spy_classify_impl(
            user_message,
            pending_clarification=None,
            history=None,
        ):
            # 必须原样透传 history：
            # 主链路现在会把对话历史交给分类器做指代消解，
            # spy 不接收就会 TypeError，整条链路跑不起来。
            result = real_classify(
                user_message,
                pending_clarification=pending_clarification,
                history=history,
            )
            intents.append(result.intent)
            return result

        def spy_tool(
            tool_name: str,
            arguments: dict,
            conversation_id: str | None = None,
        ) -> str:
            trace.append({"tool": tool_name, "arguments": arguments})
            if mock_cfg.get("tool_error"):
                return json.dumps(
                    {"error": "工具执行失败：模拟故障（用于容错评测）"},
                    ensure_ascii=False,
                )
            # conversation_id 必须透传：
            # 写操作类工具靠它绑定草稿和读取确认消息，
            # spy 吞掉这个参数，售后申请类用例会全部失败。
            return real_execute_tool(
                tool_name=tool_name,
                arguments=arguments,
                conversation_id=conversation_id,
            )

        sw = Stopwatch()
        try:
            with patch(
                "app.agent.agent.classify_intent", side_effect=spy_classify_impl
            ), patch("app.agent.agent.execute_tool", side_effect=spy_tool):
                for msg in turns:
                    answer = run_agent(
                        user_message=msg,
                        conversation_id=conv,
                        # 样本可声明归属人：查 ORD003 的用例必须以 USER002 执行，
                        # 否则测到的是权限拦截，而不是业务逻辑是否正确。
                        user_id=case.get("user_id", "USER001"),
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = sw.ms()
        clear_conversation(conv)

        exp = case["expected"]
        actual_tools = [t["tool"] for t in trace]
        exp_tools = exp.get("tools", [])

        # 意图：链路内最后一次的真实意图（多轮取最后一轮）
        actual_intent = intents[-1] if intents else ""
        intent_ok = (actual_intent == exp.get("intent")) if exp.get("intent") else None

        # 工具：集合比较 + 是否必须有工具
        tools_required = exp.get("tools_required", True)
        if tools_required:
            tool_ok = set(exp_tools) == set(actual_tools)
        else:
            # 不要求调工具的场景：调了就算错（除显式允许）
            tool_ok = len(actual_tools) == 0

        # 参数：关键参数是否出现在实际调用里
        key_params = exp.get("key_params", {})
        if key_params:
            param_ok = any(
                all(call["arguments"].get(k) == v for k, v in key_params.items())
                for call in trace
            )
        else:
            param_ok = True

        # 回答
        groups = exp.get("must_have", [])
        answer_ok, failed_groups = _contains_any(answer, groups)

        passed = (
            error is None
            and intent_ok is not False
            and tool_ok
            and param_ok
            and answer_ok
        )

        # 归因
        if error:
            reason = f"执行异常：{error}"
        elif intent_ok is False:
            reason = f"意图不符：预期 {exp.get('intent')}，实际 {actual_intent}"
        elif not tool_ok:
            reason = f"工具不符：预期 {exp_tools}，实际 {actual_tools}"
        elif not param_ok:
            reason = f"关键参数不符：期望 {key_params}"
        elif not answer_ok:
            reason = f"回答缺少关键信息：{failed_groups}"
        else:
            reason = ""

        # 分项累计
        if intent_ok is not None:
            intent_total += 1
            intent_ok_n += int(bool(intent_ok))
        tool_total += 1
        tool_ok_n += int(tool_ok)
        param_total += 1
        param_ok_n += int(param_ok)
        answer_total += 1
        answer_ok_n += int(answer_ok)

        is_multi = isinstance(input_data, list) and len(input_data) > 1
        if is_multi:
            multi_total += 1
            multi_pass += int(passed)
        if case["category"] in ("tool_error", "no_knowledge", "out_of_scope"):
            exception_total += 1
            exception_pass += int(passed)

        case_results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "user_input": input_data,
                "expected_intent": exp.get("intent"),
                "actual_intent": actual_intent,
                "expected_tools": exp_tools,
                "actual_tools": actual_tools,
                "expected_key_params": key_params,
                "actual_calls": trace,
                "expected_answer_contains": groups,
                "final_answer": answer,
                "intent_ok": intent_ok,
                "tools_ok": tool_ok,
                "params_ok": param_ok,
                "answer_ok": answer_ok,
                "elapsed_ms": elapsed,
                "passed": passed,
                "reason": reason,
            }
        )

        print(
            f"[{'PASS' if passed else 'FAIL'}] {case['id']} "
            f"intent={actual_intent} tools={actual_tools} {elapsed:.0f}ms"
            + ("" if passed else f"  ← {reason}")
        )

    metrics = {
        "e2e_pass_rate": ratio(
            sum(1 for c in case_results if c["passed"]), len(cases)
        ),
        "intent_accuracy": ratio(intent_ok_n, intent_total),
        "tool_trajectory_accuracy": ratio(tool_ok_n, tool_total),
        "key_param_accuracy": ratio(param_ok_n, param_total),
        "answer_content_accuracy": ratio(answer_ok_n, answer_total),
        "multi_turn_pass_rate": ratio(multi_pass, multi_total),
        "exception_case_pass_rate": ratio(exception_pass, exception_total),
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
            "多轮样本按「最后一轮意图 + 全量轨迹」判定，中间轮次的错误不单独计分。",
            "工具故障场景通过桩函数注入，属 mock 测试，非真实故障演练。",
            "回答判定仍为关键词组，未做语义等价判定。",
            "单次运行，未重复采样，不能反映稳定性。",
        ],
    )
    path = save_result(NAME, run_id, payload)

    print("\n" + "=" * 68)
    print(metric_table(metrics, "Agent 端到端指标（分环节）"))
    print()
    print(failure_table(case_results))
    print(f"\n结果已保存：{path}")
    return payload


if __name__ == "__main__":
    run()
