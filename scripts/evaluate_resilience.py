"""容错能力评测（新增）。

用 unittest.mock 注入 8 类故障，验证降级路径是否按预期工作。

必须明确：**这是 mock 注入测试，不等于真实故障演练。**
真实的网络抖动、限流、DNS 故障、上游降级不在本测试覆盖范围内。

运行：
    python scripts/evaluate_resilience.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAIError  # noqa: E402

from app.agent.agent import MODEL_UNAVAILABLE_MESSAGE, run_agent  # noqa: E402
from app.agent.conversation_store import clear_conversation  # noqa: E402
from app.agent.tool_executor import execute_tool  # noqa: E402
from evaluation.common import (  # noqa: E402
    Stopwatch,
    build_payload,
    new_run_id,
    save_result,
)
from evaluation.dataset import dataset_version, load_dataset  # noqa: E402
from evaluation.metrics import ratio  # noqa: E402
from evaluation.report import failure_table, metric_table  # noqa: E402

NAME = "resilience"

# 记录最近一次 run_agent 的回答，供失败案例落盘用
LAST_ANSWER = ""

# 降级提示词：只要出现其一，说明给用户的是可读提示而不是崩溃
DEGRADE_MARKERS = ["稍后重试", "暂时无法", "暂时不可用", "服务暂时"]


class FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, index: int):
        self.id = f"call_{index}"
        self.type = "function"
        self.function = FakeFunction("get_order", '{"order_id": "ORD001"}')


class FakeMessage:
    """永远要求调用工具的假消息，用于触发超轮数。"""

    def __init__(self):
        self.content = None
        self.tool_calls = [FakeToolCall(0)]

    def model_dump(self, exclude_none: bool = True) -> dict:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "get_order",
                        "arguments": '{"order_id": "ORD001"}',
                    },
                }
            ],
        }


class FakeChoice:
    def __init__(self):
        self.message = FakeMessage()


class FakeResponse:
    def __init__(self):
        self.choices = [FakeChoice()]


def _run_with(question: str, conv_suffix: str, **patches) -> tuple[str, str | None, float]:
    """在给定 patch 下跑一次 run_agent，返回 (回答, 异常, 耗时)。"""
    global LAST_ANSWER
    conv = f"resilience_{conv_suffix}"
    clear_conversation(conv)
    sw = Stopwatch()
    answer, error = "", None
    ctx = []
    for target, kwargs in patches.items():
        ctx.append(patch(target, **kwargs))

    try:
        for p in ctx:
            p.start()
        try:
            answer = run_agent(user_message=question, conversation_id=conv)
        finally:
            for p in reversed(ctx):
                p.stop()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = sw.ms()
    clear_conversation(conv)
    LAST_ANSWER = answer
    return answer, error, elapsed


def _(fault: str) -> tuple[str, str | None, float, str, bool]:
    """执行单个故障场景，返回 (说明, 异常, 耗时, 实际表现摘要, 是否通过)。"""
    if fault == "llm_timeout":
        answer, error, ms = _run_with(
            "帮我查一下 ORD001 的订单状态",
            "f01",
            **{"app.core.llm_client.get_llm_client": {"side_effect": OpenAIError("模拟超时")}},
        )
        if error:
            return "主模型超时后未降级，直接抛异常", error, ms, error, False
        degraded = MODEL_UNAVAILABLE_MESSAGE in answer
        passed = not degraded  # 期望切到备用模型成功，而不是走兜底文案
        return (
            "主模型强制抛错，观察是否降级到备用模型",
            None,
            ms,
            "走兜底文案（备用模型也不可用）" if degraded else "备用模型接管并给出回答",
            passed,
        )

    if fault == "llm_500":
        answer, error, ms = _run_with(
            "ORD001 能退款吗",
            "f02",
            **{"app.core.llm_client.get_llm_client": {"side_effect": OpenAIError("模拟 500")}},
        )
        if error:
            return "主模型 500 后未降级", error, ms, error, False
        degraded = MODEL_UNAVAILABLE_MESSAGE in answer
        return (
            "主模型强制抛错（500 语义），观察降级",
            None,
            ms,
            "走兜底文案" if degraded else "备用模型接管",
            not degraded,
        )

    if fault == "all_llm_down":
        side = {"side_effect": OpenAIError("模拟全部不可用")}
        answer, error, ms = _run_with(
            "商品签收超过7天还能退款吗",
            "f03",
            **{
                "app.core.llm_client.get_llm_client": side,
                "app.core.llm_client.get_zhipu_client": side,
                "app.core.llm_client.get_qwen_client": side,
            },
        )
        has_marker = any(m in answer for m in DEGRADE_MARKERS)
        is_model_msg = MODEL_UNAVAILABLE_MESSAGE in answer
        passed = error is None and has_marker
        if error:
            detail = f"异常穿透：{error}"
        elif is_model_msg:
            detail = "返回「模型不可用」提示（准确）"
        elif has_marker:
            detail = (
                "给出了降级提示，但归因不准：用户会以为是自己表达问题 → "
                + answer[:50].replace("\n", " ")
            )
        else:
            detail = "未给出任何降级提示"
        return ("三个模型全部不可用", error, ms, detail, passed)

    if fault == "tool_exception":
        answer, error, ms = _run_with(
            "帮我查一下 ORD001 的订单状态",
            "f04",
            **{
                "app.agent.agent.execute_tool": {
                    "side_effect": RuntimeError("模拟工具崩溃")
                }
            },
        )
        passed = error is None
        return (
            "工具执行时抛 RuntimeError",
            error,
            ms,
            "异常被隔离，Agent 仍返回回答" if passed else f"异常穿透到调用方：{error}",
            passed,
        )

    if fault == "tool_bad_json":
        answer, error, ms = _run_with(
            "查 ORD001 订单",
            "f05",
            **{
                "app.agent.agent.execute_tool": {
                    "return_value": "这不是一个合法的 JSON 结果"
                }
            },
        )
        passed = error is None
        return (
            "工具返回不可解析文本",
            error,
            ms,
            "未崩溃，模型基于脏数据作答" if passed else f"崩溃：{error}",
            passed,
        )

    if fault == "unknown_tool":
        sw = Stopwatch()
        result = execute_tool(tool_name="not_a_real_tool", arguments={})
        ms = sw.ms()
        passed = '"error"' in result
        return (
            "直接以非法工具名调用执行器",
            None,
            ms,
            f"返回：{result[:80]}",
            passed,
        )

    if fault == "max_rounds":
        answer, error, ms = _run_with(
            "查 ORD001 订单，再看物流，再看能不能退",
            "f07",
            **{
                "app.agent.agent.create_chat_completion_with_fallback": {
                    "return_value": FakeResponse()
                }
            },
        )
        passed = error is None and ("上限" in answer or "稍后重试" in answer)
        return (
            "模型永远返回 tool_calls，验证 5 轮上限",
            error,
            ms,
            answer[:60] if answer else (error or "无回答"),
            passed,
        )

    if fault == "rag_missing_model":
        # 制造模型目录缺失：让 retriever 初始化时抛错
        sw = Stopwatch()
        answer, error = "", None
        try:
            with patch(
                "app.rag.retriever.MODEL_DIR",
                new=Path("__not_exists__/bge"),
            ), patch("app.agent.tool_executor._rag_service", None):
                answer = run_agent(
                    user_message="退货政策是什么",
                    conversation_id="resilience_f08",
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        ms = sw.ms()
        clear_conversation("resilience_f08")
        passed = error is None
        return (
            "RAG 模型目录缺失",
            error,
            ms,
            "工具层捕获并继续" if passed else f"异常穿透：{error}",
            passed,
        )

    return "未实现该故障类型", None, 0.0, "skip", False


def run() -> dict:
    cases = load_dataset(NAME)
    version = dataset_version(NAME)
    run_id = new_run_id()

    print("=" * 68)
    print("ShopMate 容错能力评测（mock 注入）")
    print(f"run_id={run_id}  数据集版本={version}  场景数={len(cases)}")
    print("=" * 68)

    case_results: list[dict] = []
    for case in cases:
        note, error, ms, actual, passed = _(case["fault"])
        case_results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "fault": case["fault"],
                "question": case["question"],
                "expect": case["expect"],
                "actual": actual,
                "answer": LAST_ANSWER,
                "exception": error,
                "elapsed_ms": ms,
                "passed": passed,
                "reason": "" if passed else actual,
            }
        )
        print(
            f"[{'PASS' if passed else 'FAIL'}] {case['id']} {case['category']} "
            f"{ms:.0f}ms  → {actual[:70]}"
        )

    metrics = {
        "resilience_pass_rate": ratio(
            sum(1 for c in case_results if c["passed"]), len(cases)
        ),
        "exception_leak_count": sum(
            1 for c in case_results if c["exception"] is not None
        ),
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
            "全部为 unittest.mock 注入，属单元级验证，不等于真实故障演练。",
            "未覆盖真实网络抖动、限流、DNS 故障、上游熔断。",
            "F01/F02 依赖备用模型真实可用性；若备用 Key 失效，结果会表现为走兜底文案。",
            "未测试并发下的降级行为。",
            "F08 只验证工具层是否吞掉异常，未验证 RAG 不可用时的回答质量。",
        ],
    )
    path = save_result(NAME, run_id, payload)

    print("\n" + "=" * 68)
    print(metric_table(metrics, "容错指标"))
    print()
    print(failure_table(case_results))
    print(f"\n结果已保存：{path}")
    return payload


if __name__ == "__main__":
    run()
