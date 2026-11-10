
import sys
import json
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.agent.agent import run_agent
from app.agent.conversation_store import clear_conversation


TEST_CASES = [
    {
        "id": 1,
        "question": "帮我查一下 ORD001 的订单状态",
        "expected_calls": [
            {
                "tool": "get_order",
                "arguments": {"order_id": "ORD001"},
            }
        ],
    },
    {
        "id": 2,
        "question": "ORD002 的物流到哪里了？",
        "expected_calls": [
            {
                "tool": "get_logistics",
                "arguments": {"order_id": "ORD002"},
            }
        ],
    },
    {
        "id": 3,
        "question": "ORD001 现在可以退款吗？",
        "expected_calls": [
            {
                "tool": "check_refund_eligibility",
                "arguments": {"order_id": "ORD001"},
            }
        ],
    },
    {
        "id": 4,
        "question": "商品签收超过7天还能退款吗？",
        "expected_calls": [
            {
                "tool": "search_after_sales_policy",
                "arguments": {
                    "query": "商品签收超过7天还能退款吗"
                },
            }
        ],
    },
    {
        "id": 5,
        "question": (
            "ORD001 现在可以退款吗？"
            "另外，商品签收超过7天的退款政策是什么？"
        ),
        "expected_calls": [
            {
                "tool": "check_refund_eligibility",
                "arguments": {"order_id": "ORD001"},
            },
            {
                "tool": "search_after_sales_policy",
                "arguments": {
                    "query": "商品签收超过7天的退款政策是什么"
                },
            },
        ],
    },
]


def normalize_calls(calls):
    """统一调用记录格式，便于比较。"""

    return sorted(
        [
            {
                "tool": call["tool"],
                "arguments": call["arguments"],
            }
            for call in calls
        ],
        key=lambda item: item["tool"],
    )


def evaluate_tool_arguments():
    total = len(TEST_CASES)
    correct = 0

    print("=" * 60)
    print("ShopMate Agent 工具参数评估")
    print("=" * 60)

    for case in TEST_CASES:
        conversation_id = f"argument_eval_{case['id']}"
        clear_conversation(conversation_id)

        actual_calls = []

        def fake_execute_tool(tool_name: str, arguments: dict) -> str:
            """拦截工具调用，记录名称和参数，不执行真实业务。"""

            actual_calls.append(
                {
                    "tool": tool_name,
                    "arguments": arguments,
                }
            )

            if tool_name == "search_after_sales_policy":
                return json.dumps(
                    {
                        "query": arguments.get("query", ""),
                        "answer": {
                            "answer": "模拟政策检索结果",
                            "sources": [],
                        },
                    },
                    ensure_ascii=False,
                )

            return json.dumps(
                {
                    "mock": True,
                    "tool": tool_name,
                    "arguments": arguments,
                },
                ensure_ascii=False,
            )

        try:
            with patch(
                "app.agent.agent.execute_tool",
                side_effect=fake_execute_tool,
            ):
                run_agent(
                    user_message=case["question"],
                    conversation_id=conversation_id,
                )

        except Exception as exc:
            print(f"\n[ERROR] 测试编号：{case['id']}")
            print(f"异常信息：{exc}")
            clear_conversation(conversation_id)
            continue

        expected_calls = normalize_calls(case["expected_calls"])
        actual_calls_normalized = normalize_calls(actual_calls)

        is_correct = actual_calls_normalized == expected_calls

        if is_correct:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"\n[{status}] 测试编号：{case['id']}")
        print(f"用户问题：{case['question']}")
        print(f"预期调用：{expected_calls}")
        print(f"实际调用：{actual_calls_normalized}")

        clear_conversation(conversation_id)

    accuracy = correct / total if total else 0

    print("\n" + "=" * 60)
    print("工具参数评估结果")
    print("=" * 60)
    print(f"测试总数：{total}")
    print(f"参数正确：{correct}")
    print(f"参数错误：{total - correct}")
    print(f"工具参数准确率：{accuracy:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_tool_arguments()
