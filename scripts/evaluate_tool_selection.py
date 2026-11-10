
import sys
import json
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.agent.agent import run_agent
from app.agent.conversation_store import clear_conversation


# 工具选择评估数据集
TEST_CASES = [
    {
        "id": 1,
        "question": "帮我查一下 ORD001 的订单状态",
        "expected_tools": ["get_order"],
    },
    {
        "id": 2,
        "question": "ORD002 的物流到哪里了？",
        "expected_tools": ["get_logistics"],
    },
    {
        "id": 3,
        "question": "ORD001 现在可以退款吗？",
        "expected_tools": ["check_refund_eligibility"],
    },
    {
        "id": 4,
        "question": "商品签收超过7天还能退款吗？",
        "expected_tools": ["search_after_sales_policy"],
    },
    {
        "id": 5,
        "question": (
            "ORD001 现在可以退款吗？"
            "另外，商品签收超过7天的退款政策是什么？"
        ),
        "expected_tools": [
            "check_refund_eligibility",
            "search_after_sales_policy",
        ],
    },
]


def evaluate_tool_selection():
    """评估 Agent 是否选择了正确的工具。"""

    total = len(TEST_CASES)
    correct = 0

    print("=" * 60)
    print("ShopMate Agent 工具选择评估")
    print("=" * 60)

    for case in TEST_CASES:
        conversation_id = f"tool_eval_{case['id']}"

        # 清理测试会话，避免不同测试互相影响
        clear_conversation(conversation_id)

        actual_tools = []

        def fake_execute_tool(tool_name: str, arguments: dict) -> str:
            """
            拦截工具执行，只记录工具名称。
            不执行真实业务操作。
            """

            actual_tools.append(tool_name)

            # 为 RAG 工具返回符合 Agent 解析逻辑的模拟结果
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
            # 临时替换 Agent 中的工具执行函数
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

        expected_tools = set(case["expected_tools"])
        actual_tool_set = set(actual_tools)

        # 工具集合必须完全匹配
        is_correct = actual_tool_set == expected_tools

        if is_correct:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"\n[{status}] 测试编号：{case['id']}")
        print(f"用户问题：{case['question']}")
        print(f"预期工具：{sorted(expected_tools)}")
        print(f"实际工具：{sorted(actual_tool_set)}")

        if len(actual_tools) != len(actual_tool_set):
            print(f"工具调用记录：{actual_tools}")

        clear_conversation(conversation_id)

    accuracy = correct / total if total else 0

    print("\n" + "=" * 60)
    print("工具选择评估结果")
    print("=" * 60)
    print(f"测试总数：{total}")
    print(f"选择正确：{correct}")
    print(f"选择错误：{total - correct}")
    print(f"工具选择准确率：{accuracy:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_tool_selection()
