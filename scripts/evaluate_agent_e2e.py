
import sys
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.agent.agent import run_agent
from app.agent.conversation_store import clear_conversation


# Agent 端到端测试用例
TEST_CASES = [
    {
        "id": 1,
        "name": "订单查询",
        "question": "帮我查询一下订单 ORD001 的订单信息",
        "required_groups": [
            ["ORD001"],
            ["199.90", "199.9", "199元"],
        ],
    },
    {
        "id": 2,
        "name": "物流查询",
        "question": "帮我查询订单 ORD002 的物流信息",
        "required_groups": [
            ["ORD002"],
            ["SF123456"],
        ],
    },
    {
        "id": 3,
        "name": "退款资格检查",
        "question": "订单 ORD001 可以退款吗？请检查退款资格",
        "required_groups": [
            ["ORD001"],
            ["符合", "可以申请", "具备退款"],
        ],
    },
    {
        "id": 4,
        "name": "售后政策 RAG",
        "question": "商品签收超过7天还能退款吗？",
        "required_groups": [
            ["超过7天"],
            ["不支持普通退款", "不支持普通自助退款", "不支持退款", "无法直接退款"],
            ["人工客服", "人工审核", "转人工"],
        ],
    },
    {
        "id": 5,
        "name": "未实现功能边界",
        "question": "请帮我申请一张发票",
        "required_groups": [
            ["发票"],
            ["暂不支持", "不支持", "无法直接"],
        ],
    },
]


def normalize_text(text):
    """去除空格和换行，减少格式差异造成的误判。"""

    return re.sub(r"\s+", "", text)


def check_required_groups(answer, required_groups):
    """每组关键词至少命中一个，才算通过。"""

    normalized_answer = normalize_text(answer)
    failed_groups = []

    for group in required_groups:
        matched = any(
            normalize_text(keyword) in normalized_answer
            for keyword in group
        )

        if not matched:
            failed_groups.append(group)

    return failed_groups


def evaluate_single_case(case):
    """执行一条独立的 Agent 端到端测试。"""

    conversation_id = f"e2e_test_{case['id']}"

    # 每条测试使用独立会话，避免历史消息相互干扰
    clear_conversation(conversation_id)

    print("\n" + "=" * 65)
    print(f"测试编号：{case['id']}")
    print(f"测试场景：{case['name']}")
    print(f"用户问题：{case['question']}")

    try:
        answer = run_agent(
            user_message=case["question"],
            conversation_id=conversation_id,
        )

        print("\nAgent 最终回答：")
        print(answer)

        failed_groups = check_required_groups(
            answer,
            case["required_groups"],
        )

        if failed_groups:
            print("\n未命中的关键词组：")

            for group in failed_groups:
                print(f"  {group}")

        passed = len(failed_groups) == 0

        print(
            "\n评估结果：",
            "PASS" if passed else "FAIL",
        )

        return passed

    except Exception as exc:
        print(f"\n测试发生异常：{exc}")
        return False

    finally:
        clear_conversation(conversation_id)


def evaluate_multi_turn():
    """测试多轮澄清后继续完成业务查询。"""

    conversation_id = "e2e_test_multi_turn"

    clear_conversation(conversation_id)

    print("\n" + "=" * 65)
    print("测试编号：6")
    print("测试场景：多轮澄清与物流查询")

    try:
        # 第一轮：用户没有说明具体问题
        first_question = "我的订单有问题"

        print(f"\n第一轮用户问题：{first_question}")

        first_answer = run_agent(
            user_message=first_question,
            conversation_id=conversation_id,
        )

        print("\nAgent 澄清回复：")
        print(first_answer)

        # 第二轮：用户补充问题和订单编号
        second_question = "物流一直没更新，订单号是ORD002"

        print(f"\n第二轮用户问题：{second_question}")

        second_answer = run_agent(
            user_message=second_question,
            conversation_id=conversation_id,
        )

        print("\nAgent 最终回答：")
        print(second_answer)

        failed_groups = check_required_groups(
            second_answer,
            [
                ["ORD002"],
                ["SF123456"],
            ],
        )

        passed = len(failed_groups) == 0

        if failed_groups:
            print("\n未命中的关键词组：")

            for group in failed_groups:
                print(f"  {group}")

        print(
            "\n评估结果：",
            "PASS" if passed else "FAIL",
        )

        return passed

    except Exception as exc:
        print(f"\n测试发生异常：{exc}")
        return False

    finally:
        clear_conversation(conversation_id)


def main():
    print("ShopMate Agent 端到端评估")
    print("=" * 65)

    print("注意：本脚本会调用真实大模型和业务工具。")
    print("预计执行多次模型请求，请注意 API 用量。")

    passed = 0
    failed = 0

    # 第一部分：单轮业务测试
    print("\n一、单轮业务测试")

    for case in TEST_CASES:
        result = evaluate_single_case(case)

        if result:
            passed += 1
        else:
            failed += 1

    # 第二部分：多轮对话测试
    print("\n二、多轮对话测试")

    multi_turn_passed = evaluate_multi_turn()

    if multi_turn_passed:
        passed += 1
    else:
        failed += 1

    # 第三部分：汇总
    total = passed + failed

    print("\n" + "=" * 65)
    print("Agent 端到端评估总结")
    print("=" * 65)

    print(f"测试总数：{total}")
    print(f"通过数量：{passed}")
    print(f"失败数量：{failed}")
    print(f"通过率：{passed / total * 100:.2f}%")

    print("\n注意：")
    print("1. 本脚本通过关键词检查 Agent 的最终回答。")
    print("2. 关键词命中不等于工具选择和业务判断完全正确。")
    print("3. 需要结合实际回答和工具执行日志进行人工复核。")


if __name__ == "__main__":
    main()
