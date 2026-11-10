
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.agent.intent_classifier import classify_intent


TEST_CASES = [
    {
        "id": 1,
        "question": "帮我查一下 ORD001 的订单状态",
        "expected": "order_query",
    },
    {
        "id": 2,
        "question": "ORD002 的物流到哪里了？",
        "expected": "logistics_query",
    },
    {
        "id": 3,
        "question": "ORD001 现在可以退款吗？",
        "expected": "refund_query",
    },
    {
        "id": 4,
        "question": "我想退货，应该怎么申请？",
        "expected": "return_query",
    },
    {
        "id": 5,
        "question": "商品坏了，我想换一个新的",
        "expected": "exchange_query",
    },
    {
        "id": 6,
        "question": "商品签收超过7天还能退款吗？",
        "expected": "policy_query",
    },
    {
        "id": 7,
        "question": "你是谁？你能做什么？",
        "expected": "policy_query",
    },
    {
        "id": 8,
        "question": "我的订单有问题",
        "expected": "unclear",
    },
    {
        "id": 9,
        "question": "你好",
        "expected": "smalltalk",
    },
    {
        "id": 10,
        "question": "帮我写一首关于春天的诗",
        "expected": "out_of_scope",
    },
    {
        "id": 11,
        "question": "asdfghjkl",
        "expected": "nonsense",
    },
]


# 多轮对话测试数据
CONTEXT_TEST_CASES = [
    {
        "id": 12,
        "original_user_message": "我的订单有问题",
        "clarification_question": "请问你的订单具体遇到了什么情况？",
        "question": "物流",
        "expected": "logistics_query",
    },
    {
        "id": 13,
        "original_user_message": "我的订单有问题",
        "clarification_question": "请问你的订单具体遇到了什么情况？",
        "question": "退款",
        "expected": "refund_query",
    },
    {
        "id": 14,
        "original_user_message": "我的订单有问题",
        "clarification_question": "请问你的订单具体遇到了什么情况？",
        "question": "商品坏了",
        "expected": "unclear",
    },
    {
        "id": 15,
        "original_user_message": "我的订单有问题",
        "clarification_question": "请问你的订单具体遇到了什么情况？",
        "question": "物流一直没更新",
        "expected": "logistics_query",
    },
]


def evaluate_cases(cases, use_context=False):
    """执行一组意图识别测试。"""

    correct = 0
    errors = 0

    for case in cases:
        pending_clarification = None

        if use_context:
            pending_clarification = {
                "original_user_message": case["original_user_message"],
                "clarification_question": case["clarification_question"],
            }

        result = classify_intent(
            case["question"],
            pending_clarification=pending_clarification,
        )

        actual = result.intent
        expected = case["expected"]

        is_correct = (
            actual == expected
            and result.error is None
        )

        if is_correct:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"

        if result.error is not None:
            errors += 1

        print(f"\n[{status}] 测试编号：{case['id']}")

        if use_context:
            print(f"原始问题：{case['original_user_message']}")
            print(f"澄清问题：{case['clarification_question']}")

        print(f"用户当前消息：{case['question']}")
        print(f"预期意图：{expected}")
        print(f"实际意图：{actual}")
        print(f"置信度：{result.confidence:.2f}")

        if result.error:
            print(f"异常信息：{result.error}")

    return correct, errors


def evaluate_intent():
    """评估单轮和多轮意图识别能力。"""

    print("=" * 60)
    print("ShopMate 意图识别评估")
    print("=" * 60)

    print("\n第一部分：单轮意图识别")

    single_correct, single_errors = evaluate_cases(
        TEST_CASES,
        use_context=False,
    )

    print("\n第二部分：多轮意图识别")

    context_correct, context_errors = evaluate_cases(
        CONTEXT_TEST_CASES,
        use_context=True,
    )

    total = len(TEST_CASES) + len(CONTEXT_TEST_CASES)
    correct = single_correct + context_correct
    errors = single_errors + context_errors

    accuracy = correct / total if total else 0

    print("\n" + "=" * 60)
    print("最终评估结果")
    print("=" * 60)

    print(f"单轮测试：{single_correct}/{len(TEST_CASES)}")
    print(f"多轮测试：{context_correct}/{len(CONTEXT_TEST_CASES)}")
    print(f"测试总数：{total}")
    print(f"识别正确：{correct}")
    print(f"识别错误：{total - correct}")
    print(f"调用或解析异常：{errors}")
    print(f"总体准确率：{accuracy:.2%}")

    print("=" * 60)


if __name__ == "__main__":
    evaluate_intent()
