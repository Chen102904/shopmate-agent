
import sys
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.rag.rag_chain import RAGService


TEST_CASES = [
    {
        "id": 1,
        "question": "商品签收超过7天还能退款吗？",
        "expected_chunk_id": "after_sales_policy.md_0",
        "required_groups": [
            ["超过7天", "签收7天"],
            ["不支持普通退款", "无法直接退款", "不支持退款"],
            ["人工客服", "人工审核", "转人工"],
        ],
    },
    {
        "id": 2,
        "question": "商品有质量问题，应该怎么退货？",
        "expected_chunk_id": "after_sales_policy.md_1",
        "required_groups": [
            ["质量问题", "商品质量"],
            ["退货", "申请退货"],
        ],
    },
    {
        "id": 3,
        "question": "商品坏了，想换一个新的怎么办？",
        "expected_chunk_id": "after_sales_policy.md_2",
        "required_groups": [
            ["换货", "申请换货"],
            ["审核", "售后人员"],
        ],
    },
    {
        "id": 4,
        "question": "物流一直没有更新怎么办？",
        "expected_chunk_id": "after_sales_policy.md_3",
        "required_groups": [
            ["物流", "运输"],
            ["人工客服", "联系人工"],
        ],
    },
    {
        "id": 5,
        "question": "什么情况下需要联系人工客服？",
        "expected_chunk_id": "after_sales_policy.md_4",
        "required_groups": [
            ["人工客服"],
            ["退款资格", "签收超过7天", "物流异常", "处理结果"],
        ],
    },
    {
        "id": 6,
        "question": "ShopMate公司成立于哪一年？",
        "expected_chunk_id": None,
        "required_groups": [],
        "expect_rejection": True,
    },
]


def normalize_text(text):
    """去除空格、换行等空白字符，减少格式差异造成的误判。"""

    return re.sub(r"\s+", "", text)


def check_required_groups(answer, required_groups):
    """检查回答是否包含每组至少一个关键词。"""

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


def check_sources(sources, expected_chunk_id):
    """检查引用来源是否符合预期。"""

    if expected_chunk_id is None:
        return len(sources) == 0

    chunk_ids = [
        item.get("chunk_id")
        for item in sources
    ]

    return expected_chunk_id in chunk_ids


def evaluate_case(rag, case):
    """执行单条 RAG 回答质量测试。"""

    print("\n" + "=" * 65)
    print(f"测试编号：{case['id']}")
    print(f"用户问题：{case['question']}")

    try:
        result = rag.answer(case["question"])

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        print("\n模型回答：")
        print(answer)

        print("\n引用来源：")

        if sources:
            for source in sources:
                print(
                    f"  {source['source']} | "
                    f"{source['chunk_id']} | "
                    f"相似度：{source['score']}"
                )
        else:
            print("  无引用来源")

        if case.get("expect_rejection", False):
            rejection_passed = (
                len(sources) == 0
                and "知识库中暂时没有找到相关信息" in answer
            )

            print(
                "\n拒答检查：",
                "PASS" if rejection_passed else "FAIL",
            )

            return rejection_passed

        failed_groups = check_required_groups(
            answer,
            case["required_groups"],
        )

        content_passed = len(failed_groups) == 0

        print(
            "\n回答内容检查：",
            "PASS" if content_passed else "FAIL",
        )

        if failed_groups:
            print("未命中的关键词组：")
            for group in failed_groups:
                print(f"  {group}")

        source_passed = check_sources(
            sources,
            case["expected_chunk_id"],
        )

        print(
            "来源引用检查：",
            "PASS" if source_passed else "FAIL",
        )

        return content_passed and source_passed

    except Exception as exc:
        print(f"\n测试发生异常：{exc}")
        return False


def main():
    print("ShopMate RAG 回答质量评估")
    print("=" * 65)

    print("正在初始化 RAG 服务……")

    rag = RAGService()

    print("RAG 服务初始化完成。")

    passed = 0
    failed = 0

    for case in TEST_CASES:
        result = evaluate_case(rag, case)

        if result:
            passed += 1
            print("\n测试结果：PASS")
        else:
            failed += 1
            print("\n测试结果：FAIL")

    total = len(TEST_CASES)

    print("\n" + "=" * 65)
    print("RAG 回答质量评估总结")
    print("=" * 65)

    print(f"测试总数：{total}")
    print(f"通过数量：{passed}")
    print(f"失败数量：{failed}")
    print(f"通过率：{passed / total * 100:.2f}%")

    print("\n注意：")
    print("1. 本脚本通过关键词检查回答中的关键政策信息。")
    print("2. 关键词检查不能完全判断回答是否忠实于知识库。")
    print("3. 最终仍需人工检查回答是否存在遗漏或编造。")


if __name__ == "__main__":
    main()
