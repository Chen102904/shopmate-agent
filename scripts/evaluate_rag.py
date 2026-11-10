
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.rag.retriever import (
    KnowledgeRetriever,
    DEFAULT_SCORE_THRESHOLD,
)


# 正常问题检索测试集
RETRIEVAL_CASES = [
    {
        "id": 1,
        "question": "商品签收超过7天还能退款吗？",
        "expected_chunk_id": "after_sales_policy.md_0",
        "expected_keyword": "退款政策",
    },
    {
        "id": 2,
        "question": "商品有质量问题，应该怎么退货？",
        "expected_chunk_id": "after_sales_policy.md_1",
        "expected_keyword": "退货政策",
    },
    {
        "id": 3,
        "question": "商品坏了，想换一个新的怎么办？",
        "expected_chunk_id": "after_sales_policy.md_2",
        "expected_keyword": "换货政策",
    },
    {
        "id": 4,
        "question": "物流一直没有更新怎么办？",
        "expected_chunk_id": "after_sales_policy.md_3",
        "expected_keyword": "物流政策",
    },
    {
        "id": 5,
        "question": "什么情况下需要联系人工客服？",
        "expected_chunk_id": "after_sales_policy.md_4",
        "expected_keyword": "人工客服政策",
    },
]


# 无关问题拒答测试集
REJECTION_CASES = [
    {"id": 6, "question": "ShopMate公司成立于哪一年？"},
    {"id": 7, "question": "苹果手机最新款多少钱？"},
    {"id": 8, "question": "美国总统是谁？"},
    {"id": 9, "question": "今天北京天气怎么样？"},
]


def evaluate_retrieval(retriever):
    """评估正常问题的检索效果。"""

    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_ranks = []

    print("\n一、正常问题检索评估")
    print("=" * 60)

    for case in RETRIEVAL_CASES:
        results = retriever.search(
            query=case["question"],
            top_k=3,
        )

        result_ids = [
            result["chunk_id"]
            for result in results
        ]

        expected_id = case["expected_chunk_id"]

        if expected_id in result_ids:
            rank = result_ids.index(expected_id) + 1
            reciprocal_rank = 1 / rank
        else:
            rank = None
            reciprocal_rank = 0

        if rank == 1:
            hit_at_1 += 1

        if rank is not None and rank <= 3:
            hit_at_3 += 1

        reciprocal_ranks.append(reciprocal_rank)

        matched = rank is not None

        print(
            f"[{'PASS' if matched else 'FAIL'}] "
            f"测试编号：{case['id']}"
        )
        print(f"用户问题：{case['question']}")
        print(f"预期章节：{case['expected_keyword']}")

        if rank is not None:
            print(f"正确文本块排名：第 {rank} 名")
            print(f"RR：{reciprocal_rank:.4f}")
        else:
            print("正确文本块未出现在 Top-3 中")

        print("实际检索结果：")

        for result in results:
            print(
                f"  {result['chunk_id']} | "
                f"相似度：{result['score']:.4f}"
            )

        print("-" * 60)

    total = len(RETRIEVAL_CASES)

    hit_at_1_rate = hit_at_1 / total
    hit_at_3_rate = hit_at_3 / total
    mrr = sum(reciprocal_ranks) / total

    return {
        "hit_at_1": hit_at_1_rate,
        "hit_at_3": hit_at_3_rate,
        "mrr": mrr,
        "passed": hit_at_3,
        "total": total,
    }


def evaluate_rejection(retriever):
    """评估知识库外问题的过滤能力。"""

    passed = 0

    print("\n二、无关问题拒答检索评估")
    print("=" * 60)

    for case in REJECTION_CASES:
        results = retriever.search(
            query=case["question"],
            top_k=3,
        )

        matched = len(results) == 0

        if matched:
            passed += 1

        print(
            f"[{'PASS' if matched else 'FAIL'}] "
            f"测试编号：{case['id']}"
        )
        print(f"用户问题：{case['question']}")

        if results:
            print("实际检索结果：")
            for result in results:
                print(
                    f"  {result['chunk_id']} | "
                    f"相似度：{result['score']:.4f}"
                )
        else:
            print("实际检索结果：空列表 []")

        print("-" * 60)

    return passed


def main():
    print("正在初始化 ShopMate RAG 检索评估……")

    retriever = KnowledgeRetriever()
    chunk_count = retriever.build_index()

    print(f"知识库文本块数量：{chunk_count}")
    print(f"当前相似度阈值：{DEFAULT_SCORE_THRESHOLD}")

    retrieval_metrics = evaluate_retrieval(retriever)
    rejection_passed = evaluate_rejection(retriever)

    retrieval_total = retrieval_metrics["total"]
    rejection_total = len(REJECTION_CASES)
    total = retrieval_total + rejection_total

    total_passed = (
        retrieval_metrics["passed"] + rejection_passed
    )

    print("\n三、RAG 综合评估结果")
    print("=" * 60)

    print("正常问题检索指标：")
    print(
        f"Hit@1："
        f"{retrieval_metrics['hit_at_1'] * 100:.2f}%"
    )
    print(
        f"Hit@3："
        f"{retrieval_metrics['hit_at_3'] * 100:.2f}%"
    )
    print(f"MRR：{retrieval_metrics['mrr']:.4f}")

    print("-" * 60)

    print(
        f"无关问题过滤："
        f"{rejection_passed}/{rejection_total}"
    )

    rejection_accuracy = (
        rejection_passed / rejection_total * 100
    )

    print(f"拒答检索准确率：{rejection_accuracy:.2f}%")

    print("-" * 60)

    print(f"测试总数：{total}")
    print(f"总通过数：{total_passed}")
    print(
        f"总体通过率："
        f"{total_passed / total * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
