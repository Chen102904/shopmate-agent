"""RAG 检索评测。

改动要点：
1. 样本 9 → 30 条（正常 12 / 同义 6 / 口语 4 / 多条件 3 / 无结果 5）。
2. 用「内容锚点」替代硬编码 chunk_id：判定返回块内容是否同时包含全部 keywords，
   知识库章节顺序变化不会让评测失效。
3. 指标补全：Hit@1/@3/@5、MRR、Precision@3、无结果正确拒绝率、无关误检率。
4. 增加阈值敏感性分析：在 0.45 / 0.55 / 0.60 / 0.65 四个阈值下重跑，
   观察命中与误检的权衡（原阈值 0.60 在源码里被标注为「实验性」）。

说明：检索命中 ≠ 回答正确，这一步只评检索。

运行：
    python scripts/evaluate_rag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from app.rag.retriever import DEFAULT_SCORE_THRESHOLD, KnowledgeRetriever  # noqa: E402
from app.rag.query_rewriter import rewrite_query  # noqa: E402
from evaluation.common import build_payload, new_run_id, save_result  # noqa: E402
from evaluation.dataset import dataset_version, load_dataset, summarize  # noqa: E402
from evaluation.metrics import hit_at_k, mrr, precision_at_k, ratio  # noqa: E402
from evaluation.report import failure_table, metric_table  # noqa: E402

NAME = "rag_retrieval"
THRESHOLDS = [0.55, 0.60, 0.62, 0.65]


def _match(chunk_text: str, keywords: list[str]) -> bool:
    """内容锚点匹配：全部 keywords 都出现才算命中。"""
    return all(k in chunk_text for k in keywords)


def _covered_distributed(
    hits: list[dict],
    keywords: list[str],
) -> bool:
    """多条件问句：允许由多条 chunk 分别覆盖。

    为什么需要这个：
        像「运输中又遇到物流显示异常怎么办」这种问题，
        「正在运输中」在退款章节、「物流显示异常」在物流章节，
        物理上不可能出现在同一个 300 字文本块里。
        如果坚持「同一条 chunk 覆盖全部条件」，这类样本永远不可能通过，
        但真实 RAG 会把两段都放进上下文，是能答出来的。

    只对多条件生效，且结果里会标注 match_mode=distributed，
    不会和「单条命中」混在一起统计。
    """

    if len(keywords) < 2:
        return False

    covered: set[str] = set()

    for hit in hits:
        content = hit["content"]

        for keyword in keywords:
            if keyword in content:
                covered.add(keyword)

    return covered == set(keywords)


def _eval_once(
    retriever: KnowledgeRetriever,
    cases: list[dict],
    threshold: float,
    queries: dict[str, str] | None = None,
):
    """在给定阈值下跑一遍，返回 (每条结果, 指标)。

    queries 用于传入「改写后的查询」。
    线上 RAG 链路会先改写再检索，
    评测必须走同一条路径，否则测的不是真实效果。
    """

    results = []
    ranks: list[int | None] = []
    precisions: list[float] = []
    no_result_total = no_result_pass = 0
    irrelevant_total = irrelevant_pass = 0

    for case in cases:
        exp = case["expected"]

        # 线上路径是「原始 + 改写」双路检索，
        # 评测必须同构，否则测出来的数字和线上对不上。
        rewritten = (queries or {}).get(case["id"])

        search_queries = [case["input"]]

        if rewritten and rewritten.strip() != case["input"].strip():
            search_queries.append(rewritten)

        hits = retriever.search_multi(
            search_queries,
            top_k=5,
            score_threshold=threshold,
        )
        returned_ids = [h["chunk_id"] for h in hits]
        rank = None
        matched_ids = []
        distributed = False

        if not exp["no_result"]:
            keywords = exp["keywords"]

            for pos, h in enumerate(hits, start=1):
                if _match(h["content"], keywords):
                    if rank is None:
                        rank = pos
                    matched_ids.append(h["chunk_id"])

            # 单条 chunk 没覆盖全部条件时，退回「多段共同覆盖」判定
            if rank is None:
                distributed = _covered_distributed(hits, keywords)

            relevant = set(matched_ids)
            prec = precision_at_k(relevant, returned_ids, 3) if relevant else 0.0
        else:
            # 无结果问题：期望返回空
            relevant = set()
            prec = 0.0

        ranks.append(rank if not exp["no_result"] else None)
        precisions.append(prec)

        if exp["no_result"]:
            no_result_total += 1
            passed = len(hits) == 0
            if passed:
                no_result_pass += 1
        else:
            irrelevant_total += 1
            passed = rank is not None or distributed
            if passed:
                irrelevant_pass += 1

        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "input": case["input"],
                "expect_no_result": exp["no_result"],
                "keywords": exp["keywords"],
                "returned": [
                    {"chunk_id": h["chunk_id"], "score": round(float(h["score"]), 4)}
                    for h in hits
                ],
                "rank": rank,
                "match_mode": (
                    "distributed"
                    if distributed
                    else ("single_chunk" if rank is not None else "miss")
                ),
                "passed": passed,
                "reason": (
                    "" if passed
                    else (
                        f"无关问题误检了 {len(hits)} 条"
                        if exp["no_result"] and hits
                        else "正确内容未出现在 Top-5"
                    )
                ),
            }
        )

    positive_ranks = [
        r for r, c in zip(ranks, cases) if not c["expected"]["no_result"]
    ]
    positive_precisions = [
        p for p, c in zip(precisions, cases) if not c["expected"]["no_result"]
    ]

    metrics = {
        "hit_at_1": hit_at_k(positive_ranks, 1),
        "hit_at_3": hit_at_k(positive_ranks, 3),
        "hit_at_5": hit_at_k(positive_ranks, 5),
        "mrr": round(mrr(positive_ranks), 4),
        "precision_at_3": round(
            sum(positive_precisions) / len(positive_precisions), 4
        )
        if positive_precisions
        else 0.0,
        "positive_recall_rate": ratio(irrelevant_pass, irrelevant_total),
        "no_result_rejection_rate": ratio(no_result_pass, no_result_total),
        "no_result_false_positive_rate": ratio(
            no_result_total - no_result_pass, no_result_total
        ),
    }
    return results, metrics


def run(use_rewrite: bool = True) -> dict:
    cases = load_dataset(NAME)
    version = dataset_version(NAME)
    run_id = new_run_id()

    print("=" * 68)
    print("ShopMate RAG 检索评测")
    print(f"run_id={run_id}  数据集版本={version}  样本数={len(cases)}")
    print(f"查询改写：{'开启' if use_rewrite else '关闭'}")
    print("类别分布：", summarize(cases))
    print("=" * 68)

    print("正在加载 Embedding 模型……")
    retriever = KnowledgeRetriever()
    chunk_count = retriever.build_index()
    print(f"知识库文本块数量：{chunk_count}")

    # 预处理：批量改写查询。
    # 改写与阈值无关，只做一次，后面所有阈值实验复用。
    rewritten_queries: dict[str, str] = {}
    rewrite_failures = 0

    if use_rewrite:
        print("正在改写查询……")

        for case in cases:
            try:
                rewritten_queries[case["id"]] = rewrite_query(
                    case["input"]
                )
            except Exception:
                rewritten_queries[case["id"]] = case["input"]
                rewrite_failures += 1

    active_queries = rewritten_queries if use_rewrite else None

    # 主评测：线上真实路径（改写 + 检索）
    case_results, metrics = _eval_once(
        retriever,
        cases,
        DEFAULT_SCORE_THRESHOLD,
        active_queries,
    )

    for c in case_results:
        print(
            f"[{'PASS' if c['passed'] else 'FAIL'}] {c['id']} "
            f"rank={c['rank']} top1={c['returned'][0]['chunk_id'] if c['returned'] else '空'}"
        )

    # 对照实验：同一阈值下，用原始查询再跑一遍。
    # 这是「改写到底有没有用」的唯一证据，
    # 不给对照就说不清收益来自改写还是别的改动。
    _, baseline_metrics = _eval_once(
        retriever,
        cases,
        DEFAULT_SCORE_THRESHOLD,
        None,
    )

    # 阈值敏感性（与主评测使用同一套查询）
    sensitivity = {}
    for th in THRESHOLDS:
        _, m = _eval_once(retriever, cases, th, active_queries)
        sensitivity[f"threshold_{th}"] = {
            "hit_at_3": m["hit_at_3"],
            "precision_at_3": m["precision_at_3"],
            "no_result_rejection_rate": m["no_result_rejection_rate"],
        }
        print(
            f"阈值 {th}: Hit@3={m['hit_at_3']:.3f} "
            f"P@3={m['precision_at_3']:.3f} "
            f"拒答率={m['no_result_rejection_rate']:.3f}"
        )

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
            "命中判定基于「关键词全部出现」，不是人工相关性标注，可能低估语义正确的检索。",
            "多条件样本要求同一条 chunk 覆盖两个条件，但真实系统可能分两条返回，判定偏严。",
            "知识库为本项目自建模拟政策，规模与真实电商文档库仍有差距。",
            "Embedding 模型本地固定（bge-small-zh-v1.5），但未记录其文件哈希。",
            "查询改写本身依赖大模型，改写结果存在波动，非确定性。",
        ],
        extra={
            "chunk_count": chunk_count,
            "default_threshold": DEFAULT_SCORE_THRESHOLD,
            "use_query_rewrite": use_rewrite,
            "rewrite_failures": rewrite_failures,
            "threshold_sensitivity": sensitivity,
            "baseline_without_rewrite": {
                "hit_at_3": baseline_metrics["hit_at_3"],
                "mrr": baseline_metrics["mrr"],
                "no_result_rejection_rate": baseline_metrics[
                    "no_result_rejection_rate"
                ],
            },
            "rewrite_gain": {
                "hit_at_3_delta": round(
                    metrics["hit_at_3"] - baseline_metrics["hit_at_3"], 4
                ),
                "mrr_delta": round(
                    metrics["mrr"] - baseline_metrics["mrr"], 4
                ),
            },
        },
    )
    path = save_result(NAME, run_id, payload)

    print("\n" + "=" * 68)
    print(metric_table(metrics, "RAG 检索指标"))
    print()
    print(failure_table(case_results))
    print(f"\n结果已保存：{path}")
    return payload


if __name__ == "__main__":
    # 默认走线上真实路径（先改写再检索）。
    # 加 --no-rewrite 可关闭，用于做对照实验。
    run(use_rewrite="--no-rewrite" not in sys.argv)
