"""RAG 回答质量评测。

改动要点（相对旧脚本）：
1. 样本 6 → 30 条（25 正常 + 5 拒答）。
2. 判定从「关键词命中」升级为四类规则：
   - must_have      每组任选其一（同义容错）
   - must_have_all  每项都必须出现（关键条件不可省）
   - must_not_have  出现即失败（编造检测）
   - 拒答检查       知识库无答案时必须明确拒答
3. 增加「矛盾提示」检测：同一回答里同时出现肯定与否定表述，记为 warning，不计入通过判定。
4. 明确记录：本评测为规则检查，不是语义等价判定；未使用 LLM-as-Judge。

运行：
    python scripts/evaluate_rag_answer.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from app.rag.rag_chain import RAGService  # noqa: E402
from evaluation.common import (  # noqa: E402
    Stopwatch,
    build_payload,
    new_run_id,
    save_result,
)
from evaluation.dataset import dataset_version, load_dataset, summarize  # noqa: E402
from evaluation.metrics import ratio  # noqa: E402
from evaluation.report import failure_table, metric_table  # noqa: E402

NAME = "rag_answer"

REJECTION_MARKERS = [
    # 知识库没覆盖时的标准拒答
    "没有找到", "无法确认", "不清楚", "没有相关", "暂时没有",
    # 问题超出服务范围时的拒答
    # 实际回答常用「超出服务范围」「没法帮你」「无法提供」，
    # 只认「没有找到」会把这些正确拒答判成失败。
    "超出", "服务范围", "没法", "无法帮", "无法提供", "不属于",
]

# 矛盾提示：同时出现"可以"类与"不可以"类表述
POSITIVE = ["可以退", "支持退款", "能够申请"]
NEGATIVE = ["不支持", "无法退", "不能退"]

# 否定前缀。
# 「不支持退款」里包含子串「支持退款」，
# 只做 in 判断会把它当成正面表述，
# 进而把「虚拟商品一经使用不支持退款」这种正确句子判成自相矛盾。
NEGATION_CHARS = "不无未非没难"


def _has_positive(sentence: str) -> bool:
    """判断句子里有没有「未被否定的」正面表述。"""

    for word in POSITIVE:

        start = 0

        while True:

            index = sentence.find(word, start)

            if index == -1:
                break

            prefix = sentence[max(0, index - 1):index]

            if prefix not in NEGATION_CHARS:
                return True

            start = index + 1

    return False


def _detect_contradiction(answer: str) -> bool:
    """检测「同一句话」里的自相矛盾。

    两处修正，都是改造前误报的原因：

    1. 按句判而不是按整段判。
       真实回答经常分场景说：
       「质量问题可以退；非质量问题不支持」。
       整段同时出现正负表述是正常的条件分支，不是矛盾。
       Q10/Q11 两条高质量回答就是这样被误报的。

    2. 排除否定前缀。
       「不支持退款」里包含「支持退款」，
       只做子串判断会把它当成正面表述。
    """

    for sentence in re.split(r"[。！？；\n]", answer or ""):

        has_negative = any(n in sentence for n in NEGATIVE)

        if has_negative and _has_positive(sentence):
            return True

    return False


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


# 条款复述上下文词。
# 模型在复述知识库条文时会出现「已使用商品可能无法退货」这类句子，
# 这是引用规则，不是对当前用户下结论，不应判为违规。
CLAUSE_CONTEXT = [
    "已使用", "影响二次销售", "特殊商品", "定制类", "贴身",
    "虚拟商品", "赠品", "拆封", "生鲜",
]


def _is_clause_quote(answer: str, keyword: str) -> bool:
    """判断禁止词是否出现在「复述条款」的上下文里。"""

    normalized_answer = _norm(answer)
    position = normalized_answer.find(keyword)

    if position < 0:
        return False

    window = normalized_answer[
        max(0, position - 40): position + 40
    ]

    return any(word in window for word in CLAUSE_CONTEXT)


def _check(case: dict, answer: str, sources: list[dict]) -> dict:
    exp = case["expected"]
    a = _norm(answer)

    must_have_failed = [
        group
        for group in exp.get("must_have", [])
        if not any(_norm(k) in a for k in group)
    ]
    # must_have_all 支持两种写法：字符串（必须出现）或列表（组内任选其一）
    must_all_failed: list = []
    for item in exp.get("must_have_all", []):
        if isinstance(item, list):
            if not any(_norm(k) in a for k in item):
                must_all_failed.append(item)
        elif _norm(item) not in a:
            must_all_failed.append(item)
    forbidden_hit = [
        k
        for k in exp.get("must_not_have", [])
        if _norm(k) in a and not _is_clause_quote(answer, _norm(k))
    ]

    if exp.get("expect_rejection"):
        rejected = any(m in answer for m in REJECTION_MARKERS)
        passed = rejected and not forbidden_hit
    else:
        passed = not must_have_failed and not must_all_failed and not forbidden_hit

    contradiction = _detect_contradiction(answer)

    if passed:
        reason = ""
    elif forbidden_hit:
        reason = f"出现禁止表述：{forbidden_hit}"
    elif must_all_failed:
        reason = f"缺少关键条件：{must_all_failed}"
    elif must_have_failed:
        reason = f"缺少必要事实：{must_have_failed}"
    else:
        reason = "未明确拒答"

    return {
        "passed": passed,
        "must_have_failed": must_have_failed,
        "must_all_failed": must_all_failed,
        "forbidden_hit": forbidden_hit,
        "contradiction_warning": contradiction,
        "sources_count": len(sources),
        "reason": reason,
    }


def run() -> dict:
    cases = load_dataset(NAME)
    version = dataset_version(NAME)
    run_id = new_run_id()

    print("=" * 68)
    print("ShopMate RAG 回答质量评测")
    print(f"run_id={run_id}  数据集版本={version}  样本数={len(cases)}")
    print("类别分布：", summarize(cases))
    print("=" * 68)

    print("正在初始化 RAG 服务（加载 Embedding 模型）……")
    rag = RAGService()

    case_results: list[dict] = []
    normal_total = normal_pass = 0
    reject_total = reject_pass = 0
    forbidden_total = forbidden_violations = 0
    contradictions = 0

    for case in cases:
        exp = case["expected"]
        sw = Stopwatch()
        error = None
        answer, sources = "", []
        try:
            result = rag.answer(case["input"])
            answer = result.get("answer", "")
            sources = result.get("sources", [])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = sw.ms()

        if error:
            check = {
                "passed": False,
                "must_have_failed": [],
                "must_all_failed": [],
                "forbidden_hit": [],
                "contradiction_warning": False,
                "sources_count": 0,
                "reason": f"调用异常：{error}",
            }
        else:
            check = _check(case, answer, sources)

        if exp.get("expect_rejection"):
            reject_total += 1
            reject_pass += int(check["passed"])
        else:
            normal_total += 1
            normal_pass += int(check["passed"])

        if exp.get("must_not_have"):
            forbidden_total += 1
            forbidden_violations += int(bool(check["forbidden_hit"]))
        contradictions += int(check["contradiction_warning"])

        case_results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "input": case["input"],
                "answer": answer,
                "sources": [
                    {"chunk_id": s.get("chunk_id"), "score": s.get("score")}
                    for s in sources
                ],
                **check,
                "elapsed_ms": elapsed,
            }
        )

        print(
            f"[{'PASS' if check['passed'] else 'FAIL'}] {case['id']} "
            f"来源 {len(sources)} 条 {elapsed:.0f}ms"
            + ("" if check["passed"] else f" ← {check['reason']}")
        )

    metrics = {
        "pass_rate": ratio(
            sum(1 for c in case_results if c["passed"]), len(cases)
        ),
        "answer_quality_accuracy": ratio(normal_pass, normal_total),
        "rejection_accuracy": ratio(reject_pass, reject_total),
        "forbidden_fact_violation_rate": ratio(
            forbidden_violations, forbidden_total
        ),
        "contradiction_warning_rate": ratio(contradictions, len(cases)),
        "avg_sources": round(
            sum(c["sources_count"] for c in case_results) / len(cases), 2
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
            "判定为规则检查（关键词组/禁止词），不能识别「同义但未命中关键词」的正确回答。",
            "未使用 LLM-as-Judge，因此也没有语义等价评分。",
            "关键词表由人工从知识库原文反推，存在设计者偏差。",
            "知识库仅 5 章，答案空间很小，实际业务中难度更高。",
            "矛盾检测为启发式（正向词与否定词并存），会产生误报，仅作提示不计分。",
        ],
    )
    path = save_result(NAME, run_id, payload)

    print("\n" + "=" * 68)
    print(metric_table(metrics, "RAG 回答指标"))
    print()
    print(failure_table(case_results))
    print(f"\n结果已保存：{path}")
    return payload


if __name__ == "__main__":
    run()
