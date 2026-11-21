"""温度对比实验：回答生成用 0 和 0.4 到底差在哪。

不是只比「通过率」，而是分五个维度看：

    pass_rate        必要事实 / 禁止表述都满足的比例（业务正确性）
    forbidden_rate   出现禁止表述的比例（「一定可以退」「已帮你退款」这类）
    avg_len          平均回答长度
    structure_ratio  结构词（结论 / 条件 / 下一步）出现率 —— 越高越模板化
    self_similarity  同一问题重复问 3 次，回答之间的相似度 —— 越高越"死板"

为什么单独测 self_similarity：
    只看通过率看不出温度的影响，因为温度 0 也能全过。
    温度真正改变的是「同一个问题每次问，回答长得一样不一样」。
    这个指标才回答得了「语气是不是太死板」。

注意：
    只是回答这一步调温度。
    意图分类、查询改写、工具参数抽取仍然是 0，
    否则分类一变，测的就不是「回答风格」而是「整条链路随机性」了。
"""

import json
import pathlib
import statistics
import sys
import time

from difflib import SequenceMatcher


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

sys.stdout.reconfigure(encoding="utf-8")

from app.rag.rag_chain import RAGService  # noqa: E402
from evaluation.dataset import load_dataset  # noqa: E402
from evaluate_rag_answer import REJECTION_MARKERS, _check  # noqa: E402


# 参与对比的温度
TEMPERATURES = [0.0, 0.4]

# 结构词：出现得越多，说明回答越像填模板
STRUCTURE_MARKERS = ["结论", "条件", "下一步", "适用条件", "**"]

# 用来做「重复提问」的样例句。
# 选的是不同难度：一个直接、一个口语、一个多条件。
DIVERSITY_QUESTIONS = [
    "退货运费谁出",
    "东西坏了想退怎么办",
    "签收超过7天还能退款吗",
]

DIVERSITY_REPEATS = 3


def _structure_ratio(answer: str) -> float:
    hits = sum(1 for marker in STRUCTURE_MARKERS if marker in answer)
    return hits / len(STRUCTURE_MARKERS)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _run_quality(rag: RAGService, cases: list[dict], temperature: float) -> dict:
    """跑完整回答质量集，返回逐条结果和汇总。"""

    rows = []

    for case in cases:

        started = time.perf_counter()

        try:
            result = rag.answer(case["input"], temperature=temperature)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            error = None

        except Exception as exc:
            answer, sources = "", []
            error = f"{type(exc).__name__}: {exc}"

        elapsed = (time.perf_counter() - started) * 1000

        check = _check(case, answer, sources) if not error else {
            "passed": False,
            "forbidden_hit": [],
            "contradiction_warning": False,
            "reason": error,
        }

        # rag_answer 数据集用 expect_rejection 标记「应该拒答」的样本
        is_reject_case = bool(case["expected"].get("expect_rejection"))

        rows.append(
            {
                "id": case["id"],
                "input": case["input"],
                "passed": bool(check["passed"]),
                "forbidden": bool(check.get("forbidden_hit")),
                "contradiction": bool(check.get("contradiction_warning")),
                "rejected": any(m in answer for m in REJECTION_MARKERS),
                "is_reject_case": is_reject_case,
                "length": len(answer),
                "structure": _structure_ratio(answer),
                "elapsed_ms": round(elapsed, 1),
                "error": error,
                "answer": answer,
            }
        )

    total = len(rows)
    reject_cases = [r for r in rows if r["is_reject_case"]]
    normal_cases = [r for r in rows if not r["is_reject_case"]]

    return {
        "temperature": temperature,
        "total": total,
        "pass_rate": round(sum(r["passed"] for r in rows) / total, 4),
        "forbidden_rate": round(sum(r["forbidden"] for r in rows) / total, 4),
        "contradiction_rate": round(sum(r["contradiction"] for r in rows) / total, 4),
        "rejection_accuracy": round(
            sum(r["rejected"] for r in reject_cases) / len(reject_cases), 4
        )
        if reject_cases
        else None,
        "normal_accuracy": round(
            sum(r["passed"] for r in normal_cases) / len(normal_cases), 4
        )
        if normal_cases
        else None,
        "avg_length": round(statistics.mean(r["length"] for r in rows), 1),
        "avg_structure_ratio": round(
            statistics.mean(r["structure"] for r in rows), 4
        ),
        "avg_latency_ms": round(statistics.mean(r["elapsed_ms"] for r in rows), 1),
        "rows": rows,
    }


def _run_diversity(rag: RAGService, temperature: float) -> dict:
    """同一问题重复问，看回答之间有多像。

    相似度越接近 1，说明每次回答几乎一字不差（模板化）。
    温度越高，相似度应该越低。
    """

    per_question = []

    for question in DIVERSITY_QUESTIONS:

        answers = []

        for _ in range(DIVERSITY_REPEATS):
            try:
                result = rag.answer(question, temperature=temperature)
                answers.append(result.get("answer", ""))
            except Exception:
                answers.append("")

        pairs = []

        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                pairs.append(_similarity(answers[i], answers[j]))

        per_question.append(
            {
                "question": question,
                "avg_pairwise_similarity": round(statistics.mean(pairs), 4)
                if pairs
                else None,
                "answers": answers,
            }
        )

    values = [
        q["avg_pairwise_similarity"]
        for q in per_question
        if q["avg_pairwise_similarity"] is not None
    ]

    return {
        "temperature": temperature,
        "overall_similarity": round(statistics.mean(values), 4) if values else None,
        "per_question": per_question,
    }


def main() -> None:
    cases = load_dataset("rag_answer")

    print("=" * 68)
    print("温度对比实验")
    print(f"样本：{len(cases)} 条　温度：{TEMPERATURES}")
    print("=" * 68)

    print("\n正在初始化 RAG 服务……")
    rag = RAGService()

    quality = []
    diversity = []

    for temperature in TEMPERATURES:

        print(f"\n[温度 {temperature}] 跑质量集……")
        result = _run_quality(rag, cases, temperature)
        quality.append(result)

        print(
            f"  通过率 {result['pass_rate']:.1%}｜"
            f"违规 {result['forbidden_rate']:.1%}｜"
            f"平均长度 {result['avg_length']}｜"
            f"结构词 {result['avg_structure_ratio']:.0%}"
        )

        print(f"[温度 {temperature}] 跑重复提问……")
        div = _run_diversity(rag, temperature)
        diversity.append(div)

        print(f"  三次回答相似度 {div['overall_similarity']}")

    # 对比表
    print("\n" + "=" * 68)
    print("**对比结果**\n")
    print("| 指标 | " + " | ".join(f"温度 {t}" for t in TEMPERATURES) + " |")
    print("|---|" + "---|" * len(TEMPERATURES))

    def row(label, key, fmt="{:.4f}"):
        cells = []
        for item in quality:
            value = item.get(key)
            cells.append(fmt.format(value) if isinstance(value, (int, float)) else "-")
        print(f"| {label} | " + " | ".join(cells) + " |")

    row("通过率", "pass_rate", "{:.1%}")
    row("违规表述率", "forbidden_rate", "{:.1%}")
    row("矛盾率", "contradiction_rate", "{:.1%}")
    row("正常样本正确率", "normal_accuracy", "{:.1%}")
    row("拒答准确率", "rejection_accuracy", "{:.1%}")
    row("平均长度（字）", "avg_length", "{:.1f}")
    row("结构词出现率", "avg_structure_ratio", "{:.1%}")
    row("平均耗时（ms）", "avg_latency_ms", "{:.0f}")

    print(
        "| 重复提问相似度 | "
        + " | ".join(str(d["overall_similarity"]) for d in diversity)
        + " |"
    )

    # 保存
    out_dir = PROJECT_ROOT / "results" / "temperature_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "sample_size": len(cases),
        "temperatures": TEMPERATURES,
        "quality": quality,
        "diversity": diversity,
    }

    out_file = out_dir / "result.json"
    out_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n结果已保存：{out_file}")


if __name__ == "__main__":
    main()
