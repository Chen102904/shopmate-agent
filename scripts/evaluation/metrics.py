"""指标计算。

所有函数都是纯函数，不依赖模型，便于单元测试。

约定：
- 标签用字符串。
- 分类指标采用 macro 平均（每类权重相同），并同时给出各类别明细。
- 检索指标基于「正确结果在返回列表中的排名」。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def per_class_prf(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Iterable[str],
) -> dict[str, dict[str, float]]:
    """计算每个类别的 Precision / Recall / F1 / Support。

    说明：某类从未被预测时 P 记为 0（不是 1），
    避免「完全不预测某一类」反而拿到高分。
    """
    labels = list(labels)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        support[t] += 1
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[p] += 1

    out: dict[str, dict[str, float]] = {}
    for label in labels:
        p_denom = tp[label] + fp[label]
        r_denom = tp[label] + fn[label]
        precision = tp[label] / p_denom if p_denom else 0.0
        recall = tp[label] / r_denom if r_denom else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        out[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support[label],
        }
    return out


def macro_average(per_class: dict[str, dict[str, float]]) -> dict[str, float]:
    """只对 support > 0 的类别做 macro 平均。"""
    used = [v for v in per_class.values() if v["support"] > 0]
    if not used:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    n = len(used)
    return {
        "precision": round(sum(v["precision"] for v in used) / n, 4),
        "recall": round(sum(v["recall"] for v in used) / n, 4),
        "f1": round(sum(v["f1"] for v in used) / n, 4),
    }


def hit_at_k(ranks: Sequence[int | None], k: int) -> float:
    """ranks 里是正确结果的排名（1 起），None 表示未命中。"""
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks)


def mrr(ranks: Sequence[int | None]) -> float:
    if not ranks:
        return 0.0
    total = 0.0
    for r in ranks:
        if r is not None and r > 0:
            total += 1.0 / r
    return total / len(ranks)


def precision_at_k(relevant: set[str], returned: Sequence[str], k: int) -> float:
    topk = returned[:k]
    if not topk:
        return 0.0
    return sum(1 for d in topk if d in relevant) / len(topk)


def recall_at_k(relevant: set[str], returned: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = set(returned[:k])
    return len(topk & relevant) / len(relevant)


def set_prf(expected: Iterable[str], actual: Iterable[str]) -> dict[str, float]:
    """集合级 P/R/F1，用于工具选择（不看顺序时）。

    注意：这个函数会丢失重复信息，只在明确「顺序无关」的场景使用。
    """
    exp = set(expected)
    act = set(actual)
    tp = len(exp & act)
    precision = tp / len(act) if act else (1.0 if not exp else 0.0)
    recall = tp / len(exp) if exp else (1.0 if not act else 0.0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def sequence_match(expected: Sequence[str], actual: Sequence[str]) -> bool:
    """顺序敏感的序列比较，含重复与次数。"""
    return list(expected) == list(actual)


def avg(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def summarize_params(
    triples_correct: int,
    triples_total: int,
    calls_exact: int,
    calls_actual: int,
    cases_exact: int,
    cases_total: int,
) -> dict[str, Any]:
    """工具参数的三层计分，单位不混用。"""
    return {
        "argument_level_accuracy": ratio(triples_correct, triples_total),
        "call_level_exact_match": ratio(calls_exact, calls_actual),
        "case_level_exact_match": ratio(cases_exact, cases_total),
        "_raw": {
            "triples_correct": triples_correct,
            "triples_total": triples_total,
            "calls_exact": calls_exact,
            "calls_actual": calls_actual,
            "cases_exact": cases_exact,
            "cases_total": cases_total,
        },
    }
