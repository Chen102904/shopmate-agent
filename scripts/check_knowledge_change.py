"""知识库变更后的回归检查。

解决的问题：

    改知识库最容易犯的错不是「改错字」，而是「改掉了检索锚点」。

    真实案例：把「尚未付款」改写成「未付款」，
    看起来只是措辞更简洁，但「尚未」是这句话里 BM25 和向量都在用的关键词。
    改完 Hit@3 从 0.68 掉到 0.44——而当时没有任何机制
    在合并前提示这件事。

    版本号能告诉你「现在是哪一版」，
    但不能告诉你「这一版是不是比上一版差」。
    这个脚本补的就是后半截。

用法：

    python scripts/check_knowledge_change.py

    没有历史基线时：跑一次评测，建立基线。
    有历史基线时：跑一次评测，和最近一个不同版本的结果对比，
                  并对退化项给出明确提示。

判定口径：

    Hit@3 掉 5 个点以上 → 判为退化，建议人工复核改动
    这只是提醒，不是自动拦截：知识库改动可能是有意的
    （比如某条政策真的收紧了），工具不该替人做业务判断。
"""

import json
import sys

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from evaluation.common import RESULTS_DIR  # noqa: E402


# 关注的核心指标。顺序即展示顺序。
TRACKED_METRICS = (
    ("hit_at_1", "Hit@1"),
    ("hit_at_3", "Hit@3"),
    ("hit_at_5", "Hit@5"),
    ("mrr", "MRR"),
    ("precision_at_3", "P@3"),
    ("no_result_rejection_rate", "无关拒答率"),
)

# Hit@3 掉这么多点就提示复核。
# 5 个点大约等于 30 条样本里错 1-2 条，低于这个数基本是运行波动。
REGRESSION_THRESHOLD = 0.05

# 多指标同向下降的判定门槛。
#
# 为什么阈值可以更低（3 点）：
#     单个指标降 3 点确实可能是波动，
#     但「至少 3 个指标同时同向、平均降 3 点」的概率要低得多。
#     实测案例：改掉一个措辞，5 个指标整齐掉 4 个点，
#     单看每项都没到 5 点硬阈值，只看单项一定漏判。
CONSISTENT_DROP_THRESHOLD = 0.03
CONSISTENT_DROP_MIN_COUNT = 3


def _current_version() -> str:
    from app.rag.knowledge_version import compute_knowledge_version

    return compute_knowledge_version(force=True)


def _load_history() -> list[dict]:
    """扫 results/ 下所有 RAG 检索评测结果。"""

    history: list[dict] = []

    if not RESULTS_DIR.exists():
        return history

    for directory in RESULTS_DIR.iterdir():

        if not directory.is_dir():
            continue

        result_file = directory / "rag_retrieval.json"

        if not result_file.exists():
            continue

        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        environment = payload.get("environment", {})

        history.append(
            {
                "run_id": payload.get("run_id", directory.name),
                "kb_version": environment.get("knowledge_version", "unknown"),
                "metrics": payload.get("metrics", {}),
                "passed": payload.get("passed_cases"),
                "total": payload.get("total_cases"),
            }
        )

    # run_id 里带时间戳，字典序即时间序
    history.sort(key=lambda item: item["run_id"])

    return history


def _pick_baseline(history: list[dict], current_version: str) -> dict | None:
    """挑最近一次「不同知识库版本」的结果作为基线。

    为什么不挑「上一次运行」：
        同一个版本可能跑了很多次，
        拿同版本的上一跑来比，比的是模型波动，不是知识库改动。
    """

    candidates = [
        item
        for item in history
        if item["kb_version"] != current_version
        and item["kb_version"] != "unknown"
    ]

    return candidates[-1] if candidates else None


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if abs(value) <= 1 else str(value)
    return str(value)


def _delta_text(old, new) -> tuple[str, str]:
    """返回 (变化文本, 判定)。判定用于表格里标方向。"""

    if old is None or new is None:
        return "—", ""

    diff = float(new) - float(old)

    # 0.0005 以下的差异在四位小数展示里看不出来，按持平处理
    if abs(diff) < 0.0005:
        return "±0", "持平"

    text = f"{diff:+.4f}"
    return text, ("提升" if diff > 0 else "下降")


def main() -> None:
    current = _current_version()
    history = _load_history()

    print("=" * 66)
    print("知识库变更回归检查")
    print(f"当前知识库版本：{current}")
    print("=" * 66)

    baseline = _pick_baseline(history, current)

    if baseline is None:
        print()
        print("没有找到其他版本的评测基线。")
        print("接下来会跑一次检索评测，为当前版本建立基线。")
        print()

    else:
        print()
        print(f"基线版本：{baseline['kb_version']}")
        print(f"基线运行：{baseline['run_id']}")
        print()
        print("即将跑一次检索评测，与基线对比。")
        print("（30 条本地向量检索，不调用大模型，约 1-2 分钟）")
        print()

    # 加 --yes 跳过确认，方便在 CI 或脚本里调用。
    # 非交互环境（没有 TTY）读不到输入，也直接继续，
    # 否则自动化调用会在这里 EOFError 崩掉。
    if "--yes" not in sys.argv:
        try:
            input("按回车开始，Ctrl+C 取消……")
        except EOFError:
            print("（非交互环境，直接开始）")

    # 延迟导入：只有真要跑评测时才加载 sentence-transformers
    from evaluate_rag import run as run_rag_eval

    payload = run_rag_eval()
    current_metrics = payload.get("metrics", {})

    print()
    print("=" * 66)
    print("对比结果")
    print("=" * 66)

    if baseline is None:
        print()
        print(f"已建立基线：{payload.get('passed_cases')}/"
              f"{payload.get('total_cases')} 通过")
        print()
        print("下次改完知识库再跑一次，就能看出有没有变差。")
        return

    old_metrics = baseline["metrics"]

    print()
    print(f"{'指标':<14}{'基线':>10}{'当前':>10}{'变化':>12}   判定")
    print("-" * 60)

    # 命中类指标的变化，用于「多指标同向」判断。
    # 不含拒答率：它只有 1.0 和更低两种情况，掉一点就是严重问题，
    # 单独处理比混在平均里更清楚。
    deltas: dict[str, float] = {}

    for key, label in TRACKED_METRICS:

        old = old_metrics.get(key)
        new = current_metrics.get(key)

        delta_text, direction = _delta_text(old, new)

        print(
            f"{label:<14}{_fmt(old):>10}{_fmt(new):>10}"
            f"{delta_text:>12}   {direction}"
        )

        if (
            key != "no_result_rejection_rate"
            and old is not None
            and new is not None
        ):
            deltas[key] = float(new) - float(old)

    print()
    print(
        f"通过数：{baseline['passed']}/{baseline['total']} "
        f"→ {payload.get('passed_cases')}/{payload.get('total_cases')}"
    )
    print()

    # ---------- 判定 ----------
    #
    # 两条路，任一成立都要提示：
    #
    #   A. 单项硬阈值：Hit@3 掉 5 个点以上 → 明确判退化。
    #
    #   B. 多指标同向：至少 3 个指标同时下降，且平均降幅 ≥ 3 个点。
    #
    # 为什么需要 B：
    #     实测过——把「尚未付款」改成「未付款」，
    #     Hit@1/@3/@5、MRR 全部整齐掉 4 个点，P@3 掉 4.7 个点。
    #     单看每项都是 4 点，低于 5 点阈值，A 不触发；
    #     但 5 个指标同时同向、幅度还一样，这不是随机波动，
    #     是知识库改动导致的系统性变化。
    #     只看单项绝对幅度会漏掉这种情况。

    hard_drop = deltas.get("hit_at_3", 0.0) < -REGRESSION_THRESHOLD

    dropped_keys = [key for key, value in deltas.items() if value < 0]
    avg_delta = (
        sum(deltas.values()) / len(deltas) if deltas else 0.0
    )

    consistent_drop = (
        len(dropped_keys) >= CONSISTENT_DROP_MIN_COUNT
        and avg_delta <= -CONSISTENT_DROP_THRESHOLD
    )

    label_map = dict(TRACKED_METRICS)

    if hard_drop:
        print("⚠️  检测到检索命中退化：")
        print(
            f"   - Hit@3 从 {_fmt(old_metrics.get('hit_at_3'))} "
            f"掉到 {_fmt(current_metrics.get('hit_at_3'))}"
            f"（{(deltas.get('hit_at_3', 0)) * 100:+.1f} 个点）"
        )
        print()

    elif consistent_drop:
        print("⚠️  疑似退化：多个指标同时下降")
        print(
            f"   - {'、'.join(label_map[k] for k in dropped_keys)} "
            f"共 {len(dropped_keys)} 项下降，平均 {avg_delta * 100:+.1f} 个点"
        )
        print()
        print("   单项降幅都没到 5 个点，但方向一致、幅度接近，")
        print("   这不是随机波动该有的样子——很可能改动了检索锚点。")
        print()

    if hard_drop or consistent_drop:
        print("建议按这个顺序排查：")
        print("  1. 用 `python scripts/knowledge_version.py diff <基线版本>` 看动了哪些文件")
        print("  2. 看是不是改掉了原有条文里的关键词（「尚未」「7 天」「不支持」这类）")
        print("  3. 用 `git diff -- data/knowledge/` 逐行确认措辞改动")
        print()
        print("注意：这不一定是你改错了。")
        print("      如果某条政策本来就要收紧，指标下降是正常结果，")
        print("      但要同步更新评测集的期望，否则以后每次都报退化。")
    else:
        print("没有检测到明显退化。")
        print("      判定口径：Hit@3 掉 5 点以上，或 ≥3 个指标同向下降且平均超 3 点。")
        print()
        print("提醒：单次运行有 ±4 个点的波动，")
        print("      接近阈值的差异建议重复跑一次再下结论。")


if __name__ == "__main__":
    main()
