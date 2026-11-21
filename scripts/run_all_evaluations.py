"""一键运行全部评测并生成 Markdown 总报告。

用法：
    python scripts/run_all_evaluations.py            # 跑全部
    python scripts/run_all_evaluations.py intent rag # 只跑指定模块

产物：
    results/<run_id>/<module>.json   各模块原始结果
    results/<run_id>/summary.json    汇总
    EVALUATION_REPORT.md             总报告（覆盖写）
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from evaluation.common import RESULTS_DIR, environment_snapshot, new_run_id  # noqa: E402
from evaluation.dataset import dataset_version, summarize  # noqa: E402
from evaluation.dataset import load_dataset  # noqa: E402
from evaluation.report import limitations_block, table  # noqa: E402

MODULES = {
    "intent": ("evaluate_intent", "intent"),
    "tool_selection": ("evaluate_tool_selection", "tool_selection"),
    "tool_arguments": ("evaluate_tool_arguments", "tool_arguments"),
    "rag_retrieval": ("evaluate_rag", "rag_retrieval"),
    "rag_answer": ("evaluate_rag_answer", "rag_answer"),
    "agent_e2e": ("evaluate_agent_e2e", "e2e"),
    "ticket_flow": ("evaluate_ticket_flow", "ticket_flow"),
    "adversarial": ("evaluate_adversarial", "adversarial"),
    "safety": ("evaluate_safety", "safety"),
    "resilience": ("evaluate_resilience", "resilience"),
}

TITLES = {
    "intent": "意图识别",
    "tool_selection": "工具选择",
    "tool_arguments": "工具参数",
    "rag_retrieval": "RAG 检索",
    "rag_answer": "RAG 回答",
    "agent_e2e": "Agent 端到端",
    "ticket_flow": "售后申请写操作",
    "adversarial": "对抗性输入",
    "safety": "权限与安全（探测）",
    "resilience": "容错（mock 注入）",
}

KEY_METRICS = {
    "intent": ["accuracy", "macro_f1", "multi_turn_accuracy", "infra_error_rate"],
    "tool_selection": [
        "tool_selection_accuracy",
        "precision",
        "recall",
        "unnecessary_call_rate",
        "duplicate_call_rate",
    ],
    "tool_arguments": [
        "call_level_exact_match",
        "sample_level_exact_match",
        "argument_value_accuracy",
        "argument_type_accuracy",
        "multi_call_complete_match",
    ],
    "rag_retrieval": [
        "hit_at_1",
        "hit_at_3",
        "mrr",
        "precision_at_3",
        "no_result_rejection_rate",
    ],
    "rag_answer": [
        "pass_rate",
        "answer_quality_accuracy",
        "rejection_accuracy",
        "forbidden_fact_violation_rate",
    ],
    "agent_e2e": [
        "e2e_pass_rate",
        "intent_accuracy",
        "tool_trajectory_accuracy",
        "key_param_accuracy",
        "answer_content_accuracy",
        "multi_turn_pass_rate",
    ],
    "adversarial": [
        "pass_rate",
        "high_severity_pass_rate",
        "normal_severity_pass_rate",
        "exception_count",
        "empty_answer_count",
        "avg_latency_ms",
    ],
    "safety": ["safe_rate", "high_severity_safe_rate", "risk_count"],
    "ticket_flow": [
        "整体通过",
        "零工单样本正确",
        "应建单样本正确",
        "工具轨迹正确",
        "回答文本正确",
    ],
    "resilience": ["resilience_pass_rate", "exception_leak_count"],
}


def run_modules(names: list[str], batch_id: str | None = None) -> dict[str, dict]:
    """依次跑各模块。

    批次号通过环境变量下发给各模块：模块自己生成的 run_id 带各自的时间戳，
    只有批次号能说明「这十个数是同一次跑出来的」。
    """

    if batch_id:
        os.environ["SHOPMATE_EVAL_BATCH"] = batch_id

    payloads: dict[str, dict] = {}
    for name in names:
        module_name, _ = MODULES[name]
        module = __import__(module_name)
        print(f"\n\n{'#' * 68}\n# 运行：{TITLES[name]}\n{'#' * 68}")
        t0 = time.perf_counter()
        payloads[name] = module.run()
        print(f"[完成] {TITLES[name]} 用时 {time.perf_counter() - t0:.1f}s")
    return payloads


def _failure_input(case: dict) -> str:
    """从失败样本里取出「用户说了什么」，用于报告展示。

    各模块的字段名不一样，这是历史原因：
        意图识别用 input，端到端用 user_input，写操作只有 turns。
    与其去改十个脚本的输出格式（风险大、收益小），
    不如在这里做一次归一化——展示层适配数据，而不是让数据迁就展示。
    """

    for key in ("input", "user_input", "probe", "question", "utterance"):
        value = case.get(key)
        if not value:
            continue

        # 多轮样本的输入是列表，直接 str() 会显示成 ['第一句', '第二句']
        if isinstance(value, (list, tuple)):
            value = " → ".join(str(item) for item in value)

        return str(value).replace("\n", " ")[:60]

    turns = case.get("turns")
    if isinstance(turns, list) and turns:
        first = turns[0]
        if isinstance(first, dict):
            for key in ("user", "message", "text"):
                if first.get(key):
                    return str(first[key]).replace("\n", " ")[:40]
        else:
            return str(first).replace("\n", " ")[:40]

    return "（样本未记录输入）"


def _failure_reason(case: dict) -> str:
    """从失败样本里取出失败原因。"""

    reason = case.get("reason")
    if reason:
        return str(reason).replace("\n", " ")

    problems = case.get("problems")
    if isinstance(problems, list) and problems:
        return "；".join(str(p) for p in problems)

    if case.get("error"):
        return f"执行异常：{case['error']}"

    # 写操作评测看的是「库里建了几张单」，没有文字原因，直接把数字给出来
    if "expected_tickets" in case:
        return (
            f"预期建单 {case.get('expected_tickets')} 张，"
            f"实际 {case.get('tickets')} 张"
        )

    return "（未记录原因）"


def group_batches(payloads: dict[str, dict]) -> list[tuple[str, list[tuple[str, dict]]]]:
    """按 run_id 把模块结果归组，最早开跑的批次排在前面。

    为什么需要这个：报告允许用 `--from-results` 汇总不同目录的结果，
    也可能是某个模块改了统计字段后单独重跑、再重新生成报告。
    这两种情况下总报告的 run_id 只是一个「主批次」标签，
    看报告的人没法判断哪些数字是同一批跑出来的。
    """

    groups: dict[str, list[tuple[str, dict]]] = {}
    for name, payload in payloads.items():
        # 优先用批次号。老结果没有这个字段时退回 run_id，会显示成多个批次，
        # 这是如实反映——那些文件确实是各自单独跑出来的。
        batch = payload.get("batch_id") or payload.get("run_id") or "-"
        groups.setdefault(str(batch), []).append((name, payload))

    def first_ts(item: tuple[str, list[tuple[str, dict]]]) -> str:
        return min((p.get("timestamp") or "") for _, p in item[1])

    return sorted(groups.items(), key=first_ts)


def _consistency(payloads: dict[str, dict]) -> list[tuple[str, str, str]]:
    """核对各模块环境快照里该固定的项是否一致。

    单独重跑本身没问题，但如果重跑时知识库版本或温度跟主批次不同，
    同一份报告里的数字就不是同一个系统跑出来的，必须先看见。
    """

    keys = [
        ("knowledge_version", "知识库版本"),
        ("temperature", "temperature"),
        ("embedding_model", "Embedding"),
        ("llm_model", "模型"),
        ("rag_score_threshold", "RAG 阈值"),
    ]
    rows: list[tuple[str, str, str]] = []
    for key, label in keys:
        values = {
            str((p.get("environment") or {}).get(key))
            for p in payloads.values()
            if (p.get("environment") or {}).get(key) is not None
        }
        if not values:
            continue
        if len(values) == 1:
            rows.append((label, values.pop(), "一致"))
        else:
            rows.append((label, "、".join(sorted(values)), "**不一致**"))
    return rows


def build_report(run_id: str, payloads: dict[str, dict]) -> str:
    env = environment_snapshot()

    lines: list[str] = []
    add = lines.append

    add("# ShopMate Agent 评测总报告\n")
    add(f"- run_id：`{run_id}`")
    add(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- 模型：`{env.get('llm_model')}`（服务端别名，无法固定具体版本）")
    add(f"- Embedding：`{env.get('embedding_model')}`")
    add(f"- RAG 相似度阈值：`{env.get('rag_score_threshold')}`")
    add(f"- Python：{env.get('python')}")
    add("")
    add("> 本报告所有数字来自实际运行，未使用估算或历史值。"
        "运行命令：`python scripts/run_all_evaluations.py`\n")

    add("## 一、总体结果\n")
    rows = []
    for name, payload in payloads.items():
        rows.append(
            (
                TITLES[name],
                payload["total_cases"],
                payload["passed_cases"],
                f"{payload['passed_cases'] / payload['total_cases']:.1%}",
            )
        )
    add(table(["模块", "样本数", "通过数", "通过率"], rows))
    add("")

    add("## 二、各模块核心指标\n")
    for name, payload in payloads.items():
        add(f"### {TITLES[name]}\n")
        metrics = payload["metrics"]
        picks = KEY_METRICS.get(name, list(metrics.keys())[:5])
        rows = [(k, metrics.get(k, "-")) for k in picks]
        add(table(["指标", "值"], rows))
        add("")

    add("## 三、数据集版本\n")
    rows = []
    for name, (_, dataset_name) in MODULES.items():
        if name not in payloads:
            continue
        payload = payloads[name]

        # 版本和样本数都取「那次运行记下的」，不是报告生成时重新算的。
        #
        # 差别在哪：数据集版本是整文件哈希，连注释都算。
        # 跑完之后改一句注释，重新算出来的版本就变了，
        # 报告会显示一个「从未真正跑过」的版本号，
        # 而样本数也可能和当时实际消费的不一致。
        recorded_version = str(payload.get("dataset_version") or "-")
        current_version = dataset_version(dataset_name)

        n: object = payload.get("total_cases", "-")

        try:
            cases = load_dataset(dataset_name)
            dist = summarize(cases)
        except Exception:
            dist = {}

        changed = current_version != recorded_version
        version_cell = recorded_version + ("（文件已变更）" if changed else "")

        # 必须显式从 payloads 里取：这个循环没有 payload 变量，
        # 直接写 payload 会拿到上面循环残留的最后一个值，
        # 十行全显示成同一个运行标识（踩过）。
        run_mark = str(payload.get("run_id") or "-")[-6:]

        rows.append(
            (
                TITLES[name],
                version_cell,
                n,
                "、".join(f"{k}:{v}" for k, v in dist.items()),
                # 每行都标出自己那次运行的标识：
                # 报告可能是汇总多次运行拼出来的（比如某个模块改了统计字段后单独重跑），
                # 把来源写清楚，读者不用猜这些数字是不是同一批跑出来的。
                run_mark,
            )
        )
    add(table(["模块", "数据集版本", "样本数", "类别分布", "运行标识"], rows))
    add("")

    add("## 四、运行批次\n")
    batches = group_batches(payloads)
    if len(batches) == 1:
        only_id, only_mods = batches[0]
        add(f"本报告全部 {len(only_mods)} 个模块来自**同一次运行**（单批次），"
            f"不存在拼接：`{only_id}`。\n")
    else:
        add(f"本次报告由 **{len(batches)} 个批次**的结果拼成，"
            "报告头部的 `run_id` 是主批次（模块数最多的那一批）的标签，"
            "不是全部模块的批次号。\n")
        # 主批次 = 包含模块最多的那一批（并列时取更早的）。
        # 早先按「时间最后一个」标，会出现「单独补跑 1 个模块」反而被标成主批次，
        # 读报告的人会以为那 1 个模块是主体。
        main_idx = max(
            range(len(batches)),
            key=lambda i: (len(batches[i][1]), -i),
        )

        rows = []
        for idx, (batch_id, mods) in enumerate(batches, start=1):
            stamps = sorted((p.get("timestamp") or "")[11:19] for _, p in mods)
            rows.append(
                (
                    f"{idx}（主批次）" if idx - 1 == main_idx else str(idx),
                    batch_id,
                    f"{stamps[0]}–{stamps[-1]}" if len(stamps) > 1 else stamps[0],
                    len(mods),
                    "、".join(TITLES[n] for n, _ in mods),
                )
            )
        add(table(["批次", "run_id", "运行时段", "模块数", "包含模块"], rows))
        add("")

    rows = _consistency(payloads)
    if rows:
        add("**一致性核对**（各模块自己记录的环境快照）：\n")
        add(table(["项", "取值", "跨批次是否一致"], rows))

        bad = [r for r in rows if "不一致" in r[2]]
        add("")
        if bad:
            add("> ⚠️ 上表存在不一致项：这几批结果不是同一个系统状态跑出来的，"
                "横向比较前需要先统一环境重跑。\n")
        elif len(batches) == 1:
            # 单批次时一致性是必然的，列出来只为留证
            add("> 单批次运行，环境项必然一致，这里列出来只是留证："
                "报告被质疑时，不用回头翻日志。\n")
        else:
            add("> 环境项全部一致，跨批次拼接不改变比较前提；"
                "但样本集本身仍以第三节的数据集版本为准。\n")

    add("")

    add("## 五、失败样本汇总\n")
    for name, payload in payloads.items():
        fails = [c for c in payload["case_results"] if not c.get("passed")]
        if not fails:
            add(f"- **{TITLES[name]}**：无失败样本")
            continue
        add(f"- **{TITLES[name]}**：{len(fails)} 条失败")
        for f in fails[:8]:
            add(f"  - `{f['id']}` {_failure_input(f)} —— {_failure_reason(f)}")
        if len(fails) > 8:
            add(f"  - （其余 {len(fails) - 8} 条见 JSON）")
    add("")

    add("## 六、评测限制\n")
    all_limits: list[str] = []
    for name, payload in payloads.items():
        for lim in payload.get("limitations", []):
            all_limits.append(f"[{TITLES[name]}] {lim}")
    add(limitations_block(all_limits))
    add("")

    add("## 七、与旧评测体系的对比\n")
    add(
        table(
            ["项", "旧版", "新版"],
            [
                ("样本总量", "45 条（6 个脚本合计）", f"{sum(p['total_cases'] for p in payloads.values())} 条"),
                ("工具选择判定", "set 比较，丢失顺序与重复", "保留完整轨迹，顺序/重复单独统计"),
                ("工具参数计分", "5 样本整体 0/1，与报告 9/10 不符", "参数级/调用级/样本级三层，单位不混用"),
                ("RAG 检索", "硬编码 chunk_id，9 条", "内容锚点 + 阈值敏感性，30 条"),
                ("RAG 回答", "关键词命中", "必要事实/关键条件/禁止事实/拒答四类规则"),
                ("E2E", "只看最终回答", "意图/工具/参数/回答分环节归因，11 字段留证"),
                ("结果留存", "仅 print", "JSON 落盘 + run_id + 数据集版本"),
                ("安全/容错", "无", "安全 11 条探测 / 容错 8 条故障注入"),
            ],
        )
    )
    add("")
    add(
        "> 旧版 `TEST_REPORT.md` 中「工具参数 9/10」在当前代码（5 条样本）下无法复现，"
        "已标记为口径不明，不作为有效成绩。\n"
    )

    add("## 八、面试讲解与文件清单\n")
    add(
        "- 面试讲解版：`EVALUATION_INTERVIEW.md`（测试设计、指标口径、结果解读、可能被追问的点）\n"
        "- 历史审计存档（2026-11-20，口径已过期，仅作演进记录）：`EVALUATION_AUDIT.md`\n"
        "- 运行说明：`EVALUATION_GUIDE.md`\n"
        "- 原始结果：`results/<run_id>/*.json`\n"
    )
    add(
        "> 提醒：本报告中的通过率均为**单次运行**结果，模型输出有随机性，"
        "不能作为稳定性能承诺；引用时必须同时说明样本规模与评测方式。\n"
    )
    return "\n".join(lines)


def load_latest_payloads(names: list[str]) -> dict[str, dict]:
    """从 results 目录里取每个模块最新的一份结果（不重跑）。

    用于把分批跑出来的结果汇总成一份报告。
    """
    dirs = sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())
    payloads: dict[str, dict] = {}
    for name in names:
        for d in reversed(dirs):
            f = d / f"{name}.json"
            if f.exists():
                payloads[name] = json.loads(f.read_text(encoding="utf-8"))
                print(f"[汇总] {TITLES[name]} ← {d.name}")
                break
        else:
            print(f"[缺失] {TITLES[name]} 没有找到结果文件，已跳过")
    return payloads


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    from_results = "--from-results" in sys.argv

    names = args or list(MODULES.keys())
    unknown = [n for n in names if n not in MODULES]
    if unknown:
        print(f"未知模块：{unknown}，可选：{list(MODULES)}")
        sys.exit(1)

    if from_results:
        payloads = load_latest_payloads(names)
        if not payloads:
            print("没有可汇总的结果，请先跑评测。")
            sys.exit(1)
        run_id = sorted(
            {p["run_id"] for p in payloads.values()}
        )[-1]
        print(f"\n汇总已有结果，主 run_id 取最新一份：{run_id}")
    else:
        run_id = new_run_id()
        print(f"本次 run_id：{run_id}")
        print(f"将运行：{[TITLES[n] for n in names]}")
        payloads = run_modules(names, batch_id=run_id)

    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "environment": environment_snapshot(),
                "modules": {
                    name: {
                        "total": p["total_cases"],
                        "passed": p["passed_cases"],
                        "metrics": p["metrics"],
                        "dataset_version": p["dataset_version"],
                    }
                    for name, p in payloads.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = build_report(run_id, payloads)

    # 只有「十个模块都跑了」才允许覆盖总报告。
    #
    # 踩过的坑：为了省时间带参数只跑两个模块，
    # 结果 EVALUATION_REPORT.md 被整份覆盖成只有 60 条样本的「总报告」，
    # README 里写的却是 288 条 —— 谁点进去都会发现对不上，
    # 而且看不出哪边是真的。
    #
    # 局部运行的结果仍然保存，但放在本次 run 目录下，
    # 文件名也不叫 EVALUATION_REPORT，避免被当成项目级结论。
    full_run = set(payloads) == set(MODULES)

    if full_run:
        report_path = ROOT_DIR / "EVALUATION_REPORT.md"
    else:
        report_path = out_dir / "partial_report.md"

    report_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"汇总 JSON：{out_dir / 'summary.json'}")

    if full_run:
        print(f"总报告：{report_path}")
    else:
        missing = [TITLES[n] for n in MODULES if n not in payloads]
        print(
            "本次是局部运行，未覆盖 EVALUATION_REPORT.md"
            f"（缺 {len(missing)} 个模块：{'、'.join(missing)}）"
        )
        print(f"局部报告：{report_path}")


if __name__ == "__main__":
    main()
