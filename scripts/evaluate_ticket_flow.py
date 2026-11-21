"""售后申请写操作评测。

和其他评测模块的区别：

    别的模块看「回答对不对」，
    这个模块只看「库里多了几张工单」。

    因为写操作错了，话说得再好也没用：
    多建一张就是多一条人工审核，多一次误退款。

判定维度：
    tickets        新建工单数（期望绝大多数是 0）
    must_have      最后一轮回答里该说的关键信息
    must_not_have  最后一轮回答里绝对不能出现的幻觉式表态
    tools_required 全程必须真实调用到的工具

为什么每条样本都清库：
    建单走 (user_id, order_id, request_type) 部分唯一索引，
    上一条样本留下的工单会让下一条的创建变成「重放」，
    工单数变成 0，看起来像通过，其实是被上一轮污染了。

运行：
    python scripts/evaluate_ticket_flow.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from app.agent.agent import run_agent  # noqa: E402
from app.agent.conversation_store import clear_conversation  # noqa: E402
from app.agent.tool_executor import execute_tool as real_execute_tool  # noqa: E402
from app.core.db import get_connection  # noqa: E402
from evaluation.common import (  # noqa: E402
    Stopwatch,
    build_payload,
    new_run_id,
    save_result,
)
from evaluation.dataset import dataset_version, load_dataset, summarize  # noqa: E402
from evaluation.metrics import ratio  # noqa: E402
from evaluation.report import metric_table, table  # noqa: E402

NAME = "ticket_flow"
DATASET = "ticket_flow"


def _norm(text: str) -> str:
    """去掉空白和标点，避免「已 提交」这类排版差异导致漏判。"""

    return re.sub(r"[\s，。、；：！？,.;:!?*#`|>「」（）()【】\[\]-]+", "", text or "")


def _groups(groups: list) -> list[list[str]]:
    """把一维/二维的关键词组统一成二维。

    为什么需要这一步：
        数据集里有人写 [["已提交成功"]]，也有人写 ["已提交成功"]。
        如果判定函数直接 `for keyword in group` 遍历一维数组，
        拿到的是字符串，再遍历就变成单个汉字——
        "已提交成功" 会被拆成 ['已','提','交','成','功']，
        于是回答里出现任何一个字都算命中，判定全错。

        真实踩过：第一版就是这个 bug，14 条里 9 条误报。
    """

    out: list[list[str]] = []

    for group in groups or []:
        if isinstance(group, str):
            out.append([group])
        else:
            out.append(list(group))

    return out


def _must_have_ok(answer: str, groups: list) -> tuple[bool, list[list[str]]]:
    """每组至少命中一个词；组之间是 AND。"""

    a = _norm(answer)

    failed = [
        group
        for group in _groups(groups)
        if not any(_norm(keyword) in a for keyword in group)
    ]

    return (not failed), failed


def _must_not_have_ok(
    answer: str,
    groups: list,
) -> tuple[bool, list[str]]:
    """所有组里一个词都不能出现。"""

    a = _norm(answer)

    hit = [
        keyword
        for group in _groups(groups)
        for keyword in group
        if _norm(keyword) in a
    ]

    return (not hit), hit


def _reset_ticket_tables() -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM after_sales_tickets")
        connection.execute("DELETE FROM idempotency_keys")
        connection.execute("DELETE FROM ticket_drafts")


def _ticket_count() -> int:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS c FROM after_sales_tickets"
        ).fetchone()

    return int(row["c"])


def _list_tickets() -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT ticket_id, order_id, request_type, reason, status
            FROM after_sales_tickets
            ORDER BY created_at
            """
        ).fetchall()

    return [dict(row) for row in rows]


def run() -> dict:
    cases = load_dataset(DATASET)
    version = dataset_version(DATASET)
    run_id = new_run_id()

    print("=" * 68)
    print("ShopMate 售后申请写操作评测")
    print(f"run_id={run_id}  数据集版本={version}  样本数={len(cases)}")
    print("类别分布：", summarize(cases))
    print("=" * 68)

    rows: list[dict] = []
    failures: list[dict] = []

    passed_n = 0
    zero_ticket_ok = 0
    zero_ticket_total = 0
    one_ticket_ok = 0
    one_ticket_total = 0
    tool_ok_n = 0
    tool_total = 0
    text_ok_n = 0

    for case in cases:
        case_id = case["id"]
        # 数据集统一用 input 字段；多轮样本给的是列表
        raw_input = case["input"]
        turns = raw_input if isinstance(raw_input, list) else [raw_input]
        user_id = case.get("user_id", "USER001")
        expected = case["expected"]

        conv = f"ticketflow_{case_id}"

        _reset_ticket_tables()
        clear_conversation(conv)

        trace: list[dict] = []
        answers: list[str] = []
        error: str | None = None

        def spy_tool(
            tool_name: str,
            arguments: dict,
            conversation_id: str | None = None,
        ) -> str:
            trace.append({"tool": tool_name, "arguments": arguments})
            return real_execute_tool(
                tool_name=tool_name,
                arguments=arguments,
                conversation_id=conversation_id,
            )

        sw = Stopwatch()

        try:
            with patch("app.agent.agent.execute_tool", side_effect=spy_tool):
                for turn in turns:
                    answer = run_agent(
                        user_message=turn,
                        conversation_id=conv,
                        user_id=user_id,
                    )
                    answers.append(answer)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

        elapsed = sw.ms()

        created = _ticket_count()
        actual_tools = [item["tool"] for item in trace]
        last_answer = answers[-1] if answers else ""

        # ---------- 判定 ----------

        exp_tickets = int(expected["tickets"])
        ticket_ok = (created == exp_tickets)

        if exp_tickets == 0:
            zero_ticket_total += 1
            zero_ticket_ok += int(ticket_ok)
        else:
            one_ticket_total += 1
            one_ticket_ok += int(ticket_ok)

        tools_req = expected.get("tools_required", [])
        tools_missing = [t for t in tools_req if t not in actual_tools]
        tool_ok = not tools_missing

        if tools_req:
            tool_total += 1
            tool_ok_n += int(tool_ok)

        have_ok, have_missing = _must_have_ok(
            last_answer, expected.get("must_have", [])
        )
        have_ok2, forbidden_hit = _must_not_have_ok(
            last_answer, expected.get("must_not_have", [])
        )

        text_ok = have_ok and have_ok2

        if text_ok:
            text_ok_n += 1

        passed = ticket_ok and tool_ok and text_ok and error is None

        if passed:
            passed_n += 1
        else:
            reasons = []

            if not ticket_ok:
                reasons.append(f"工单数 {created}，期望 {exp_tickets}")
            if not tool_ok:
                reasons.append(f"缺工具 {tools_missing}")
            if not have_ok:
                reasons.append(f"回答缺关键词 {have_missing}")
            if not have_ok2:
                reasons.append(f"回答出现禁用词 {forbidden_hit}")
            if error:
                reasons.append(f"异常 {error}")

            failures.append(
                {
                    "id": case_id,
                    "category": case["category"],
                    "reason": "；".join(reasons),
                    "answer": last_answer[:120],
                    "tools": ",".join(actual_tools),
                }
            )

        rows.append(
            {
                "id": case_id,
                "category": case["category"],
                "input": case["input"][0] if case.get("input") else "",
                "turns": len(turns),
                "tickets": created,
                "expected_tickets": exp_tickets,
                "tools": ",".join(actual_tools) or "-",
                "passed": passed,
                # 失败原因要跟着样本一起存下来。
                # 之前只写进单独的 failures 列表，而报告读的是 case_results，
                # 结果「失败样本汇总」里这一栏是空的，看着像没跑完。
                "reason": "" if passed else "；".join(reasons),
            }
        )

        mark = "PASS" if passed else "FAIL"
        print(
            f"[{mark}] {case_id} 工单={created}/{exp_tickets} "
            f"工具={','.join(actual_tools) or '-'} {elapsed:.0f}ms"
        )

        if not passed:
            print(f"       理由：{failures[-1]['reason']}")
            print(f"       回答：{last_answer[:150]}")

        clear_conversation(conv)

    _reset_ticket_tables()

    total = len(cases)

    metrics = {
        "样本数": total,
        "整体通过": f"{passed_n}/{total} = {ratio(passed_n, total):.1%}",
        "零工单样本正确": f"{zero_ticket_ok}/{zero_ticket_total} = "
        f"{ratio(zero_ticket_ok, zero_ticket_total):.1%}",
        "应建单样本正确": f"{one_ticket_ok}/{one_ticket_total} = "
        f"{ratio(one_ticket_ok, one_ticket_total):.1%}",
        "工具轨迹正确": f"{tool_ok_n}/{tool_total} = "
        f"{ratio(tool_ok_n, tool_total):.1%}",
        "回答文本正确": f"{text_ok_n}/{total} = {ratio(text_ok_n, total):.1%}",
    }

    print()
    print(metric_table(metrics, "写操作指标"))

    if failures:
        print(
            table(
                ("样本", "类别", "失败原因"),
                [
                    (f["id"], f["category"], f["reason"][:90])
                    for f in failures
                ],
            )
        )

    payload = build_payload(
        evaluation_name=NAME,
        run_id=run_id,
        dataset_version=version,
        total_cases=total,
        passed_cases=passed_n,
        failed_cases=total - passed_n,
        metrics=metrics,
        case_results=[{**row, "passed": row["passed"]} for row in rows],
        error_details=failures,
        limitations=[
            "样本量 14 条，覆盖正常流程、闸门拦截、越权与边界，"
            "但不覆盖长期运行后的草稿堆积。",
            "工单数用「跑前清库 + 跑后计数」统计，"
            "评测与业务共用一个 SQLite 文件，不能与线上实例同时跑。",
            "must_have 是关键词匹配，模型换一种说法仍会被判失败，"
            "失败样本需要人工看原文复核。",
        ],
    )

    save_result(NAME, run_id, payload)

    return payload


if __name__ == "__main__":
    result = run()
    print()
    print(f"结果已保存：results/{NAME}/")
