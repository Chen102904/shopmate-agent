"""会话长期记忆（滚动摘要）测试。

核心要守的一条：

    分段摘要必须「合并」而不是「重写」。

    长对话是分批摘要的（覆盖 0-12、12-22、22-32……）。
    如果每批都直接覆盖上一次的结果，
    那么聊到 30 轮，第 1 轮说的订单号就没了——
    这恰恰是这个模块要解决的问题，不能反过来制造它。

    这个 bug 真实存在过：实现每次都用新摘要替换旧摘要，
    而模块文档写着「早期的关键事实不会丢」。
"""

from app.agent.conversation_store import clear_conversation
from app.agent.memory import (
    SUMMARY_KEEP_RECENT,
    SUMMARY_TRIGGER_MESSAGES,
    build_memory_context,
    clear_memory,
)


def _conv(name: str) -> str:
    conversation_id = f"test-memory-{name}"
    clear_conversation(conversation_id)
    return conversation_id


def _history(count: int) -> list[dict]:
    messages = []
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append(
            {"role": role, "content": f"第{index}条消息"}
        )
    return messages


def test_short_history_returns_as_is():
    conv = _conv("short")

    history = _history(6)

    summary, recent = build_memory_context(conv, history)

    assert summary is None
    assert recent == history

    clear_conversation(conv)


def test_long_history_produces_summary(monkeypatch):
    conv = _conv("long")

    calls = []

    def fake_summary(messages, previous_summary=""):
        calls.append(len(messages))
        return "用户咨询订单 ORD001 的退款问题，尚未解决。"

    monkeypatch.setattr(
        "app.agent.memory._generate_summary",
        fake_summary,
    )

    history = _history(SUMMARY_TRIGGER_MESSAGES + 6)

    summary, recent = build_memory_context(conv, history)

    assert summary is not None
    assert "ORD001" in summary
    assert len(recent) == SUMMARY_KEEP_RECENT
    assert calls, "应当调用一次摘要生成"

    clear_conversation(conv)


def test_summary_is_reused_not_regenerated(monkeypatch):
    """同一段历史不应反复摘要，否则每轮都多一次模型调用。"""

    conv = _conv("reuse")

    calls = []

    def fake_summary(messages, previous_summary=""):
        calls.append(1)
        return "摘要内容"

    monkeypatch.setattr(
        "app.agent.memory._generate_summary",
        fake_summary,
    )

    history = _history(SUMMARY_TRIGGER_MESSAGES + 6)

    build_memory_context(conv, history)
    build_memory_context(conv, history)

    assert len(calls) == 1

    clear_conversation(conv)


def test_second_summary_merges_previous(monkeypatch):
    """第二次摘要必须把上一次的结果作为输入传进去。

    这是修 bug 的关键断言。
    如果实现又变回「只摘要新的一段、然后覆盖」，
    这里会拿到空串，测试立刻挂。
    """

    conv = _conv("merge")

    previous_seen: list[str] = []

    def fake_summary(messages, previous_summary=""):
        previous_seen.append(previous_summary)
        return f"摘要{len(previous_seen)}"

    monkeypatch.setattr(
        "app.agent.memory._generate_summary",
        fake_summary,
    )

    # 第一次：触发摘要，覆盖较早部分
    build_memory_context(conv, _history(20))

    # 第二次：历史继续增长，超出已覆盖范围，需要再摘要一段
    build_memory_context(conv, _history(32))

    assert len(previous_seen) == 2
    assert previous_seen[0] == "", "第一次摘要是全新生成，不该有前序"
    assert previous_seen[1] == "摘要1", "第二次摘要丢了上一次的结果"

    clear_conversation(conv)


def test_summary_failure_degrades_gracefully(monkeypatch):
    """首次摘要失败：退化成只带最近消息，不阻塞对话。"""

    conv = _conv("fail")

    monkeypatch.setattr(
        "app.agent.memory._generate_summary",
        lambda messages, previous_summary="": "",
    )

    history = _history(SUMMARY_TRIGGER_MESSAGES + 6)

    summary, recent = build_memory_context(conv, history)

    assert summary is None
    assert len(recent) == SUMMARY_KEEP_RECENT

    clear_conversation(conv)


def test_summary_failure_keeps_previous_summary(monkeypatch):
    """增量摘要失败时，不能把已有摘要一起丢掉。

    清空等于把早期信息全丢，
    比「摘要不更新」糟糕得多。
    """

    conv = _conv("keep")

    monkeypatch.setattr(
        "app.agent.memory._generate_summary",
        lambda messages, previous_summary="": "第一版摘要",
    )

    build_memory_context(conv, _history(20))

    # 第二次生成失败
    monkeypatch.setattr(
        "app.agent.memory._generate_summary",
        lambda messages, previous_summary="": "",
    )

    summary, _ = build_memory_context(conv, _history(32))

    assert summary == "第一版摘要"

    clear_conversation(conv)


def test_clear_memory_removes_summary(monkeypatch):
    conv = _conv("clearmem")

    monkeypatch.setattr(
        "app.agent.memory._generate_summary",
        lambda messages, previous_summary="": "旧摘要",
    )

    history = _history(SUMMARY_TRIGGER_MESSAGES + 6)

    build_memory_context(conv, history)
    clear_memory(conv)

    _, recent = build_memory_context(conv, history)

    assert len(recent) == SUMMARY_KEEP_RECENT

    clear_conversation(conv)
