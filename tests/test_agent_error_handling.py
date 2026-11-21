"""Agent 层异常兜底测试。

守的核心是一条：

    **任何异常都不允许穿透到 API 层。**

    穿透的后果不是「日志里多一条」，
    而是用户收到裸的 500「服务器内部错误」，
    而且前面已经花掉的模型调用全部作废。

真实缺口（2026-11-20 审计发现）：
    run_agent 里唯一的 try 只包住了工具调用循环，
    「取会话历史」「写用户消息」这些数据库操作都在 try 外面。
    数据库一抖动就直接 500。

修法是把主流程包一层（run_agent → _run_agent_impl），
下面这些用例就是守那一层的。
"""

import pytest

import app.agent.agent as agent_module


def _boom(*args, **kwargs):
    raise RuntimeError("模拟数据库故障")


def _conv(name: str) -> str:
    return f"test-err-{name}"


def test_unexpected_error_returns_fallback(monkeypatch):
    """主流程抛意外异常时，返回友好文案而不是向上抛。"""

    monkeypatch.setattr(agent_module, "get_conversation", _boom)

    result = agent_module.run_agent(
        "ORD001 到哪了",
        conversation_id=_conv("main"),
        user_id="USER001",
    )

    assert result == agent_module.UNEXPECTED_ERROR_MESSAGE


def test_fallback_survives_db_failure(monkeypatch):
    """降级回复自己落库失败时，仍要返回文案。

    这条容易被忽略：
        能走到降级分支通常说明服务已经不健康，
        这时候写数据库很可能也失败。
        如果降级函数不自己兜住，它会变成新的异常源，
        等于兜了个寂寞。
    """

    monkeypatch.setattr(agent_module, "get_conversation", _boom)
    monkeypatch.setattr(agent_module, "add_message", _boom)

    result = agent_module.run_agent(
        "ORD001 到哪了",
        conversation_id=_conv("dbdown"),
        user_id="USER001",
    )

    assert result == agent_module.UNEXPECTED_ERROR_MESSAGE


def test_model_error_fallback_survives_db_failure(monkeypatch):
    """模型不可用的降级同样要能扛住数据库失败。"""

    monkeypatch.setattr(agent_module, "add_message", _boom)

    result = agent_module._save_model_error_response(_conv("model"))

    assert result == agent_module.MODEL_UNAVAILABLE_MESSAGE


def test_fallback_message_does_not_leak_internals():
    """降级文案里不能出现路径、SQL、异常类型这些内部信息。"""

    text = agent_module.UNEXPECTED_ERROR_MESSAGE

    for leaked in ("Error", "Traceback", "sqlite", "C:\\", "/app/", "None"):
        assert leaked not in text


def test_intent_classifier_failure_degrades_to_clarification(monkeypatch):
    """意图分类失败时，退化成「没听懂」而不是整轮失败。

    这条不是本次修的，但值得有个用例守着——
    它是「异常在正确的层级被消化」的正面样板。
    """

    from app.agent import intent_classifier

    def _broken(*args, **kwargs):
        raise RuntimeError("模型挂了")

    monkeypatch.setattr(
        intent_classifier,
        "create_chat_completion_with_fallback",
        _broken,
    )

    result = intent_classifier.classify_intent("我要退款")

    assert result.intent == "unclear"
    assert result.need_clarification is True


@pytest.mark.parametrize(
    "stage",
    ["get_conversation", "get_pending_clarification", "add_message"],
)
def test_each_early_stage_is_covered(monkeypatch, stage):
    """主流程开头的每个数据库操作都要被兜住。

    参数化是为了防止「补了一个、漏了另一个」——
    当初的缺口就是「只包住了工具循环，前面的没包」。
    """

    monkeypatch.setattr(agent_module, stage, _boom)

    result = agent_module.run_agent(
        "ORD001 到哪了",
        conversation_id=_conv(stage),
        user_id="USER001",
    )

    assert result == agent_module.UNEXPECTED_ERROR_MESSAGE


# ============================================================
# 降级不能泄露内部信息（2026-11-21 审计发现）
# ============================================================


def test_tool_failure_does_not_leak_exception_text(monkeypatch):
    """工具抛异常时，返回给模型的结果里不能出现异常原文。

    真实现象：
        embedding 模型挂掉时，异常是
        「embedding 挂了 /root/models/x.bin」。
        原来直接把 str(exc) 拼进 error 字段，
        模型照着念，用户收到「售后政策查询失败（工具报错）」。

    用户不该看到「工具报错」这种内部术语，
    更不该看到文件路径、SQL 片段、第三方服务地址。
    """

    from app.agent import tool_executor

    def _explode(*args, **kwargs):
        raise RuntimeError("embedding 挂了 /root/models/x.bin")

    monkeypatch.setitem(tool_executor.TOOL_MAP, "get_order", _explode)

    result = tool_executor.execute_tool(
        "get_order",
        {"order_id": "ORD001"},
        user_id="USER001",
    )

    assert "/root/" not in result
    assert "embedding" not in result
    assert "x.bin" not in result

    # 但要保留 error_type：模型需要判断「能不能重试」
    assert "RuntimeError" in result
    assert "tool_unavailable" in result


def test_ticket_confirm_failure_does_not_leak_exception_text(monkeypatch):
    """建单失败时不能把异常原文给模型。

    这里可能撞上数据库错误，异常里有表名和 SQL 片段。
    用户只需要知道「这次没提交成功」。
    """

    from app.agent import ticket_tools

    monkeypatch.setattr(
        ticket_tools,
        "complete_draft",
        lambda *a, **k: None,
    )

    def _explode(*args, **kwargs):
        raise RuntimeError(
            "sqlite3.OperationalError: no such table: after_sales_tickets"
        )

    monkeypatch.setattr(
        ticket_tools,
        "create_after_sales_ticket",
        _explode,
        raising=False,
    )

    # 走一遍 draft → confirm，让 confirm 真的进到建单分支
    draft = ticket_tools.draft_after_sales_ticket(
        order_id="ORD001",
        request_type="refund",
        reason="商品破损",
        user_id="USER001",
        conversation_id="test-err-leak",
    )

    from app.agent.conversation_store import add_message

    add_message(
        "test-err-leak",
        {"role": "user", "content": "确认"},
    )

    result = ticket_tools.confirm_after_sales_ticket(
        user_id="USER001",
        conversation_id="test-err-leak",
        draft_id=draft.get("draft_id"),
    )

    text = str(result)

    assert "sqlite3" not in text
    assert "no such table" not in text
    assert "after_sales_tickets" not in text

    # 必须明确说「没提交成功」，不能含糊成「可能延迟」
    assert "没有创建工单" in text


def test_rerank_failure_degrades_and_logs(caplog):
    """重排推理失败要降级为原顺序，并且留下日志。

    降级对用户是无感的，但运维必须知道
    「这批回答没有经过重排」——否则检索质量悄悄下滑，
    只能靠人工发现。
    """

    import logging

    from app.rag.reranker import Reranker

    class _BrokenModel:
        def predict(self, pairs):
            raise RuntimeError("显存不足")

    reranker = Reranker()
    reranker.model = _BrokenModel()
    reranker._load_failed = False

    candidates = [
        {"content": "条款 A", "chunk_id": "a"},
        {"content": "条款 B", "chunk_id": "b"},
        {"content": "条款 C", "chunk_id": "c"},
    ]

    with caplog.at_level(logging.WARNING):
        result = reranker.rerank("退款多久", candidates, top_k=3)

    # 降级：原顺序、条数不变，不能返回空
    assert len(result) == 3
    assert [item["chunk_id"] for item in result] == ["a", "b", "c"]

    assert any("重排推理失败" in record.message for record in caplog.records)


def test_summary_failure_keeps_previous_summary(caplog):
    """摘要生成失败要沿用上一次摘要，而不是清空。

    清空的后果是「聊久了就失忆」，
    用户会重复说订单号、助手会重复问已经答过的问题。
    """

    import logging

    import app.agent.memory as memory

    monkeypatch_target = memory._generate_summary

    try:
        memory._generate_summary = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("摘要模型挂了")
        )

        conversation_id = "test-err-summary"

        from app.agent.conversation_store import (
            add_message,
            clear_conversation,
            update_conversation_state,
        )

        clear_conversation(conversation_id)

        # 先存一段「上一版摘要」，模拟之前成功过一次
        update_conversation_state(
            conversation_id,
            {memory.STATE_KEY: {"text": "用户在问 ORD001 的退款", "covered": 4}},
        )

        # 造足够多的消息，触发摘要
        history = []

        for index in range(20):
            message = {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"第 {index} 条消息",
            }
            add_message(conversation_id, message)
            history.append(message)

        with caplog.at_level(logging.WARNING):
            summary, recent = memory.build_memory_context(
                conversation_id,
                history,
            )

        # 关键断言：旧摘要还在，没被清空
        assert summary == "用户在问 ORD001 的退款"
        assert recent

        clear_conversation(conversation_id)

    finally:
        memory._generate_summary = monkeypatch_target
