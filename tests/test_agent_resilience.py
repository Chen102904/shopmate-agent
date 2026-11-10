
from types import SimpleNamespace

from openai import APIConnectionError, APITimeoutError

from app.agent.agent import MODEL_UNAVAILABLE_MESSAGE, run_agent
from app.agent.intent_classifier import IntentResult
from app.agent.conversation_store import (
    clear_conversation,
    get_conversation,
)


def setup_mock_agent(monkeypatch, error):
    """模拟意图识别成功，同时模拟主备模型均调用失败。"""

    monkeypatch.setattr(
        "app.agent.agent.classify_intent",
        lambda *args, **kwargs: IntentResult(
            intent="order_query",
            confidence=0.99,
            need_clarification=False,
        ),
    )

    def mock_completion(**kwargs):
        raise error

    monkeypatch.setattr(
        "app.agent.agent.create_chat_completion_with_fallback",
        mock_completion,
    )


def test_agent_handles_model_connection_failure(monkeypatch):
    """测试主备模型调用失败时，Agent 能够正常降级。"""

    conversation_id = "test_model_connection_failure"
    clear_conversation(conversation_id)

    error = APIConnectionError(
        message="模拟网络连接失败",
        request=None,
    )

    setup_mock_agent(monkeypatch, error)

    result = run_agent(
        "帮我查一下 ORD001",
        conversation_id,
    )

    assert result == MODEL_UNAVAILABLE_MESSAGE

    history = get_conversation(conversation_id)

    assert any(
        message["role"] == "user"
        and message["content"] == "帮我查一下 ORD001"
        for message in history
    )

    assert any(
        message["role"] == "assistant"
        and message["content"] == MODEL_UNAVAILABLE_MESSAGE
        for message in history
    )

    clear_conversation(conversation_id)


def test_agent_handles_model_timeout(monkeypatch):
    """测试模型请求超时时，Agent 能够正常降级。"""

    conversation_id = "test_model_timeout"
    clear_conversation(conversation_id)

    error = APITimeoutError(request=None)

    setup_mock_agent(monkeypatch, error)

    result = run_agent(
        "帮我查一下 ORD001",
        conversation_id,
    )

    assert result == MODEL_UNAVAILABLE_MESSAGE

    history = get_conversation(conversation_id)

    assert any(
        message["role"] == "assistant"
        and message["content"] == MODEL_UNAVAILABLE_MESSAGE
        for message in history
    )

    clear_conversation(conversation_id)


def test_agent_recovers_after_model_failure(monkeypatch):
    """测试模型第一次失败、第二次恢复后，Agent 能正常工作。"""

    conversation_id = "test_model_recovery"
    clear_conversation(conversation_id)

    # 第一次调用：模拟主备模型均失败
    error = APIConnectionError(
        message="模拟网络连接失败",
        request=None,
    )

    setup_mock_agent(monkeypatch, error)

    first_result = run_agent(
        "帮我查一下 ORD001",
        conversation_id,
    )

    assert first_result == MODEL_UNAVAILABLE_MESSAGE

    # 第二次调用：模拟模型恢复正常
    mock_message = SimpleNamespace(
        content="模型服务已经恢复，ShopMate 可以正常工作了。",
        tool_calls=None,
    )

    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=mock_message)
        ]
    )

    # 模拟新的 Fallback 调用入口恢复成功
    monkeypatch.setattr(
        "app.agent.agent.create_chat_completion_with_fallback",
        lambda **kwargs: mock_response,
    )

    second_result = run_agent(
        "请再帮我查一下 ORD001",
        conversation_id,
    )

    assert second_result == "模型服务已经恢复，ShopMate 可以正常工作了。"

    # 验证恢复后的回答已经保存到会话
    history = get_conversation(conversation_id)

    assert any(
        message["role"] == "assistant"
        and message["content"] == second_result
        for message in history
    )

    clear_conversation(conversation_id)
