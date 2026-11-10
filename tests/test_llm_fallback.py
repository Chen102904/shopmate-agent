
import pytest

from openai import APIConnectionError, APITimeoutError

from app.core import llm_client


def create_mock_client(create_function):
    """创建模拟的 OpenAI 客户端。"""

    return type(
        "MockClient",
        (),
        {
            "chat": type(
                "MockChat",
                (),
                {
                    "completions": type(
                        "MockCompletions",
                        (),
                        {
                            "create": staticmethod(create_function),
                        },
                    )(),
                },
            )(),
        },
    )()


def test_deepseek_success_without_fallback(monkeypatch):
    """测试 DeepSeek 正常时，不调用智谱和千问。"""

    expected_response = object()

    primary_calls = []
    fallback_calls = []

    def mock_primary_create(**kwargs):
        primary_calls.append(kwargs)
        return expected_response

    def mock_fallback_client():
        fallback_calls.append(True)
        raise AssertionError("DeepSeek 成功时不应该调用备用模型")

    monkeypatch.setattr(
        llm_client,
        "get_llm_client",
        lambda: create_mock_client(mock_primary_create),
    )

    monkeypatch.setattr(
        llm_client,
        "get_zhipu_client",
        mock_fallback_client,
    )

    monkeypatch.setattr(
        llm_client,
        "get_qwen_client",
        mock_fallback_client,
    )

    result = llm_client.create_chat_completion_with_fallback(
        model=llm_client.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": "你好"}],
    )

    assert result is expected_response
    assert len(primary_calls) == 1
    assert fallback_calls == []

    assert primary_calls[0]["model"] == llm_client.DEEPSEEK_MODEL


def test_deepseek_failure_then_zhipu_success(monkeypatch):
    """测试 DeepSeek 失败后，自动切换到智谱。"""

    primary_error = APIConnectionError(
        message="模拟 DeepSeek 连接失败",
        request=None,
    )

    expected_response = object()

    fallback_calls = []

    def mock_primary_create(**kwargs):
        raise primary_error

    def mock_fallback_create(**kwargs):
        fallback_calls.append(kwargs)
        return expected_response

    def mock_qwen_client():
        raise AssertionError("智谱成功时不应该调用千问")

    monkeypatch.setattr(
        llm_client,
        "get_llm_client",
        lambda: create_mock_client(mock_primary_create),
    )

    monkeypatch.setattr(
        llm_client,
        "get_zhipu_client",
        lambda: create_mock_client(mock_fallback_create),
    )

    monkeypatch.setattr(
        llm_client,
        "get_qwen_client",
        mock_qwen_client,
    )

    result = llm_client.create_chat_completion_with_fallback(
        model=llm_client.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": "查询订单"}],
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },
    )

    assert result is expected_response
    assert len(fallback_calls) == 1

    assert fallback_calls[0]["model"] == llm_client.ZHIPU_MODEL
    assert "extra_body" not in fallback_calls[0]


def test_deepseek_and_zhipu_fail_then_qwen_success(monkeypatch):
    """测试 DeepSeek 和智谱失败后，自动切换到千问。"""

    primary_error = APIConnectionError(
        message="模拟 DeepSeek 连接失败",
        request=None,
    )

    zhipu_error = APITimeoutError(request=None)

    expected_response = object()

    qwen_calls = []

    def mock_primary_create(**kwargs):
        raise primary_error

    def mock_zhipu_create(**kwargs):
        raise zhipu_error

    def mock_qwen_create(**kwargs):
        qwen_calls.append(kwargs)
        return expected_response

    monkeypatch.setattr(
        llm_client,
        "get_llm_client",
        lambda: create_mock_client(mock_primary_create),
    )

    monkeypatch.setattr(
        llm_client,
        "get_zhipu_client",
        lambda: create_mock_client(mock_zhipu_create),
    )

    monkeypatch.setattr(
        llm_client,
        "get_qwen_client",
        lambda: create_mock_client(mock_qwen_create),
    )

    result = llm_client.create_chat_completion_with_fallback(
        model=llm_client.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": "你好"}],
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },
    )

    assert result is expected_response
    assert len(qwen_calls) == 1

    assert qwen_calls[0]["model"] == llm_client.QWEN_MODEL
    assert "extra_body" not in qwen_calls[0]


def test_all_models_fail_raise_primary_error(monkeypatch):
    """测试三个模型全部失败时，抛出 DeepSeek 原始异常。"""

    primary_error = APIConnectionError(
        message="模拟 DeepSeek 连接失败",
        request=None,
    )

    zhipu_error = APITimeoutError(request=None)
    qwen_error = APIConnectionError(
        message="模拟千问连接失败",
        request=None,
    )

    def mock_primary_create(**kwargs):
        raise primary_error

    def mock_zhipu_create(**kwargs):
        raise zhipu_error

    def mock_qwen_create(**kwargs):
        raise qwen_error

    monkeypatch.setattr(
        llm_client,
        "get_llm_client",
        lambda: create_mock_client(mock_primary_create),
    )

    monkeypatch.setattr(
        llm_client,
        "get_zhipu_client",
        lambda: create_mock_client(mock_zhipu_create),
    )

    monkeypatch.setattr(
        llm_client,
        "get_qwen_client",
        lambda: create_mock_client(mock_qwen_create),
    )

    with pytest.raises(APIConnectionError) as exc_info:
        llm_client.create_chat_completion_with_fallback(
            model=llm_client.DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": "你好"}],
        )

    assert exc_info.value is primary_error
