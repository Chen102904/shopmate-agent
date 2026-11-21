"""会话级工具结果缓存测试。"""

from app.agent.conversation_store import clear_conversation
from app.agent.tool_cache import (
    build_cache_key,
    clear_tool_cache,
    get_cached_result,
    store_tool_result,
)


def _conv(name: str) -> str:
    conversation_id = f"test-tool-cache-{name}"
    clear_conversation(conversation_id)
    return conversation_id


def test_cache_key_normalizes_order_id():
    key = build_cache_key("get_order", {"order_id": " ord001 "})

    assert key == "get_order:ORD001"


def test_non_cacheable_tool_returns_none():
    key = build_cache_key(
        "search_after_sales_policy",
        {"query": "退款政策"},
    )

    assert key is None


def test_missing_order_id_returns_none():
    assert build_cache_key("get_order", {}) is None
    assert build_cache_key("get_order", {"order_id": ""}) is None


def test_store_and_read_back():
    conv = _conv("roundtrip")
    key = build_cache_key("get_order", {"order_id": "ORD001"})

    store_tool_result(conv, key, '{"order_id": "ORD001"}')

    assert get_cached_result(conv, key) == '{"order_id": "ORD001"}'

    clear_conversation(conv)


def test_error_result_is_not_cached():
    """错误结果不缓存，否则用户一个 TTL 周期内会一直被拒。"""

    conv = _conv("error")
    key = build_cache_key("get_order", {"order_id": "ORD999"})

    store_tool_result(
        conv,
        key,
        '{"error": "订单不存在或无权访问"}',
    )

    assert get_cached_result(conv, key) is None

    clear_conversation(conv)


def test_expired_entry_is_miss(monkeypatch):
    conv = _conv("expired")
    key = build_cache_key("get_order", {"order_id": "ORD001"})

    store_tool_result(conv, key, '{"order_id": "ORD001"}')

    # 把 TTL 调成负数，等价于已经过期
    monkeypatch.setattr(
        "app.agent.tool_cache.CACHE_TTL_SECONDS",
        -1,
    )

    assert get_cached_result(conv, key) is None

    clear_conversation(conv)


def test_clear_tool_cache():
    conv = _conv("clear")
    key = build_cache_key("get_order", {"order_id": "ORD001"})

    store_tool_result(conv, key, '{"order_id": "ORD001"}')
    clear_tool_cache(conv)

    assert get_cached_result(conv, key) is None

    clear_conversation(conv)
