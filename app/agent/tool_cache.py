"""会话级工具结果缓存。

解决的问题：
    多轮对话里模型经常重复查询同一个订单。
    例如第一轮「查下 ORD001」，第二轮「能不能退款」，
    模型会再调一次 get_order —— 它既不知道退款工具内部已经查过订单，
    也不记得上一轮已经拿到了结果。

    代价：
        - 多一次数据库/接口调用
        - 工具结果又进一遍上下文，挤占窗口
        - 极端情况下把工具轮次耗在上限之前

    做法：
        不靠「提示模型别查」（只能降低概率），
        而是在它真要查时由缓存代为返回（能消除成本）。

TTL 设计：
    订单状态、物流状态是实时变化的，不能永久缓存。
    默认 300 秒：
        - 短到不会拿过期状态回答用户；
        - 长到足以覆盖一次连续对话。
    超过 TTL 的条目视为未命中，会重新真实查询。

注意：
    这是「结果缓存」，不是「伪造轨迹」。
    命中缓存时确实没有发起那次查询，
    执行轨迹里会记录 tool_cache_hit，可追溯。
"""

import time

from typing import Any

from app.agent.conversation_store import (
    get_conversation_state,
    update_conversation_state,
)


# 缓存有效期（秒）
CACHE_TTL_SECONDS = 300

# 只缓存「查询类」工具。
# 有副作用的操作（创建工单、退款）绝不能缓存，
# 否则用户第二次申请会拿到第一次的工单号。
CACHEABLE_TOOLS = {
    "get_order",
    "get_logistics",
    "check_refund_eligibility",
}

# 单个会话最多保留多少条缓存，防止状态无限膨胀
MAX_CACHE_ENTRIES = 30


def build_cache_key(
    tool_name: str,
    arguments: dict,
) -> str | None:
    """构造缓存键。

    返回 None 表示这次调用不适合缓存。
    """

    if tool_name not in CACHEABLE_TOOLS:
        return None

    if not isinstance(arguments, dict):
        return None

    order_id = arguments.get("order_id")

    if not order_id or not isinstance(order_id, str):
        return None

    return f"{tool_name}:{order_id.strip().upper()}"


def _load_cache(conversation_id: str) -> dict[str, Any]:
    """读取缓存的原始字典。"""

    state = get_conversation_state(conversation_id)

    cache = state.get("tool_cache")

    if not isinstance(cache, dict):
        return {}

    return cache


def _is_fresh(entry: Any, now: float) -> bool:
    """判断缓存条目是否仍在有效期内。"""

    if not isinstance(entry, dict):
        return False

    created_at = entry.get("ts")

    if not isinstance(created_at, (int, float)):
        return False

    return (now - created_at) < CACHE_TTL_SECONDS


def get_cached_result(
    conversation_id: str,
    cache_key: str | None,
) -> str | None:
    """取缓存结果。未命中或已过期返回 None。"""

    if not cache_key:
        return None

    cache = _load_cache(conversation_id)

    entry = cache.get(cache_key)

    if not entry:
        return None

    if not _is_fresh(entry, time.time()):
        return None

    result = entry.get("result")

    return result if isinstance(result, str) else None


def store_tool_result(
    conversation_id: str,
    cache_key: str | None,
    result: str,
) -> None:
    """写入缓存结果。"""

    if not cache_key or not isinstance(result, str):
        return

    # 只缓存成功结果。
    # 错误结果（参数错、越权、工具故障）不缓存：
    # 下一次同样的请求可能因为状态变化而成功，
    # 缓存错误会让用户在一个 TTL 周期内一直被拒。
    if '"error"' in result:
        return

    cache = _load_cache(conversation_id)

    now = time.time()

    # 清掉过期条目
    cache = {
        key: value
        for key, value in cache.items()
        if _is_fresh(value, now)
    }

    cache[cache_key] = {
        "result": result,
        "ts": now,
    }

    # 超出上限时丢弃最早写入的
    if len(cache) > MAX_CACHE_ENTRIES:
        ordered = sorted(
            cache.items(),
            key=lambda item: item[1].get("ts", 0),
        )
        cache = dict(ordered[-MAX_CACHE_ENTRIES:])

    update_conversation_state(
        conversation_id,
        {"tool_cache": cache},
    )


def clear_tool_cache(conversation_id: str) -> None:
    """清空某个会话的工具缓存。"""

    update_conversation_state(
        conversation_id,
        {},
        remove_keys=["tool_cache"],
    )
