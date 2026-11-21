
from collections import deque
from copy import deepcopy
from contextvars import ContextVar
from datetime import datetime, timezone
from threading import Lock
from typing import Any


# 当前请求的执行事件列表
execution_events_var: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "execution_events",
    default=None,
)


# 最近完成的请求记录，最多保存100条
MAX_COMPLETED_TRACES = 100

_completed_traces: dict[str, dict[str, Any]] = {}
_trace_order: deque[str] = deque()

_trace_lock = Lock()


def start_execution_trace() -> None:
    """开始记录一次 Agent 请求的执行过程。"""
    execution_events_var.set([])


def record_execution_event(
    event: str,
    status: str,
    duration_ms: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """记录一个执行事件。"""
    events = execution_events_var.get()

    if events is None:
        return

    record = {
        "event": event,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if duration_ms is not None:
        record["duration_ms"] = round(duration_ms, 2)

    if details:
        record["details"] = deepcopy(details)

    events.append(record)


def get_execution_trace() -> list[dict[str, Any]]:
    """获取当前请求的完整执行记录。"""
    events = execution_events_var.get()

    if events is None:
        return []

    return deepcopy(events)


def save_completed_execution_trace(
    request_id: str,
    status: str,
    duration_ms: float,
    user_id: str | None = None,
) -> None:
    """保存已完成的请求记录，最多保留最近100条。

    记录 user_id 的目的：
        执行轨迹里包含工具调用参数（例如订单号）和最终回答，
        属于用户私有数据，查询时必须按归属校验。
    """

    from app.core.identity import get_current_user_id

    trace = {
        "request_id": request_id,
        "user_id": user_id or get_current_user_id(),
        "status": status,
        "duration_ms": round(duration_ms, 2),
        "events": get_execution_trace(),
    }

    with _trace_lock:
        if request_id in _completed_traces:
            _trace_order.remove(request_id)

        _completed_traces[request_id] = deepcopy(trace)
        _trace_order.append(request_id)

        while len(_trace_order) > MAX_COMPLETED_TRACES:
            oldest_request_id = _trace_order.popleft()
            _completed_traces.pop(oldest_request_id, None)


def get_completed_execution_trace(
    request_id: str,
) -> dict[str, Any] | None:
    """根据 request_id 查询已完成的请求记录。"""
    with _trace_lock:
        trace = _completed_traces.get(request_id)

        if trace is None:
            return None

        return deepcopy(trace)


def get_latest_completed_execution_trace() -> dict[str, Any] | None:
    """获取最近一次完成的请求记录。"""
    with _trace_lock:
        if not _trace_order:
            return None

        latest_request_id = _trace_order[-1]
        trace = _completed_traces.get(latest_request_id)

        if trace is None:
            return None

        return deepcopy(trace)


def clear_execution_trace() -> None:
    """清理当前请求的执行记录。"""
    execution_events_var.set(None)
