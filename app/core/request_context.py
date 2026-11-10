
import logging
import time
import uuid

from contextvars import ContextVar
from functools import wraps

from app.core.execution_trace import (
    start_execution_trace,
    record_execution_event,
    save_completed_execution_trace,
    clear_execution_trace,
)


request_id_var = ContextVar(
    "request_id",
    default="-",
)


def get_request_id() -> str:
    """获取当前请求的 Trace ID。"""
    return request_id_var.get()


def with_request_context(func):
    """
    请求追踪装饰器。

    1. 自动生成 request_id。
    2. 设置当前请求上下文。
    3. 初始化执行记录。
    4. 记录请求开始和结束。
    5. 保存完整执行记录。
    6. 请求结束后恢复上下文。

    _include_request_metadata=True 时，
    返回 {"result": 原始结果, "request_id": 请求 ID}。
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        include_metadata = kwargs.pop(
            "_include_request_metadata",
            False,
        )

        request_id = uuid.uuid4().hex[:12]

        token = request_id_var.set(request_id)

        start_execution_trace()

        start_time = time.perf_counter()

        logger = logging.getLogger("app.request")

        logger.info(
            "请求开始 | request_id=%s",
            request_id,
        )

        try:
            result = func(*args, **kwargs)

            duration_ms = (
                time.perf_counter() - start_time
            ) * 1000

            record_execution_event(
                event="request",
                status="success",
                duration_ms=duration_ms,
                details={
                    "request_id": request_id,
                },
            )

            save_completed_execution_trace(
                request_id=request_id,
                status="success",
                duration_ms=duration_ms,
            )

            logger.info(
                "请求结束 | request_id=%s | "
                "status=success | duration=%.2fms",
                request_id,
                duration_ms,
            )

            if include_metadata:
                return {
                    "result": result,
                    "request_id": request_id,
                }

            return result

        except Exception as exc:
            duration_ms = (
                time.perf_counter() - start_time
            ) * 1000

            record_execution_event(
                event="request",
                status="error",
                duration_ms=duration_ms,
                details={
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                },
            )

            save_completed_execution_trace(
                request_id=request_id,
                status="error",
                duration_ms=duration_ms,
            )

            logger.error(
                "请求异常 | request_id=%s | "
                "status=error | duration=%.2fms | error_type=%s",
                request_id,
                duration_ms,
                type(exc).__name__,
            )

            raise

        finally:
            clear_execution_trace()
            request_id_var.reset(token)

    return wrapper
