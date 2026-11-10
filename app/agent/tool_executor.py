import json
import logging
import time

from app.core.request_context import get_request_id
from app.core.execution_trace import record_execution_event

from app.services.order_service import get_order_by_id
from app.services.logistics_service import get_logistics_by_order_id
from app.services.refund_service import check_refund_eligibility


logger = logging.getLogger(__name__)


# RAG 服务实例：第一次调用时才初始化，后续重复使用
_rag_service = None


def search_after_sales_policy(query: str) -> dict:
    """
    检索售后政策知识库。
    使用延迟初始化，避免每次调用工具都重新加载 Embedding 模型。
    """
    global _rag_service

    if _rag_service is None:
        from app.rag.rag_chain import RAGService

        _rag_service = RAGService()

    answer = _rag_service.answer(query)

    return {
        "query": query,
        "answer": answer,
    }


# 工具名称与 Python 函数的映射关系
TOOL_MAP = {
    "get_order": get_order_by_id,
    "get_logistics": get_logistics_by_order_id,
    "check_refund_eligibility": check_refund_eligibility,
    "search_after_sales_policy": search_after_sales_policy,
}


def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    根据工具名称调用对应的 Python 函数。

    Args:
        tool_name: 大模型返回的工具名称
        arguments: 大模型返回的工具参数

    Returns:
        JSON 格式的工具执行结果
    """
    start_time = time.perf_counter()
    request_id = get_request_id()

    # 1. 检查工具是否存在
    tool_function = TOOL_MAP.get(tool_name)

    if tool_function is None:
        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        record_execution_event(
            event="tool_result",
            status="error",
            duration_ms=duration_ms,
            details={
                "tool": tool_name,
                "error_type": "unknown_tool",
            },
        )

        logger.warning(
            "工具执行失败 | tool=%s | status=unknown_tool | duration=%.2fms",
            tool_name,
            duration_ms,
        )

        return json.dumps(
            {"error": f"未知工具: {tool_name}"},
            ensure_ascii=False,
        )

    # 2. 记录工具开始执行事件
    record_execution_event(
        event="tool_call",
        status="started",
        details={
            "tool": tool_name,
        },
    )

    # 只记录参数名称，不记录参数值。
    argument_keys = sorted(arguments.keys())

    logger.info(
        "工具开始执行 | request_id=%s | tool=%s | argument_keys=%s",
        request_id,
        tool_name,
        argument_keys,
    )

    try:
        # 3. 执行工具
        result = tool_function(**arguments)

        # 4. 将 Pydantic 模型转换为字典
        if hasattr(result, "model_dump"):
            result = result.model_dump(mode="json")

        # 5. 转换为 JSON 字符串
        result_json = json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

        # 6. 计算执行耗时
        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # 7. 记录工具执行成功事件
        record_execution_event(
            event="tool_result",
            status="success",
            duration_ms=duration_ms,
            details={
                "tool": tool_name,
            },
        )

        logger.info(
            "工具执行成功 | request_id=%s | tool=%s | "
            "status=success | duration=%.2fms",
            request_id,
            tool_name,
            duration_ms,
        )

        return result_json

    except Exception as exc:
        # 8. 记录工具执行失败事件
        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        record_execution_event(
            event="tool_result",
            status="error",
            duration_ms=duration_ms,
            details={
                "tool": tool_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )

        logger.exception(
            "工具执行失败 | request_id=%s | tool=%s | "
            "status=error | duration=%.2fms",
            request_id,
            tool_name,
            duration_ms,
        )

        return json.dumps(
            {"error": f"工具执行失败：{str(exc)}"},
            ensure_ascii=False,
        )