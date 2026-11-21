import json
import logging
import time

from app.core.identity import (
    ORDER_NOT_ACCESSIBLE_MESSAGE,
    get_current_user_id,
)
from app.core.request_context import get_request_id
from app.core.execution_trace import record_execution_event

from app.services.order_service import get_order_by_id
from app.services.logistics_service import query_logistics
from app.services.refund_service import check_refund_eligibility

from app.agent.ticket_tools import (
    cancel_after_sales_ticket,
    confirm_after_sales_ticket,
    draft_after_sales_ticket,
    get_after_sales_tickets,
)

from app.agent.tool_validator import validate_tool_arguments


logger = logging.getLogger(__name__)


# 必须按用户隔离的工具集合。
# 这些工具查询的是「某个用户的私有数据」，
# 执行时必须把当前登录用户传进去做归属校验。
#
# 政策检索工具不在此列：售后政策是公共知识，
# 任何用户都可以查询，不需要归属校验。
USER_SCOPED_TOOLS = {
    "get_order",
    "get_logistics",
    "check_refund_eligibility",
    "draft_after_sales_ticket",
    "confirm_after_sales_ticket",
    "get_after_sales_tickets",
    "cancel_after_sales_ticket",
}


# 需要会话上下文的工具。
# 写操作必须是「本会话内」的：
#   - 草稿绑定了 conversation_id，换个会话就用不了；
#   - 确认闸门要读本会话最近一条用户消息。
# 所以这两个工具除了 user_id 还要拿到 conversation_id。
CONVERSATION_SCOPED_TOOLS = {
    "draft_after_sales_ticket",
    "confirm_after_sales_ticket",
    # 查工单也要会话：要顺带告诉模型「你有没有没提交的草稿」，
    # 草稿是按会话存的。
    "get_after_sales_tickets",
    "cancel_after_sales_ticket",
}


# 有副作用的工具。
# 单独列出来是为了让日志、追踪、缓存三处都能一眼识别：
# 这类工具不能重试、不能缓存、必须留下审计记录。
WRITE_TOOLS = {
    "confirm_after_sales_ticket",
    "cancel_after_sales_ticket",
}


# 不需要用户身份的工具
PUBLIC_TOOLS = {
    "search_after_sales_policy",
}


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
    # 用结构化版本：能区分「越权」和「还没发货」
    "get_logistics": query_logistics,
    "check_refund_eligibility": check_refund_eligibility,
    "search_after_sales_policy": search_after_sales_policy,
    "draft_after_sales_ticket": draft_after_sales_ticket,
    "confirm_after_sales_ticket": confirm_after_sales_ticket,
    "get_after_sales_tickets": get_after_sales_tickets,
    "cancel_after_sales_ticket": cancel_after_sales_ticket,
}


def _build_not_accessible_result(
    tool_name: str,
    arguments: dict,
) -> dict:
    """构造「订单不存在或无权访问」的统一返回体。"""

    result = {
        "error": ORDER_NOT_ACCESSIBLE_MESSAGE,
        "reason": ORDER_NOT_ACCESSIBLE_MESSAGE,
        "status": "not_accessible",
    }

    order_id = arguments.get("order_id")

    if order_id:
        result["order_id"] = order_id

    return result


def execute_tool(
    tool_name: str,
    arguments: dict,
    user_id: str | None = None,
    conversation_id: str | None = None,
) -> str:
    """
    根据工具名称调用对应的 Python 函数。

    参数：
        tool_name: 大模型返回的工具名称
        arguments: 大模型返回的工具参数
        user_id: 当前用户编号。
                 不传时从请求上下文读取（由 with_user_identity 写入）。
        conversation_id: 当前会话编号。
                 只有写操作类工具需要（草稿绑定会话、确认闸门读会话消息）。
                 查询类工具不传也能跑，保持向后兼容。

    返回：
        JSON 格式的工具执行结果。

    数据归属校验放在这一层而不是文件层：
        工具是模型唯一能触达数据的地方，
        把它卡在工具入口，任何新增的越权路径都会被拦住。
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

    # 2. 取出当前用户
    current_user_id = user_id or get_current_user_id()

    # 3. 记录工具开始执行事件
    record_execution_event(
        event="tool_call",
        status="started",
        details={
            "tool": tool_name,
        },
    )

    # 4. 参数校验
    #    必须放在调用业务函数之前：
    #    直接调用会在缺参数/多参数时抛 TypeError，
    #    那条错误信息是给开发者看的，模型读不懂也无法据此修正。
    validation = validate_tool_arguments(tool_name, arguments)

    if not validation.ok:
        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        record_execution_event(
            event="tool_result",
            status="invalid_arguments",
            duration_ms=duration_ms,
            details={
                "tool": tool_name,
                "error_type": "invalid_arguments",
                "error_count": len(validation.errors),
            },
        )

        logger.warning(
            "工具参数校验未通过 | request_id=%s | tool=%s | "
            "status=invalid_arguments | errors=%s",
            request_id,
            tool_name,
            validation.errors,
        )

        return json.dumps(
            {
                "error": "工具参数不合法",
                "details": validation.errors,
                "hint": "请修正参数后重新调用该工具。",
            },
            ensure_ascii=False,
        )

    # 多余参数已在校验里剔除，这里记一条提示不影响执行
    if validation.errors:
        logger.info(
            "工具参数已清理 | request_id=%s | tool=%s | notes=%s",
            request_id,
            tool_name,
            validation.errors,
        )

    arguments = validation.cleaned

    # 只记录参数名称，不记录参数值。
    argument_keys = sorted(arguments.keys())

    logger.info(
        "工具开始执行 | request_id=%s | tool=%s | argument_keys=%s",
        request_id,
        tool_name,
        argument_keys,
    )

    try:
        # 4. 按工具类型决定注入哪些上下文
        call_kwargs = dict(arguments)

        if tool_name in USER_SCOPED_TOOLS:
            call_kwargs["user_id"] = current_user_id

        if tool_name in CONVERSATION_SCOPED_TOOLS:

            # 写操作没有会话就拒绝执行。
            # 不能退化成「随便挑一个会话」：
            # 那等于绕过了草稿的会话绑定。
            if not conversation_id:
                duration_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                record_execution_event(
                    event="tool_result",
                    status="error",
                    duration_ms=duration_ms,
                    details={
                        "tool": tool_name,
                        "error_type": "missing_conversation_context",
                    },
                )

                return json.dumps(
                    {
                        "error": "缺少会话上下文，无法执行该操作",
                        "hint": "请重试；若持续失败请联系人工客服。",
                    },
                    ensure_ascii=False,
                )

            call_kwargs["conversation_id"] = conversation_id

        result = tool_function(**call_kwargs)

        # 5. 订单不存在或不属于当前用户时，返回统一错误体。
        #    直接返回 None 会让模型收到 "null"，它只会含糊地说"查询失败"。
        if result is None:
            duration_ms = (
                time.perf_counter() - start_time
            ) * 1000

            record_execution_event(
                event="tool_result",
                status="denied",
                duration_ms=duration_ms,
                details={
                    "tool": tool_name,
                    "error_type": "not_accessible",
                },
            )

            logger.warning(
                "工具数据归属校验未通过 | request_id=%s | tool=%s | "
                "status=denied | duration=%.2fms",
                request_id,
                tool_name,
                duration_ms,
            )

            return json.dumps(
                _build_not_accessible_result(tool_name, arguments),
                ensure_ascii=False,
            )

        # 6. 将 Pydantic 模型转换为字典
        if hasattr(result, "model_dump"):
            result = result.model_dump(mode="json")

        # 7. 工具有几种方式表达「订单不存在或无权访问」：
        #      - 直接返回 None
        #      - 返回 {"status": "not_accessible"}（退款、物流用的是这种）
        #    统一按同一套错误体回给模型，避免它收到
        #    eligible=null 这种字段后自己猜成「无法判断」。
        if (
            isinstance(result, dict)
            and result.get("status") == "not_accessible"
        ):
            duration_ms = (
                time.perf_counter() - start_time
            ) * 1000

            record_execution_event(
                event="tool_result",
                status="denied",
                duration_ms=duration_ms,
                details={
                    "tool": tool_name,
                    "error_type": "not_accessible",
                },
            )

            logger.warning(
                "工具数据归属校验未通过 | request_id=%s | tool=%s | "
                "status=denied | duration=%.2fms",
                request_id,
                tool_name,
                duration_ms,
            )

            return json.dumps(
                _build_not_accessible_result(tool_name, arguments),
                ensure_ascii=False,
            )

        # 8. 转换为 JSON 字符串
        result_json = json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

        # 9. 计算执行耗时
        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # 10. 记录工具执行成功事件
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
        # 11. 记录工具执行失败事件
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

        # 异常原文只留在日志和 trace 里，不进模型的上下文。
        #
        # 原来是 f"工具执行失败：{str(exc)}"，实测发现模型会照着念——
        # embedding 模型挂掉时，用户收到「售后政策知识库暂时查询失败（工具报错）」。
        # 「工具报错」是内部术语，用户看不懂；更糟的是异常里可能带
        # 文件路径、SQL 片段、第三方服务地址，等于把内部结构说给用户听。
        #
        # 返回体保留 error_type，是为了让模型能区分「该重试」和「别重试」；
        # 但提示词明确要求它不要把这个字段念给用户。
        return json.dumps(
            {
                "error": "服务暂时不可用，这次查询没成功",
                "status": "tool_unavailable",
                "error_type": type(exc).__name__,
                "hint": (
                    "这次工具调用失败了。如实告诉用户暂时查不到，"
                    "建议稍后重试或联系人工客服；"
                    "不要把 error_type 或任何内部错误细节说给用户。"
                    "换个工具能回答的部分，可以先答。"
                ),
            },
            ensure_ascii=False,
        )
