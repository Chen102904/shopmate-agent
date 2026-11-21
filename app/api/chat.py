
import asyncio
import logging
import os

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agent.agent import run_agent
from app.api.deps import require_user_id
from app.agent.conversation_store import (
    conversation_exists,
    get_conversation_owner,
    get_conversation_snapshot,
    clear_conversation,
)


router = APIRouter(tags=["聊天会话"])

logger = logging.getLogger(__name__)


# 单次对话处理的超时上限（秒）。
#
# 为什么要设：
#     Agent 内部最多 5 轮工具调用，每轮可能触发一次模型请求，
#     再加上 RAG 检索和查询改写，极端情况下会跑很久。
#     不设上限的话，一次异常请求会长期占住线程和会话锁，
#     后续同一会话的请求全部排队。
#
# 为什么是 120 秒：
#     端到端评测里最慢的一条（政策问答 + 改写 + 检索）约 26 秒，
#     120 秒留了足够余量，不会误杀正常请求。
CHAT_TIMEOUT_SECONDS = float(
    os.getenv("SHOPMATE_CHAT_TIMEOUT", "120")
)


def _ensure_conversation_accessible(
    conversation_id: str,
    user_id: str,
) -> None:
    """确认会话属于当前用户。

    不区分「会话不存在」和「不属于你」，
    统一返回 404，避免通过状态码差异探测会话 ID 是否有效。
    """

    if not conversation_exists(conversation_id):
        raise HTTPException(
            status_code=404,
            detail="会话不存在",
        )

    owner = get_conversation_owner(conversation_id)

    if owner != user_id:
        raise HTTPException(
            status_code=404,
            detail="会话不存在",
        )


class ChatRequest(BaseModel):
    """聊天请求参数。"""

    message: str = Field(
        min_length=1,
        max_length=5000,
        description="用户发送的消息",
    )

    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="会话 ID，不传时自动创建",
    )


class ChatResponse(BaseModel):
    """聊天响应。"""

    conversation_id: str
    request_id: str
    user_id: str
    answer: str


class ConversationResponse(BaseModel):
    """会话历史响应。"""

    conversation_id: str
    message_count: int
    messages: list[dict]


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    user_id: str = Depends(require_user_id),
):
    """
    发送消息并获取 Agent 回答。

    不传 conversation_id 时自动创建新会话。
    传入已有 conversation_id 时继续原来的对话。

    身份：
        由请求头 X-User-Id 指定，决定能查询哪些订单。
        使用别人的 conversation_id 也会被拒绝，避免读到别人的上下文。

    超时：
        超过 CHAT_TIMEOUT_SECONDS 返回 504，
        避免异常请求长期占住线程和会话锁。
    """

    conversation_id = (
        request.conversation_id or uuid4().hex
    )

    # 传入已有会话时，先确认归属再继续对话。
    # 若会话不存在，则允许创建（等价于新会话）。
    if request.conversation_id:
        owner = get_conversation_owner(conversation_id)

        if owner is not None and owner != user_id:
            raise HTTPException(
                status_code=404,
                detail="会话不存在",
            )

    try:
        # run_agent 内部都是同步调用（SQLite、OpenAI SDK），
        # 用 to_thread 放到线程池跑，避免阻塞事件循环；
        # 再套 wait_for 给整体设一个上限。
        execution = await asyncio.wait_for(
            asyncio.to_thread(
                run_agent,
                user_message=request.message,
                conversation_id=conversation_id,
                user_id=user_id,
                _include_request_metadata=True,
            ),
            timeout=CHAT_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:
        # 注意：超时只是「不再等」，线程本身停不下来。
        # Python 没有强杀线程的手段，run_agent 会继续跑到自然结束，
        # 期间它仍持有该会话的锁。
        #
        # 所以这里必须记一条日志：否则运维看到的是「偶发 504」，
        # 而真相是「有一批请求在后台跑完了但结果被丢弃」。
        logger.warning(
            "对话处理超时 | conversation=%s | timeout=%ss | "
            "后台线程仍在运行，会继续占用会话锁",
            conversation_id,
            CHAT_TIMEOUT_SECONDS,
        )

        raise HTTPException(
            status_code=504,
            detail="处理超时，请稍后重试",
        )

    except HTTPException:
        # run_agent 目前不抛 HTTPException，但保留这一层：
        # 将来如果内部开始用 HTTPException 表达业务错误，
        # 不会被下面的兜底改写状态码。
        raise

    except Exception as exc:
        # 走到这里说明连「降级回复」都没能返回。
        # 按服务不可用处理（503），而不是让全局处理器回 500：
        # 503 明确告诉客户端「过一会儿再来」，
        # 500 会让对方以为是自己请求写错了。
        logger.exception(
            "对话处理出现未预期异常 | conversation=%s | error_type=%s",
            conversation_id,
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail="服务暂时不可用，请稍后重试",
        )

    return ChatResponse(
        conversation_id=conversation_id,
        request_id=execution["request_id"],
        user_id=user_id,
        answer=execution["result"],
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
)
def query_conversation(
    conversation_id: str,
    user_id: str = Depends(require_user_id),
):
    """查询指定会话的历史消息。只能查询自己的会话。"""

    _ensure_conversation_accessible(
        conversation_id,
        user_id,
    )

    messages = get_conversation_snapshot(
        conversation_id
    )

    if messages is None:
        raise HTTPException(
            status_code=404,
            detail="会话不存在",
        )

    return ConversationResponse(
        conversation_id=conversation_id,
        message_count=len(messages),
        messages=messages,
    )


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    user_id: str = Depends(require_user_id),
):
    """清空指定会话。只能清空自己的会话。"""

    _ensure_conversation_accessible(
        conversation_id,
        user_id,
    )

    clear_conversation(conversation_id)

    return {
        "success": True,
        "conversation_id": conversation_id,
        "message": "会话已清空",
    }
