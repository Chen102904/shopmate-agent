
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.agent import run_agent
from app.agent.conversation_store import (
    conversation_exists,
    get_conversation_snapshot,
    clear_conversation,
)


router = APIRouter(tags=["聊天会话"])


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
def chat(request: ChatRequest):
    """
    发送消息并获取 Agent 回答。

    不传 conversation_id 时自动创建新会话。
    传入已有 conversation_id 时继续原来的对话。
    """

    conversation_id = (
        request.conversation_id or uuid4().hex
    )

    execution = run_agent(
        user_message=request.message,
        conversation_id=conversation_id,
        _include_request_metadata=True,
    )

    return ChatResponse(
        conversation_id=conversation_id,
        request_id=execution["request_id"],
        answer=execution["result"],
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
)
def query_conversation(conversation_id: str):
    """查询指定会话的历史消息。"""

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
def delete_conversation(conversation_id: str):
    """清空指定会话。"""

    if not conversation_exists(conversation_id):
        raise HTTPException(
            status_code=404,
            detail="会话不存在",
        )

    clear_conversation(conversation_id)

    return {
        "success": True,
        "conversation_id": conversation_id,
        "message": "会话已清空",
    }
