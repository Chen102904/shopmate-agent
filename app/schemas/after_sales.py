from datetime import datetime

from pydantic import BaseModel, Field

from app.models.after_sales import AfterSalesType, TicketStatus


class AfterSalesCreateRequest(BaseModel):
    order_id: str = Field(min_length=1)

    user_id: str | None = Field(
        default=None,
        deprecated=True,
        description=(
            "已废弃：身份改由请求头 X-User-Id 决定，"
            "本字段会被忽略。保留只是为了兼容旧客户端，不要依赖它。"
        ),
    )

    request_type: AfterSalesType
    reason: str = Field(min_length=1)


class AfterSalesResponse(BaseModel):
    ticket_id: str
    order_id: str
    user_id: str
    request_type: AfterSalesType
    reason: str
    status: TicketStatus
    created_at: datetime
    # 最后一次状态变更时间。刚创建时为 null。
    updated_at: datetime | None = None
    # 审核意见。用户被驳回时最想知道「为什么」。
    review_note: str | None = None


class ReviewRequest(BaseModel):
    """人工审核请求。只有管理员接口用。"""

    decision: str = Field(
        description="审核结论，只能是 approved 或 rejected",
    )

    review_note: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "审核意见。驳回时建议填写，会展示给用户。"
            "不填时驳回会自动补一句默认说明。"
        ),
    )