from pydantic import BaseModel, Field


class RefundCheckRequest(BaseModel):
    order_id: str = Field(min_length=1)


class RefundCheckResponse(BaseModel):
    order_id: str

    eligible: bool | None = Field(
        description=(
            "True 表示符合退款条件；"
            "False 表示不符合；"
            "None 表示无法判断（订单不存在或无权访问）"
        )
    )

    reason: str = Field(
        description="给用户看的中文原因",
    )

    status: str | None = Field(
        default=None,
        description="机器可读的状态标识，便于前端区分具体分支",
    )

    days_since_delivery: int | None = Field(
        default=None,
        description="签收后经过的天数，仅签收类型订单返回",
    )
