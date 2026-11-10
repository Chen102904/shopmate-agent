from pydantic import BaseModel, Field


class RefundCheckRequest(BaseModel):
    order_id: str = Field(min_length=1)


class RefundCheckResponse(BaseModel):
    order_id: str
    eligible: bool
    reason: str