from pydantic import BaseModel, Field

from app.models.after_sales import AfterSalesType, TicketStatus


class AfterSalesCreateRequest(BaseModel):
    order_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    request_type: AfterSalesType
    reason: str = Field(min_length=1)


class AfterSalesResponse(BaseModel):
    ticket_id: str
    order_id: str
    user_id: str
    request_type: AfterSalesType
    reason: str
    status: TicketStatus