from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.schemas.after_sales import (
    AfterSalesCreateRequest,
    AfterSalesResponse,
)
from app.services.after_sales_service import (
    create_after_sales_ticket,
    get_after_sales_ticket,
)

router = APIRouter(prefix="/after-sales", tags=["售后工单"])


@router.post("", response_model=AfterSalesResponse)
def create_ticket(request: AfterSalesCreateRequest):
    """创建售后工单。"""
    try:
        ticket = create_after_sales_ticket(
            order_id=request.order_id,
            user_id=request.user_id,
            request_type=request.request_type,
            reason=request.reason,
        )
        return ticket

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@router.get("/{ticket_id}", response_model=AfterSalesResponse)
def query_ticket(ticket_id: UUID):
    """根据工单编号查询售后工单。"""
    ticket = get_after_sales_ticket(str(ticket_id))

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="售后工单不存在",
        )

    return ticket