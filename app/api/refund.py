from fastapi import APIRouter

from app.schemas.refund import (
    RefundCheckRequest,
    RefundCheckResponse,
)
from app.services.refund_service import check_refund_eligibility

router = APIRouter(prefix="/refunds", tags=["退款"])


@router.post("/check", response_model=RefundCheckResponse)
def check_refund(request: RefundCheckRequest):
    result = check_refund_eligibility(request.order_id)
    return result