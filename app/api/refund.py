from fastapi import APIRouter, Depends

from app.api.deps import require_user_id
from app.schemas.refund import (
    RefundCheckRequest,
    RefundCheckResponse,
)
from app.services.refund_service import check_refund_eligibility

router = APIRouter(prefix="/refunds", tags=["退款"])


@router.post("/check", response_model=RefundCheckResponse)
def check_refund(
    request: RefundCheckRequest,
    user_id: str = Depends(require_user_id),
):
    """检查某个订单是否符合退款条件。

    只能检查自己的订单。
    订单不存在或不属于当前用户时，返回 eligible=None，
    而不是 eligible=False，避免把「查不到」说成「不符合退款条件」。
    """

    result = check_refund_eligibility(
        request.order_id,
        user_id,
    )

    return result
