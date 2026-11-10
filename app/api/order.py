from fastapi import APIRouter, HTTPException

from app.services.order_service import get_order_by_id

router = APIRouter(prefix="/orders", tags=["订单"])


@router.get("/{order_id}")
def query_order(order_id: str):
    order = get_order_by_id(order_id)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="订单不存在",
        )

    return order