from fastapi import APIRouter, HTTPException

from app.services.logistics_service import get_logistics_by_order_id

router = APIRouter(prefix="/logistics", tags=["物流"])


@router.get("/{order_id}")
def query_logistics(order_id: str):
    logistics = get_logistics_by_order_id(order_id)

    if logistics is None:
        raise HTTPException(
            status_code=404,
            detail="该订单暂无物流信息",
        )

    return logistics