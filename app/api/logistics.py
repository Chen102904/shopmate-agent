from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import require_user_id
from app.services.logistics_service import query_logistics

router = APIRouter(prefix="/logistics", tags=["物流"])


@router.get("/{order_id}")
def query_order_logistics(
    order_id: str,
    user_id: str = Depends(require_user_id),
):
    """查询订单对应的物流信息。

    两种「查不到」要分开说：

        订单不存在 / 不属于你
            → 404，文案统一，不透露订单是否存在。

        订单存在但还没发货
            → 也是 404（物流资源确实不存在），
              但文案明确说「还没发货」。

    为什么值得分开：
        原来两种情况共用一句
        「订单不存在、无权访问，或该订单暂无物流信息」。
        用户看到这句话，第一反应是「我订单号写错了吗」，
        然后去核对、去重输——而他的订单号是对的，只是货没发。
        把三种可能糊成一句，等于把排查成本转嫁给用户。

    不透露订单是否存在这条仍然守住：
        「不属于你」和「不存在」的文案完全一致。
    """

    result = query_logistics(order_id, user_id)

    status = result.get("status")

    if status == "not_accessible":
        raise HTTPException(
            status_code=404,
            detail="订单不存在或无权访问",
        )

    if status == "no_logistics":
        raise HTTPException(
            status_code=404,
            detail=result.get("reason", "该订单暂无物流信息"),
        )

    return result["logistics"]
