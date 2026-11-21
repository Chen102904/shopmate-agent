from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import require_user_id
from app.services.order_service import get_order_by_id, list_orders_by_user

router = APIRouter(prefix="/orders", tags=["订单"])


@router.get("")
def list_my_orders(
    user_id: str = Depends(require_user_id),
):
    """列出当前用户名下的订单，最近的在前。

    只返回本人的订单：
        user_id 只从请求头取，不接受查询参数指定，
        否则改个 URL 就能翻别人的订单列表。
    """

    return list_orders_by_user(user_id)


@router.get("/{order_id}")
def query_order(
    order_id: str,
    user_id: str = Depends(require_user_id),
):
    """查询订单详情。

    只能查询自己的订单。
    订单不存在和订单不属于当前用户，都返回相同的 404，
    避免通过状态码差异探测系统里存在哪些订单号。
    """

    order = get_order_by_id(order_id, user_id)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="订单不存在",
        )

    return order
