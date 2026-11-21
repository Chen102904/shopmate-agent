from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import require_user_id
from app.services.order_service import get_order_by_id, list_orders_by_user
from app.services.order_view import build_order_view

router = APIRouter(prefix="/orders", tags=["订单"])


@router.get("")
def list_my_orders(
    user_id: str = Depends(require_user_id),
):
    """列出当前用户名下的订单，最近的在前。

    每张单都带上售后叠加后的展示状态和最新一张工单，
    前端一次请求就能画出「订单 + 售后进度」的卡片，
    不用再按订单逐个查工单。

    只返回本人的订单：
        user_id 只从请求头取，不接受查询参数指定，
        否则改个 URL 就能翻别人的订单列表。
    """

    orders = list_orders_by_user(user_id)

    return [build_order_view(order, user_id) for order in orders]


@router.get("/{order_id}")
def query_order(
    order_id: str,
    user_id: str = Depends(require_user_id),
):
    """查询订单详情。

    只能查询自己的订单。
    订单不存在和订单不属于当前用户，都返回相同的 404，
    避免通过状态码差异探测系统里存在哪些订单号。

    返回体在订单本身字段之外，多带了售后叠加出来的展示状态，
    所以「详情」和「列表」里的同一张单，状态口径是一致的。
    """

    order = get_order_by_id(order_id, user_id)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="订单不存在或无权访问",
        )

    return {
        **order.model_dump(mode="json"),
        **build_order_view(order, user_id),
    }
