from datetime import datetime

from app.models.order import OrderStatus
from app.services.order_service import get_order_by_id
from data.mock_data import LOGISTICS


def check_refund_eligibility(
    order_id: str,
    user_id: str,
) -> dict:
    """根据订单状态和实际签收时间判断退款资格。

    参数：
        order_id: 订单编号
        user_id: 当前用户编号，必须是该订单的归属人

    返回字段说明：
        eligible: True 表示符合退款条件，False 表示不符合，
                  None 表示无法判断（订单不存在或不属于当前用户）。
        reason: 给用户看的中文原因。
        status: 机器可读的状态，便于上层区分「不符合条件」和「查不到」。

    为什么 eligible 用 None 而不是 False：
        如果订单查不到时返回 False，
        模型会把「查不到这个订单」说成「这个订单不符合退款条件」，
        属于把系统问题说成业务结论，会误导用户。
    """

    # 1. 查询订单，并确认归属
    order = get_order_by_id(order_id, user_id)

    if order is None:
        return {
            "order_id": order_id,
            "eligible": None,
            "reason": "订单不存在或无权访问",
            "status": "not_accessible",
        }

    # 2. 已取消订单
    if order.status == OrderStatus.CANCELLED:
        return {
            "order_id": order_id,
            "eligible": False,
            "reason": "订单已经取消",
            "status": "cancelled",
        }

    # 3. 未付款或已付款但未发货
    if order.status in (
        OrderStatus.PENDING,
        OrderStatus.PAID,
    ):
        return {
            "order_id": order_id,
            "eligible": True,
            "reason": "订单尚未发货，可以申请退款",
            "status": "not_shipped",
        }

    # 4. 商品正在运输
    if order.status == OrderStatus.SHIPPED:
        return {
            "order_id": order_id,
            "eligible": False,
            "reason": "商品正在运输中，暂不支持退款",
            "status": "in_transit",
        }

    # 5. 商品已经送达
    if order.status == OrderStatus.DELIVERED:
        logistics = LOGISTICS.get(order_id)

        if logistics is None or logistics.delivered_at is None:
            return {
                "order_id": order_id,
                "eligible": None,
                "reason": "缺少实际签收时间，暂时无法判断",
                "status": "missing_delivery_time",
            }

        days_since_delivery = (
            datetime.now().date() - logistics.delivered_at.date()
        ).days

        if 0 <= days_since_delivery <= 7:
            return {
                "order_id": order_id,
                "eligible": True,
                "reason": "签收后7天内，可以申请退款",
                "status": "within_return_window",
                "days_since_delivery": days_since_delivery,
            }

        return {
            "order_id": order_id,
            "eligible": False,
            "reason": "已经超过签收后7天的退款期限",
            "status": "expired_return_window",
            "days_since_delivery": days_since_delivery,
        }

    # 6. 兜底处理
    return {
        "order_id": order_id,
        "eligible": None,
        "reason": "无法识别的订单状态",
        "status": "unknown_status",
    }
