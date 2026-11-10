from datetime import datetime

from app.models.order import OrderStatus
from data.mock_data import ORDERS, LOGISTICS


def check_refund_eligibility(order_id: str) -> dict:
    """根据订单状态和实际签收时间判断退款资格。"""

    # 1. 查询订单
    order = ORDERS.get(order_id)

    if order is None:
        return {
            "order_id": order_id,
            "eligible": False,
            "reason": "订单不存在",
        }

    # 2. 已取消订单
    if order.status == OrderStatus.CANCELLED:
        return {
            "order_id": order_id,
            "eligible": False,
            "reason": "订单已经取消",
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
        }

    # 4. 商品正在运输
    if order.status == OrderStatus.SHIPPED:
        return {
            "order_id": order_id,
            "eligible": False,
            "reason": "商品正在运输中，暂不支持退款",
        }

    # 5. 商品已经送达
    if order.status == OrderStatus.DELIVERED:
        logistics = LOGISTICS.get(order_id)

        if logistics is None or logistics.delivered_at is None:
            return {
                "order_id": order_id,
                "eligible": False,
                "reason": "缺少实际签收时间，暂时无法判断",
            }

        days_since_delivery = (
            datetime.now().date() - logistics.delivered_at.date()
        ).days

        if 0 <= days_since_delivery <= 7:
            return {
                "order_id": order_id,
                "eligible": True,
                "reason": "签收后7天内，可以申请退款",
            }

        return {
            "order_id": order_id,
            "eligible": False,
            "reason": "已经超过签收后7天的退款期限",
        }

    # 6. 兜底处理
    return {
        "order_id": order_id,
        "eligible": False,
        "reason": "无法识别的订单状态",
    }