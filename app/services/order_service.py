from app.models.order import Order
from data.mock_data import ORDERS


def get_order_by_id(order_id: str) -> Order | None:
    """根据订单编号查询订单。"""
    return ORDERS.get(order_id)