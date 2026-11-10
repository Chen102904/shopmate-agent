from app.models.logistics import Logistics
from data.mock_data import LOGISTICS


def get_logistics_by_order_id(order_id: str) -> Logistics | None:
    """根据订单编号查询物流信息。"""
    return LOGISTICS.get(order_id)