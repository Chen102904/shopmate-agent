from app.models.logistics import Logistics
from app.services.order_service import get_order_by_id
from data.mock_data import LOGISTICS


def get_logistics_by_order_id(
    order_id: str,
    user_id: str,
) -> Logistics | None:
    """根据订单编号查询物流信息，并校验订单归属。

    物流数据本身不带用户字段，
    所以必须先确认订单属于当前用户，才能返回对应的物流信息。
    否则只要知道订单号，就能拿到别人的快递单号和收件进度。
    """

    if get_order_by_id(order_id, user_id) is None:
        return None

    return LOGISTICS.get(order_id)


def query_logistics(
    order_id: str,
    user_id: str,
) -> dict:
    """查询物流，返回**能区分原因**的结构化结果。

    为什么不能只返回 None：
        get_logistics_by_order_id 返回 None 时有两种完全不同的原因：
            1. 订单不存在 / 不属于当前用户 → 越权
            2. 订单存在但还没发货 → 没有物流数据

        工具层原来把两种都当越权处理，于是一个刚付款的用户问
        「我的快递到哪了」，会收到「订单不存在或无权访问，请核对订单号」。
        他的订单号是对的，只是货还没发。
        这种回答会让人以为自己的单丢了。

    返回：
        {"status": "ok", "logistics": {...}}
        {"status": "no_logistics", "order_status": "paid", "reason": "..."}
        {"status": "not_accessible"}
    """

    order = get_order_by_id(order_id, user_id)

    if order is None:
        # 不存在和不属于你返回同一个结果，不泄露订单是否存在
        return {"status": "not_accessible"}

    logistics = LOGISTICS.get(order_id)

    if logistics is None:
        return {
            "status": "no_logistics",
            "order_id": order_id,
            "order_status": order.status.value,
            "reason": (
                "订单还没有物流信息。"
                "通常是商家尚未发货，发货后才能查到快递单号。"
            ),
        }

    return {
        "status": "ok",
        "order_id": order_id,
        "logistics": logistics.model_dump(mode="json"),
    }
