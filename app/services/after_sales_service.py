from uuid import uuid4

from app.models.after_sales import AfterSalesTicket, AfterSalesType
from app.services.refund_service import check_refund_eligibility
from data.mock_data import ORDERS


# 暂时使用内存字典保存工单，后续再接入数据库
TICKETS: dict[str, AfterSalesTicket] = {}


def create_after_sales_ticket(
    order_id: str,
    user_id: str,
    request_type: AfterSalesType,
    reason: str,
) -> AfterSalesTicket:
    """创建售后工单。"""

    # 1. 检查订单是否存在
    order = ORDERS.get(order_id)

    if order is None:
        raise ValueError("订单不存在")

    # 2. 检查当前用户是否拥有该订单
    if order.user_id != user_id:
        raise ValueError("无权为该订单申请售后")

    # 3. 如果是退款申请，先检查退款资格
    if request_type == AfterSalesType.REFUND:
        eligibility = check_refund_eligibility(order_id, user_id)

        if not eligibility["eligible"]:
            raise ValueError(eligibility["reason"])

    # 4. 创建工单
    ticket = AfterSalesTicket(
        ticket_id=str(uuid4()),
        order_id=order_id,
        user_id=user_id,
        request_type=request_type,
        reason=reason,
    )

    # 5. 保存工单
    TICKETS[ticket.ticket_id] = ticket

    return ticket

def get_after_sales_ticket(ticket_id: str) -> AfterSalesTicket | None:
    """根据工单编号查询售后工单。"""
    return TICKETS.get(ticket_id)