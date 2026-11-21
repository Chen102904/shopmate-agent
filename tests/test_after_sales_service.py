import pytest

from app.models.after_sales import AfterSalesType, TicketStatus
from app.services.after_sales_service import (
    create_after_sales_ticket,
    get_after_sales_ticket,
)


def test_create_refund_ticket_success():
    ticket = create_after_sales_ticket(
        order_id="ORD001",
        user_id="USER001",
        request_type=AfterSalesType.REFUND,
        reason="不想要了",
    )

    assert ticket.order_id == "ORD001"
    assert ticket.user_id == "USER001"
    assert ticket.request_type == AfterSalesType.REFUND
    assert ticket.status == TicketStatus.PENDING

    # 工单已落库：能用编号查回来，且归属正确
    saved = get_after_sales_ticket(ticket.ticket_id, "USER001")
    assert saved is not None
    assert saved.ticket_id == ticket.ticket_id


def test_create_ticket_for_nonexistent_order():
    with pytest.raises(ValueError, match="订单不存在或无权访问"):
        create_after_sales_ticket(
            order_id="ORD999",
            user_id="USER001",
            request_type=AfterSalesType.REFUND,
            reason="申请退款",
        )


def test_user_cannot_create_ticket_for_another_users_order():
    """跨用户创建工单必须被拒绝，且不透露该订单是否存在。"""

    with pytest.raises(ValueError, match="订单不存在或无权访问"):
        create_after_sales_ticket(
            order_id="ORD003",
            user_id="USER001",
            request_type=AfterSalesType.REFUND,
            reason="申请退款",
        )


def test_cannot_refund_shipped_order():
    with pytest.raises(ValueError, match="商品正在运输中"):
        create_after_sales_ticket(
            order_id="ORD002",
            user_id="USER001",
            request_type=AfterSalesType.REFUND,
            reason="申请退款",
        )


def test_get_after_sales_ticket_success():
    ticket = create_after_sales_ticket(
        order_id="ORD001",
        user_id="USER001",
        request_type=AfterSalesType.REFUND,
        reason="不想要了",
    )

    result = get_after_sales_ticket(ticket.ticket_id, "USER001")

    assert result is not None
    assert result.ticket_id == ticket.ticket_id
    assert result.order_id == "ORD001"


def test_get_nonexistent_after_sales_ticket():
    result = get_after_sales_ticket(
        "INVALID-TICKET-ID",
        "USER001",
    )

    assert result is None


def test_cannot_read_another_users_ticket():
    """拿到别人的工单号也不能查，避免工单号泄露即数据泄露。"""

    ticket = create_after_sales_ticket(
        order_id="ORD001",
        user_id="USER001",
        request_type=AfterSalesType.REFUND,
        reason="不想要了",
    )

    result = get_after_sales_ticket(
        ticket.ticket_id,
        "USER002",
    )

    assert result is None
