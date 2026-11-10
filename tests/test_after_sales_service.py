import pytest

from app.models.after_sales import AfterSalesType, TicketStatus
from app.services.after_sales_service import (
    TICKETS,
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
    assert ticket.ticket_id in TICKETS


def test_create_ticket_for_nonexistent_order():
    with pytest.raises(ValueError, match="订单不存在"):
        create_after_sales_ticket(
            order_id="ORD999",
            user_id="USER001",
            request_type=AfterSalesType.REFUND,
            reason="申请退款",
        )


def test_user_cannot_create_ticket_for_another_users_order():
    with pytest.raises(ValueError, match="无权为该订单申请售后"):
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

from app.services.after_sales_service import get_after_sales_ticket


def test_get_after_sales_ticket_success():
    ticket = create_after_sales_ticket(
        order_id="ORD001",
        user_id="USER001",
        request_type=AfterSalesType.REFUND,
        reason="不想要了",
    )

    result = get_after_sales_ticket(ticket.ticket_id)

    assert result is not None
    assert result.ticket_id == ticket.ticket_id
    assert result.order_id == "ORD001"


def test_get_nonexistent_after_sales_ticket():
    result = get_after_sales_ticket("INVALID-TICKET-ID")

    assert result is None