from datetime import datetime

import pytest
from pydantic import ValidationError

from app.models.after_sales import (
    AfterSalesTicket,
    AfterSalesType,
    TicketStatus,
)


def test_create_after_sales_ticket():
    ticket = AfterSalesTicket(
        ticket_id="TK001",
        order_id="ORD001",
        user_id="USER001",
        request_type="refund",
        reason="商品存在质量问题",
    )

    assert ticket.ticket_id == "TK001"
    assert ticket.request_type == AfterSalesType.REFUND
    assert ticket.status == TicketStatus.PENDING
    assert isinstance(ticket.created_at, datetime)


def test_ticket_rejects_invalid_request_type():
    with pytest.raises(ValidationError):
        AfterSalesTicket(
            ticket_id="TK002",
            order_id="ORD001",
            user_id="USER001",
            request_type="unknown",
            reason="商品存在质量问题",
        )


def test_ticket_rejects_empty_reason():
    with pytest.raises(ValidationError):
        AfterSalesTicket(
            ticket_id="TK003",
            order_id="ORD001",
            user_id="USER001",
            request_type="refund",
            reason="",
        )