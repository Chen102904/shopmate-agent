from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.models.order import Order, OrderStatus


def test_create_order():
    order = Order(
        order_id="ORD001",
        user_id="USER001",
        status="paid",
        total_amount="99.90",
        order_time=datetime.now(),
    )

    assert order.order_id == "ORD001"
    assert order.status == OrderStatus.PAID
    assert order.total_amount == Decimal("99.90")


def test_order_rejects_negative_amount():
    with pytest.raises(ValidationError):
        Order(
            order_id="ORD002",
            user_id="USER001",
            status="paid",
            total_amount="-10.00",
            order_time=datetime.now(),
        )


def test_order_rejects_invalid_status():
    with pytest.raises(ValidationError):
        Order(
            order_id="ORD003",
            user_id="USER001",
            status="unknown",
            total_amount="99.90",
            order_time=datetime.now(),
        )