from datetime import datetime, timedelta
from decimal import Decimal

from app.models.order import Order
from app.models.logistics import Logistics


ORDERS = {
    "ORD001": Order(
        order_id="ORD001",
        user_id="USER001",
        status="paid",
        total_amount=Decimal("199.90"),
        order_time=datetime.now() - timedelta(days=2),
    ),
    "ORD002": Order(
        order_id="ORD002",
        user_id="USER001",
        status="shipped",
        total_amount=Decimal("299.00"),
        order_time=datetime.now() - timedelta(days=5),
    ),
    "ORD003": Order(
        order_id="ORD003",
        user_id="USER002",
        status="delivered",
        total_amount=Decimal("89.90"),
        order_time=datetime.now() - timedelta(days=10),
    ),
}


LOGISTICS = {
    "ORD002": Logistics(
        order_id="ORD002",
        tracking_number="SF123456",
        company="顺丰速运",
        status="in_transit",
        shipped_at=datetime.now() - timedelta(days=2),
    ),
    "ORD003": Logistics(
        order_id="ORD003",
        tracking_number="YT654321",
        company="圆通速递",
        status="delivered",
        shipped_at=datetime.now() - timedelta(days=8),
        delivered_at=datetime.now() - timedelta(days=7),
    ),
}