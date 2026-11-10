from pydantic import ValidationError
import pytest

from app.models.logistics import Logistics, LogisticsStatus


def test_create_logistics():
    item = Logistics(
        order_id="ORD001",
        tracking_number="SF123456",
        company="顺丰速运",
        status="in_transit",
    )

    assert item.order_id == "ORD001"
    assert item.status == LogisticsStatus.IN_TRANSIT
    assert item.shipped_at is None


def test_logistics_rejects_invalid_status():
    with pytest.raises(ValidationError):
        Logistics(
            order_id="ORD001",
            tracking_number="SF123456",
            company="顺丰速运",
            status="unknown",
        )


def test_logistics_rejects_empty_tracking_number():
    with pytest.raises(ValidationError):
        Logistics(
            order_id="ORD001",
            tracking_number="",
            company="顺丰速运",
            status="in_transit",
        )