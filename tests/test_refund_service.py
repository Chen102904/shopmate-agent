from app.services.refund_service import check_refund_eligibility


def test_paid_order_can_refund():
    result = check_refund_eligibility("ORD001")

    assert result["eligible"] is True


def test_shipped_order_cannot_refund():
    result = check_refund_eligibility("ORD002")

    assert result["eligible"] is False


def test_delivered_order_can_refund_within_seven_days():
    result = check_refund_eligibility("ORD003")

    assert result["eligible"] is True


def test_nonexistent_order_cannot_refund():
    result = check_refund_eligibility("ORD999")

    assert result["eligible"] is False
    assert result["reason"] == "订单不存在"