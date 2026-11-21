from app.services.refund_service import check_refund_eligibility


def test_paid_order_can_refund():
    result = check_refund_eligibility("ORD001", "USER001")

    assert result["eligible"] is True


def test_shipped_order_cannot_refund():
    result = check_refund_eligibility("ORD002", "USER001")

    assert result["eligible"] is False


def test_delivered_order_can_refund_within_seven_days():
    """ORD003 属于 USER002，必须用归属人身份查询。"""

    result = check_refund_eligibility("ORD003", "USER002")

    assert result["eligible"] is True


def test_nonexistent_order_cannot_refund():
    result = check_refund_eligibility("ORD999", "USER001")

    # 订单查不到时 eligible 必须是 None，不能是 False。
    # 返回 False 会让模型把「查不到」说成「不符合退款条件」。
    assert result["eligible"] is None
    assert result["reason"] == "订单不存在或无权访问"
    assert result["status"] == "not_accessible"


def test_cannot_check_another_users_order():
    """用 USER001 查 USER002 的订单，必须和「订单不存在」表现一致。"""

    result = check_refund_eligibility("ORD003", "USER001")

    assert result["eligible"] is None
    assert result["reason"] == "订单不存在或无权访问"
