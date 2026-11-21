"""演示模式测试。

守两条线：

1. **打开演示模式时**，访客能测完整流程。
   不这么做的话，别人打开链接查什么都是「订单不存在」，没法测。

2. **关闭演示模式时**，行为必须和原来完全一样。
   这些放宽都会削弱安全性（身份可伪造、限流可绕过），
   不能因为加了演示功能就让生产环境也松掉。
"""

import pytest

import app.core.demo_mode as demo


@pytest.fixture
def demo_on(monkeypatch):
    """打开演示模式。"""

    monkeypatch.setattr(demo, "DEMO_MODE", True)
    monkeypatch.setattr(demo, "DEMO_OWNER_ID", "USER001")
    yield
    # monkeypatch 会自动还原


def test_guest_prefix_detection():
    assert demo.is_guest("guest-abc12345") is True
    assert demo.is_guest("USER001") is False
    assert demo.is_guest(None) is False
    assert demo.is_guest("") is False


def test_resolve_without_demo_mode_keeps_original(monkeypatch):
    """关闭演示模式时，访客身份不能被映射成演示账号。

    否则任何人都能自称 guest-xxx 查到演示账号的订单。
    """

    monkeypatch.setattr(demo, "DEMO_MODE", False)

    assert demo.resolve_demo_owner("guest-abc") == "guest-abc"
    assert demo.resolve_demo_owner("USER002") == "USER002"


def test_resolve_with_demo_mode_maps_guest(demo_on):
    assert demo.resolve_demo_owner("guest-abc") == "USER001"


def test_resolve_with_demo_mode_keeps_real_user(demo_on):
    """演示模式下，真实用户 id 不能被映射。

    否则所有真实用户都会看到同一个人的订单。
    """

    assert demo.resolve_demo_owner("USER002") == "USER002"


def test_guest_can_read_demo_owner_order(demo_on):
    """访客能查到演示账号的订单。"""

    from app.services.order_service import get_order_by_id

    order = get_order_by_id("ORD001", "guest-abc12345")

    assert order is not None
    assert order.order_id == "ORD001"


def test_guest_cannot_read_other_user_order(demo_on):
    """越权保护仍然有效：访客查不到别人的订单。

    演示模式放宽的是「访客 → 演示账号」这一条映射，
    不是「谁都能查任何订单」。
    """

    from app.services.order_service import get_order_by_id

    # ORD003 属于 USER002
    order = get_order_by_id("ORD003", "guest-abc12345")

    assert order is None


def test_guest_without_demo_mode_sees_nothing(monkeypatch):
    """关闭演示模式时，访客查不到任何订单。"""

    monkeypatch.setattr(demo, "DEMO_MODE", False)

    from app.services.order_service import get_order_by_id

    assert get_order_by_id("ORD001", "guest-abc12345") is None


def test_tickets_isolated_between_guests(demo_on):
    """两个访客的工单互相看不到。

    这是演示时最容易出问题的地方：
    如果按演示账号存工单，A 提交的退款会出现在 B 的「我的申请」里。
    """

    from app.core.db import get_connection
    from app.models.after_sales import AfterSalesType
    from app.services.after_sales_service import create_after_sales_ticket
    from app.services.ticket_store import list_recent_tickets

    with get_connection() as connection:
        for table in (
            "after_sales_tickets",
            "ticket_drafts",
            "idempotency_keys",
        ):
            connection.execute(f"DELETE FROM {table}")

    create_after_sales_ticket(
        "ORD001",
        "guest-aaaa1111",
        AfterSalesType.REFUND,
        "商品破损",
    )
    create_after_sales_ticket(
        "ORD002",
        "guest-bbbb2222",
        AfterSalesType.RETURN,
        "尺码不合适",
    )

    a_tickets = list_recent_tickets("guest-aaaa1111")
    b_tickets = list_recent_tickets("guest-bbbb2222")

    assert len(a_tickets) == 1
    assert len(b_tickets) == 1
    assert a_tickets[0].order_id == "ORD001"
    assert b_tickets[0].order_id == "ORD002"

    with get_connection() as connection:
        for table in (
            "after_sales_tickets",
            "ticket_drafts",
            "idempotency_keys",
        ):
            connection.execute(f"DELETE FROM {table}")


def test_rate_limit_grouped_by_guest_in_demo_mode(monkeypatch, demo_on):
    """演示模式下，限流按访客分组，不按 IP。

    走公网隧道时所有访客的来源 IP 相同，
    按 IP 分组等于十几个人共用 20 次/分钟，
    第一个人聊几句后面的人就被 429 挡住。
    """

    from app.core.api_security import _get_rate_limit_key

    class _Client:
        host = "127.0.0.1"

    class _Request:
        url = type("U", (), {"path": "/chat"})()

        def __init__(self, guest):
            self.headers = {"X-User-Id": guest}
            self.client = _Client()

    key_a = _get_rate_limit_key(_Request("guest-aaaa"))
    key_b = _get_rate_limit_key(_Request("guest-bbbb"))

    assert key_a != key_b
    assert "guest-aaaa" in key_a
