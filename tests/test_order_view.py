"""订单视图测试：订单列表、状态叠加、模拟审核入口。

这一组测的是「用户能看到什么」，与 test_order_api.py 的分工：
    那边测「能不能查到这张单、会不会越权」
    这边测「多张单一起列出来时长什么样、售后之后状态怎么变」

最要紧的三条：

    1. 列表只能出现自己的单（不能靠改 URL 翻别人的订单）
    2. 有售后单时，展示状态要跟着变，而不是永远「待发货」
    3. 模拟审核入口默认关闭，没开开关时接口等于不存在
"""

import pytest

from fastapi.testclient import TestClient

from app.models.after_sales import AfterSalesType
from app.services.after_sales_service import (
    cancel_after_sales_ticket,
    create_after_sales_ticket,
    review_after_sales_ticket,
)
from main import app


client = TestClient(app)

USER = "USER001"
OTHER_USER = "USER002"


def _headers(user_id: str = USER) -> dict:
    return {"X-User-Id": user_id}


def _create(
    order_id: str = "ORD001",
    user_id: str = USER,
    request_type: AfterSalesType = AfterSalesType.REFUND,
):
    return create_after_sales_ticket(
        order_id=order_id,
        user_id=user_id,
        request_type=request_type,
        reason="商品破损",
    )


def _order_view(order_id: str, user_id: str = USER) -> dict:
    response = client.get(f"/orders/{order_id}", headers=_headers(user_id))

    assert response.status_code == 200

    return response.json()


# ============================================================
# 订单列表
# ============================================================


def test_list_only_returns_own_orders():
    response = client.get("/orders", headers=_headers(USER))

    assert response.status_code == 200

    order_ids = [item["order_id"] for item in response.json()]

    assert order_ids == ["ORD001", "ORD002"]
    assert "ORD003" not in order_ids


def test_list_is_sorted_by_order_time_desc():
    """ORD001 是 2 天前下的，ORD002 是 5 天前，所以 001 在前。"""

    order_ids = [
        item["order_id"]
        for item in client.get("/orders", headers=_headers(USER)).json()
    ]

    assert order_ids[0] == "ORD001"


def test_other_user_sees_only_their_order():
    order_ids = [
        item["order_id"]
        for item in client.get("/orders", headers=_headers(OTHER_USER)).json()
    ]

    assert order_ids == ["ORD003"]


def test_list_item_has_display_fields():
    item = client.get("/orders", headers=_headers(USER)).json()[0]

    assert item["status_label"] == "待发货"
    assert item["display_status"] == "paid"
    assert item["display_status_label"] == "待发货"
    assert item["total_amount"] == "199.90"
    assert item["ticket"] is None


# ============================================================
# 售后叠加出来的展示状态
# ============================================================


def test_pending_ticket_turns_order_into_refunding():
    _create()

    item = _order_view("ORD001")

    assert item["status"] == "paid"            # 交易状态没变
    assert item["display_status"] == "refunding"
    assert item["display_status_label"] == "退款中"
    assert item["ticket"]["status"] == "pending"
    assert item["ticket"]["status_label"] == "待审核"


def test_approved_ticket_turns_order_into_refunded():
    ticket = _create()
    review_after_sales_ticket(ticket.ticket_id, "approved")

    item = _order_view("ORD001")

    assert item["display_status"] == "refunded"
    assert item["display_status_label"] == "已退款"


def test_rejected_ticket_falls_back_to_order_status():
    """驳回之后订单回到「待发货」，不能一直挂个退款中。"""

    ticket = _create()
    review_after_sales_ticket(ticket.ticket_id, "rejected", "材料不足")

    item = _order_view("ORD001")

    assert item["display_status"] == "paid"
    assert item["display_status_label"] == "待发货"
    assert item["ticket"]["status"] == "rejected"
    assert item["ticket"]["review_note"] == "材料不足"


def test_cancelled_ticket_falls_back_to_order_status():
    ticket = _create()
    cancel_after_sales_ticket(ticket.ticket_id, USER)

    item = _order_view("ORD001")

    assert item["display_status"] == "paid"


def test_resubmit_after_cancel_uses_latest_ticket():
    """撤销后重新申请：要以最新那张为准，不能还显示「已撤销」。"""

    first = _create()
    cancel_after_sales_ticket(first.ticket_id, USER)

    _create()

    item = _order_view("ORD001")

    assert item["display_status"] == "refunding"


def test_exchange_ticket_is_not_shown_as_refund():
    """换货不走「退款中」，否则用户会以为钱在退。"""

    _create(request_type=AfterSalesType.EXCHANGE)

    item = _order_view("ORD001")

    assert item["display_status"] == "after_sales"
    assert item["display_status_label"] == "售后中"


def test_refund_hint_explains_why_not_eligible():
    """已发货在途的单不能直接退，要把原因直接给出来。"""

    item = _order_view("ORD002")

    assert item["can_request_refund"] is False
    assert item["refund_hint"]


# ============================================================
# 模拟审核入口（演示用，默认关闭）
# ============================================================


def test_demo_review_is_404_when_disabled(monkeypatch):
    """没开 SHOPMATE_DEMO_REVIEW 时，这个路由对外不存在。

    这条是整组里最要紧的：
        演示用的自批入口如果能被默认打开，
        整套「审核权限和用户身份分离」的设计就白做了。
    """

    monkeypatch.delenv("SHOPMATE_DEMO_REVIEW", raising=False)

    ticket = _create()

    response = client.post(
        f"/demo/review/{ticket.ticket_id}",
        json={"decision": "approved"},
        headers=_headers(USER),
    )

    assert response.status_code == 404

    # 状态没被改动
    assert _order_view("ORD001")["display_status"] == "refunding"


def test_demo_status_reports_switch(monkeypatch):
    monkeypatch.delenv("SHOPMATE_DEMO_REVIEW", raising=False)
    assert client.get("/demo/status").json()["review_enabled"] is False

    monkeypatch.setenv("SHOPMATE_DEMO_REVIEW", "1")
    assert client.get("/demo/status").json()["review_enabled"] is True


def test_demo_review_approves_own_ticket(monkeypatch):
    monkeypatch.setenv("SHOPMATE_DEMO_REVIEW", "1")

    ticket = _create()

    response = client.post(
        f"/demo/review/{ticket.ticket_id}",
        json={"decision": "approved"},
        headers=_headers(USER),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "approved"

    assert _order_view("ORD001")["display_status_label"] == "已退款"


def test_demo_review_cannot_touch_other_users_ticket(monkeypatch):
    """模拟审核也不能越过归属校验。"""

    monkeypatch.setenv("SHOPMATE_DEMO_REVIEW", "1")

    ticket = _create(order_id="ORD003", user_id=OTHER_USER)

    response = client.post(
        f"/demo/review/{ticket.ticket_id}",
        json={"decision": "approved"},
        headers=_headers(USER),
    )

    assert response.status_code == 404


def test_demo_review_rejects_bad_decision(monkeypatch):
    monkeypatch.setenv("SHOPMATE_DEMO_REVIEW", "1")

    ticket = _create()

    response = client.post(
        f"/demo/review/{ticket.ticket_id}",
        json={"decision": "looks_fine"},
        headers=_headers(USER),
    )

    assert response.status_code == 400


def test_demo_review_cannot_review_twice(monkeypatch):
    monkeypatch.setenv("SHOPMATE_DEMO_REVIEW", "1")

    ticket = _create()

    client.post(
        f"/demo/review/{ticket.ticket_id}",
        json={"decision": "approved"},
        headers=_headers(USER),
    )

    response = client.post(
        f"/demo/review/{ticket.ticket_id}",
        json={"decision": "rejected"},
        headers=_headers(USER),
    )

    assert response.status_code == 409


# ============================================================
# 越权
# ============================================================


@pytest.mark.parametrize("order_id", ["ORD003"])
def test_detail_of_other_users_order_is_404(order_id):
    response = client.get(f"/orders/{order_id}", headers=_headers(USER))

    assert response.status_code == 404
