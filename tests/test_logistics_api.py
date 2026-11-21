from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _headers(user_id: str = "USER001") -> dict:
    return {"X-User-Id": user_id}


def test_query_existing_logistics():
    response = client.get(
        "/logistics/ORD002",
        headers=_headers(),
    )

    assert response.status_code == 200

    data = response.json()
    assert data["order_id"] == "ORD002"
    assert data["tracking_number"] == "SF123456"
    assert data["status"] == "in_transit"


def test_query_not_shipped_logistics():
    """ORD001 是「已付款未发货」，物流查不到但订单是本人的。

    这里必须明确说「还没发货」，而不是「订单不存在或无权访问」——
    后者的文案会让用户去核对订单号，而他的订单号是对的。
    """

    response = client.get(
        "/logistics/ORD001",
        headers=_headers(),
    )

    assert response.status_code == 404

    detail = response.json()["detail"]

    assert "发货" in detail
    assert "不存在" not in detail


def test_not_accessible_message_stays_generic():
    """越权和不存在的文案必须完全一致，否则能靠差异枚举订单号。"""

    other_users_order = client.get(
        "/logistics/ORD003",
        headers=_headers("USER001"),
    )

    nonexistent = client.get(
        "/logistics/ORD999",
        headers=_headers("USER001"),
    )

    assert other_users_order.status_code == 404
    assert nonexistent.status_code == 404
    assert (
        other_users_order.json()["detail"]
        == nonexistent.json()["detail"]
        == "订单不存在或无权访问"
    )


def test_cannot_query_another_users_logistics():
    """ORD003 有物流信息，但属于 USER002。"""

    response = client.get(
        "/logistics/ORD003",
        headers=_headers("USER001"),
    )

    assert response.status_code == 404


def test_owner_can_query_own_logistics():
    response = client.get(
        "/logistics/ORD003",
        headers=_headers("USER002"),
    )

    assert response.status_code == 200
    assert response.json()["tracking_number"] == "YT654321"
