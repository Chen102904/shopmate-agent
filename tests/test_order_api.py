from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def _headers(user_id: str = "USER001") -> dict:
    return {"X-User-Id": user_id}


def test_query_existing_order():
    response = client.get(
        "/orders/ORD001",
        headers=_headers(),
    )

    assert response.status_code == 200

    data = response.json()
    assert data["order_id"] == "ORD001"
    assert data["user_id"] == "USER001"
    assert data["status"] == "paid"


def test_query_nonexistent_order():
    response = client.get(
        "/orders/ORD999",
        headers=_headers(),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "订单不存在或无权访问"


def test_cannot_query_another_users_order():
    """ORD003 属于 USER002，USER001 查询必须失败。"""

    response = client.get(
        "/orders/ORD003",
        headers=_headers("USER001"),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "订单不存在或无权访问"


def test_owner_can_query_own_order():
    response = client.get(
        "/orders/ORD003",
        headers=_headers("USER002"),
    )

    assert response.status_code == 200
    assert response.json()["order_id"] == "ORD003"


def test_missing_header_falls_back_to_demo_user():
    """不带身份时回落到演示账号 USER001，可以查自己的订单。"""

    response = client.get("/orders/ORD001")

    assert response.status_code == 200


def test_missing_header_cannot_see_other_users_order():
    """不带身份时也要受归属限制，不能因为没传就算通。"""

    response = client.get("/orders/ORD003")

    assert response.status_code == 404
