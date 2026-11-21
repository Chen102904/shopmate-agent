from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def _headers(user_id: str = "USER001") -> dict:
    return {"X-User-Id": user_id}


def test_check_refund_for_paid_order():
    response = client.post(
        "/refunds/check",
        headers=_headers(),
        json={"order_id": "ORD001"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "ORD001"
    assert data["eligible"] is True


def test_check_refund_for_shipped_order():
    response = client.post(
        "/refunds/check",
        headers=_headers(),
        json={"order_id": "ORD002"},
    )

    assert response.status_code == 200
    assert response.json()["eligible"] is False


def test_check_refund_for_nonexistent_order():
    response = client.post(
        "/refunds/check",
        headers=_headers(),
        json={"order_id": "ORD999"},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["eligible"] is None
    assert data["reason"] == "订单不存在或无权访问"
    assert data["status"] == "not_accessible"


def test_check_refund_for_another_users_order():
    """用 USER001 查 USER002 的 ORD003，必须查不到。"""

    response = client.post(
        "/refunds/check",
        headers=_headers("USER001"),
        json={"order_id": "ORD003"},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["eligible"] is None
    assert data["status"] == "not_accessible"


def test_check_refund_rejects_empty_order_id():
    response = client.post(
        "/refunds/check",
        headers=_headers(),
        json={"order_id": ""},
    )

    assert response.status_code == 422
