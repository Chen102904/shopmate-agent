from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_check_refund_for_paid_order():
    response = client.post(
        "/refunds/check",
        json={"order_id": "ORD001"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "ORD001"
    assert data["eligible"] is True


def test_check_refund_for_shipped_order():
    response = client.post(
        "/refunds/check",
        json={"order_id": "ORD002"},
    )

    assert response.status_code == 200
    assert response.json()["eligible"] is False


def test_check_refund_for_nonexistent_order():
    response = client.post(
        "/refunds/check",
        json={"order_id": "ORD999"},
    )

    assert response.status_code == 200
    assert response.json()["eligible"] is False
    assert response.json()["reason"] == "订单不存在"


def test_check_refund_rejects_empty_order_id():
    response = client.post(
        "/refunds/check",
        json={"order_id": ""},
    )

    assert response.status_code == 422