from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_query_existing_order():
    response = client.get("/orders/ORD001")

    assert response.status_code == 200

    data = response.json()
    assert data["order_id"] == "ORD001"
    assert data["user_id"] == "USER001"
    assert data["status"] == "paid"


def test_query_nonexistent_order():
    response = client.get("/orders/ORD999")

    assert response.status_code == 404
    assert response.json()["detail"] == "订单不存在"