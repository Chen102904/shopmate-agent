from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_query_existing_logistics():
    response = client.get("/logistics/ORD002")

    assert response.status_code == 200

    data = response.json()
    assert data["order_id"] == "ORD002"
    assert data["tracking_number"] == "SF123456"
    assert data["status"] == "in_transit"


def test_query_nonexistent_logistics():
    response = client.get("/logistics/ORD001")

    assert response.status_code == 404
    assert response.json()["detail"] == "该订单暂无物流信息"