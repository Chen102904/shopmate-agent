from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_create_after_sales_ticket_success():
    response = client.post(
        "/after-sales",
        json={
            "order_id": "ORD001",
            "user_id": "USER001",
            "request_type": "refund",
            "reason": "不想要了",
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["order_id"] == "ORD001"
    assert data["user_id"] == "USER001"
    assert data["request_type"] == "refund"
    assert data["status"] == "pending"
    assert data["ticket_id"]


def test_create_ticket_for_nonexistent_order():
    response = client.post(
        "/after-sales",
        json={
            "order_id": "ORD999",
            "user_id": "USER001",
            "request_type": "refund",
            "reason": "申请退款",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "订单不存在"


def test_create_ticket_for_another_users_order():
    response = client.post(
        "/after-sales",
        json={
            "order_id": "ORD003",
            "user_id": "USER001",
            "request_type": "refund",
            "reason": "申请退款",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "无权为该订单申请售后"


def test_create_ticket_for_shipped_order():
    response = client.post(
        "/after-sales",
        json={
            "order_id": "ORD002",
            "user_id": "USER001",
            "request_type": "refund",
            "reason": "申请退款",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "商品正在运输中，暂不支持退款"


def test_query_existing_ticket():
    create_response = client.post(
        "/after-sales",
        json={
            "order_id": "ORD001",
            "user_id": "USER001",
            "request_type": "refund",
            "reason": "不想要了",
        },
    )

    assert create_response.status_code == 200

    ticket_id = create_response.json()["ticket_id"]

    response = client.get(f"/after-sales/{ticket_id}")

    assert response.status_code == 200
    assert response.json()["ticket_id"] == ticket_id
    assert response.json()["order_id"] == "ORD001"


def test_query_nonexistent_ticket():
    response = client.get(
        "/after-sales/00000000-0000-4000-8000-000000000000"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "售后工单不存在"


def test_query_ticket_with_invalid_uuid():
    response = client.get("/after-sales/INVALID-TICKET-ID")

    assert response.status_code == 422