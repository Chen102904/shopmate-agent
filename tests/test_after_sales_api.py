from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _headers(user_id: str = "USER001") -> dict:
    """构造带身份的请求头。"""

    return {"X-User-Id": user_id}


def test_create_after_sales_ticket_success():
    response = client.post(
        "/after-sales",
        headers=_headers(),
        json={
            "order_id": "ORD001",
            "request_type": "refund",
            "reason": "不想要了",
        },
    )

    assert response.status_code == 201

    data = response.json()
    assert data["order_id"] == "ORD001"
    assert data["user_id"] == "USER001"
    assert data["request_type"] == "refund"
    assert data["status"] == "pending"
    assert data["ticket_id"]


def test_create_ticket_for_nonexistent_order():
    response = client.post(
        "/after-sales",
        headers=_headers(),
        json={
            "order_id": "ORD999",
            "request_type": "refund",
            "reason": "申请退款",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "订单不存在或无权访问"


def test_create_ticket_for_another_users_order():
    response = client.post(
        "/after-sales",
        headers=_headers("USER001"),
        json={
            "order_id": "ORD003",
            "request_type": "refund",
            "reason": "申请退款",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "订单不存在或无权访问"


def test_body_user_id_cannot_override_header():
    """关键的越权测试。

    请求体里把 user_id 填成 USER002（ORD003 的真实归属人），
    但请求头是 USER001。
    服务端必须以请求头为准，拒绝这次操作，
    否则用户只要在请求体里改一个字段就能替别人申请售后。
    """

    response = client.post(
        "/after-sales",
        headers=_headers("USER001"),
        json={
            "order_id": "ORD003",
            "user_id": "USER002",
            "request_type": "refund",
            "reason": "申请退款",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "订单不存在或无权访问"


def test_create_ticket_for_shipped_order():
    response = client.post(
        "/after-sales",
        headers=_headers(),
        json={
            "order_id": "ORD002",
            "request_type": "refund",
            "reason": "申请退款",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "商品正在运输中，暂不支持退款"


def test_query_existing_ticket():
    create_response = client.post(
        "/after-sales",
        headers=_headers(),
        json={
            "order_id": "ORD001",
            "request_type": "refund",
            "reason": "不想要了",
        },
    )

    assert create_response.status_code == 201

    ticket_id = create_response.json()["ticket_id"]

    response = client.get(
        f"/after-sales/{ticket_id}",
        headers=_headers(),
    )

    assert response.status_code == 200
    assert response.json()["ticket_id"] == ticket_id
    assert response.json()["order_id"] == "ORD001"


def test_cannot_query_another_users_ticket():
    """工单号泄露时，别人也查不到这张工单。"""

    create_response = client.post(
        "/after-sales",
        headers=_headers("USER001"),
        json={
            "order_id": "ORD001",
            "request_type": "refund",
            "reason": "不想要了",
        },
    )

    ticket_id = create_response.json()["ticket_id"]

    response = client.get(
        f"/after-sales/{ticket_id}",
        headers=_headers("USER002"),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "售后工单不存在或无权访问"


def test_query_nonexistent_ticket():
    response = client.get(
        "/after-sales/00000000-0000-4000-8000-000000000000",
        headers=_headers(),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "售后工单不存在或无权访问"


def test_query_ticket_with_invalid_uuid():
    response = client.get(
        "/after-sales/INVALID-TICKET-ID",
        headers=_headers(),
    )

    assert response.status_code == 422
