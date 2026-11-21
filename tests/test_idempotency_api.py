"""幂等键在 HTTP 层的行为。

服务层保证「不重复建单」，这一层保证客户端能区分
「新建」和「重放」，并且在用错 key 时拿到明确的 409。
"""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _headers(user_id: str = "USER001", key: str | None = None) -> dict:
    headers = {"X-User-Id": user_id}

    if key is not None:
        headers["Idempotency-Key"] = key

    return headers


def _body(reason: str = "不想要了") -> dict:
    return {
        "order_id": "ORD001",
        "request_type": "refund",
        "reason": reason,
    }


def test_first_request_returns_201():
    response = client.post(
        "/after-sales",
        headers=_headers(key="api-key-1"),
        json=_body(),
    )

    assert response.status_code == 201
    assert response.headers["Idempotent-Replayed"] == "false"


def test_replayed_request_returns_200_same_ticket():
    first = client.post(
        "/after-sales",
        headers=_headers(key="api-key-2"),
        json=_body(),
    )

    second = client.post(
        "/after-sales",
        headers=_headers(key="api-key-2"),
        json=_body(),
    )

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.headers["Idempotent-Replayed"] == "true"
    assert second.json()["ticket_id"] == first.json()["ticket_id"]


def test_same_key_different_body_returns_409():
    client.post(
        "/after-sales",
        headers=_headers(key="api-key-3"),
        json=_body("不想要了"),
    )

    response = client.post(
        "/after-sales",
        headers=_headers(key="api-key-3"),
        json=_body("商品破损"),
    )

    assert response.status_code == 409


def test_overlong_key_is_rejected():
    response = client.post(
        "/after-sales",
        headers=_headers(key="k" * 300),
        json=_body(),
    )

    assert response.status_code == 400


def test_empty_key_is_treated_as_absent():
    """空的 Idempotency-Key 等于没传，不能用来绕过去重。"""

    first = client.post(
        "/after-sales",
        headers=_headers(key=""),
        json=_body(),
    )

    second = client.post(
        "/after-sales",
        headers=_headers(key=""),
        json=_body(),
    )

    assert first.status_code == 201

    # 没有键时走业务去重，仍然返回同一张工单
    assert second.json()["ticket_id"] == first.json()["ticket_id"]
