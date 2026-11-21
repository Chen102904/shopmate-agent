"""签名令牌与鉴权测试。"""

from fastapi.testclient import TestClient

from app.core.auth import (
    DEMO_USERS,
    TokenError,
    issue_token,
    verify_token,
)
from main import app


client = TestClient(app)

DEMO_PASSWORD = DEMO_USERS["USER001"]


def _token(user_id: str = "USER001") -> str:
    response = client.post(
        "/auth/token",
        json={"user_id": user_id, "password": DEMO_PASSWORD},
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def test_issue_token_with_correct_password():
    response = client.post(
        "/auth/token",
        json={"user_id": "USER001", "password": DEMO_PASSWORD},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["token_type"] == "bearer"
    assert data["user_id"] == "USER001"
    assert data["expires_in"] > 0
    assert data["access_token"]


def test_wrong_password_rejected():
    response = client.post(
        "/auth/token",
        json={"user_id": "USER001", "password": "wrong"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "用户名或密码错误"


def test_unknown_user_rejected():
    response = client.post(
        "/auth/token",
        json={"user_id": "NOBODY", "password": DEMO_PASSWORD},
    )

    assert response.status_code == 401


def test_token_grants_access_to_own_order():
    token = _token("USER001")

    response = client.get(
        "/orders/ORD001",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200


def test_token_cannot_access_other_users_order():
    token = _token("USER001")

    response = client.get(
        "/orders/ORD003",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404


def test_token_user_wins_over_header():
    """同时带令牌和 X-User-Id 时，以令牌为准。

    否则攻击者只要加一个 X-User-Id 就能覆盖令牌里的身份。
    """

    token = _token("USER002")

    response = client.get(
        "/orders/ORD003",
        headers={
            "Authorization": f"Bearer {token}",
            "X-User-Id": "USER001",
        },
    )

    assert response.status_code == 200
    assert response.json()["user_id"] == "USER002"


def test_tampered_token_rejected():
    token = _token("USER001")

    # 改掉签名最后一位
    tampered = token[:-1] + ("0" if token[-1] != "0" else "1")

    response = client.get(
        "/orders/ORD001",
        headers={"Authorization": f"Bearer {tampered}"},
    )

    assert response.status_code == 401


def test_tampered_payload_rejected():
    """改载荷里的 uid 会被签名拦住。"""

    token = _token("USER001")
    payload, signature = token.rsplit(".", 1)

    other_payload = _token("USER002").rsplit(".", 1)[0]

    forged = f"{other_payload}.{signature}"

    response = client.get(
        "/orders/ORD003",
        headers={"Authorization": f"Bearer {forged}"},
    )

    assert response.status_code == 401


def test_expired_token_rejected():
    token = issue_token("USER001", ttl_seconds=-1)

    response = client.get(
        "/orders/ORD001",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


def test_bad_token_does_not_fall_back_to_header():
    """坏令牌必须直接拒绝，不能退回自报身份。"""

    response = client.get(
        "/orders/ORD001",
        headers={
            "Authorization": "Bearer not-a-real-token",
            "X-User-Id": "USER001",
        },
    )

    assert response.status_code == 401


def test_malformed_authorization_header():
    response = client.get(
        "/orders/ORD001",
        headers={"Authorization": "Basic abc"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "身份令牌格式不正确"


def test_verify_roundtrip():
    token = issue_token("USER002")

    assert verify_token(token) == "USER002"


def test_verify_rejects_garbage():
    for bad in ("", "abc", "a.b", "...."):
        try:
            verify_token(bad)
        except TokenError:
            continue
        raise AssertionError(f"应当拒绝：{bad!r}")
