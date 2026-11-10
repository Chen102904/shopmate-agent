
import pytest

from fastapi.testclient import TestClient

from main import app
from app.core import api_security


client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_security_state(monkeypatch):
    """每个测试前清理限流记录和身份验证环境变量。"""
    monkeypatch.delenv("SHOPMATE_API_KEY", raising=False)
    monkeypatch.delenv("SHOPMATE_REQUIRE_API_KEY", raising=False)
    monkeypatch.setenv("SHOPMATE_ENABLE_DOCS", "true")

    api_security.reset_rate_limits()

    yield

    api_security.reset_rate_limits()


def test_api_access_without_configured_key():
    response = client.get(
        "/conversations/security_test_missing"
    )
    assert response.status_code == 404


def test_api_rejects_missing_key(monkeypatch):
    """配置 API Key 后，缺少密钥应被拒绝。"""
    monkeypatch.setenv(
        "SHOPMATE_API_KEY",
        "test-secret-key",
    )

    response = client.get(
        "/conversations/security_test_missing"
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "API 身份验证失败"


def test_api_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv(
        "SHOPMATE_API_KEY",
        "correct-test-key",
    )

    response = client.get(
        "/conversations/security_test_missing",
        headers={
            "X-API-Key": "wrong-test-key"
        },
    )

    assert response.status_code == 401


def test_api_accepts_correct_key(monkeypatch):
    monkeypatch.setenv(
        "SHOPMATE_API_KEY",
        "correct-test-key",
    )

    response = client.get(
        "/conversations/security_test_missing",
        headers={
            "X-API-Key": "correct-test-key"
        },
    )

    assert response.status_code == 404


def test_api_key_required_but_not_configured(monkeypatch):
    monkeypatch.setenv(
        "SHOPMATE_REQUIRE_API_KEY",
        "true",
    )

    response = client.get(
        "/conversations/security_test_missing"
    )

    assert response.status_code == 503


def test_rate_limit_returns_429(monkeypatch):
    monkeypatch.setattr(
        api_security,
        "DEFAULT_REQUESTS_PER_MINUTE",
        1,
    )

    first_response = client.get(
        "/conversations/rate_limit_test"
    )

    second_response = client.get(
        "/conversations/rate_limit_test"
    )

    assert first_response.status_code == 404
    assert second_response.status_code == 429
    assert "Retry-After" in second_response.headers


def test_security_headers_are_added():
    response = client.get(
        "/conversations/security_header_test"
    )

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"
