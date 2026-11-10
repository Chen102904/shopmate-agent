
from fastapi.testclient import TestClient

from main import app

from app.core.execution_trace import (
    save_completed_execution_trace,
)


client = TestClient(app)


def test_query_existing_trace():
    # 12 位小写十六进制字符
    request_id = "abcdef123456"

    save_completed_execution_trace(
        request_id=request_id,
        status="success",
        duration_ms=100,
    )

    response = client.get(
        f"/traces/{request_id}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["request_id"] == request_id
    assert data["status"] == "success"
    assert data["duration_ms"] == 100


def test_query_nonexistent_trace():
    response = client.get(
        "/traces/000000000000"
    )

    assert response.status_code == 404
    assert data_detail(response) == "执行记录不存在或已过期"


def data_detail(response):
    return response.json()["detail"]


def test_trace_rejects_invalid_request_id():
    response = client.get(
        "/traces/invalid"
    )

    assert response.status_code == 422
