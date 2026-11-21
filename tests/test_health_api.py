"""健康检查与超时降级测试。"""

import time

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def _headers(user_id: str = "USER001") -> dict:
    return {"X-User-Id": user_id}


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["checks"]["database"]["ok"] is True
    assert data["checks"]["knowledge_base"]["ok"] is True
    assert data["checks"]["llm_config"]["ok"] is True


def test_health_covers_ticket_store():
    """工单表不可用时要能提前发现，不能等用户点提交才 500。"""

    response = client.get("/health")

    store = response.json()["checks"]["ticket_store"]

    assert store["ok"] is True
    assert store["tickets"] >= 0
    assert store["idempotency_keys"] >= 0


def test_health_reports_reranker():
    response = client.get("/health")

    reranker = response.json()["checks"]["reranker"]

    assert "enabled" in reranker


def test_reranker_missing_does_not_make_service_unhealthy(monkeypatch):
    """重排模型是可选项，缺失只能降级，不能判服务不健康。"""

    import app.api.health as health_module

    monkeypatch.setattr(
        health_module,
        "_check_reranker",
        lambda: {"ok": False, "enabled": True, "reason": "模型文件不存在"},
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["checks"]["reranker"]["ok"] is False


def test_health_reports_knowledge_base_stats():
    response = client.get("/health")

    kb = response.json()["checks"]["knowledge_base"]

    assert kb["documents"] >= 1
    assert kb["sections"] >= 1


def test_health_does_not_leak_secrets():
    """健康检查只返回「是否配置」，不能把密钥本身吐出来。"""

    body = client.get("/health").text

    assert "DEEPSEEK_API_KEY=" not in body
    assert "sk-" not in body


def test_health_unhealthy_when_knowledge_base_missing(monkeypatch):
    """知识库不可用时应返回 503，让探活判定为不健康。"""

    import app.api.health as health_module

    monkeypatch.setattr(
        health_module,
        "KNOWLEDGE_DIR",
        health_module.BASE_DIR / "__no_such_dir__",
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"]["knowledge_base"]["ok"] is False


def test_chat_timeout_returns_504(monkeypatch):
    """处理超时返回 504，而不是让请求无限挂着。"""

    def slow_run_agent(**kwargs):
        time.sleep(1.0)
        return {
            "result": "太慢了",
            "request_id": "abcdef123456",
        }

    monkeypatch.setattr(
        "app.api.chat.run_agent",
        slow_run_agent,
    )
    monkeypatch.setattr(
        "app.api.chat.CHAT_TIMEOUT_SECONDS",
        0.1,
    )

    response = client.post(
        "/chat",
        headers=_headers(),
        json={"message": "你好"},
    )

    assert response.status_code == 504
    assert response.json()["detail"] == "处理超时，请稍后重试"
