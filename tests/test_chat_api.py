
from fastapi.testclient import TestClient

from main import app
from app.agent.conversation_store import (
    add_message,
    clear_conversation,
)


client = TestClient(app)


def test_chat_creates_new_conversation(monkeypatch):
    def fake_run_agent(
        user_message,
        conversation_id,
        _include_request_metadata=False,
    ):
        add_message(
            conversation_id,
            {
                "role": "user",
                "content": user_message,
            },
        )

        add_message(
            conversation_id,
            {
                "role": "assistant",
                "content": "你好，我是 ShopMate。",
            },
        )

        return {
            "result": "你好，我是 ShopMate。",
            "request_id": "abcdef123456",
        }

    monkeypatch.setattr(
        "app.api.chat.run_agent",
        fake_run_agent,
    )

    response = client.post(
        "/chat",
        json={
            "message": "你好",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["conversation_id"]
    assert data["request_id"] == "abcdef123456"
    assert data["answer"] == "你好，我是 ShopMate。"


def test_chat_continues_existing_conversation(monkeypatch):
    conversation_id = "test-conversation-001"

    clear_conversation(conversation_id)

    def fake_run_agent(
        user_message,
        conversation_id,
        _include_request_metadata=False,
    ):
        return {
            "result": "订单查询结果",
            "request_id": "abcdef123456",
        }

    monkeypatch.setattr(
        "app.api.chat.run_agent",
        fake_run_agent,
    )

    response = client.post(
        "/chat",
        json={
            "message": "查询订单",
            "conversation_id": conversation_id,
        },
    )

    assert response.status_code == 200

    assert (
        response.json()["conversation_id"]
        == conversation_id
    )

    clear_conversation(conversation_id)


def test_query_conversation():
    conversation_id = "test-conversation-002"

    clear_conversation(conversation_id)

    add_message(
        conversation_id,
        {
            "role": "user",
            "content": "你好",
        },
    )

    response = client.get(
        f"/conversations/{conversation_id}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["conversation_id"] == conversation_id
    assert data["message_count"] == 1
    assert data["messages"][0]["content"] == "你好"

    clear_conversation(conversation_id)


def test_query_nonexistent_conversation():
    response = client.get(
        "/conversations/nonexistent-conversation"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "会话不存在"


def test_delete_conversation():
    conversation_id = "test-conversation-003"

    add_message(
        conversation_id,
        {
            "role": "user",
            "content": "测试",
        },
    )

    response = client.delete(
        f"/conversations/{conversation_id}"
    )

    assert response.status_code == 200
    assert response.json()["success"] is True

    query_response = client.get(
        f"/conversations/{conversation_id}"
    )

    assert query_response.status_code == 404


def test_chat_rejects_empty_message():
    response = client.post(
        "/chat",
        json={
            "message": "",
        },
    )

    assert response.status_code == 422
