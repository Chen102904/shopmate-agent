
from fastapi.testclient import TestClient

from main import app
from app.agent.conversation_store import (
    add_message,
    clear_conversation,
)


client = TestClient(app)


def _headers(user_id: str = "USER001") -> dict:
    return {"X-User-Id": user_id}


def test_chat_creates_new_conversation(monkeypatch):
    def fake_run_agent(
        user_message,
        conversation_id,
        user_id=None,
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
        headers=_headers(),
        json={
            "message": "你好",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["conversation_id"]
    assert data["request_id"] == "abcdef123456"
    assert data["user_id"] == "USER001"
    assert data["answer"] == "你好，我是 ShopMate。"


def test_chat_continues_existing_conversation(monkeypatch):
    conversation_id = "test-conversation-001"

    clear_conversation(conversation_id)

    def fake_run_agent(
        user_message,
        conversation_id,
        user_id=None,
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
        headers=_headers(),
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


def test_chat_rejects_another_users_conversation(monkeypatch):
    """拿到别人的 conversation_id 也不能续聊，否则能读到别人的上下文。"""

    conversation_id = "test-conversation-foreign"

    clear_conversation(conversation_id)

    add_message(
        conversation_id,
        {
            "role": "user",
            "content": "我的订单号是 ORD001",
        },
    )

    def fake_run_agent(**kwargs):
        raise AssertionError("不应该执行到这里")

    monkeypatch.setattr(
        "app.api.chat.run_agent",
        fake_run_agent,
    )

    response = client.post(
        "/chat",
        headers=_headers("USER002"),
        json={
            "message": "继续",
            "conversation_id": conversation_id,
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "会话不存在"

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
        f"/conversations/{conversation_id}",
        headers=_headers(),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["conversation_id"] == conversation_id
    assert data["message_count"] == 1
    assert data["messages"][0]["content"] == "你好"

    clear_conversation(conversation_id)


def test_cannot_query_another_users_conversation():
    conversation_id = "test-conversation-004"

    clear_conversation(conversation_id)

    add_message(
        conversation_id,
        {
            "role": "user",
            "content": "私密内容",
        },
    )

    response = client.get(
        f"/conversations/{conversation_id}",
        headers=_headers("USER002"),
    )

    assert response.status_code == 404

    clear_conversation(conversation_id)


def test_query_nonexistent_conversation():
    response = client.get(
        "/conversations/nonexistent-conversation",
        headers=_headers(),
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
        f"/conversations/{conversation_id}",
        headers=_headers(),
    )

    assert response.status_code == 200
    assert response.json()["success"] is True

    query_response = client.get(
        f"/conversations/{conversation_id}",
        headers=_headers(),
    )

    assert query_response.status_code == 404


def test_cannot_delete_another_users_conversation():
    conversation_id = "test-conversation-005"

    clear_conversation(conversation_id)

    add_message(
        conversation_id,
        {
            "role": "user",
            "content": "不要被删掉",
        },
    )

    response = client.delete(
        f"/conversations/{conversation_id}",
        headers=_headers("USER002"),
    )

    assert response.status_code == 404

    # 确认数据还在
    check = client.get(
        f"/conversations/{conversation_id}",
        headers=_headers(),
    )

    assert check.status_code == 200

    clear_conversation(conversation_id)


def test_chat_rejects_empty_message():
    response = client.post(
        "/chat",
        headers=_headers(),
        json={
            "message": "",
        },
    )

    assert response.status_code == 422


def test_chat_end_to_end_without_mocking_agent(monkeypatch):
    """不 mock run_agent，验证 /chat 的真实调用链能跑通。

    为什么要专门写这一条：

        上面所有用例都把 run_agent 替换成了假的，
        结果就是「/chat 传给 run_agent 的参数，从来没被真的接收过」。
        with_request_context 需要 pop 掉 _include_request_metadata，
        而外层装饰器会按函数签名做 bind_partial——
        只要签名里没这个参数就抛 TypeError，整个接口 500/503。

        这个 bug 一直潜伏着，直到手工调接口才炸出来。

    为了不产生模型调用，把意图分类打桩成「没听懂」，
    流程走到澄清分支就返回，不进入模型请求。
    """

    from app.agent import agent as agent_module
    from app.agent.intent_classifier import IntentResult

    monkeypatch.setattr(
        agent_module,
        "classify_intent",
        lambda *args, **kwargs: IntentResult(
            intent="unclear",
            confidence=0.0,
            need_clarification=True,
            clarification_question="请问你想查询订单、物流，还是售后政策？",
        ),
    )

    conversation_id = "e2e-chat-real"

    response = client.post(
        "/chat",
        headers=_headers(),
        json={
            "message": "你好",
            "conversation_id": conversation_id,
        },
    )

    assert response.status_code == 200

    body = response.json()

    # 响应结构完整：说明 _include_request_metadata 那条链路是通的
    assert body["conversation_id"] == conversation_id
    assert body["request_id"]
    assert body["user_id"] == "USER001"
    assert body["answer"]

    clear_conversation(conversation_id)


def test_chat_returns_503_when_agent_breaks(monkeypatch):
    """Agent 彻底崩了也要返回结构化错误，不能裸 500。

    run_agent 内部已经兜了一层，这里再把它整个替换成抛异常，
    验证接口层还有第二道防线。
    """

    import app.api.chat as chat_module

    def _boom(*args, **kwargs):
        raise RuntimeError("模拟系统级故障")

    monkeypatch.setattr(chat_module, "run_agent", _boom)

    response = client.post(
        "/chat",
        headers={"X-User-Id": "USER-ERR"},
        json={"message": "测试"},
    )

    assert response.status_code == 503
    assert response.json()["detail"]
