"""工单生命周期测试：取消、审核、以及管理员权限。

这一组测的是「工单建完之后发生的事」，与 test_ticket_tools.py 的分工：
    那边测「怎么建单」（草稿 → 确认 → 落库）
    这边测「建完之后能不能撤、能不能审」

最要紧的两条：

    1. 用户取消后可以重新申请（唯一索引不能把 cancelled 当活跃）
    2. 普通用户绝不能自己审核通过自己的工单（自批退款）
"""

import pytest

from fastapi.testclient import TestClient

from app.core.db import get_connection
from app.models.after_sales import AfterSalesType, TicketStatus
from app.services.after_sales_service import (
    cancel_after_sales_ticket,
    create_after_sales_ticket,
    review_after_sales_ticket,
)
from app.services.ticket_store import InvalidStatusTransition


USER = "USER001"
OTHER_USER = "USER002"


def _create(
    order_id: str = "ORD001",
    user_id: str = USER,
    reason: str = "商品破损",
):
    return create_after_sales_ticket(
        order_id=order_id,
        user_id=user_id,
        request_type=AfterSalesType.REFUND,
        reason=reason,
    )


# ============================================================
# 取消
# ============================================================


def test_cancel_pending_ticket():
    ticket = _create()

    cancelled, changed = cancel_after_sales_ticket(ticket.ticket_id, USER)

    assert changed is True
    assert cancelled.status == TicketStatus.CANCELLED
    assert cancelled.updated_at is not None


def test_cancel_is_idempotent():
    """重复取消不算失败，否则客户端重试会看到报错。"""

    ticket = _create()
    cancel_after_sales_ticket(ticket.ticket_id, USER)

    again, changed = cancel_after_sales_ticket(ticket.ticket_id, USER)

    assert changed is False
    assert again.status == TicketStatus.CANCELLED


def test_cannot_cancel_other_users_ticket():
    """别人的工单返回 None，和「不存在」同一个结果。

    分开返回等于告诉对方「这个工单存在，只是不是你的」。
    """

    ticket = _create()

    result, changed = cancel_after_sales_ticket(ticket.ticket_id, OTHER_USER)

    assert result is None
    assert changed is False


def test_cancelled_ticket_frees_the_unique_slot():
    """取消之后可以重新申请。

    这条依赖唯一索引只约束活跃状态
    （pending / approved / processing）。
    如果 cancelled 也被算作活跃，
    用户撤销一次就再也申请不了了。
    """

    first = _create()
    cancel_after_sales_ticket(first.ticket_id, USER)

    second = _create(reason="重新考虑后还是要退")

    assert second.ticket_id != first.ticket_id
    assert second.status == TicketStatus.PENDING


def test_cannot_cancel_approved_ticket():
    """已经审核通过的不能再撤销。

    否则用户可以「先通过再撤销」，把审核记录搅乱。
    """

    ticket = _create()
    review_after_sales_ticket(ticket.ticket_id, "approved")

    with pytest.raises(InvalidStatusTransition):
        cancel_after_sales_ticket(ticket.ticket_id, USER)


# ============================================================
# 审核
# ============================================================


def test_review_approve():
    ticket = _create()

    reviewed, changed = review_after_sales_ticket(
        ticket.ticket_id,
        "approved",
        "材料齐全",
    )

    assert changed is True
    assert reviewed.status == TicketStatus.APPROVED
    assert reviewed.review_note == "材料齐全"


def test_review_reject_gets_default_note():
    """驳回时没写理由，要补一句默认说明。

    用户看到「已驳回」却没有任何原因，只能再来问客服。
    """

    ticket = _create()

    reviewed, _ = review_after_sales_ticket(ticket.ticket_id, "rejected")

    assert reviewed.status == TicketStatus.REJECTED
    assert reviewed.review_note


def test_review_is_idempotent():
    ticket = _create()
    review_after_sales_ticket(ticket.ticket_id, "approved")

    again, changed = review_after_sales_ticket(ticket.ticket_id, "approved")

    assert changed is False
    assert again.status == TicketStatus.APPROVED


def test_review_rejects_invalid_decision():
    ticket = _create()

    with pytest.raises(ValueError):
        review_after_sales_ticket(ticket.ticket_id, "completed")


def test_cannot_review_cancelled_ticket():
    """已取消的单不能再审核通过。

    允许的话等于「用户撤销了，但审核员又批了」，
    最终可能真的打款出去。
    """

    ticket = _create()
    cancel_after_sales_ticket(ticket.ticket_id, USER)

    with pytest.raises(InvalidStatusTransition):
        review_after_sales_ticket(ticket.ticket_id, "approved")


def test_review_unknown_ticket_returns_none():
    result, changed = review_after_sales_ticket(
        "00000000-0000-0000-0000-000000000000",
        "approved",
    )

    assert result is None
    assert changed is False


# ============================================================
# REST 接口
# ============================================================


@pytest.fixture
def client():
    """TestClient，并且每个用例前清空限流计数。

    为什么要清：
        限流是按 IP 记的，整个测试套件跑下来，
        前面的用例已经把额度用光，
        这里就会莫名其妙收到 429 —— 和被测逻辑完全无关。
    """

    import main

    from app.core.api_security import reset_rate_limits

    reset_rate_limits()

    return TestClient(main.app)


def test_list_tickets_only_own(client):
    _create(order_id="ORD001")
    _create(order_id="ORD003", user_id=OTHER_USER)

    response = client.get(
        "/after-sales",
        headers={"X-User-Id": USER},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["user_id"] == USER


def test_list_tickets_filter_by_status(client):
    ticket = _create(order_id="ORD001")
    cancel_after_sales_ticket(ticket.ticket_id, USER)

    cancelled = client.get(
        "/after-sales?status=cancelled",
        headers={"X-User-Id": USER},
    )
    pending = client.get(
        "/after-sales?status=pending",
        headers={"X-User-Id": USER},
    )

    assert len(cancelled.json()) == 1
    assert cancelled.json()[0]["ticket_id"] == ticket.ticket_id
    assert pending.json() == []


def test_cancel_endpoint(client):
    ticket = _create()

    response = client.post(
        f"/after-sales/{ticket.ticket_id}/cancel",
        headers={"X-User-Id": USER},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.headers["Idempotent-Replayed"] == "false"


def test_cancel_endpoint_is_idempotent(client):
    ticket = _create()
    client.post(
        f"/after-sales/{ticket.ticket_id}/cancel",
        headers={"X-User-Id": USER},
    )

    response = client.post(
        f"/after-sales/{ticket.ticket_id}/cancel",
        headers={"X-User-Id": USER},
    )

    assert response.status_code == 200
    assert response.headers["Idempotent-Replayed"] == "true"


def test_cancel_other_users_ticket_returns_404(client):
    ticket = _create()

    response = client.post(
        f"/after-sales/{ticket.ticket_id}/cancel",
        headers={"X-User-Id": OTHER_USER},
    )

    assert response.status_code == 404


def test_review_endpoint_hidden_when_not_configured(client, monkeypatch):
    """没配 ADMIN_TOKEN 时接口必须是 404。

    403 会暴露「这里有个审核接口」，
    404 让它在未启用时对外界完全不存在。
    """

    from app.api import after_sales as module

    monkeypatch.setattr(module, "ADMIN_TOKEN", "")

    ticket = _create()

    response = client.post(
        f"/after-sales/{ticket.ticket_id}/review",
        json={"decision": "approved"},
    )

    assert response.status_code == 404


def test_review_endpoint_rejects_wrong_token(client, monkeypatch):
    from app.api import after_sales as module

    monkeypatch.setattr(module, "ADMIN_TOKEN", "s3cret-token")

    ticket = _create()

    response = client.post(
        f"/after-sales/{ticket.ticket_id}/review",
        json={"decision": "approved"},
        headers={"X-Admin-Token": "wrong"},
    )

    assert response.status_code == 403


def test_review_endpoint_rejects_user_identity(client, monkeypatch):
    """关键安全用例：普通用户的身份头不能当管理员凭证用。

    如果这条挂了，任何用户都能审核通过自己的退款申请。
    """

    from app.api import after_sales as module

    monkeypatch.setattr(module, "ADMIN_TOKEN", "s3cret-token")

    ticket = _create()

    response = client.post(
        f"/after-sales/{ticket.ticket_id}/review",
        json={"decision": "approved"},
        headers={"X-User-Id": USER},
    )

    assert response.status_code == 403


def test_review_endpoint_works_with_admin_token(client, monkeypatch):
    from app.api import after_sales as module

    monkeypatch.setattr(module, "ADMIN_TOKEN", "s3cret-token")

    ticket = _create()

    response = client.post(
        f"/after-sales/{ticket.ticket_id}/review",
        json={"decision": "rejected", "review_note": "商品已使用"},
        headers={"X-Admin-Token": "s3cret-token"},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "rejected"
    assert body["review_note"] == "商品已使用"


def test_review_endpoint_rejects_bad_decision(client, monkeypatch):
    from app.api import after_sales as module

    monkeypatch.setattr(module, "ADMIN_TOKEN", "s3cret-token")

    ticket = _create()

    response = client.post(
        f"/after-sales/{ticket.ticket_id}/review",
        json={"decision": "退款吧"},
        headers={"X-Admin-Token": "s3cret-token"},
    )

    assert response.status_code == 400


def test_review_after_cancel_returns_409(client, monkeypatch):
    from app.api import after_sales as module

    monkeypatch.setattr(module, "ADMIN_TOKEN", "s3cret-token")

    ticket = _create()
    cancel_after_sales_ticket(ticket.ticket_id, USER)

    response = client.post(
        f"/after-sales/{ticket.ticket_id}/review",
        json={"decision": "approved"},
        headers={"X-Admin-Token": "s3cret-token"},
    )

    assert response.status_code == 409


def test_cancel_after_review_returns_409(client):
    """已审核的工单再撤销，必须是 409 而不是 500。

    真实踩过：取消接口没捕获状态冲突，
    异常直接穿透到全局处理器，用户收到 500「服务器内部错误」。
    这种情况是「不能这么做」，不是「服务器坏了」。
    """

    ticket = _create()
    review_after_sales_ticket(ticket.ticket_id, "approved", "材料齐全")

    response = client.post(
        f"/after-sales/{ticket.ticket_id}/cancel",
        headers={"X-User-Id": USER},
    )

    assert response.status_code == 409
