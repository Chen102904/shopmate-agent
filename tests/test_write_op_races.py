"""写操作边界验收：并发、竞争、过期。

这份测试和 `test_idempotency.py` / `test_ticket_draft.py` 的分工：

    那两份证明「分支写对了」——单线程依次调用，每道闸门都拦得住。
    这份证明「同时来两个人的时候，副作用只发生一次」。

为什么必须单独写：
    两阶段确认（draft → confirm）的核心承诺是「用户点两下也只建一张单」。
    这个承诺在单线程测试里是测不出来的：
    单线程永远走不到「两个请求同时通过检查」那条路。

覆盖六种情况：
    1. 同一草稿被 8 个线程同时确认
    2. 不带 draft_id 时 8 个线程同时确认
    3. 正在被消费的草稿不能被「僵死回收」抢走
    4. 过期草稿确认只会被拒，不会建单
    5. 先撤销、后确认，不能留下「用户以为撤了、其实单还在」
    6. 同一张工单被同时撤销 / 同时审核
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest

from app.agent.conversation_store import add_message
from app.agent.ticket_draft import (
    DraftBusy,
    cancel_pending_drafts,
    create_draft,
    reserve_draft,
)
from app.agent.ticket_tools import (
    cancel_after_sales_ticket,
    confirm_after_sales_ticket,
    draft_after_sales_ticket,
)
from app.core.db import get_connection
from app.models.after_sales import AfterSalesType
from app.services.after_sales_service import (
    cancel_after_sales_ticket as cancel_service,
    create_after_sales_ticket,
    review_after_sales_ticket,
)


USER = "USER001"
OTHER_USER = "USER002"

# 每个用例用独立会话，避免消息跨用例残留影响「最后一条用户消息」判定。
CONVERSATION = "conv-race-000"


@pytest.fixture(autouse=True)
def fresh_conversation():
    import uuid

    global CONVERSATION
    CONVERSATION = f"conv-race-{uuid.uuid4().hex[:8]}"
    yield


def _say(message: str) -> None:
    add_message(CONVERSATION, {"role": "user", "content": message})


def _ticket_count() -> int:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS c FROM after_sales_tickets"
        ).fetchone()
    return int(row["c"])


def _ticket_statuses() -> list[str]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT status FROM after_sales_tickets ORDER BY created_at"
        ).fetchall()
    return [row["status"] for row in rows]


def _make_draft(order_id: str = "ORD001", reason: str = "商品破损") -> str:
    result = draft_after_sales_ticket(
        order_id=order_id,
        request_type="refund",
        reason=reason,
        user_id=USER,
        conversation_id=CONVERSATION,
    )
    assert result.get("status") == "awaiting_confirmation", result
    return result["draft_id"]


def _age_draft(draft_id: str, seconds: int, status: str | None = None) -> None:
    """把草稿的创建时间往前拨，模拟「用户看了很久才点确认」。"""

    old = (datetime.now() - timedelta(seconds=seconds)).isoformat(timespec="seconds")

    with get_connection() as connection:
        if status:
            connection.execute(
                "UPDATE ticket_drafts SET created_at = ?, status = ? WHERE draft_id = ?",
                (old, status, draft_id),
            )
        else:
            connection.execute(
                "UPDATE ticket_drafts SET created_at = ? WHERE draft_id = ?",
                (old, draft_id),
            )


# ============================================================
# 1 / 2 并发确认
# ============================================================


def test_concurrent_confirm_same_draft_creates_one_ticket():
    """8 个线程拿同一个 draft_id 同时确认，只能建一张工单。"""

    draft_id = _make_draft()
    _say("确认")

    def confirm(_):
        return confirm_after_sales_ticket(
            user_id=USER,
            conversation_id=CONVERSATION,
            draft_id=draft_id,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(confirm, range(8)))

    assert _ticket_count() == 1

    statuses = [r.get("status") for r in results]
    # 允许：抢到的那个返回 submitted，其余 busy 或幂等重放
    assert set(statuses) <= {"submitted", "draft_busy"}
    assert statuses.count("submitted") >= 1

    # 只有一个线程是真正建单的（其余要么 busy，要么拿到幂等重放）
    real = [r for r in results if r.get("status") == "submitted" and not r.get("idempotent_replay")]
    assert len(real) == 1


def test_concurrent_confirm_without_draft_id_creates_one_ticket():
    """模型不传 draft_id（真实情况）时，并发确认同样只建一张。"""

    _make_draft()
    _say("确认，提交吧")

    def confirm(_):
        return confirm_after_sales_ticket(
            user_id=USER,
            conversation_id=CONVERSATION,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(confirm, range(8)))

    assert _ticket_count() == 1
    assert set(r.get("status") for r in results) <= {"submitted", "draft_busy"}


# ============================================================
# 3 僵死回收不能抢走「正在被消费」的草稿
# ============================================================


def test_draft_consumed_now_is_not_stolen_even_if_old():
    """草稿是 10 分钟前生成的，但刚刚被别的请求抢占。

    真实场景：用户看了 5 分钟才点确认，此时草稿的「创建时间」很早，
    但消费才刚开始。僵死判定如果拿创建时间算，
    第二个并发请求会以为「上一个执行者死了」，把草稿抢走。
    """

    draft_id = _make_draft()
    _say("确认")

    # 模拟：草稿很老，但刚被另一个请求抢占
    _age_draft(draft_id, seconds=600)
    reserve_draft(draft_id, USER, CONVERSATION)

    with pytest.raises(DraftBusy):
        reserve_draft(draft_id, USER, CONVERSATION)


def test_stale_consuming_draft_can_be_reclaimed():
    """真的僵死（消费开始时间也超过阈值）时，仍然允许回收。

    这一条和上一条是一对：不修就抢活人，修过头就永远回收不了。
    """

    draft_id = _make_draft()
    _say("确认")
    reserve_draft(draft_id, USER, CONVERSATION)

    # 把「消费开始时间」也拨老
    old = (datetime.now() - timedelta(seconds=600)).isoformat(timespec="seconds")
    with get_connection() as connection:
        connection.execute(
            "UPDATE ticket_drafts SET consuming_at = ? WHERE draft_id = ?",
            (old, draft_id),
        )

    draft = reserve_draft(draft_id, USER, CONVERSATION)
    assert draft.status == "consuming"


# ============================================================
# 4 过期草稿
# ============================================================


def test_expired_draft_creates_no_ticket():
    """草稿过期后确认，只能被拒，不能落库。"""

    draft_id = _make_draft()
    _say("确认")

    past = (datetime.now() - timedelta(seconds=5)).isoformat(timespec="seconds")
    with get_connection() as connection:
        connection.execute(
            "UPDATE ticket_drafts SET expires_at = ? WHERE draft_id = ?",
            (past, draft_id),
        )

    result = confirm_after_sales_ticket(
        user_id=USER,
        conversation_id=CONVERSATION,
        draft_id=draft_id,
    )

    assert result.get("status") == "draft_expired"
    assert _ticket_count() == 0


# ============================================================
# 5 撤销与确认的竞争
# ============================================================


def test_cancel_voids_pending_draft_so_later_confirm_does_nothing():
    """用户说「撤销」时申请还没提交。

    此时如果只回「没有可撤销的申请」，草稿还留着，
    模型下一轮完全可能把它确认掉——用户以为自己撤了，单却建出来了。
    """

    draft_id = _make_draft()
    _say("撤销")

    result = cancel_after_sales_ticket(
        user_id=USER,
        conversation_id=CONVERSATION,
        order_id="ORD001",
    )

    assert result.get("status") == "draft_cancelled", result

    # 用户再回一句「确认」，不应该建单
    _say("确认")
    confirm_after_sales_ticket(
        user_id=USER,
        conversation_id=CONVERSATION,
        draft_id=draft_id,
    )

    assert _ticket_count() == 0


def test_cancel_pending_drafts_is_scoped_to_user():
    """作废草稿只能作废自己的，不能顺手清掉别人的。"""

    mine = create_draft(
        draft_id="drf_scope_mine",
        conversation_id="conv-mine",
        user_id=USER,
        order_id="ORD001",
        request_type="refund",
        reason="商品破损",
    )

    theirs = create_draft(
        draft_id="drf_scope_theirs",
        conversation_id="conv-theirs",
        user_id=OTHER_USER,
        order_id="ORD001",
        request_type="refund",
        reason="商品破损",
    )

    removed = cancel_pending_drafts(
        conversation_id="conv-mine",
        user_id=USER,
        order_id="ORD001",
    )

    assert removed == 1

    with get_connection() as connection:
        rows = {
            row["draft_id"]: row["status"]
            for row in connection.execute(
                "SELECT draft_id, status FROM ticket_drafts"
            )
        }

    assert rows[mine.draft_id] == "cancelled"
    assert rows[theirs.draft_id] == "open"


# ============================================================
# 6 同一张工单被并发处理
# ============================================================


def _make_ticket() -> str:
    ticket = create_after_sales_ticket(
        order_id="ORD001",
        user_id=USER,
        request_type=AfterSalesType.REFUND,
        reason="商品破损",
    )
    return ticket.ticket_id


def test_concurrent_cancel_changes_status_once():
    """8 个线程同时撤销同一张单，真正生效的只有一次。"""

    ticket_id = _make_ticket()

    def cancel(_):
        _, changed = cancel_service(ticket_id, USER)
        return changed

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(cancel, range(8)))

    assert sum(1 for r in results if r) == 1
    assert _ticket_statuses() == ["cancelled"]


def test_concurrent_approve_changes_status_once():
    """8 个线程同时审核通过，真正生效的只有一次。"""

    ticket_id = _make_ticket()

    def approve(_):
        _, changed = review_after_sales_ticket(ticket_id, "approved")
        return changed

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(approve, range(8)))

    assert sum(1 for r in results if r) == 1
    assert _ticket_statuses() == ["approved"]


def test_cancel_and_approve_race_leaves_terminal_state():
    """撤销和审核同时发生，最后只能是其中一种终态，不能既撤又批。"""

    ticket_id = _make_ticket()

    def cancel(_):
        try:
            cancel_service(ticket_id, USER)
        except Exception:
            pass

    def approve(_):
        try:
            review_after_sales_ticket(ticket_id, "approved")
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(cancel, i) for i in range(4)]
        futures += [pool.submit(approve, i) for i in range(4)]
        for f in futures:
            f.result()

    statuses = _ticket_statuses()
    assert len(statuses) == 1
    assert statuses[0] in {"cancelled", "approved"}


# ============================================================
# 7 接口层：并发调用不能出现 500
# ============================================================


@pytest.fixture
def client():
    """TestClient，用例前清空限流计数（限流是每 IP 60 次/分）。"""

    import main

    from app.core.api_security import reset_rate_limits

    reset_rate_limits()

    from fastapi.testclient import TestClient

    return TestClient(main.app)


def test_concurrent_cancel_endpoint_never_returns_500(client):
    """8 个线程同时调 /after-sales/{id}/cancel。

    只要求「都不 500」和「最终状态是 cancelled」：
    其中一个是真正改状态的，其余命中幂等分支返回同样的结果。
    """

    ticket_id = _make_ticket()

    def hit(_):
        response = client.post(
            f"/after-sales/{ticket_id}/cancel",
            headers={"X-User-Id": USER},
        )
        return response.status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(hit, range(8)))

    assert 500 not in codes
    assert set(codes) <= {200, 409}
    assert _ticket_statuses() == ["cancelled"]
