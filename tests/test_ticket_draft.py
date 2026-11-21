"""工单草稿存储层测试。

测的是状态机和归属约束，不涉及模型：
    open → consuming → consumed
    归属绑定 user + conversation
    过期、僵死、并发、幂等重放
"""

import pytest

from datetime import datetime, timedelta

from app.agent.ticket_draft import (
    DraftBusy,
    DraftExpired,
    DraftNotFound,
    complete_draft,
    count_drafts,
    cleanup_expired_drafts,
    create_draft,
    get_draft,
    release_draft,
    reserve_draft,
)
from app.core.db import get_connection


USER = "USER001"
OTHER_USER = "USER002"
CONVERSATION = "conv-001"
OTHER_CONVERSATION = "conv-002"


def _make_draft(
    draft_id: str = "drf_test001",
    user_id: str = USER,
    conversation_id: str = CONVERSATION,
):
    return create_draft(
        draft_id=draft_id,
        conversation_id=conversation_id,
        user_id=user_id,
        order_id="ORD001",
        request_type="refund",
        reason="商品破损",
    )


def _expire(draft_id: str) -> None:
    """把草稿的过期时间改到过去。"""

    past = (datetime.now() - timedelta(seconds=1)).isoformat()

    with get_connection() as connection:
        connection.execute(
            "UPDATE ticket_drafts SET expires_at = ? WHERE draft_id = ?",
            (past, draft_id),
        )


# ============================================================
# 创建与读取
# ============================================================


def test_create_then_get():
    draft = _make_draft()

    loaded = get_draft(draft.draft_id, USER, CONVERSATION)

    assert loaded.order_id == "ORD001"
    assert loaded.request_type == "refund"
    assert loaded.status == "open"
    assert loaded.ticket_id is None


def test_expires_in_seconds_is_positive():
    draft = _make_draft()

    assert 0 < draft.expires_in_seconds() <= 600


# ============================================================
# 归属约束
# ============================================================


def test_other_user_cannot_read_draft():
    """草稿是私有凭证，别人拿到 id 也读不到。"""

    _make_draft()

    with pytest.raises(DraftNotFound):
        get_draft("drf_test001", OTHER_USER, CONVERSATION)


def test_other_conversation_cannot_read_draft():
    """换个会话也用不了同一张草稿。

    否则用户在 A 会话生成的草稿，
    能在 B 会话被模型拿去确认，上下文就串了。
    """

    _make_draft()

    with pytest.raises(DraftNotFound):
        get_draft("drf_test001", USER, OTHER_CONVERSATION)


def test_missing_draft_raises_not_found():
    with pytest.raises(DraftNotFound):
        get_draft("drf_not_exist", USER, CONVERSATION)


def test_error_message_does_not_leak_existence():
    """「不存在」和「不是你的」必须给同一句话，否则可以靠错误信息试探。"""

    _make_draft()

    with pytest.raises(DraftNotFound) as not_mine:
        get_draft("drf_test001", OTHER_USER, CONVERSATION)

    with pytest.raises(DraftNotFound) as missing:
        get_draft("drf_nope", OTHER_USER, CONVERSATION)

    assert str(not_mine.value) == str(missing.value)


# ============================================================
# 抢占状态机
# ============================================================


def test_reserve_moves_to_consuming():
    _make_draft()

    draft = reserve_draft("drf_test001", USER, CONVERSATION)

    assert draft.status == "consuming"


def test_second_reserve_is_busy():
    """同一个草稿被两个请求同时确认时，第二个要被挡住。"""

    _make_draft()

    reserve_draft("drf_test001", USER, CONVERSATION)

    with pytest.raises(DraftBusy):
        reserve_draft("drf_test001", USER, CONVERSATION)


def test_reserve_after_complete_replays():
    """已经建过单的草稿，再确认一次返回 consumed 状态（幂等重放）。"""

    _make_draft()

    reserve_draft("drf_test001", USER, CONVERSATION)
    complete_draft("drf_test001", "ticket-abc")

    draft = reserve_draft("drf_test001", USER, CONVERSATION)

    assert draft.status == "consumed"
    assert draft.ticket_id == "ticket-abc"


def test_release_allows_retry_after_failure():
    """建单失败释放后，用户能重新确认。"""

    _make_draft()

    reserve_draft("drf_test001", USER, CONVERSATION)
    release_draft("drf_test001")

    draft = reserve_draft("drf_test001", USER, CONVERSATION)

    assert draft.status == "consuming"


def test_release_does_not_undo_completed_draft():
    """已完成的草稿不能被 release 打回 open，否则会重复建单。"""

    _make_draft()

    reserve_draft("drf_test001", USER, CONVERSATION)
    complete_draft("drf_test001", "ticket-abc")
    release_draft("drf_test001")

    loaded = get_draft("drf_test001", USER, CONVERSATION)

    assert loaded.status == "consumed"
    assert loaded.ticket_id == "ticket-abc"


# ============================================================
# 过期
# ============================================================


def test_expired_draft_cannot_be_reserved():
    _make_draft()
    _expire("drf_test001")

    with pytest.raises(DraftExpired):
        reserve_draft("drf_test001", USER, CONVERSATION)


def test_stale_consuming_draft_can_be_reclaimed():
    """进程在「抢占成功」和「写回工单号」之间崩溃时，草稿会卡在 consuming。

    超过僵死阈值后必须允许重新抢占，
    否则这张草稿永远确认不了。
    """

    _make_draft()
    reserve_draft("drf_test001", USER, CONVERSATION)

    # 把 created_at 往前推，模拟长时间卡住
    long_ago = (
        datetime.now() - timedelta(seconds=120)
    ).isoformat()

    with get_connection() as connection:
        connection.execute(
            "UPDATE ticket_drafts SET created_at = ? WHERE draft_id = ?",
            (long_ago, "drf_test001"),
        )

    draft = reserve_draft("drf_test001", USER, CONVERSATION)

    assert draft.status == "consuming"


# ============================================================
# 清理
# ============================================================


def test_cleanup_removes_only_expired_and_unconsumed():
    _make_draft("drf_open")
    _make_draft("drf_done")

    reserve_draft("drf_done", USER, CONVERSATION)
    complete_draft("drf_done", "ticket-x")

    _expire("drf_open")
    _expire("drf_done")

    removed = cleanup_expired_drafts()

    assert removed == 1
    assert count_drafts() == 1


def test_count_drafts_filters_by_user():
    _make_draft("drf_a", user_id=USER)
    _make_draft("drf_b", user_id=OTHER_USER)

    assert count_drafts() == 2
    assert count_drafts(USER) == 1
