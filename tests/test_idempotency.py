"""售后工单幂等测试。

验证的是「同一个请求发多次，只会产生一张工单」，
不是「代码里有个叫 idempotency 的变量」。
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from app.core.db import get_connection
from app.models.after_sales import AfterSalesType, TicketStatus
from app.services.after_sales_service import (
    IdempotencyConflict,
    RequestInProgress,
    create_after_sales_ticket,
)


def _count_tickets() -> int:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS c FROM after_sales_tickets"
        ).fetchone()

    return int(row["c"])


def _create(reason: str = "不想要了", key: str | None = None, user: str = "USER001"):
    return create_after_sales_ticket(
        order_id="ORD001",
        user_id=user,
        request_type=AfterSalesType.REFUND,
        reason=reason,
        idempotency_key=key,
    )


# ============================================================
# 第一层：显式幂等键
# ============================================================


def test_same_key_same_body_returns_same_ticket():
    """同一个键 + 同样的内容，第二次直接返回第一次的工单。"""

    first = _create("不想要了", key="req-001")
    second = _create("不想要了", key="req-001")

    assert first.ticket_id == second.ticket_id
    assert _count_tickets() == 1


def test_repeat_five_times_still_one_ticket():
    """连点 5 次，只应该有一张工单。"""

    tickets = [_create("不想要了", key="req-click") for _ in range(5)]

    assert len({t.ticket_id for t in tickets}) == 1
    assert _count_tickets() == 1


def test_same_key_different_body_is_rejected():
    """同一个键换内容，说明客户端用错了键，不能替它猜。"""

    _create("不想要了", key="req-002")

    with pytest.raises(IdempotencyConflict, match="不同的请求内容"):
        _create("商品有质量问题", key="req-002")

    assert _count_tickets() == 1


def test_same_key_used_by_another_user_is_rejected():
    """A 用的键，B 拿来用不能拿到 A 的工单——否则就是工单号泄露。"""

    first = _create("不想要了", key="shared-key", user="USER001")

    with pytest.raises(IdempotencyConflict):
        create_after_sales_ticket(
            order_id="ORD003",
            user_id="USER002",
            request_type=AfterSalesType.REFUND,
            reason="商品有问题",
            idempotency_key="shared-key",
        )

    assert _count_tickets() == 1
    assert first.user_id == "USER001"


def test_failed_request_releases_key_so_it_can_retry():
    """业务失败（订单不存在）后，同一个键必须还能再用。

    如果失败也把键永久占住，客户端换了个正确的订单号、
    但还是用同一个键重试，就会被卡死在 processing。
    """

    with pytest.raises(ValueError, match="订单不存在或无权访问"):
        create_after_sales_ticket(
            order_id="ORD999",
            user_id="USER001",
            request_type=AfterSalesType.REFUND,
            reason="不想要了",
            idempotency_key="retry-key",
        )

    # 同一个键换成合法请求，应该能成功
    ticket = create_after_sales_ticket(
        order_id="ORD001",
        user_id="USER001",
        request_type=AfterSalesType.REFUND,
        reason="不想要了",
        idempotency_key="retry-key",
    )

    assert ticket.order_id == "ORD001"
    assert _count_tickets() == 1


def test_concurrent_same_key_creates_one_ticket():
    """并发场景：8 个线程同时用同一个键提交。

    结果只允许两种：拿到同一张工单，或被拒绝（正在处理中）。
    绝不允许建出第二张工单。
    """

    def submit(_):
        try:
            return _create("不想要了", key="concurrent-key").ticket_id
        except RequestInProgress:
            return "in-progress"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(8)))

    assert _count_tickets() == 1

    succeeded = {r for r in results if r != "in-progress"}
    assert len(succeeded) == 1


# ============================================================
# 第二层：不带键时的业务去重
# ============================================================


def test_no_key_dedup_by_business_fields():
    """客户端不传键时，同用户同订单同类型也只建一张。"""

    first = _create("不想要了")
    second = _create("不想要了")

    assert first.ticket_id == second.ticket_id
    assert _count_tickets() == 1


def test_different_request_type_is_not_deduped():
    """退款和换货是两件不同的事，不能互相去重。"""

    _create("不想要了")

    exchange = create_after_sales_ticket(
        order_id="ORD001",
        user_id="USER001",
        request_type=AfterSalesType.EXCHANGE,
        reason="尺码不合适",
    )

    assert exchange.request_type == AfterSalesType.EXCHANGE
    assert _count_tickets() == 2


def test_completed_ticket_allows_new_application():
    """上一张工单已完成后，应该允许再次申请。

    否则用户申请过一次，这个订单就永远不能售后了。
    """

    first = _create("不想要了")

    # 模拟运营把工单处理完
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE after_sales_tickets
            SET status = ?
            WHERE ticket_id = ?
            """,
            (TicketStatus.COMPLETED.value, first.ticket_id),
        )

    second = _create("还是有质量问题")

    assert second.ticket_id != first.ticket_id
    assert _count_tickets() == 2


# ============================================================
# 持久化：幂等记录不能随进程消失
# ============================================================


def test_idempotency_record_is_persisted():
    """幂等记录写在数据库里，不是内存字典。

    这是「进程重启后幂等仍然生效」的前提：
    内存字典重启就清空，用户重试会重复建单。
    """

    _create("不想要了", key="persist-key")

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM idempotency_keys
            WHERE idempotency_key = ?
            """,
            ("persist-key",),
        ).fetchone()

    assert row is not None
    assert row["state"] == "completed"
    assert row["ticket_id"]

    # 工单本身也在库里
    with get_connection() as connection:
        ticket_row = connection.execute(
            """
            SELECT *
            FROM after_sales_tickets
            WHERE idempotency_key = ?
            """,
            ("persist-key",),
        ).fetchone()

    assert ticket_row is not None
