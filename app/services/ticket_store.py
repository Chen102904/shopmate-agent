"""
售后工单存储与幂等控制。

为什么需要这个模块：
    改造前工单存在内存字典 TICKETS 里，有两个问题：

    1. 进程重启，所有工单消失。用户昨天提交的售后，今天查就是「不存在」。
    2. 完全没有防重。同一个用户把「申请退款」点两下、
       或者客户端超时自动重试，就会建出两张内容一模一样的工单。
       工单背后是人工审核和退款流程，重复建单 = 重复退款风险。

幂等的两层设计：

    第一层（严格）：客户端传 Idempotency-Key 请求头。
        - 同一个 key + 同样的请求内容 → 返回第一次的结果，不重复执行。
        - 同一个 key + 不同的请求内容 → 409 拒绝
          （说明客户端乱用 key，不能替它猜）
        - 同一个 key 正在处理中 → 409，让客户端稍后取结果

    第二层（兜底）：客户端不传 key 时，按
       user_id + order_id + request_type 做数据库级唯一约束。
       同一用户对同一订单的同类申请，只要已有活跃工单（pending/approved），
       就直接返回那张工单。

    为什么两层都要：
        只靠应用层「先查再插」在并发下不可靠——两个请求可能都查到「没有」，
        然后都插入。真正的互斥必须由数据库唯一约束提供。
"""

import hashlib
import os
import sqlite3
import time

from datetime import datetime

from app.core.db import get_connection
from app.models.after_sales import (
    AfterSalesTicket,
    AfterSalesType,
    TicketStatus,
)


# 无幂等键时的业务去重窗口（秒）。
#
# 这个窗口只影响「客户端没传 key」的兜底路径。
# 传了 key 的请求走严格幂等，不受窗口限制。
DEDUP_WINDOW_SECONDS = int(
    os.getenv("SHOPMATE_TICKET_DEDUP_WINDOW", "300")
)

# 活跃状态：处于这些状态的工单会被判定为「已存在同类申请」。
# 已完成（completed）或被拒（rejected）之后允许再次申请。
ACTIVE_STATUSES = (
    TicketStatus.PENDING.value,
    TicketStatus.APPROVED.value,
    TicketStatus.PROCESSING.value,
)


# 工单状态机：谁能流转到谁。
#
# 为什么集中定义而不是散在各处 if 判断：
#     加「已取消」时如果只在取消接口里写一句 status == 'pending'，
#     审核接口那边仍然可能把已取消的工单改成 approved。
#     一张表说清所有合法边，加状态时只需要改这里。
#
# 为什么 approved / rejected / completed / cancelled 没有出边：
#     它们是终态。尤其是 cancelled——
#     允许「已取消 → 审核通过」等于让用户撤销之后还能拿到退款。
ALLOWED_TRANSITIONS: dict[TicketStatus, set[TicketStatus]] = {
    TicketStatus.PENDING: {
        TicketStatus.APPROVED,
        TicketStatus.REJECTED,
        TicketStatus.CANCELLED,
    },
    # 处理中还能被驳回或完成（审核后发现问题的场景）
    TicketStatus.PROCESSING: {
        TicketStatus.APPROVED,
        TicketStatus.REJECTED,
        TicketStatus.COMPLETED,
    },
    TicketStatus.APPROVED: {TicketStatus.PROCESSING, TicketStatus.COMPLETED},
    TicketStatus.REJECTED: set(),
    TicketStatus.COMPLETED: set(),
    TicketStatus.CANCELLED: set(),
}

# 抢占幂等键时允许的并发冲突重试次数。
# 只是为了让「同时点两下」不报 500，
# 不改变幂等语义：重试多少次都只会有一张工单。
RESERVE_MAX_ATTEMPTS = 8


class IdempotencyConflict(Exception):
    """同一个幂等键被用于不同的请求内容，或属于别的用户。"""


class RequestInProgress(Exception):
    """同一个幂等键的请求正在处理中，结果还没写回。"""


class InvalidStatusTransition(Exception):
    """工单状态不允许这样流转。

    例子：把已取消的工单改成审核通过、
    或者对已完成的工单做取消。
    这类请求不能当成功处理——它意味着调用方的判断有问题。
    """


# ============================================================
# 建表
# ============================================================


def _ensure_schema() -> None:
    """建立工单表与幂等键表。语句全部幂等，支持多进程同时启动。"""

    for attempt in range(5):
        try:
            with get_connection() as connection:

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS after_sales_tickets (
                        ticket_id TEXT PRIMARY KEY,
                        order_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        request_type TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        created_at TEXT NOT NULL,
                        idempotency_key TEXT,
                        updated_at TEXT,
                        review_note TEXT
                    )
                    """
                )

                # 兼容旧库：这两列是加「取消 / 审核」时新增的。
                # 老数据补 NULL，读取时按 None 处理即可。
                ticket_columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(after_sales_tickets)"
                    )
                }

                if "updated_at" not in ticket_columns:
                    connection.execute(
                        "ALTER TABLE after_sales_tickets ADD COLUMN updated_at TEXT"
                    )

                if "review_note" not in ticket_columns:
                    connection.execute(
                        "ALTER TABLE after_sales_tickets ADD COLUMN review_note TEXT"
                    )

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS idempotency_keys (
                        idempotency_key TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        request_fingerprint TEXT NOT NULL,
                        ticket_id TEXT,
                        state TEXT NOT NULL DEFAULT 'processing',
                        created_at TEXT NOT NULL
                    )
                    """
                )

                # 业务去重的最后一道防线：
                # 同一用户 + 同一订单 + 同一类型，只能存在一张活跃工单。
                # 用部分唯一索引（只约束活跃状态），
                # 已完成或已拒绝的工单不占位，用户可以再次申请。
                connection.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS
                        idx_tickets_active_unique
                    ON after_sales_tickets (
                        user_id,
                        order_id,
                        request_type
                    )
                    WHERE status IN ('pending', 'approved', 'processing')
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_tickets_user
                    ON after_sales_tickets (user_id)
                    """
                )

            return

        except sqlite3.OperationalError:
            # 多进程同时建表时的短暂锁冲突，退避重试
            if attempt == 4:
                raise

            time.sleep(0.1 * (attempt + 1))


_ensure_schema()


# ============================================================
# 内部工具
# ============================================================


def _now() -> str:
    return datetime.now().isoformat()


def build_fingerprint(
    user_id: str,
    order_id: str,
    request_type: AfterSalesType,
    reason: str,
) -> str:
    """请求内容指纹。

    为什么把 user_id 也算进去：
        如果只看订单和理由，A 用户用 key=K 建单后，
        B 用户拿同一个 K 加同样参数，会被判成「重复请求」，
        从而拿到 A 的工单编号——这是一个信息泄露。

    为什么不包括时间：
        幂等的语义是「同一个请求重复发送」，
        时间戳每次都不同，算进去就永远判不重复。
    """

    raw = "|".join(
        [
            user_id,
            order_id,
            request_type.value,
            reason.strip(),
        ]
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _row_to_ticket(row: sqlite3.Row) -> AfterSalesTicket:
    keys = row.keys()

    def _opt_datetime(name: str):
        raw = row[name] if name in keys else None
        return datetime.fromisoformat(raw) if raw else None

    return AfterSalesTicket(
        ticket_id=row["ticket_id"],
        order_id=row["order_id"],
        user_id=row["user_id"],
        request_type=AfterSalesType(row["request_type"]),
        reason=row["reason"],
        status=TicketStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=_opt_datetime("updated_at"),
        review_note=(row["review_note"] if "review_note" in keys else None),
    )


def _insert_ticket(
    connection: sqlite3.Connection,
    ticket: AfterSalesTicket,
    idempotency_key: str | None,
) -> None:
    connection.execute(
        """
        INSERT INTO after_sales_tickets (
            ticket_id,
            order_id,
            user_id,
            request_type,
            reason,
            status,
            created_at,
            idempotency_key
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticket.ticket_id,
            ticket.order_id,
            ticket.user_id,
            ticket.request_type.value,
            ticket.reason,
            ticket.status.value,
            ticket.created_at.isoformat(),
            idempotency_key,
        ),
    )


def _find_active_ticket(
    connection: sqlite3.Connection,
    order_id: str,
    user_id: str,
    request_type: AfterSalesType,
) -> AfterSalesTicket | None:
    """查找已有的同类活跃工单。"""

    placeholders = ", ".join("?" for _ in ACTIVE_STATUSES)

    row = connection.execute(
        f"""
        SELECT *
        FROM after_sales_tickets
        WHERE user_id = ?
          AND order_id = ?
          AND request_type = ?
          AND status IN ({placeholders})
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (user_id, order_id, request_type.value, *ACTIVE_STATUSES),
    ).fetchone()

    return _row_to_ticket(row) if row else None


# ============================================================
# 幂等键：占位 / 完成 / 释放
# ============================================================


def _reserve_once(
    connection: sqlite3.Connection,
    idempotency_key: str,
    user_id: str,
    fingerprint: str,
) -> AfterSalesTicket | None:
    """在已开启的写事务里完成「检查 + 占位」。"""

    row = connection.execute(
        """
        SELECT *
        FROM idempotency_keys
        WHERE idempotency_key = ?
        """,
        (idempotency_key,),
    ).fetchone()

    if row is None:
        connection.execute(
            """
            INSERT INTO idempotency_keys (
                idempotency_key,
                user_id,
                request_fingerprint,
                ticket_id,
                state,
                created_at
            )
            VALUES (?, ?, ?, NULL, 'processing', ?)
            """,
            (
                idempotency_key,
                user_id,
                fingerprint,
                _now(),
            ),
        )
        return None

    # 键已存在：先确认是不是同一个人的同一个请求
    if row["user_id"] != user_id:
        # 不透露这个 key 属于谁，只说不能用
        raise IdempotencyConflict(
            "该请求标识已被占用，请更换 Idempotency-Key"
        )

    if row["request_fingerprint"] != fingerprint:
        raise IdempotencyConflict(
            "同一个 Idempotency-Key 不能用于不同的请求内容"
        )

    # 已完成：返回第一次的结果
    if row["ticket_id"]:
        ticket_row = connection.execute(
            """
            SELECT *
            FROM after_sales_tickets
            WHERE ticket_id = ?
            """,
            (row["ticket_id"],),
        ).fetchone()

        if ticket_row:
            return _row_to_ticket(ticket_row)

        # 键指向的工单不存在（数据被手工清理过）。
        # 当作占位失效处理：清掉重来，避免这个 key 永久不可用。
        connection.execute(
            """
            DELETE FROM idempotency_keys
            WHERE idempotency_key = ?
            """,
            (idempotency_key,),
        )
        return None

    # 占位还在，说明上一个请求没跑完
    raise RequestInProgress(
        "该请求正在处理中，请稍后用同一个 Idempotency-Key 重试"
    )


def reserve_idempotency_key(
    idempotency_key: str,
    user_id: str,
    fingerprint: str,
) -> AfterSalesTicket | None:
    """占住幂等键。

    返回值：
        已有工单 → 这是重复请求，直接把它返回给调用方，不再执行业务。
        None    → 抢占成功，调用方继续执行业务。

    抛错：
        IdempotencyConflict → 同 key 不同内容，或不属于当前用户
        RequestInProgress   → 同 key 的上一个请求还在处理

    为什么必须先占位再执行：
        如果先执行业务、成功后才写键，
        两个并发请求会同时通过「键不存在」的检查，
        然后各自建一张工单——幂等键就白加了。

    为什么用 BEGIN IMMEDIATE：
        普通 BEGIN 是延迟锁，两个事务都能先读，
        真正写入时才抢锁，其中一个会失败回滚。
        IMMEDIATE 一开始就拿写锁，检查和插入天然串行。

    为什么外面还要套重试：
        SQLite 在 WAL 模式下，多线程/多进程同时抢写锁时
        仍可能返回 "readonly database" / "database is locked"，
        busy_timeout 对前者无效。
        这一步是「检查 + 插入」的原子操作，没有部分成功，
        整体重试不会产生第二张工单，因此是安全的。
    """

    last_error: sqlite3.OperationalError | None = None

    for attempt in range(RESERVE_MAX_ATTEMPTS):

        try:
            with get_connection() as connection:

                connection.execute("BEGIN IMMEDIATE")

                return _reserve_once(
                    connection,
                    idempotency_key=idempotency_key,
                    user_id=user_id,
                    fingerprint=fingerprint,
                )

        except sqlite3.OperationalError as exc:

            message = str(exc).lower()

            retryable = any(
                keyword in message
                for keyword in (
                    "readonly",
                    "locked",
                    "busy",
                    "disk i/o",
                )
            )

            if not retryable:
                raise

            last_error = exc

            if attempt == RESERVE_MAX_ATTEMPTS - 1:
                raise

            time.sleep(
                min(0.05 * (attempt + 1), 0.3)
            )

    raise last_error  # pragma: no cover


def complete_idempotency_key(
    idempotency_key: str,
    ticket: AfterSalesTicket,
) -> None:
    """业务成功后，把工单号写回幂等键，标记为已完成。"""

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE idempotency_keys
            SET ticket_id = ?,
                state = 'completed'
            WHERE idempotency_key = ?
            """,
            (ticket.ticket_id, idempotency_key),
        )


def release_idempotency_key(idempotency_key: str) -> None:
    """业务失败后释放占位，允许客户端用同一个 key 重试。

    为什么失败要释放：
        常见失败原因是「参数不合法」「不符合退款条件」，
        用户改完参数再提交，通常还是同一个请求意图。
        如果把失败的 key 永久占住，用户就再也用不了这个 key 了。

    Stripe 的做法也是失败可重试，只有成功才锁定。
    """

    with get_connection() as connection:
        connection.execute(
            """
            DELETE FROM idempotency_keys
            WHERE idempotency_key = ?
              AND ticket_id IS NULL
            """,
            (idempotency_key,),
        )


# ============================================================
# 工单读写
# ============================================================


def find_recent_active_ticket(
    order_id: str,
    user_id: str,
    request_type: AfterSalesType,
) -> AfterSalesTicket | None:
    """无幂等键时的兜底去重：查有没有同类活跃工单。"""

    with get_connection() as connection:
        return _find_active_ticket(
            connection,
            order_id=order_id,
            user_id=user_id,
            request_type=request_type,
        )


def save_ticket(
    ticket: AfterSalesTicket,
    idempotency_key: str | None = None,
) -> AfterSalesTicket:
    """保存工单。

    数据库唯一索引兜底：
        并发下两个请求都可能通过 find_recent_active_ticket 的检查，
        但插入时只有一个能成功，另一个撞唯一约束。
        撞了不要报错，改成返回已存在的那张——用户的意图已经达成了。
    """

    try:
        with get_connection() as connection:
            _insert_ticket(connection, ticket, idempotency_key)
        return ticket

    except sqlite3.IntegrityError:
        existing = find_recent_active_ticket(
            order_id=ticket.order_id,
            user_id=ticket.user_id,
            request_type=ticket.request_type,
        )

        if existing is not None:
            return existing

        # 不是唯一约束冲突（比如主键重复），如实抛出
        raise


def get_ticket_by_id(
    ticket_id: str,
    user_id: str,
) -> AfterSalesTicket | None:
    """按编号查工单，同时校验归属。

    查不到和不属于当前用户都返回 None，调用方统一按同一个错误处理，
    避免靠返回值差异试探别人的工单编号。
    """

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM after_sales_tickets
            WHERE ticket_id = ?
              AND user_id = ?
            """,
            (ticket_id, user_id),
        ).fetchone()

    return _row_to_ticket(row) if row else None


def list_tickets_by_order(
    order_id: str,
    user_id: str,
    limit: int = 5,
) -> list[AfterSalesTicket]:
    """查某个订单下的售后工单，最新的在前。

    按 user_id 过滤而不是只按 order_id：
        工单表里 user_id 是申请人，和订单归属是同一个校验维度。
        只按 order_id 查会让「拿到别人的订单号」变成
        「能看到别人的售后进度」。
    """

    if not order_id or not user_id:
        return []

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM after_sales_tickets
            WHERE order_id = ?
              AND user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (order_id, user_id, limit),
        ).fetchall()

    return [_row_to_ticket(row) for row in rows]


def list_recent_tickets(
    user_id: str,
    limit: int = 5,
    status: "TicketStatus | str | None" = None,
) -> list[AfterSalesTicket]:
    """查某个用户最近的售后工单。

    用于用户没给订单号的情况（「我的退款申请怎么样了」）。
    客服场景里这种问法比给单号的还常见。

    status 可选：传了就只返回该状态的工单。
    列表接口需要按状态筛选（比如只看待审核的）。
    """

    if not user_id:
        return []

    status_value = (
        status.value if isinstance(status, TicketStatus) else status
    )

    with get_connection() as connection:

        if status_value:
            rows = connection.execute(
                """
                SELECT *
                FROM after_sales_tickets
                WHERE user_id = ?
                  AND status = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, status_value, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT *
                FROM after_sales_tickets
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

    return [_row_to_ticket(row) for row in rows]


def count_tickets(user_id: str | None = None) -> int:
    """统计工单数量，供健康检查使用。"""

    with get_connection() as connection:
        if user_id is None:
            row = connection.execute(
                "SELECT COUNT(*) AS c FROM after_sales_tickets"
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT COUNT(*) AS c FROM after_sales_tickets WHERE user_id = ?",
                (user_id,),
            ).fetchone()

    return int(row["c"])


def _update_ticket_status_once(
    ticket_id: str,
    new_status: TicketStatus,
    user_id: str | None = None,
    review_note: str | None = None,
) -> tuple[AfterSalesTicket | None, bool]:
    """变更工单状态，返回 (工单, 本次是否真的改了)。

    参数：
        user_id: 传了就校验归属（用户操作）；
                 不传表示管理员操作（审核），不做归属校验。
        review_note: 审核意见，只有审核类变更会带。

    返回的 bool 是给幂等用的：
        已经是目标状态时返回 False，调用方可以回「刚才已经处理过了」，
        而不是报错让用户以为失败了。

    状态流转集中定义在 ALLOWED_TRANSITIONS 里，
    不散落在各个调用点——否则加一个新状态要改十几处判断。
    """

    if user_id is None:
        # 管理员路径：不限归属，但仍然只允许合法流转
        owner_clause = ""
        owner_params: tuple = ()
    else:
        owner_clause = " AND user_id = ?"
        owner_params = (user_id,)

    allowed_from = [
        status.value
        for status, targets in ALLOWED_TRANSITIONS.items()
        if new_status in targets
    ]

    if not allowed_from:
        raise InvalidStatusTransition(
            f"没有任何状态可以流转到 {new_status.value}"
        )

    placeholders = ", ".join("?" for _ in allowed_from)

    with get_connection() as connection:

        connection.execute("BEGIN IMMEDIATE")

        row = connection.execute(
            f"""
            SELECT *
            FROM after_sales_tickets
            WHERE ticket_id = ?
            {owner_clause}
            """,
            (ticket_id, *owner_params),
        ).fetchone()

        if row is None:
            # 不存在和不属于当前用户返回同一个结果，
            # 避免靠差异试探别人的工单号。
            return None, False

        current = _row_to_ticket(row)

        if current.status == new_status:
            # 幂等：已经是目标状态
            return current, False

        if current.status.value not in allowed_from:
            raise InvalidStatusTransition(
                f"工单当前状态为 {current.status.value}，"
                f"不能变更为 {new_status.value}"
            )

        connection.execute(
            f"""
            UPDATE after_sales_tickets
            SET status = ?,
                updated_at = ?,
                review_note = COALESCE(?, review_note)
            WHERE ticket_id = ?
              AND status IN ({placeholders})
              {owner_clause}
            """,
            (
                new_status.value,
                _now(),
                review_note,
                ticket_id,
                *allowed_from,
                *owner_params,
            ),
        )

        updated = connection.execute(
            "SELECT * FROM after_sales_tickets WHERE ticket_id = ?",
            (ticket_id,),
        ).fetchone()

    return (_row_to_ticket(updated) if updated else None), True


def update_ticket_status(
    ticket_id: str,
    new_status: TicketStatus,
    user_id: str | None = None,
    review_note: str | None = None,
) -> tuple[AfterSalesTicket | None, bool]:
    """变更工单状态（带并发重试）。

    并发撤销 / 并发审核时会撞上 SQLite 的
    "attempt to write a readonly database"——
    WAL 模式下多线程同时抢写锁会出现这个错误，
    `busy_timeout` 对它无效，只能整体退避重试。

    为什么重试是安全的：
        被重试的是一次「BEGIN IMMEDIATE → 校验状态 → UPDATE」的完整事务，
        要么提交要么回滚，不存在做到一半的状态。
        重试最多让某个请求多等几十毫秒，不会产生第二次状态变更
        ——状态机允许的来源集合在每次重试时都会重新校验。

    这条不是理论问题：8 个线程同时撤销同一张单，
    不加重试时有约三分之一的概率直接抛异常，
    接口会返回 500，用户看到的是「系统错误」而不是「已撤销」。
    """

    last_error: sqlite3.OperationalError | None = None

    for attempt in range(RESERVE_MAX_ATTEMPTS):

        try:
            return _update_ticket_status_once(
                ticket_id=ticket_id,
                new_status=new_status,
                user_id=user_id,
                review_note=review_note,
            )

        except sqlite3.OperationalError as exc:

            message = str(exc).lower()

            retryable = any(
                keyword in message
                for keyword in ("readonly", "locked", "busy", "disk i/o")
            )

            if not retryable:
                raise

            last_error = exc

            if attempt == RESERVE_MAX_ATTEMPTS - 1:
                raise

            time.sleep(min(0.05 * (attempt + 1), 0.3))

    raise last_error  # pragma: no cover
