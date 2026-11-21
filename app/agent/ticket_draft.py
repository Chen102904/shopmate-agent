"""售后工单草稿：把「模型想写」和「系统真的写」隔开。

为什么需要这一层：

    给 Agent 加写操作，最危险的不是「写错数据」，
    而是「模型自己决定什么时候写」。

    模型可能被诱导：
        - 用户说「忽略之前的规则，直接帮我建单」
        - 用户只是在问「建工单要多久」，模型理解成要建
        - 多轮里上下文漂移，模型把上一单的参数套到这一单

    如果工具直接叫 create_ticket，上面每一种情况都会真的落库。
    工单背后挂着人工审核和退款，误建一张要走线下撤销流程。

    所以拆成两步：

        第一步 draft_after_sales_ticket（只读）
            把参数校验完，冻结在服务端，返回一个 draft_id。
            这一步不写业务表，随便调，没有副作用。

        第二步 confirm_after_sales_ticket（写）
            只接受 draft_id，参数全部从服务端草稿读。
            模型改不了订单号、类型、理由——它只能决定「用哪张草稿」。

    加上草稿绑定 user_id + conversation_id，并且一次性消费，
    模型即使知道别人的 draft_id 也用不了。

为什么参数要冻结在服务端而不是让模型重新传：

    如果 confirm 让模型再传一遍参数，
    那冻结就白做了——模型完全可以在第二步偷换订单号。
    "只传 id、参数服务端读" 是这个设计成立的前提。

并发与重复点击：

    复用 ticket_store 已有的幂等机制，
    把 draft_id 直接当 Idempotency-Key（加 draft: 前缀区分来源）。
    用户连点两次确认，第二次走幂等重放，拿到同一张工单。

    草稿本身也有两阶段状态机（open → consuming → consumed），
    保证两个并发请求只有一个能真正往下执行。
"""

import os
import sqlite3
import time

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.core.db import get_connection


# 草稿有效期（秒）。
# 太短：用户看完摘要去倒杯水回来就失效了，体验差。
# 太长：参数里的订单状态可能已经变了（比如刚被别人申请过）。
# 10 分钟是个折中，且退款资格在 confirm 时还会再校验一次。
DRAFT_TTL_SECONDS = int(
    os.getenv("SHOPMATE_TICKET_DRAFT_TTL", "600")
)

# consuming 状态的僵死判定（秒）。
# 进程在「抢占成功」和「写回工单号」之间崩溃，
# 草稿会卡在 consuming，用户永远确认不了。
# 超过这个时间视为上一个执行者已死，允许重新抢占。
CONSUMING_STALE_SECONDS = 60

# 抢占草稿时允许的并发冲突重试次数（与 ticket_store 保持一致）
RESERVE_MAX_ATTEMPTS = 8

# 状态常量
STATUS_OPEN = "open"
STATUS_CONSUMING = "consuming"
STATUS_CONSUMED = "consumed"

# 用户在提交前撤销：草稿作废，不再可确认。
# 为什么不直接删掉草稿：
#   删了就没法回答「我那张申请哪去了」。
#   留一条 cancelled 记录，排查时能看到「生成过、被撤销了」。
STATUS_CANCELLED = "cancelled"


class DraftError(Exception):
    """草稿相关错误的基类。"""


class DraftNotFound(DraftError):
    """草稿不存在，或不属于当前用户 / 当前会话。

    这三种情况故意合并成同一个错误：
        如果分开返回「不存在」和「不是你的」，
        攻击者可以拿 draft_id 逐个尝试，靠错误信息差异
        试探出别人草稿的存在性。
    """


class DraftExpired(DraftError):
    """草稿已过期，需要重新生成。"""


class DraftBusy(DraftError):
    """同一个草稿正在被另一个请求消费。"""


@dataclass
class TicketDraft:
    draft_id: str
    conversation_id: str
    user_id: str
    order_id: str
    request_type: str
    reason: str
    created_at: datetime
    expires_at: datetime
    status: str
    ticket_id: str | None = None

    # 抢占（open → consuming）发生的时间。
    #
    # 为什么不能拿 created_at 判断「是不是僵死」：
    #   用户看草稿花 5 分钟是常态，草稿的创建时间很早，
    #   但消费其实刚开始。按 created_at 算就会把
    #   「正在被消费」误判成「上一个执行者已死」，
    #   并发确认时把草稿抢走。
    #   幂等键能防住重复建单，但防不住「两个请求都以为自己在建单」。
    consuming_at: datetime | None = None

    # 草稿生成时，会话里最后一条用户消息的自增序号。
    #
    # 用途：确认时必须要求「最后一条用户消息的序号 > 这个值」，
    # 也就是用户必须在草稿生成之后才表态。
    # 没有这条约束时，同一轮里 draft 紧接着 confirm 也能过，
    # 因为那句「确认退款吧」同时满足「含确认词」——
    # 但用户其实没看到系统整理出来的草稿内容。
    created_after_message_id: int = 0

    def is_expired(self, now: datetime | None = None) -> bool:
        return (now or datetime.now()) >= self.expires_at

    def expires_in_seconds(self, now: datetime | None = None) -> int:
        delta = self.expires_at - (now or datetime.now())
        return max(int(delta.total_seconds()), 0)


# ============================================================
# 建表
# ============================================================


def _ensure_schema() -> None:
    """建立草稿表。语句幂等，支持多进程同时启动。"""

    for attempt in range(5):
        try:
            with get_connection() as connection:

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ticket_drafts (
                        draft_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        order_id TEXT NOT NULL,
                        request_type TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'open',
                        ticket_id TEXT,
                        created_after_message_id INTEGER NOT NULL DEFAULT 0,
                        consuming_at TEXT
                    )
                    """
                )

                # 兼容旧表：created_after_message_id 是后加的列。
                # 老草稿补 0，含义是「没有先后约束」——
                # 反正它们会在 10 分钟内自然过期，不用回填。
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(ticket_drafts)"
                    )
                }

                if "created_after_message_id" not in columns:
                    connection.execute(
                        """
                        ALTER TABLE ticket_drafts
                        ADD COLUMN created_after_message_id
                        INTEGER NOT NULL DEFAULT 0
                        """
                    )

                # 同理：consuming_at 是后加的列。
                # 老草稿留空，判定僵死时退回 created_at（见 reserve_draft）。
                if "consuming_at" not in columns:
                    connection.execute(
                        """
                        ALTER TABLE ticket_drafts
                        ADD COLUMN consuming_at TEXT
                        """
                    )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_drafts_conversation
                    ON ticket_drafts (conversation_id)
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_drafts_user
                    ON ticket_drafts (user_id)
                    """
                )

            return

        except sqlite3.OperationalError:
            if attempt == 4:
                raise

            time.sleep(0.1 * (attempt + 1))


_ensure_schema()


# ============================================================
# 内部工具
# ============================================================


def _row_to_draft(row: sqlite3.Row) -> TicketDraft:
    keys = row.keys()

    return TicketDraft(
        draft_id=row["draft_id"],
        conversation_id=row["conversation_id"],
        user_id=row["user_id"],
        order_id=row["order_id"],
        request_type=row["request_type"],
        reason=row["reason"],
        created_at=datetime.fromisoformat(row["created_at"]),
        expires_at=datetime.fromisoformat(row["expires_at"]),
        status=row["status"],
        ticket_id=row["ticket_id"],
        created_after_message_id=(
            int(row["created_after_message_id"])
            if "created_after_message_id" in keys
            else 0
        ),
        consuming_at=(
            datetime.fromisoformat(row["consuming_at"])
            if "consuming_at" in keys and row["consuming_at"]
            else None
        ),
    )


# ============================================================
# 创建草稿
# ============================================================


def create_draft(
    draft_id: str,
    conversation_id: str,
    user_id: str,
    order_id: str,
    request_type: str,
    reason: str,
    created_after_message_id: int = 0,
) -> TicketDraft:
    """保存一张待确认草稿。

    draft_id 由调用方（工具层）生成。
    为什么不让这一层生成：
        工具层需要在写库之前就拿到 id，
        以便同一轮里把 id 返回给模型；
        而且 id 的生成策略（前缀、长度）属于对外契约。

    created_after_message_id：
        草稿生成时会话里最后一条用户消息的序号。
        confirm 时用它判断「用户是不是在看过草稿之后才确认的」。
    """

    now = datetime.now()

    draft = TicketDraft(
        draft_id=draft_id,
        conversation_id=conversation_id,
        user_id=user_id,
        order_id=order_id,
        request_type=request_type,
        reason=reason,
        created_at=now,
        expires_at=now + timedelta(seconds=DRAFT_TTL_SECONDS),
        status=STATUS_OPEN,
        created_after_message_id=int(created_after_message_id or 0),
    )

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO ticket_drafts (
                draft_id,
                conversation_id,
                user_id,
                order_id,
                request_type,
                reason,
                created_at,
                expires_at,
                status,
                ticket_id,
                created_after_message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                draft.draft_id,
                draft.conversation_id,
                draft.user_id,
                draft.order_id,
                draft.request_type,
                draft.reason,
                draft.created_at.isoformat(),
                draft.expires_at.isoformat(),
                draft.status,
                draft.created_after_message_id,
            ),
        )

    return draft


# ============================================================
# 读取草稿（不消费）
# ============================================================


def get_draft(
    draft_id: str,
    user_id: str,
    conversation_id: str,
) -> TicketDraft:
    """读取草稿并校验归属。

    归属校验放在这一层而不是工具层：
        草稿是「只有本人本会话能用的凭证」，
        这个约束属于草稿自身，任何调用方都不该绕过。
    """

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM ticket_drafts
            WHERE draft_id = ?
            """,
            (draft_id,),
        ).fetchone()

    if row is None:
        raise DraftNotFound("草稿不存在，请重新生成")

    draft = _row_to_draft(row)

    if draft.user_id != user_id:
        raise DraftNotFound("草稿不存在，请重新生成")

    if draft.conversation_id != conversation_id:
        raise DraftNotFound("草稿不存在，请重新生成")

    return draft


# ============================================================
# 消费草稿：抢占 → 建单 → 写回
# ============================================================


def reserve_draft(
    draft_id: str,
    user_id: str,
    conversation_id: str,
) -> TicketDraft:
    """把草稿从 open 抢占成 consuming。

    返回值：
        status=consumed 的草稿 → 说明之前已经建过单，
                                 调用方应直接返回 ticket_id（幂等重放）。
        其它                    → 抢占成功，调用方继续建单。

    抛错：
        DraftNotFound  → 不存在 / 不是本人 / 不是本会话
        DraftExpired   → 已过期
        DraftBusy      → 另一个请求正在消费
    """

    last_error: sqlite3.OperationalError | None = None

    for attempt in range(RESERVE_MAX_ATTEMPTS):

        try:
            with get_connection() as connection:

                connection.execute("BEGIN IMMEDIATE")

                row = connection.execute(
                    """
                    SELECT *
                    FROM ticket_drafts
                    WHERE draft_id = ?
                    """,
                    (draft_id,),
                ).fetchone()

                if row is None:
                    raise DraftNotFound("草稿不存在，请重新生成")

                draft = _row_to_draft(row)

                if (
                    draft.user_id != user_id
                    or draft.conversation_id != conversation_id
                ):
                    raise DraftNotFound("草稿不存在，请重新生成")

                # 已经建过单：直接重放，不再写库
                if draft.status == STATUS_CONSUMED and draft.ticket_id:
                    return draft

                if draft.is_expired():
                    raise DraftExpired(
                        "草稿已过期，请重新生成后再确认"
                    )

                # 已被用户在提交前撤销：不能再被抢占。
                #
                # 少了这一条，撤销就只是「看起来生效」：
                # 用户说撤销 → 草稿标成 cancelled → 模型下一轮拿着
                # 旧的 draft_id 再确认一次 → 照样建单。
                if draft.status == STATUS_CANCELLED:
                    raise DraftNotFound(
                        "这张申请已经取消，如需重新申请请先生成新的草稿"
                    )

                if draft.status == STATUS_CONSUMING:

                    # 判断是不是上一个执行者崩溃留下的僵死状态。
                    #
                    # 基准必须是「开始消费的时间」，不是「草稿创建时间」：
                    # 用户看完草稿再确认花上几分钟很正常，
                    # 拿创建时间算会把正在处理的草稿当成死掉的抢走。
                    # 老数据没有 consuming_at，退回 created_at。
                    since = draft.consuming_at or draft.created_at
                    elapsed = (datetime.now() - since).total_seconds()

                    if elapsed < CONSUMING_STALE_SECONDS:
                        raise DraftBusy(
                            "该草稿正在处理中，请稍后重试"
                        )

                    # 僵死：允许重新抢占，往下走

                # 抢占成功
                started_at = datetime.now()

                connection.execute(
                    """
                    UPDATE ticket_drafts
                    SET status = ?,
                        consuming_at = ?
                    WHERE draft_id = ?
                    """,
                    (
                        STATUS_CONSUMING,
                        started_at.isoformat(timespec="seconds"),
                        draft_id,
                    ),
                )

                draft.status = STATUS_CONSUMING
                draft.consuming_at = started_at

                return draft

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


def complete_draft(draft_id: str, ticket_id: str) -> None:
    """建单成功后，把工单号写回草稿并标记已消费。"""

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE ticket_drafts
            SET status = ?,
                ticket_id = ?
            WHERE draft_id = ?
            """,
            (STATUS_CONSUMED, ticket_id, draft_id),
        )


def release_draft(draft_id: str) -> None:
    """建单失败后释放草稿，允许用户重新确认。

    失败原因通常是「不符合退款条件」「订单状态已变」，
    这类失败应该让用户看到原因后修改，而不是把草稿锁死。
    和幂等键的失败释放是同一个道理（Stripe 语义）。
    """

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE ticket_drafts
            SET status = ?,
                consuming_at = NULL
            WHERE draft_id = ?
              AND ticket_id IS NULL
            """,
            (STATUS_OPEN, draft_id),
        )


def find_latest_pending_draft(
    conversation_id: str,
    user_id: str,
) -> TicketDraft | None:
    """取本会话最近一张还可以确认的草稿。

    为什么需要它：

        draft_id 只存在于「生成草稿那一轮」的消息里。
        agent 只把最终回答写进会话历史，工具返回的原始 JSON
        不落库，所以下一轮模型根本看不到 draft_id。

        让模型凭记忆复述一串 uuid 是不现实的，
        它会编一个，然后 confirm 报「草稿不存在」——
        用户看到的就是「提交失败」，但草稿其实好好的。

    语义上也更自然：
        用户说「确认」，确认的就是刚刚那张待确认申请。
        不需要他念一遍编号。

    只返回 open / consuming 的草稿：
        consumed 代表已经提交过，再确认应该是幂等重放，
        走 draft_id 那条路；不能在这里被当成「待确认」重新提交。
    """

    now = datetime.now()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM ticket_drafts
            WHERE conversation_id = ?
              AND user_id = ?
              AND status IN (?, ?)
            ORDER BY created_at DESC
            """,
            (
                conversation_id,
                user_id,
                STATUS_OPEN,
                STATUS_CONSUMING,
            ),
        ).fetchall()

    for row in rows:

        draft = _row_to_draft(row)

        if not draft.is_expired(now):
            return draft

    return None


def find_recent_consumed_draft(
    conversation_id: str,
    user_id: str,
    within_seconds: int = DRAFT_TTL_SECONDS,
) -> TicketDraft | None:
    """找一张刚刚提交过的草稿，用于「用户又说了一次确认」的场景。

    为什么需要：
        用户连说两遍「确认」（或者网络卡了重发），
        第一遍建了单，草稿变成 consumed。
        第二遍如果只找 open/consuming 草稿，
        会得到「当前没有待确认的申请」——
        用户会以为没提交成功，再申请一次。

    为什么要限时间窗：
        不加限制的话，用户三个月后随口说一句「确认」，
        可能会被回放到很久以前那张工单，
        告诉他「您已提交，工单号 xxx」——比说「没有待确认」更糟。
    """

    cutoff = (
        datetime.now() - timedelta(seconds=within_seconds)
    ).isoformat()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM ticket_drafts
            WHERE conversation_id = ?
              AND user_id = ?
              AND status = ?
              AND created_at >= ?
            ORDER BY created_at DESC
            """,
            (conversation_id, user_id, STATUS_CONSUMED, cutoff),
        ).fetchall()

    for row in rows:

        draft = _row_to_draft(row)

        if draft.ticket_id:
            return draft

    return None


def find_reusable_draft(
    conversation_id: str,
    user_id: str,
    order_id: str,
    request_type: str,
    reason: str,
) -> TicketDraft | None:
    """找一张内容完全相同、仍然有效、且未被消费的草稿。

    为什么需要复用而不是每次都新建：

        模型很容易在每一轮都重新调用 draft——
        用户第二轮说「确认」，它可能先把申请内容「再整理一遍」，
        于是生成了第二张草稿，再拿新草稿去确认。

        而新草稿的 created_after_message_id 是当前这条「确认」消息，
        确认闸门要求「表态消息的序号必须更大」，
        新草稿永远无法满足自己刚刚创建时的那条消息——死锁。

        从业务上说也不该重复：
        同一张申请单被"重新生成"十次，用户会看到十个不同的编号。

    复用条件必须是五个字段全等，
    不能只比订单号：用户可能对同一订单先申请退款、后来又改成换货，
    那是两张不同的草稿，不能混。
    """

    now = datetime.now()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM ticket_drafts
            WHERE conversation_id = ?
              AND user_id = ?
              AND order_id = ?
              AND request_type = ?
              AND reason = ?
              AND status IN (?, ?)
            ORDER BY created_at DESC
            """,
            (
                conversation_id,
                user_id,
                order_id,
                request_type,
                reason,
                STATUS_OPEN,
                STATUS_CONSUMING,
            ),
        ).fetchall()

    for row in rows:

        draft = _row_to_draft(row)

        if not draft.is_expired(now):
            return draft

    return None


def cancel_pending_drafts(
    conversation_id: str,
    user_id: str,
    order_id: str | None = None,
) -> int:
    """作废本会话里还没提交的草稿，返回作废条数。

    为什么需要它：

        用户说「撤销」，但那张申请其实还停在草稿阶段（没提交）。
        撤销工具只找工单，找不到就回「没有可撤销的申请」——
        草稿还留在库里，模型下一轮完全可能把它确认掉。
        结果是：用户以为自己撤销了，工单却建出来了。

        这类问题在单线程测试里看不出来，只有把「撤销」和「确认」
        放在真实对话里跑才会撞上。

    只动 open 状态的：
        consuming 说明已经有请求在提交，此时作废反而制造歧义
        （工单可能已经建出来了），交给幂等键和随后的撤销处理。
    """

    sql = """
        UPDATE ticket_drafts
        SET status = ?
        WHERE conversation_id = ?
          AND user_id = ?
          AND status = ?
    """
    params: list[object] = [
        STATUS_CANCELLED,
        conversation_id,
        user_id,
        STATUS_OPEN,
    ]

    if order_id:
        sql += " AND order_id = ?"
        params.append(order_id.strip().upper())

    with get_connection() as connection:
        cursor = connection.execute(sql, params)
        return int(cursor.rowcount or 0)


def cleanup_expired_drafts(now: datetime | None = None) -> int:
    """清理过期且未消费的草稿，返回删除条数。

    幂等键和草稿都会随对话增长而累积，
    没有清理任务就是慢性内存泄漏。
    """

    cutoff = (now or datetime.now()).isoformat()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            DELETE FROM ticket_drafts
            WHERE expires_at < ?
              AND status != ?
            """,
            (cutoff, STATUS_CONSUMED),
        )
        return cursor.rowcount if cursor.rowcount is not None else 0


def count_drafts(user_id: str | None = None) -> int:
    """统计草稿数量，供健康检查使用。"""

    with get_connection() as connection:
        if user_id is None:
            row = connection.execute(
                "SELECT COUNT(*) AS c FROM ticket_drafts"
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT COUNT(*) AS c FROM ticket_drafts WHERE user_id = ?",
                (user_id,),
            ).fetchone()

    return int(row["c"])
