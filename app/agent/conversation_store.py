import json
import sqlite3
import time
import uuid

from contextlib import contextmanager
from functools import wraps
from inspect import signature
from threading import Event, Thread
from typing import Any, Iterator

# 数据库路径与连接实现在 app/core/db.py 中统一定义，
# 工单存储（app/services/ticket_store.py）复用同一份。
# 这里保留 _get_connection 这个名字，本文件内的调用点无需改动。
from app.core.db import DATABASE_PATH, get_connection as _get_connection  # noqa: F401


# ============================================================
# 数据库配置
# ============================================================


# ============================================================
# 并发锁配置
# ============================================================

# 锁最多等待时间：120 秒
LOCK_WAIT_TIMEOUT = 120

# 锁租约有效时间：90 秒
LOCK_LEASE_SECONDS = 90

# 每 20 秒续期一次
LOCK_HEARTBEAT_INTERVAL = 20

# 获取锁失败后的轮询间隔
LOCK_POLL_INTERVAL = 0.1


# ============================================================
# 数据库连接
# ============================================================
#
# _get_connection 由 app/core/db.py 提供（文件顶部已导入），
# 连接参数、事务语义只在那一处维护。


# ============================================================
# 数据库初始化
# ============================================================


def _initialize_database() -> None:
    """初始化会话、消息和并发锁数据表，支持多进程同时启动。"""

    max_attempts = 20

    for attempt in range(max_attempts):
        try:
            # 第一步：启用 WAL。
            # 单独使用连接，避免在事务中切换 journal mode。
            #
            # 关键优化：先读当前模式，已经是 WAL 就不再切换。
            # 原因：PRAGMA journal_mode = WAL 在「切换」时需要排他锁，
            # 多进程同时启动会互相冲突，表现之一就是
            # "attempt to write a readonly database"。
            # 数据库一旦是 WAL（模式写在文件头），后续进程只需读取确认，
            # 不再产生排他锁，从根上消除这类竞争。
            with _get_connection() as connection:
                current_mode = connection.execute(
                    "PRAGMA journal_mode"
                ).fetchone()[0]

                if str(current_mode).lower() != "wal":
                    connection.execute("PRAGMA journal_mode = WAL")

            # 第二步：通过 BEGIN IMMEDIATE 串行化数据库结构初始化。
            with _get_connection() as connection:
                connection.execute("BEGIN IMMEDIATE")

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        conversation_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL DEFAULT '',
                        state_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

                # 兼容旧数据库：早期版本没有 user_id 列。
                # ALTER TABLE 只能加常量默认值，
                # 这里用演示账号作为历史会话的归属人（均为演示期创建）。
                conversation_columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(conversations)"
                    )
                }

                if "user_id" not in conversation_columns:
                    connection.execute(
                        """
                        ALTER TABLE conversations
                        ADD COLUMN user_id TEXT NOT NULL DEFAULT 'USER001'
                        """
                    )

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL,
                        message_json TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

                        FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_conversation_messages_conversation_id
                    ON conversation_messages(conversation_id)
                    """
                )

                # 跨进程会话锁
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_locks (
                        conversation_id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        expires_at REAL NOT NULL
                    )
                    """
                )

            # 初始化成功
            return

        except sqlite3.OperationalError as exc:
            # 初始化语句全部是幂等的（IF NOT EXISTS），因此任何
            # OperationalError 都可以安全重试，不限于 "locked"/"busy"。
            # 并发启动时 SQLite 可能返回：
            #   database is locked / busy
            #   attempt to write a readonly database（WAL 切换过程中）
            #   disk I/O error（瞬时的 -shm 文件竞争）
            # 以上均为瞬时状态，退避后重试即可；真正的权限问题会重试
            # 20 次后抛出，总等待上限约 7.5 秒。
            if attempt == max_attempts - 1:
                raise

            # 逐渐增加等待时间，最高 0.5 秒。
            time.sleep(min(0.05 * (attempt + 1), 0.5))



def _ensure_conversation(
    connection: sqlite3.Connection,
    conversation_id: str,
    user_id: str | None = None,
) -> None:
    """确保指定会话存在，并记录会话归属人。

    使用 INSERT OR IGNORE：
    会话已存在时不会覆盖原有归属人，
    因此第一个创建会话的用户成为唯一所有者，
    后续即使别人拿到同一个 conversation_id 也无法改写归属。
    """

    from app.core.identity import get_current_user_id

    owner_id = user_id or get_current_user_id()

    connection.execute(
        """
        INSERT OR IGNORE INTO conversations (
            conversation_id,
            user_id
        )
        VALUES (?, ?)
        """,
        (conversation_id, owner_id),
    )


_initialize_database()


# ============================================================
# 跨进程会话锁
# ============================================================

def _try_acquire_conversation_lock(
    conversation_id: str,
    owner_id: str,
) -> bool:
    """
    尝试获取指定会话的锁。

    使用 SQLite 原子写入操作：
    - 锁不存在：创建锁。
    - 锁已经过期：接管锁。
    - 锁仍然有效：获取失败。

    为什么要重试：
        多进程同时启动时，另一个进程可能正在做 WAL checkpoint，
        此时本进程的 BEGIN IMMEDIATE 会收到
        "attempt to write a readonly database"。
        这是瞬时的数据库文件状态，不是逻辑错误，退避重试即可。
        busy_timeout 只对「锁等待」生效，对 readonly 无效，
        所以这一层重试不能省。
    """

    max_attempts = 8

    for attempt in range(max_attempts):

        now = time.time()

        try:
            with _get_connection() as connection:

                connection.execute("BEGIN IMMEDIATE")

                cursor = connection.execute(
                    """
                    INSERT INTO conversation_locks (
                        conversation_id,
                        owner_id,
                        expires_at
                    )
                    VALUES (?, ?, ?)

                    ON CONFLICT(conversation_id)
                    DO UPDATE SET
                        owner_id = excluded.owner_id,
                        expires_at = excluded.expires_at

                    WHERE conversation_locks.expires_at < ?
                    """,
                    (
                        conversation_id,
                        owner_id,
                        now + LOCK_LEASE_SECONDS,
                        now,
                    ),
                )

                return cursor.rowcount == 1

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

            if attempt == max_attempts - 1:
                raise

            time.sleep(min(0.05 * (attempt + 1), 0.3))

    raise RuntimeError("获取会话锁重试次数用尽")


def _renew_conversation_lock(
    conversation_id: str,
    owner_id: str,
) -> bool:
    """续期当前持有的会话锁。"""

    with _get_connection() as connection:

        cursor = connection.execute(
            """
            UPDATE conversation_locks
            SET expires_at = ?
            WHERE conversation_id = ?
              AND owner_id = ?
            """,
            (
                time.time() + LOCK_LEASE_SECONDS,
                conversation_id,
                owner_id,
            ),
        )

        return cursor.rowcount == 1


def _release_conversation_lock(
    conversation_id: str,
    owner_id: str,
) -> None:
    """释放当前持有的会话锁。"""

    with _get_connection() as connection:

        connection.execute(
            """
            DELETE FROM conversation_locks
            WHERE conversation_id = ?
              AND owner_id = ?
            """,
            (
                conversation_id,
                owner_id,
            ),
        )


@contextmanager
def conversation_lock(
    conversation_id: str,
) -> Iterator[None]:
    """
    获取指定会话的跨进程锁。

    同一个会话只能由一个请求同时处理。
    不同会话之间互不阻塞。
    """

    owner_id = uuid.uuid4().hex

    start_time = time.monotonic()

    acquired = False

    while not acquired:

        acquired = _try_acquire_conversation_lock(
            conversation_id,
            owner_id,
        )

        if acquired:
            break

        if (
            time.monotonic() - start_time
            >= LOCK_WAIT_TIMEOUT
        ):
            raise TimeoutError(
                f"获取会话锁超时：{conversation_id}"
            )

        time.sleep(LOCK_POLL_INTERVAL)

    stop_event = Event()
    lock_lost_event = Event()

    def heartbeat() -> None:
        """后台定期续期，避免长时间模型调用导致锁过期。"""

        while not stop_event.wait(
            LOCK_HEARTBEAT_INTERVAL
        ):
            try:
                renewed = _renew_conversation_lock(
                    conversation_id,
                    owner_id,
                )

                if not renewed:
                    lock_lost_event.set()
                    return

            except Exception:
                # 续期失败时停止续期。
                # 租约到期后，其他请求可以接管锁。
                lock_lost_event.set()
                return

    heartbeat_thread = Thread(
        target=heartbeat,
        name=f"conversation-lock-{conversation_id}",
        daemon=True,
    )

    heartbeat_thread.start()

    try:
        yield

        if lock_lost_event.is_set():
            raise RuntimeError(
                f"会话锁续期失败，无法保证会话并发安全：{conversation_id}"
            )

    finally:
        stop_event.set()

        heartbeat_thread.join(timeout=2)

        _release_conversation_lock(
            conversation_id,
            owner_id,
        )


def with_conversation_lock(func):
    """
    Agent 函数装饰器。

    自动根据 conversation_id 获取会话锁。
    """

    function_signature = signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):

        bound_arguments = function_signature.bind_partial(
            *args,
            **kwargs,
        )

        conversation_id = bound_arguments.arguments.get(
            "conversation_id",
            "default",
        )

        with conversation_lock(conversation_id):
            return func(*args, **kwargs)

    return wrapper


# ============================================================
# 会话查询
# ============================================================

def conversation_exists(
    conversation_id: str,
) -> bool:
    """检查指定会话是否存在。"""

    with _get_connection() as connection:

        result = connection.execute(
            """
            SELECT 1
            FROM conversations
            WHERE conversation_id = ?
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()

        return result is not None


def get_conversation_owner(
    conversation_id: str,
) -> str | None:
    """查询会话归属人。

    返回：
        归属人的 user_id；会话不存在时返回 None。
        历史数据（迁移前创建）的归属人为演示账号 USER001。
    """

    with _get_connection() as connection:

        row = connection.execute(
            """
            SELECT user_id
            FROM conversations
            WHERE conversation_id = ?
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()

        if row is None:
            return None

        return row[0] or ""


def is_conversation_owner(
    conversation_id: str,
    user_id: str,
) -> bool:
    """判断某个会话是否属于指定用户。

    会话不存在时返回 False，
    由调用方决定是「允许新建」还是「拒绝访问」。
    """

    owner = get_conversation_owner(conversation_id)

    if owner is None:
        return False

    return owner == user_id


def get_conversation(
    conversation_id: str,
) -> list[dict[str, Any]]:
    """从 SQLite 获取指定会话的完整历史消息。"""

    with _get_connection() as connection:

        _ensure_conversation(
            connection,
            conversation_id,
        )

        rows = connection.execute(
            """
            SELECT message_json
            FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

        return [
            json.loads(row["message_json"])
            for row in rows
        ]


def get_last_user_message_record(
    conversation_id: str,
    scan_limit: int = 50,
) -> dict[str, Any] | None:
    """取最近一条用户消息，连同它的自增序号。

    返回：
        {"id": 消息序号, "content": 消息正文}；没有用户消息时返回 None。

    为什么需要「序号」而不只是「内容」：
        售后申请的确认闸门要区分两句话——
            这张草稿生成之前用户说过的话
            这张草稿生成之后用户才说的「确认」
        只有后者才算真的看过草稿再同意。
        正文内容比对不出先后顺序，自增 id 可以。

    为什么只扫最近 scan_limit 条：
        用户消息一定在会话尾部，扫全部在大历史下没有意义。
        LIMIT 在 SQL 层做，避免把整段历史读进内存。
    """

    with _get_connection() as connection:

        _ensure_conversation(
            connection,
            conversation_id,
        )

        rows = connection.execute(
            """
            SELECT id, message_json
            FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, scan_limit),
        ).fetchall()

        for row in rows:

            message = json.loads(row["message_json"])

            if message.get("role") != "user":
                continue

            return {
                "id": int(row["id"]),
                "content": message.get("content") or "",
            }

        return None


def get_conversation_snapshot(
    conversation_id: str,
) -> list[dict[str, Any]] | None:
    """获取会话历史快照，不存在时返回 None。"""

    with _get_connection() as connection:

        conversation = connection.execute(
            """
            SELECT 1
            FROM conversations
            WHERE conversation_id = ?
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()

        if conversation is None:
            return None

        rows = connection.execute(
            """
            SELECT message_json
            FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

        return [
            json.loads(row["message_json"])
            for row in rows
        ]


# ============================================================
# 消息管理
# ============================================================

def add_message(
    conversation_id: str,
    message: dict[str, Any],
) -> None:
    """向指定会话添加一条消息并持久化。"""

    message_json = json.dumps(
        message,
        ensure_ascii=False,
    )

    with _get_connection() as connection:

        _ensure_conversation(
            connection,
            conversation_id,
        )

        connection.execute(
            """
            INSERT INTO conversation_messages (
                conversation_id,
                message_json
            )
            VALUES (?, ?)
            """,
            (
                conversation_id,
                message_json,
            ),
        )

        connection.execute(
            """
            UPDATE conversations
            SET updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )


# ============================================================
# 会话状态管理
# ============================================================

def get_conversation_state(
    conversation_id: str,
) -> dict[str, Any]:
    """从 SQLite 获取会话状态。"""

    with _get_connection() as connection:

        _ensure_conversation(
            connection,
            conversation_id,
        )

        row = connection.execute(
            """
            SELECT state_json
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()

        if row is None:
            return {}

        return json.loads(
            row["state_json"]
        )


def set_pending_clarification(
    conversation_id: str,
    original_user_message: str,
    clarification_question: str,
) -> None:
    """原子更新当前会话的待澄清状态。"""

    with _get_connection() as connection:

        connection.execute("BEGIN IMMEDIATE")

        _ensure_conversation(
            connection,
            conversation_id,
        )

        row = connection.execute(
            """
            SELECT state_json
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()

        state = json.loads(
            row["state_json"]
        )

        state["pending_clarification"] = {
            "original_user_message": original_user_message,
            "clarification_question": clarification_question,
        }

        connection.execute(
            """
            UPDATE conversations
            SET
                state_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
            """,
            (
                json.dumps(
                    state,
                    ensure_ascii=False,
                ),
                conversation_id,
            ),
        )


def update_conversation_state(
    conversation_id: str,
    updates: dict[str, Any],
    remove_keys: list[str] | None = None,
) -> dict[str, Any]:
    """原子更新会话状态，返回更新后的完整状态。

    为什么需要通用接口：
        原来只有 set_pending_clarification 这类专用函数，
        每加一种状态（工具缓存、对话摘要）就要复制一遍
        读-改-写的代码，容易漏掉 BEGIN IMMEDIATE 和事务提交。

    参数：
        updates: 要合并进去的键值
        remove_keys: 要删除的键

    注意：
        整个操作在 BEGIN IMMEDIATE 事务里完成，
        避免并发请求互相覆盖对方的写入。
    """

    with _get_connection() as connection:

        connection.execute("BEGIN IMMEDIATE")

        _ensure_conversation(
            connection,
            conversation_id,
        )

        row = connection.execute(
            """
            SELECT state_json
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()

        state = json.loads(row["state_json"]) if row else {}

        for key in remove_keys or []:
            state.pop(key, None)

        state.update(updates)

        connection.execute(
            """
            UPDATE conversations
            SET
                state_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
            """,
            (
                json.dumps(
                    state,
                    ensure_ascii=False,
                ),
                conversation_id,
            ),
        )

        return state


def get_pending_clarification(
    conversation_id: str,
) -> dict[str, Any] | None:
    """获取当前会话的待澄清状态。"""

    state = get_conversation_state(
        conversation_id
    )

    return state.get(
        "pending_clarification"
    )


def clear_pending_clarification(
    conversation_id: str,
) -> None:
    """原子清除当前会话的待澄清状态。"""

    with _get_connection() as connection:

        connection.execute("BEGIN IMMEDIATE")

        _ensure_conversation(
            connection,
            conversation_id,
        )

        row = connection.execute(
            """
            SELECT state_json
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()

        state = json.loads(
            row["state_json"]
        )

        state.pop(
            "pending_clarification",
            None,
        )

        connection.execute(
            """
            UPDATE conversations
            SET
                state_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
            """,
            (
                json.dumps(
                    state,
                    ensure_ascii=False,
                ),
                conversation_id,
            ),
        )


# ============================================================
# 会话清理
# ============================================================

def clear_conversation(
    conversation_id: str,
) -> None:
    """
    清空指定会话。

    删除操作同样需要获取会话锁，
    避免与正在执行的 Agent 请求发生冲突。

    同时作废该会话下的售后申请草稿。
    为什么必须一起删：
        「清空会话」在用户理解里等于「重新开始」。
        但草稿是独立存在的一张表，绑的是 conversation_id。
        只删消息不删草稿的话，会出现这种情况：
            用户说了「退 ORD001，商品破损」→ 生成草稿
            用户发现说错了，清空会话
            用户随口说一句「确认」（他以为在确认别的）
            系统把清空前那张草稿提交了
        这比「多问一遍」严重得多——是一张真的工单。
    """

    with conversation_lock(conversation_id):

        with _get_connection() as connection:

            connection.execute(
                """
                DELETE FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )

            # 草稿表可能还没建（全新库、只跑过查询的场景），
            # 缺表不该让「清空会话」这个操作失败。
            try:
                connection.execute(
                    """
                    DELETE FROM ticket_drafts
                    WHERE conversation_id = ?
                    """,
                    (conversation_id,),
                )
            except sqlite3.OperationalError:
                pass