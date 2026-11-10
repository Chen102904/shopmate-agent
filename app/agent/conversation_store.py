import json
import os
import sqlite3
import time
import uuid

from contextlib import contextmanager
from functools import wraps
from inspect import signature
from pathlib import Path
from threading import Event, Thread
from typing import Any, Iterator


# ============================================================
# 数据库配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_PATH = Path(
    os.getenv(
        "SHOPMATE_DB_PATH",
        str(PROJECT_ROOT / "data" / "shopmate.db"),
    )
)

DATABASE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


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

@contextmanager
def _get_connection() -> Iterator[sqlite3.Connection]:
    """获取 SQLite 数据库连接。"""

    connection = sqlite3.connect(
        str(DATABASE_PATH),
        timeout=10,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    connection.execute(
        "PRAGMA busy_timeout = 10000"
    )

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


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
            with _get_connection() as connection:
                connection.execute("PRAGMA journal_mode = WAL")

            # 第二步：通过 BEGIN IMMEDIATE 串行化数据库结构初始化。
            with _get_connection() as connection:
                connection.execute("BEGIN IMMEDIATE")

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        conversation_id TEXT PRIMARY KEY,
                        state_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
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
            error_message = str(exc).lower()

            # 只对 SQLite 锁冲突进行重试，其他错误直接抛出。
            if (
                "database is locked" not in error_message
                and "database is busy" not in error_message
            ):
                raise

            if attempt == max_attempts - 1:
                raise

            # 逐渐增加等待时间，最高 0.5 秒。
            time.sleep(min(0.05 * (attempt + 1), 0.5))



def _ensure_conversation(
    connection: sqlite3.Connection,
    conversation_id: str,
) -> None:
    """确保指定会话存在。"""

    connection.execute(
        """
        INSERT OR IGNORE INTO conversations (
            conversation_id
        )
        VALUES (?)
        """,
        (conversation_id,),
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
    """

    now = time.time()

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