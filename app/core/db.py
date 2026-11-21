"""
SQLite 连接与路径的统一定义。

为什么单独抽一个模块：
    会话存储（app/agent/conversation_store.py）和工单存储
    （app/services/ticket_store.py）共用同一个数据库文件。

    如果工单模块直接 import 会话模块，依赖方向就反了
    （services 层反过来依赖 agent 层），
    而且会连带触发会话模块的初始化逻辑。

    路径、连接参数、事务语义只应该有一处定义，
    否则两处不一致时会出现「同一个库，两个模块看到不同的行为」。
"""

import os
import sqlite3

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


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


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """获取 SQLite 连接，正常退出自动提交，异常自动回滚。"""

    connection = sqlite3.connect(
        str(DATABASE_PATH),
        timeout=10,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    # 等锁最多 10 秒。
    # 注意：busy_timeout 只对「等锁」生效，
    # 对 readonly / disk I/O 这类错误无效，需要调用方自己重试。
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
