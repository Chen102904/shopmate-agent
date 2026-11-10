
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_workers(tmp_path, conversation_ids, sleep_seconds=1.0):
    """启动多个独立 Python 进程，测试 SQLite 跨进程锁。"""
    db_path = tmp_path / "concurrency_test.db"
    result_path = tmp_path / "events.jsonl"

    worker_code = """
import json
import os
import time

from app.agent.conversation_store import conversation_lock

conversation_id = os.environ["TEST_CONVERSATION_ID"]
sleep_seconds = float(os.environ["TEST_SLEEP_SECONDS"])
result_path = os.environ["TEST_RESULT_PATH"]

with conversation_lock(conversation_id):
    start = time.time()
    time.sleep(sleep_seconds)
    end = time.time()

    with open(result_path, "a", encoding="utf-8") as file:
        file.write(json.dumps({
            "conversation_id": conversation_id,
            "start": start,
            "end": end
        }) + "\\n")
"""

    processes = []

    for conversation_id in conversation_ids:
        env = os.environ.copy()
        env["SHOPMATE_DB_PATH"] = str(db_path)
        env["TEST_CONVERSATION_ID"] = conversation_id
        env["TEST_SLEEP_SECONDS"] = str(sleep_seconds)
        env["TEST_RESULT_PATH"] = str(result_path)

        process = subprocess.Popen(
            [sys.executable, "-c", worker_code],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process)

    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, (
            f"子进程执行失败\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )

    with open(result_path, "r", encoding="utf-8") as file:
        events = [json.loads(line) for line in file if line.strip()]

    return events


def test_same_conversation_processes_are_serialized(tmp_path):
    """同一个会话的两个进程不能同时持有锁。"""
    events = run_workers(
        tmp_path,
        ["same_session", "same_session"],
        sleep_seconds=1.0,
    )

    assert len(events) == 2

    first, second = sorted(events, key=lambda item: item["start"])

    assert first["end"] <= second["start"]


def test_different_conversations_can_run_concurrently(tmp_path):
    """不同会话的进程可以同时持有各自的锁。"""
    events = run_workers(
        tmp_path,
        ["session_a", "session_b"],
        sleep_seconds=1.5,
    )

    assert len(events) == 2

    first, second = events

    # 两个执行区间必须存在重叠
    assert max(first["start"], second["start"]) < min(
        first["end"], second["end"]
    )


def test_expired_lock_can_be_taken_over(tmp_path):
    """模拟进程异常退出留下的过期锁，验证其他进程可以接管。"""
    db_path = tmp_path / "expired_lock_test.db"

    script = """
import sqlite3
import time

from app.agent.conversation_store import conversation_lock

db_path = __import__("os").environ["SHOPMATE_DB_PATH"]

connection = sqlite3.connect(db_path)
connection.execute(
    '''
    INSERT OR REPLACE INTO conversation_locks
    (conversation_id, owner_id, expires_at)
    VALUES (?, ?, ?)
    ''',
    ("expired_session", "dead_process", time.time() - 10)
)
connection.commit()
connection.close()

with conversation_lock("expired_session"):
    print("成功接管过期锁")
"""

    env = os.environ.copy()
    env["SHOPMATE_DB_PATH"] = str(db_path)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "成功接管过期锁" in result.stdout
