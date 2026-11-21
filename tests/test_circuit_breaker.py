"""熔断器测试。

守的核心：

    模型服务挂掉时，请求要**快速失败**，而不是让每个用户等三次超时。
    三家全挂的情况下，第 1 个请求走完三次尝试，
    之后所有请求应该在毫秒级返回。

    熔断本身不能变成新的故障源：
    误判（一次参数错误就把模型熔断）比不熔断更糟。

真实场景（2026-11-21 审计）：
    没有熔断时，模型侧故障期间每个请求都要依次尝试
    DeepSeek → 智谱 → 千问，单次可能卡几十秒。
    用户等 30 秒拿到「服务暂时不可用」，比 2 秒拿到更糟，
    而且把有限线程全耗在注定失败的调用上。
"""

import time

import pytest

from app.core.circuit_breaker import (
    AllProvidersCircuitOpen,
    CircuitBreaker,
    CircuitState,
    get_breaker,
    is_breaker_eligible,
    reset_all_breakers,
)


@pytest.fixture(autouse=True)
def _clean_breakers():
    """每个用例前后都清干净，避免相互影响。"""

    reset_all_breakers()
    yield
    reset_all_breakers()


# ============================================================
# 状态机
# ============================================================


def test_starts_closed_and_allows():
    breaker = CircuitBreaker("t1")

    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_opens_after_threshold_failures():
    breaker = CircuitBreaker("t2", failure_threshold=3)

    for _ in range(3):
        assert breaker.allow_request() is True
        breaker.record_failure()

    assert breaker.state is CircuitState.OPEN
    assert breaker.allow_request() is False


def test_success_resets_consecutive_count():
    """连续失败才熔断。中间成功一次就清零，不能累计。"""

    breaker = CircuitBreaker("t3", failure_threshold=3)

    breaker.allow_request()
    breaker.record_failure()
    breaker.allow_request()
    breaker.record_failure()

    # 成功一次，前面的失败全部作废
    breaker.allow_request()
    breaker.record_success()

    assert breaker.state is CircuitState.CLOSED

    # 再失败两次不该熔断（因为已经从 0 开始数）
    breaker.allow_request()
    breaker.record_failure()
    breaker.allow_request()
    breaker.record_failure()

    assert breaker.state is CircuitState.CLOSED


def test_half_open_after_cooldown_allows_only_one_probe():
    """冷却结束后只放一个请求去试探。

    全放进去会把刚恢复的服务再打挂，陷入
    「恢复 → 打挂 → 恢复」的循环。
    """

    breaker = CircuitBreaker("t4", failure_threshold=1, recovery_time=0.1)

    breaker.allow_request()
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN

    time.sleep(0.15)

    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow_request() is True
    # 第二个请求必须被拒：探针位已经占了
    assert breaker.allow_request() is False


def test_probe_success_closes_circuit():
    breaker = CircuitBreaker("t5", failure_threshold=1, recovery_time=0.1)

    breaker.allow_request()
    breaker.record_failure()

    time.sleep(0.15)

    assert breaker.allow_request() is True
    breaker.record_success()

    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_probe_failure_reopens_immediately():
    """试探失败要立刻重新打开，不能再等凑满阈值。"""

    breaker = CircuitBreaker("t6", failure_threshold=3, recovery_time=0.1)

    for _ in range(3):
        breaker.allow_request()
        breaker.record_failure()

    time.sleep(0.15)

    assert breaker.allow_request() is True
    breaker.record_failure()

    assert breaker.state is CircuitState.OPEN


# ============================================================
# 异常分类：哪些失败该计入模型健康度
# ============================================================


def test_bad_request_does_not_trip_breaker():
    """400 参数错误不该算模型挂了。

    调用方写错参数，换哪家模型都会因为同一个原因失败。
    如果算到模型头上，一次格式 bug 会把三家全熔断，
    故障面从「一个工具出错」放大成「整个 Agent 不可用」。
    """

    from openai import BadRequestError

    exc = BadRequestError(
        "invalid request",
        response=_FakeResponse(400),
        body={"error": "bad params"},
    )

    assert is_breaker_eligible(exc) is False


def test_other_errors_count_as_failures():
    """超时、连接、5xx 都算模型侧问题。"""

    assert is_breaker_eligible(RuntimeError("boom")) is True
    assert is_breaker_eligible(TimeoutError("timeout")) is True


class _FakeResponse:
    """BadRequestError 需要 response 对象，这里给个最小实现。"""

    def __init__(self, status_code: int):
        self.status_code = status_code
        self.headers = {}
        self.request = None


# ============================================================
# 降级链路
# ============================================================


def _no_client(*args, **kwargs):
    raise RuntimeError("模型服务不可用")


def test_fallback_skips_open_provider(monkeypatch):
    """主模型熔断后，请求应该直接走备用，不再打主模型。"""

    import app.core.llm_client as llm

    # 先手动把 DeepSeek 熔断
    deepseek = get_breaker("DeepSeek")
    for _ in range(3):
        deepseek.allow_request()
        deepseek.record_failure()

    assert deepseek.state is CircuitState.OPEN

    calls: list[str] = []

    class _OkClient:
        class chat:
            class completions:
                @staticmethod
                def create(model=None, **kwargs):
                    calls.append(model)
                    return {"ok": True}

    monkeypatch.setattr(llm, "get_llm_client", _no_client)
    monkeypatch.setattr(llm, "get_zhipu_client", lambda: _OkClient())

    result = llm.create_chat_completion_with_fallback(
        "deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result == {"ok": True}
    # 只调了智谱，没有调 DeepSeek
    assert calls == [llm.ZHIPU_MODEL]


def test_all_open_raises_without_any_call(monkeypatch):
    """三家全熔断时，一次网络请求都不该发。"""

    import app.core.llm_client as llm

    for name in ("DeepSeek", "Zhipu", "Qwen"):
        breaker = get_breaker(name)
        for _ in range(3):
            breaker.allow_request()
            breaker.record_failure()

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("熔断状态下不应该创建客户端")

    monkeypatch.setattr(llm, "get_llm_client", _must_not_be_called)
    monkeypatch.setattr(llm, "get_zhipu_client", _must_not_be_called)
    monkeypatch.setattr(llm, "get_qwen_client", _must_not_be_called)

    with pytest.raises(AllProvidersCircuitOpen):
        llm.create_chat_completion_with_fallback(
            "deepseek-chat",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_fast_fail_after_breaker_opens(monkeypatch):
    """熔断打开后，失败耗时应该从「三次尝试」降到「几乎为零」。

    这是熔断存在的唯一理由：
    用户不该为了一个注定失败的请求等 30 秒。
    """

    import app.core.llm_client as llm

    monkeypatch.setattr(llm, "get_llm_client", _no_client)
    monkeypatch.setattr(llm, "get_zhipu_client", _no_client)
    monkeypatch.setattr(llm, "get_qwen_client", _no_client)

    # 先把熔断打开（前三次会真的走完三家）
    for _ in range(3):
        with pytest.raises(Exception):
            llm.create_chat_completion_with_fallback(
                "deepseek-chat",
                messages=[{"role": "user", "content": "hi"}],
            )

    start = time.perf_counter()

    with pytest.raises(AllProvidersCircuitOpen):
        llm.create_chat_completion_with_fallback(
            "deepseek-chat",
            messages=[{"role": "user", "content": "hi"}],
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    # 没有网络请求，应该远低于 10ms
    assert elapsed_ms < 10


def test_agent_gives_model_unavailable_message_when_all_open(monkeypatch):
    """三家全熔断时，用户要收到「模型不可用」的说明，而不是「出了点状况」。"""

    import app.agent.agent as agent_module

    for name in ("DeepSeek", "Zhipu", "Qwen"):
        breaker = get_breaker(name)
        for _ in range(3):
            breaker.allow_request()
            breaker.record_failure()

    # 意图分类也会调模型，先让它失败（这样会走到 Agent 主流程的模型分支）
    def _raise(*args, **kwargs):
        raise AllProvidersCircuitOpen("全部熔断")

    monkeypatch.setattr(
        agent_module,
        "create_chat_completion_with_fallback",
        _raise,
    )

    answer = agent_module.run_agent(
        "ORD001 到哪了",
        conversation_id="test-err-circuit",
        user_id="USER001",
    )

    assert "暂时不可用" in answer

    from app.agent.conversation_store import clear_conversation

    clear_conversation("test-err-circuit")
