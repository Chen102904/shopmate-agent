"""模型调用的熔断器（Circuit Breaker）。

解决的问题：
    模型服务挂掉时，每个请求都要依次尝试 DeepSeek → 智谱 → 千问。
    如果三家都挂，单次请求要等三次超时，可能几十秒。
    用户等 30 秒拿到一句「服务暂时不可用」，比 2 秒拿到更糟。

    而且这时候打过去的请求几乎没有成功可能，
    等于把有限的连接和线程全耗在注定失败的调用上。

做法（经典三态）：
    CLOSED     正常放行。连续失败达到阈值 → 转 OPEN。
    OPEN       直接拒绝，不再调用。持续 recovery_time 后 → 转 HALF_OPEN。
    HALF_OPEN  只放一个请求去试探。成功 → 回 CLOSED；失败 → 回 OPEN。

为什么阈值是 3 次（连续，不是比例）：
    这个项目流量低（一次对话 3-4 次模型调用），
    按比例算没有统计意义。3 次连续失败已经足够说明「不是偶发抖动」。
    次数少的好处是故障时能快速止损。

为什么恢复时间 30 秒：
    模型侧故障多数是几十秒的抖动（限流、扩容、短时 5xx）。
    太短会反复打过去（每 5 秒试一次，等于没熔断）；
    太长会让恢复变慢（服务 20 秒就恢复了，用户还要多等 10 秒）。

为什么 HALF_OPEN 只放 1 个：
    刚恢复的服务最脆弱。放全部流量进去，很可能立刻又被打挂，
    然后进入「恢复-打挂-恢复」的循环。
    放一个请求确认真的好了，再全量放开。

哪些异常算失败（关键）：
    算：超时、连接失败、5xx、限流、认证失败、模型不存在。
    不算：400 参数错误。
        参数写错了，换哪家模型都一样失败，健康度应该记在调用方身上，
        不该让模型背锅——否则一次格式 bug 会把所有模型全熔断，
        故障面从「一个工具出错」放大成「整个 Agent 不可用」。

已知局限：
    - 状态在进程内存里。多进程/多实例部署时各算各的，
      所以要恢复 N 个实例就要 N×recovery_time。
      真要做全局熔断得把状态放 Redis。
    - 没有滑动窗口，只有「连续失败计数」。
      成功一次就清零，所以「每分钟失败 50%」这种缓慢劣化抓不到。
"""

from __future__ import annotations

import logging
import threading
import time

from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class AllProvidersCircuitOpen(RuntimeError):
    """所有模型供应商都在熔断中，这次调用一次网络请求都没发。

    单独定义一个异常类型，是为了让上层能区分两种失败：

        三家都试过了、都失败  → 是「模型服务出问题了」
        一家都没试（全熔断）  → 是「我们主动快速失败了」

    两者对用户的说法可以一样，但对运维的意义完全不同：
    前者要去看供应商状态页，后者要看我们的熔断配置是不是太激进。

    继承 RuntimeError 而不是 OpenAIError：
    这不是 SDK 抛的错，是我们自己的判断结果。
    上层按需捕获即可（见 agent.py 的 except 分支）。
    """


# 连续失败达到这个次数就打开熔断
DEFAULT_FAILURE_THRESHOLD = 3

# 打开后多久允许试探（秒）
DEFAULT_RECOVERY_TIME = 30.0


class CircuitBreaker:
    """单个下游依赖（这里是一个模型供应商）的熔断状态。"""

    def __init__(
        self,
        name: str,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_time: float = DEFAULT_RECOVERY_TIME,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = 0.0

        # HALF_OPEN 期间是否已经有请求在试探
        self._probing = False

        # 统计量：只用来观察，不参与判定
        self._total_calls = 0
        self._total_failures = 0
        self._total_rejected = 0

        self._lock = threading.Lock()

    # ---------------- 对外接口 ----------------

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def allow_request(self) -> bool:
        """这次调用放不放行。

        返回 False 表示应该直接走下一个供应商或直接失败，
        不要发起网络请求。
        """

        with self._lock:
            self._maybe_half_open()

            if self._state is CircuitState.CLOSED:
                self._total_calls += 1
                return True

            if self._state is CircuitState.OPEN:
                # 还在冷却期内
                self._total_rejected += 1
                return False

            # HALF_OPEN：只放一个探针
            if self._probing:
                self._total_rejected += 1
                return False

            self._probing = True
            self._total_calls += 1
            return True

    def record_success(self) -> None:
        with self._lock:
            was_open = self._state is not CircuitState.CLOSED

            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._probing = False
            self._opened_at = 0.0

            if was_open:
                logger.info(
                    "熔断关闭，恢复正常调用 | provider=%s",
                    self.name,
                )

    def record_failure(self) -> None:
        with self._lock:
            self._total_failures += 1
            self._failure_count += 1

            if self._state is CircuitState.HALF_OPEN:
                # 试探失败：立刻重新打开，不用等凑满阈值
                self._open()
                return

            if (
                self._state is CircuitState.CLOSED
                and self._failure_count >= self.failure_threshold
            ):
                self._open()

    def snapshot(self) -> dict:
        """给 /health 和日志用的状态快照。"""

        with self._lock:
            self._maybe_half_open()

            now = time.monotonic()

            remaining = 0.0

            if self._state is CircuitState.OPEN:
                remaining = max(
                    0.0,
                    self.recovery_time - (now - self._opened_at),
                )

            return {
                "provider": self.name,
                "state": self._state.value,
                "consecutive_failures": self._failure_count,
                "recover_in_seconds": round(remaining, 1),
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_rejected": self._total_rejected,
            }

    def reset(self) -> None:
        """测试用：强制回到正常状态。"""

        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._probing = False
            self._opened_at = 0.0
            self._total_calls = 0
            self._total_failures = 0
            self._total_rejected = 0

    # ---------------- 内部 ----------------

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._probing = False

        logger.warning(
            "熔断打开，接下来 %.0f 秒内不再调用 | provider=%s | "
            "consecutive_failures=%s",
            self.recovery_time,
            self.name,
            self._failure_count,
        )

    def _maybe_half_open(self) -> None:
        """冷却期满就转 HALF_OPEN，允许一个探针。

        必须在持锁状态下调用。
        """

        if self._state is not CircuitState.OPEN:
            return

        if time.monotonic() - self._opened_at < self.recovery_time:
            return

        self._state = CircuitState.HALF_OPEN
        self._probing = False

        logger.info(
            "熔断冷却结束，进入试探状态 | provider=%s",
            self.name,
        )


# ============================================================
# 注册表：每个供应商一个熔断器
# ============================================================

_BREAKERS: dict[str, CircuitBreaker] = {}
_REGISTRY_LOCK = threading.Lock()


def get_breaker(name: str) -> CircuitBreaker:
    """按供应商名取熔断器，没有就建一个。

    为什么不写成固定三个常量：
        以后加供应商只改调用处，不用回来改这个文件。
    """

    with _REGISTRY_LOCK:
        breaker = _BREAKERS.get(name)

        if breaker is None:
            breaker = CircuitBreaker(name)
            _BREAKERS[name] = breaker

        return breaker


def all_breakers_snapshot() -> list[dict]:
    """所有熔断器的状态，供 /health 展示。"""

    with _REGISTRY_LOCK:
        breakers = list(_BREAKERS.values())

    return [breaker.snapshot() for breaker in breakers]


def reset_all_breakers() -> None:
    """测试用：把所有熔断器恢复初始状态。"""

    with _REGISTRY_LOCK:
        breakers = list(_BREAKERS.values())

    for breaker in breakers:
        breaker.reset()


def is_breaker_eligible(exc: BaseException) -> bool:
    """这个异常该不该计入模型健康度。

    参数类错误（400）不计：换哪家模型都会因为同一个原因失败，
    算到模型头上会让一次调用方的 bug 把整条链路熔断。

    其余都计：超时、连接、5xx、限流、认证、模型不存在——
    这些反映的是「这个供应商现在能不能用」。
    """

    # 只对 openai SDK 的异常做细分。
    # 其他异常（比如我们自己抛的 RuntimeError）一律计入：
    # 能走到这里的异常，说明这次调用确实没成功。
    try:
        from openai import BadRequestError

        if isinstance(exc, BadRequestError):
            return False

    except Exception:
        # openai SDK 没装或版本差异，退化成「全部计入」
        pass

    return True
