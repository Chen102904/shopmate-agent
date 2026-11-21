
import os
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from app.core.circuit_breaker import (
    AllProvidersCircuitOpen,
    get_breaker,
    is_breaker_eligible,
)
from app.core.request_context import get_request_id
from app.core.execution_trace import record_execution_event


# 加载项目根目录下的 .env 文件
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


logger = logging.getLogger(__name__)


# 默认采样温度。
#
# 为什么不留空（用 API 默认值）：
#     不设时模型用默认温度（通常接近 1），
#     同样的输入每次输出都不同。
#     对「意图分类」「参数抽取」「查询改写」这类任务，
#     这是错误的：它让评测结果在两次运行之间上下浮动，
#     也让同一句话忽判 refund 忽判 policy。
#
# 实测证据（未设温度时）：
#     同一份 30 条 E2E，两次运行 24/30 与 23/30；
#     多轮通过率 33% 与 0%；RAG 回答 96.7% 与 90%；
#     RAG 检索 Hit@3 在 0.88 与 0.92 之间跳。
#
# 设成 0 后输出基本确定（服务端仍可能有微小波动），
# 评测结果才具备「可复现」的前提。
#
# 需要自然表达的地方（例如最终回复）可以显式传更高的温度。
DEFAULT_TEMPERATURE = float(
    os.getenv("SHOPMATE_TEMPERATURE", "0")
)


def _current_request_id() -> str:
    """获取当前请求的 Trace ID。"""
    return get_request_id()


# =========================
# 模型配置
# =========================

DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

ZHIPU_MODEL = "glm-4.7-flash"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.8-max")


# =========================
# 模型客户端
# =========================

def get_llm_client() -> OpenAI:
    """创建 DeepSeek 主模型客户端。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError(
            "没有找到 DEEPSEEK_API_KEY，请检查项目根目录下的 .env 文件"
        )

    return OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=30.0,
        max_retries=2,
    )


def get_zhipu_client() -> OpenAI:
    """创建智谱备用模型客户端。"""
    api_key = os.getenv("ZHIPU_API_KEY")

    if not api_key:
        raise ValueError(
            "没有找到 ZHIPU_API_KEY，请检查项目根目录下的 .env 文件"
        )

    return OpenAI(
        api_key=api_key,
        base_url=ZHIPU_BASE_URL,
        timeout=15.0,
        max_retries=0,
    )


def get_qwen_client() -> OpenAI:
    """创建阿里云百炼千问备用模型客户端。"""
    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL")

    if not api_key:
        raise ValueError(
            "没有找到 QWEN_API_KEY，请检查项目根目录下的 .env 文件"
        )

    if not base_url:
        raise ValueError(
            "没有找到 QWEN_BASE_URL，请检查项目根目录下的 .env 文件"
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        timeout=30.0,
        max_retries=0,
    )


# =========================
# 备用模型参数处理
# =========================

def _prepare_fallback_kwargs(kwargs: dict) -> dict:
    """清理 DeepSeek 专属参数。"""
    fallback_kwargs = dict(kwargs)

    extra_body = fallback_kwargs.get("extra_body")

    if isinstance(extra_body, dict):
        extra_body = dict(extra_body)

        extra_body.pop("thinking", None)

        if extra_body:
            fallback_kwargs["extra_body"] = extra_body
        else:
            fallback_kwargs.pop("extra_body", None)

    return fallback_kwargs


# =========================
# 模型调用日志与结构化记录
# =========================

def _log_model_success(
    provider: str,
    model: str,
    start_time: float,
    fallback: bool,
) -> None:
    """记录模型调用成功。"""
    duration_ms = (time.perf_counter() - start_time) * 1000

    record_execution_event(
        event="model_call",
        status="success",
        duration_ms=duration_ms,
        details={
            "request_id": _current_request_id(),
            "provider": provider,
            "model": model,
            "fallback": fallback,
        },
    )

    logger.info(
        "模型调用成功 | request_id=%s | provider=%s | "
        "model=%s | status=success | fallback=%s | duration=%.2fms",
        _current_request_id(),
        provider,
        model,
        fallback,
        duration_ms,
    )


def _log_model_failure(
    provider: str,
    model: str,
    start_time: float,
    exc: Exception,
    fallback: bool,
) -> None:
    """记录模型调用失败，不保存异常原文。"""
    duration_ms = (time.perf_counter() - start_time) * 1000
    error_type = type(exc).__name__

    record_execution_event(
        event="model_call",
        status="error",
        duration_ms=duration_ms,
        details={
            "request_id": _current_request_id(),
            "provider": provider,
            "model": model,
            "fallback": fallback,
            "error_type": error_type,
        },
    )

    logger.warning(
        "模型调用失败 | request_id=%s | provider=%s | "
        "model=%s | status=error | fallback=%s | "
        "duration=%.2fms | error_type=%s",
        _current_request_id(),
        provider,
        model,
        fallback,
        duration_ms,
        error_type,
    )


# =========================
# 主模型 + 多级备用模型
# =========================

def create_chat_completion_with_fallback(
    model: str,
    **kwargs,
):
    """
    多模型故障切换 + 熔断：

    1. 优先调用 DeepSeek。
    2. DeepSeek 失败，尝试智谱。
    3. 智谱失败，尝试千问。
    4. 三个模型全部失败，重新抛出 DeepSeek 原始异常。
    5. 某个供应商处于熔断状态时直接跳过，不发网络请求。

    第 5 条是后加的：三家全挂时，每个请求原本都要等三次超时，
    单次可能几十秒。熔断打开后，请求在毫秒级失败，
    用户 2 秒拿到「服务暂时不可用」，而不是 30 秒。

    采样温度：
        调用方没显式传 temperature 时，自动补 DEFAULT_TEMPERATURE（0）。
        这样所有走统一入口的调用（意图分类、查询改写、工具决策、
        回答生成）默认都是确定性的，不需要每处单独设置，
        也不会出现「某个新加的调用忘了设温度」的情况。
    """

    if "temperature" not in kwargs:
        kwargs["temperature"] = DEFAULT_TEMPERATURE

    # 供应商顺序就是降级顺序。
    # 写成表而不是三段重复代码：
    # 加一个供应商只需要在这里加一行，不用再抄一遍 try/except。
    providers = (
        {
            "name": "DeepSeek",
            "model": model,
            "fallback": False,
            "client_getter": get_llm_client,
        },
        {
            "name": "Zhipu",
            "model": ZHIPU_MODEL,
            "fallback": True,
            "client_getter": get_zhipu_client,
        },
        {
            "name": "Qwen",
            "model": QWEN_MODEL,
            "fallback": True,
            "client_getter": get_qwen_client,
        },
    )

    primary_error = None
    last_error = None
    attempted = 0
    skipped: list[str] = []

    for provider in providers:

        name = provider["name"]
        breaker = get_breaker(name)

        # 熔断中：直接跳过，不发请求。
        # 这是「快速失败」的关键——不打网络，不占连接，不耗线程。
        if not breaker.allow_request():
            skipped.append(name)

            logger.warning(
                "模型处于熔断状态，跳过调用 | request_id=%s | provider=%s",
                _current_request_id(),
                name,
            )

            continue

        attempted += 1

        start_time = time.perf_counter()

        # 备用模型要清掉 DeepSeek 专有参数（thinking 之类），
        # 否则备用模型会因为不认识这个字段直接 400。
        call_kwargs = (
            _prepare_fallback_kwargs(kwargs)
            if provider["fallback"]
            else kwargs
        )

        try:
            client = provider["client_getter"]()

            response = client.chat.completions.create(
                model=provider["model"],
                **call_kwargs,
            )

            breaker.record_success()

            _log_model_success(
                provider=name,
                model=provider["model"],
                start_time=start_time,
                fallback=provider["fallback"],
            )

            if provider["fallback"]:
                logger.info(
                    "Fallback 最终使用模型：%s | request_id=%s",
                    provider["model"],
                    _current_request_id(),
                )

            return response

        except Exception as exc:

            # 只把「模型侧问题」计入健康度。
            # 400 参数错误不算：换哪家都会因为同一个原因失败，
            # 算到模型头上会把一次调用方 bug 放大成「全链路不可用」。
            if is_breaker_eligible(exc):
                breaker.record_failure()

            _log_model_failure(
                provider=name,
                model=provider["model"],
                start_time=start_time,
                exc=exc,
                fallback=provider["fallback"],
            )

            if primary_error is None:
                primary_error = exc

            last_error = exc

            logger.warning(
                "模型调用失败，准备切换下一个 | request_id=%s | provider=%s",
                _current_request_id(),
                name,
            )

    # ── 走到这里说明三家都没成功 ──

    if attempted == 0:
        # 一家都没试：全在熔断状态。
        # 抛专门的异常，让上层能给出准确文案，
        # 而不是含糊的「出了点状况」。
        logger.error(
            "所有模型均处于熔断状态，本次未发起任何调用 | request_id=%s | "
            "skipped=%s",
            _current_request_id(),
            skipped,
        )

        raise AllProvidersCircuitOpen(
            "所有模型均处于熔断状态，本次未发起调用"
        )

    logger.error(
        "所有模型均调用失败 | request_id=%s | skipped=%s",
        _current_request_id(),
        skipped,
    )

    if primary_error is not None:
        raise primary_error from last_error

    raise last_error


# =========================
# 普通文本调用接口
# =========================

def ask_deepseek(prompt: str) -> str:
    """调用 DeepSeek 生成文本。"""
    start_time = time.perf_counter()

    try:
        client = get_llm_client()

        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是 ShopMate 智能电商售后助手。",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            },
        )

        _log_model_success(
            provider="DeepSeek",
            model=DEEPSEEK_MODEL,
            start_time=start_time,
            fallback=False,
        )

        return response.choices[0].message.content or ""

    except Exception as exc:
        _log_model_failure(
            provider="DeepSeek",
            model=DEEPSEEK_MODEL,
            start_time=start_time,
            exc=exc,
            fallback=False,
        )
        raise


def ask_zhipu(prompt: str) -> str:
    """调用智谱生成文本。"""
    start_time = time.perf_counter()

    try:
        client = get_zhipu_client()

        response = client.chat.completions.create(
            model=ZHIPU_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是 ShopMate 智能电商售后助手。",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        _log_model_success(
            provider="Zhipu",
            model=ZHIPU_MODEL,
            start_time=start_time,
            fallback=False,
        )

        return response.choices[0].message.content or ""

    except Exception as exc:
        _log_model_failure(
            provider="Zhipu",
            model=ZHIPU_MODEL,
            start_time=start_time,
            exc=exc,
            fallback=False,
        )
        raise


def ask_qwen(prompt: str) -> str:
    """调用千问生成文本。"""
    start_time = time.perf_counter()

    try:
        client = get_qwen_client()

        response = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是 ShopMate 智能电商售后助手。",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        _log_model_success(
            provider="Qwen",
            model=QWEN_MODEL,
            start_time=start_time,
            fallback=False,
        )

        return response.choices[0].message.content or ""

    except Exception as exc:
        _log_model_failure(
            provider="Qwen",
            model=QWEN_MODEL,
            start_time=start_time,
            exc=exc,
            fallback=False,
        )
        raise
