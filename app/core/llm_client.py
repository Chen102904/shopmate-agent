
import os
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from app.core.request_context import get_request_id
from app.core.execution_trace import record_execution_event


# 加载项目根目录下的 .env 文件
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


logger = logging.getLogger(__name__)


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
    多模型故障切换：

    1. 优先调用 DeepSeek。
    2. DeepSeek 失败，尝试智谱。
    3. 智谱失败，尝试千问。
    4. 三个模型全部失败，重新抛出 DeepSeek 原始异常。
    """
    primary_error = None

    # 第一级：DeepSeek 主模型
    start_time = time.perf_counter()

    try:
        client = get_llm_client()

        response = client.chat.completions.create(
            model=model,
            **kwargs,
        )

        _log_model_success(
            provider="DeepSeek",
            model=model,
            start_time=start_time,
            fallback=False,
        )

        return response

    except OpenAIError as exc:
        primary_error = exc

        _log_model_failure(
            provider="DeepSeek",
            model=model,
            start_time=start_time,
            exc=exc,
            fallback=False,
        )

        logger.warning(
            "DeepSeek 调用失败，准备切换到智谱备用模型 | request_id=%s",
            _current_request_id(),
        )

    fallback_kwargs = _prepare_fallback_kwargs(kwargs)

    # 第二级：智谱备用模型
    start_time = time.perf_counter()

    try:
        zhipu_client = get_zhipu_client()

        response = zhipu_client.chat.completions.create(
            model=ZHIPU_MODEL,
            **fallback_kwargs,
        )

        _log_model_success(
            provider="Zhipu",
            model=ZHIPU_MODEL,
            start_time=start_time,
            fallback=True,
        )

        logger.info(
            "Fallback 最终使用模型：%s | request_id=%s",
            ZHIPU_MODEL,
            _current_request_id(),
        )

        return response

    except Exception as zhipu_error:
        _log_model_failure(
            provider="Zhipu",
            model=ZHIPU_MODEL,
            start_time=start_time,
            exc=zhipu_error,
            fallback=True,
        )

        logger.warning(
            "智谱备用模型调用失败，准备切换到千问 | request_id=%s",
            _current_request_id(),
        )

    # 第三级：千问备用模型
    start_time = time.perf_counter()

    try:
        qwen_client = get_qwen_client()

        response = qwen_client.chat.completions.create(
            model=QWEN_MODEL,
            **fallback_kwargs,
        )

        _log_model_success(
            provider="Qwen",
            model=QWEN_MODEL,
            start_time=start_time,
            fallback=True,
        )

        logger.info(
            "Fallback 最终使用模型：%s | request_id=%s",
            QWEN_MODEL,
            _current_request_id(),
        )

        return response

    except Exception as qwen_error:
        _log_model_failure(
            provider="Qwen",
            model=QWEN_MODEL,
            start_time=start_time,
            exc=qwen_error,
            fallback=True,
        )

        logger.error(
            "千问备用模型也不可用，所有模型均调用失败 | request_id=%s",
            _current_request_id(),
        )

        if primary_error is not None:
            raise primary_error from qwen_error

        raise


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
