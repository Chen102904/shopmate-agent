
import hmac
import math
import os
import threading
import time

from collections import defaultdict, deque
from fastapi import Request
from fastapi.responses import JSONResponse, Response


# 每分钟默认允许的请求次数
DEFAULT_REQUESTS_PER_MINUTE = 60

# 聊天接口每分钟允许的请求次数
CHAT_REQUESTS_PER_MINUTE = 20

# 统计周期：60 秒
RATE_LIMIT_WINDOW_SECONDS = 60

# 记录每个 IP 的请求时间
_REQUEST_HISTORY: dict[str, deque[float]] = defaultdict(deque)

# 并发保护
_RATE_LIMIT_LOCK = threading.Lock()

# 开发环境下默认开放的文档路径
DOCUMENTATION_PATHS = {
    "/docs",
    "/redoc",
    "/openapi.json",
}


def _is_enabled(value: str | None, default: bool = False) -> bool:
    """解析环境变量中的布尔值。"""
    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def reset_rate_limits() -> None:
    """清空请求频率记录，主要用于自动化测试。"""
    with _RATE_LIMIT_LOCK:
        _REQUEST_HISTORY.clear()


def _get_rate_limit_key(request: Request) -> str:
    """获取限流分组，避免通过修改会话 ID 轻易绕过限制。"""
    path = request.url.path

    if path == "/chat":
        route_group = "chat"
    elif path.startswith("/conversations/"):
        route_group = "conversations"
    elif path.startswith("/traces/"):
        route_group = "traces"
    else:
        route_group = "default"

    client_host = (
        request.client.host
        if request.client is not None
        else "unknown"
    )

    return f"{client_host}:{route_group}"


def _check_rate_limit(
    request: Request,
) -> tuple[bool, int]:
    """
    检查请求频率。

    返回：
        (是否允许请求, 建议等待秒数)
    """
    path = request.url.path

    if path == "/chat":
        limit = CHAT_REQUESTS_PER_MINUTE
    else:
        limit = DEFAULT_REQUESTS_PER_MINUTE

    key = _get_rate_limit_key(request)
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    with _RATE_LIMIT_LOCK:
        history = _REQUEST_HISTORY[key]

        while history and history[0] <= window_start:
            history.popleft()

        if len(history) >= limit:
            retry_after = max(
                1,
                math.ceil(
                    history[0]
                    + RATE_LIMIT_WINDOW_SECONDS
                    - now
                ),
            )
            return False, retry_after

        history.append(now)

        # 定期清理已经没有有效请求记录的 IP
        if len(_REQUEST_HISTORY) > 10000:
            expired_keys = [
                item_key
                for item_key, timestamps in _REQUEST_HISTORY.items()
                if not timestamps
                or timestamps[-1] <= window_start
            ]

            for item_key in expired_keys:
                _REQUEST_HISTORY.pop(item_key, None)

    return True, 0


def _add_security_headers(response: Response) -> Response:
    """添加基础安全响应头。"""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"

    return response


async def api_security_middleware(
    request: Request,
    call_next,
):
    """统一处理 API 身份验证、请求限流和安全响应头。"""

    path = request.url.path

    docs_enabled = _is_enabled(
        os.getenv("SHOPMATE_ENABLE_DOCS"),
        default=True,
    )

    # 开发环境开放文档时，不对文档接口进行 API Key 验证和限流。
    if docs_enabled and path in DOCUMENTATION_PATHS:
        response = await call_next(request)
        return _add_security_headers(response)

    # 先进行请求频率限制
    allowed, retry_after = _check_rate_limit(request)

    if not allowed:
        response = JSONResponse(
            status_code=429,
            content={
                "detail": "请求过于频繁，请稍后重试"
            },
            headers={
                "Retry-After": str(retry_after)
            },
        )
        return _add_security_headers(response)

    # 读取专门用于 ShopMate API 访问的密钥
    expected_api_key = os.getenv(
        "SHOPMATE_API_KEY",
        "",
    ).strip()

    require_api_key = _is_enabled(
        os.getenv("SHOPMATE_REQUIRE_API_KEY"),
        default=False,
    )

    # 如果生产环境要求身份验证，却没有配置密钥，则拒绝请求
    if require_api_key and not expected_api_key:
        response = JSONResponse(
            status_code=503,
            content={
                "detail": "API 身份验证尚未正确配置"
            },
        )
        return _add_security_headers(response)

    # 配置了密钥后，所有非文档接口都必须携带正确密钥
    if expected_api_key:
        supplied_api_key = request.headers.get(
            "X-API-Key",
            "",
        )

        if not hmac.compare_digest(
            supplied_api_key,
            expected_api_key,
        ):
            response = JSONResponse(
                status_code=401,
                content={
                    "detail": "API 身份验证失败"
                },
            )
            return _add_security_headers(response)

    response = await call_next(request)

    return _add_security_headers(response)
