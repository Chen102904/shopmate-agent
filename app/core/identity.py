"""用户身份上下文模块。

作用：
    在一次请求内部，把「当前是哪个用户在问」这件事传递到任意深度的调用栈，
    不需要在每个函数签名里手动加参数。

身份来源优先级：
    1. HTTP 请求头 X-User-Id（真实部署时应由登录态 / JWT 中间件注入）
    2. 未携带时回落到演示账号 DEFAULT_DEMO_USER_ID

重要说明（务必写进 README 和面试口径）：
    本模块只做「身份传递 + 数据归属校验」，不做认证。
    X-User-Id 由客户端自行填写，可以被伪造，因此这里的隔离是
    **演示级隔离**，不是生产级鉴权。
    真实上线必须替换为签名令牌（JWT / Session），并在网关或中间件层校验签名，
    然后把校验通过的用户 ID 注入到 set_current_user_id()。
    替换点只有一处：app/api/chat.py 里读取请求头的地方。
"""

import logging
import time

from contextvars import ContextVar, Token
from functools import wraps
from inspect import signature


logger = logging.getLogger(__name__)


# 演示模式下未携带身份时使用的账号。
# 与 data/mock_data.py 中的 ORD001 / ORD002 归属一致。
DEFAULT_DEMO_USER_ID = "USER001"


# 当前请求的用户 ID。
# 默认值为空字符串，get_current_user_id() 会把空值兜底成演示账号。
user_id_var: ContextVar[str] = ContextVar(
    "user_id",
    default="",
)


# 演示模式告警只打一次，避免每条日志都刷屏。
_demo_fallback_warned = False


def set_current_user_id(
    user_id: str | None,
) -> Token:
    """设置当前请求的用户 ID，返回可用于恢复的 Token。"""

    global _demo_fallback_warned

    normalized = (user_id or "").strip()

    if not normalized:
        normalized = DEFAULT_DEMO_USER_ID

        if not _demo_fallback_warned:
            _demo_fallback_warned = True

            logger.warning(
                "未携带用户身份，已回落到演示账号 | user_id=%s | "
                "mode=demo（生产环境必须由鉴权中间件注入真实用户）",
                normalized,
            )

    return user_id_var.set(normalized)


def get_current_user_id() -> str:
    """获取当前请求的用户 ID。"""

    current = user_id_var.get()

    return current or DEFAULT_DEMO_USER_ID


def reset_current_user_id(token: Token) -> None:
    """恢复上一个身份上下文。"""

    user_id_var.reset(token)


def with_user_identity(func):
    """
    用户身份装饰器。

    从参数中取出 user_id，写入上下文变量，
    请求结束后自动恢复，避免不同请求之间串身份。

    注意：必须在函数签名中保留 user_id 参数，
    否则调用方传入的 user_id 会被当成未知参数。

    这里用 bind_partial 取值而不是直接 pop kwargs，
    原因是调用方可能按位置传入 user_id（第三个参数），
    只 pop kwargs 会漏掉这种写法，导致上下文里仍是演示账号。
    """

    function_signature = signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):

        bound_arguments = function_signature.bind_partial(
            *args,
            **kwargs,
        )

        user_id = bound_arguments.arguments.get("user_id")

        token = set_current_user_id(user_id)

        try:
            return func(*args, **kwargs)
        finally:
            reset_current_user_id(token)

    return wrapper


class ToolPermissionError(Exception):
    """工具层的数据归属校验失败。

    与「订单不存在」共用同一条对外文案，
    避免通过错误信息差异枚举出系统里真实存在的订单号。
    """


# 对外统一文案：不区分「不存在」和「不属于你」。
ORDER_NOT_ACCESSIBLE_MESSAGE = "订单不存在或无权访问"

# 兼容短横线写法，部分客户端会发 X-User-Id / x-user-id。
USER_ID_HEADER = "X-User-Id"
