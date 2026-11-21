"""FastAPI 身份依赖。

所有需要「知道当前用户是谁」的路由，统一从这里取，
不要各自解析请求头，避免出现某个路由忘了校验的情况。

身份来源优先级：
    1. `Authorization: Bearer <token>` —— 服务端签发，验签通过才认
    2. `X-User-Id` —— 演示模式，客户端自报，可伪造
    3. 都没有 —— 回落到演示账号 USER001

    带了 Authorization 但验签失败时**直接 401**，不回落。
    否则攻击者只要随便传一个坏令牌，就能退回到自报身份。

生产环境替换点：
    把 authenticate / verify_token 换成你们的统一认证客户端，
    路由和业务代码都不用改。
"""

from fastapi import Header, HTTPException

from app.core.auth import TokenError, verify_token
from app.core.identity import (
    DEFAULT_DEMO_USER_ID,
    USER_ID_HEADER,
)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_user_id(
    authorization: str | None = Header(
        default=None,
        description="Bearer 令牌，由 POST /auth/token 获取",
    ),
    x_user_id: str | None = Header(
        default=None,
        alias=USER_ID_HEADER,
        description=(
            "演示模式下的用户编号，可被伪造。"
            "生产环境应只接受 Authorization。"
        ),
    ),
) -> str:
    """解析并返回当前用户编号。"""

    if authorization and authorization.strip():

        scheme, _, token = authorization.strip().partition(" ")

        if scheme.lower() != "bearer" or not token.strip():
            raise _unauthorized("身份令牌格式不正确")

        try:
            return verify_token(token.strip())
        except TokenError:
            # 不返回具体原因（过期/签名错/格式错），
            # 避免给攻击者提供试探反馈。
            raise _unauthorized("身份令牌无效或已过期")

    return (x_user_id or "").strip() or DEFAULT_DEMO_USER_ID
