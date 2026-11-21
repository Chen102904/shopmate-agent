"""令牌签发接口。

流程：
    1. 客户端 POST /auth/token，提交 user_id + password
    2. 服务端校验通过后签发 HMAC 签名令牌
    3. 后续请求带 `Authorization: Bearer <token>`

安全说明（必须如实写进文档）：
    - 密码是统一的环境变量，用户表写死在代码里，
      没有注册、找回密码、登录失败锁定、多设备管理。
    - 这是**演示级**鉴权，不是可直接上线的认证系统。
    - 它的价值是：身份由服务端签发，客户端改不了；
      比「自己填 X-User-Id」强一个量级。
"""

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from app.core.auth import (
    TOKEN_TTL_SECONDS,
    authenticate,
    issue_token,
)


router = APIRouter(tags=["身份认证"])


class TokenRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user_id: str


@router.post("/auth/token", response_model=TokenResponse)
def create_token(request: TokenRequest):
    """用用户名密码换取访问令牌。"""

    if not authenticate(request.user_id, request.password):
        # 不区分「用户不存在」和「密码错误」，
        # 避免被用来枚举有效用户名。
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
        )

    user_id = request.user_id.strip()

    return TokenResponse(
        access_token=issue_token(user_id),
        token_type="bearer",
        expires_in=TOKEN_TTL_SECONDS,
        user_id=user_id,
    )
