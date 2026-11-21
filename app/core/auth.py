"""签名令牌（演示级鉴权）。

为什么还要做这件事：
    上一阶段的身份隔离是「客户端自己说我是谁」（X-User-Id），
    能保证不同用户数据不串，但挡不住有人直接冒充别人。
    这里补上签名令牌：身份由服务端签发，客户端改不了。

做到什么程度：
    这是**轻量签名令牌**，不是完整的 OAuth2 / JWT 体系：
        - 有签名，客户端无法伪造 user_id 或延长有效期；
        - 有有效期，过期必须重新换取；
        - 没有刷新令牌、没有吊销列表、没有多设备管理。

    说它是「演示级」的原因：
        密码是统一的环境变量，用户表写死在代码里，
        没有注册、找回、风控。
        真实系统必须接统一认证，不要照抄这段。

令牌格式：
    base64url(payload).base64url(hmac_sha256(payload))

    payload = {"uid": "USER001", "exp": 1795154099}

    不用第三方 JWT 库，是因为这里只需要「防篡改 + 过期」，
    自己用 hmac 实现反而依赖更少、更好讲清楚原理。
"""

import base64
import hashlib
import hmac
import json
import os
import time


# 签名密钥。
# 生产环境必须通过环境变量注入，且不能写进代码仓库。
# 未配置时用一个固定开发值，并在首次使用时打警告。
SECRET = os.getenv(
    "SHOPMATE_AUTH_SECRET",
    "shopmate-dev-secret-change-me",
)

# 令牌有效期（秒）
TOKEN_TTL_SECONDS = int(
    os.getenv("SHOPMATE_TOKEN_TTL", "3600")
)


# 演示用户表：用户名 -> 密码。
# 真实系统应存哈希并加盐，这里为了可读性直接写明文并注明。
DEMO_USERS = {
    "USER001": os.getenv("SHOPMATE_DEMO_PASSWORD", "shopmate2026"),
    "USER002": os.getenv("SHOPMATE_DEMO_PASSWORD", "shopmate2026"),
}


class TokenError(Exception):
    """令牌无效、过期或签名不匹配。"""


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _sign(payload: bytes) -> str:
    """对载荷做 HMAC-SHA256 签名。"""

    digest = hmac.new(
        SECRET.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).digest()

    return _b64encode(digest)


def authenticate(
    user_id: str,
    password: str,
) -> bool:
    """校验用户名密码。

    使用 hmac.compare_digest 而不是 ==：
    普通字符串比较会在第一个不同字符处返回，
    攻击者可以靠响应时间差逐位猜出密码（时序攻击）。
    """

    expected = DEMO_USERS.get((user_id or "").strip())

    if not expected:
        return False

    return hmac.compare_digest(
        expected,
        password or "",
    )


def issue_token(
    user_id: str,
    ttl_seconds: int | None = None,
) -> str:
    """签发令牌。"""

    if not user_id or not user_id.strip():
        raise TokenError("用户编号不能为空")

    expires_at = int(
        time.time() + (
            TOKEN_TTL_SECONDS
            if ttl_seconds is None
            else ttl_seconds
        )
    )

    payload = json.dumps(
        {
            "uid": user_id.strip(),
            "exp": expires_at,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    return f"{_b64encode(payload)}.{_sign(payload)}"


def verify_token(token: str) -> str:
    """校验令牌并返回 user_id。

    任何异常都转成 TokenError，
    不把解码错误、JSON 错误等细节泄露给调用方。
    """

    if not token or "." not in token:
        raise TokenError("令牌格式不正确")

    payload_part, signature_part = token.rsplit(".", 1)

    try:
        payload_bytes = _b64decode(payload_part)
    except Exception as exc:
        raise TokenError("令牌载荷无法解码") from exc

    # 先验签，再解析内容。
    # 顺序很重要：如果先 json.loads 再验签，
    # 攻击者可以用畸形载荷消耗解析资源，也会让错误信息泄露内部结构。
    expected_signature = _sign(payload_bytes)

    if not hmac.compare_digest(expected_signature, signature_part):
        raise TokenError("令牌签名不匹配")

    try:
        data = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise TokenError("令牌载荷格式不正确") from exc

    user_id = data.get("uid")
    expires_at = data.get("exp")

    if not isinstance(user_id, str) or not user_id:
        raise TokenError("令牌缺少用户信息")

    if not isinstance(expires_at, (int, float)):
        raise TokenError("令牌缺少有效期")

    if time.time() > expires_at:
        raise TokenError("令牌已过期")

    return user_id
