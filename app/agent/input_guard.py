
from dataclasses import dataclass


# 用户消息最大长度
MAX_MESSAGE_LENGTH = 2000


@dataclass
class InputCheckResult:
    """用户输入校验结果。"""

    valid: bool
    message: str
    error: str | None = None


def validate_user_input(user_message: str) -> InputCheckResult:
    """
    校验用户输入。

    当前负责：
    1. 检查空消息。
    2. 清理首尾空格。
    3. 限制消息最大长度。

    注意：
    本模块不负责判断用户的真实意图。
    """

    if not isinstance(user_message, str):
        return InputCheckResult(
            valid=False,
            message="",
            error="输入内容必须是字符串。",
        )

    # 清理首尾空格
    cleaned_message = user_message.strip()

    # 检查空消息
    if not cleaned_message:
        return InputCheckResult(
            valid=False,
            message="",
            error="请输入你的问题。",
        )

    # 检查消息长度
    if len(cleaned_message) > MAX_MESSAGE_LENGTH:
        return InputCheckResult(
            valid=False,
            message="",
            error=f"消息长度不能超过 {MAX_MESSAGE_LENGTH} 个字符。",
        )

    return InputCheckResult(
        valid=True,
        message=cleaned_message,
    )
