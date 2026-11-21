"""演示模式。

解决什么问题：
    要把这个 Agent 发给别人试，会遇到三个「只在演示场景才存在」的问题：

    1. 限流按 IP 分组。走公网隧道时，所有访客的来源 IP 都是同一个，
       20 次/分钟的聊天额度会被十几个人分，第一个人聊几句，
       后面的人就被 429 挡住。
    2. 演示订单只属于 USER001。访客没有身份，进来看不到任何订单，
       什么都测不了。
    3. 工单按用户存。如果所有人共用 USER001，A 建的工单会出现在
       B 的「我申请怎么样了」里，看起来像串号。

做法：
    给每个访客前端生成一个 `guest-xxxxxxxx` 当身份。

    - 订单：guest 共享演示账号 USER001 的订单（能看到 ORD001/ORD002，
      查 ORD003 仍然被拒，越权保护还能演示）
    - 工单：直接用 guest id 存，天然隔离，谁也看不到谁的
    - 限流：按访客 id 分组，每人独立额度

为什么用环境变量开关而不是直接改默认行为：
    这些放宽都会削弱安全性（身份可伪造、限流可绕过）。
    生产环境绝对不能开。用一个显式开关，默认关闭，
    要开必须写 SHOPMATE_DEMO=1，日志里也会打一行提醒。
"""

from __future__ import annotations

import os


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


# 演示模式开关
DEMO_MODE = _env_bool("SHOPMATE_DEMO", default=False)

# 访客共享哪个演示账号的订单
DEMO_OWNER_ID = os.getenv("SHOPMATE_DEMO_OWNER", "USER001")

# 访客身份前缀
GUEST_PREFIX = "guest-"


def is_guest(user_id: str | None) -> bool:
    """这个身份是不是演示访客。"""

    return bool(user_id) and user_id.startswith(GUEST_PREFIX)


def resolve_demo_owner(user_id: str | None) -> str | None:
    """把访客身份换成实际用于查订单的账号。

    非演示模式、或不是访客时，原样返回——
    不能因为加了演示逻辑就让正常用户查到别人的订单。
    """

    if not DEMO_MODE:
        return user_id

    if is_guest(user_id):
        return DEMO_OWNER_ID

    return user_id
