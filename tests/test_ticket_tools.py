"""售后申请工具测试：草稿 + 确认。

这份测试守的是整个写操作链路最关键的两个不变量：

    1. draft 阶段绝对不能落库（否则二次确认就是摆设）
    2. 没有明确确认时绝对不能落库（否则闸门没生效）

其余用例覆盖归属、过期、幂等重放、参数容错。
"""

import pytest
import uuid

from app.agent.conversation_store import add_message
from app.agent.ticket_draft import get_draft
from app.agent.ticket_tools import (
    confirm_after_sales_ticket,
    draft_after_sales_ticket,
    get_after_sales_tickets,
)
from app.core.db import get_connection
from app.core.identity import ORDER_NOT_ACCESSIBLE_MESSAGE


USER = "USER001"
OTHER_USER = "USER002"

# 每个用例会由 fixture 换成独立会话。
# 会话消息表不随 conftest 清理，共用会话会让上一个用例的消息
# 影响下一个用例的「最后一条用户消息」判定。
CONVERSATION = "conv-ticket-001"


@pytest.fixture(autouse=True)
def fresh_conversation():
    """每个用例用一个全新会话，避免消息跨用例残留。"""

    global CONVERSATION
    CONVERSATION = f"conv-ticket-{uuid.uuid4().hex[:8]}"
    yield


def _ticket_count() -> int:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS c FROM after_sales_tickets"
        ).fetchone()
    return int(row["c"])


def _say(message: str, conversation_id: str | None = None) -> None:
    """把一条用户消息写进会话，模拟真实对话历史。

    confirm 的确认闸门读的就是这里存的内容，
    所以测试必须走同一条路径，不能直接传字符串。
    """

    add_message(
        conversation_id or CONVERSATION,
        {"role": "user", "content": message},
    )


def _draft(
    order_id: str = "ORD001",
    request_type: str = "refund",
    reason: str = "商品破损",
    user_id: str = USER,
    conversation_id: str | None = None,
) -> dict:
    return draft_after_sales_ticket(
        order_id=order_id,
        request_type=request_type,
        reason=reason,
        user_id=user_id,
        conversation_id=conversation_id or CONVERSATION,
    )


# ============================================================
# 草稿：正常路径
# ============================================================


def test_draft_returns_awaiting_confirmation():
    result = _draft()

    assert result["status"] == "awaiting_confirmation"
    assert result["draft_id"].startswith("drf_")
    assert result["order_id"] == "ORD001"
    assert result["request_type"] == "refund"
    assert "ORD001" in result["summary"]
    assert "商品破损" in result["summary"]


def test_draft_does_not_create_ticket():
    """最重要的一条：草稿阶段数据库里必须一张工单都没有。

    如果这条挂了，说明「二次确认」只是提示词层面的装饰，
    模型一旦误判就直接落库了。
    """

    _draft()

    assert _ticket_count() == 0


def test_draft_normalizes_order_id_case():
    result = _draft(order_id="ord001")

    assert result["order_id"] == "ORD001"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("refund", "refund"),
        ("退款", "refund"),
        ("return", "return"),
        ("退货", "return"),
        ("exchange", "exchange"),
        ("换货", "exchange"),
        ("REFUND", "refund"),
    ],
)
def test_draft_accepts_type_aliases(raw, expected):
    """模型不一定严格按 enum 传值，中文和大小写都要能接住。"""

    result = _draft(request_type=raw)

    assert result["request_type"] == expected


# ============================================================
# 草稿：拒绝路径
# ============================================================


def test_draft_rejects_other_users_order():
    """ORD003 属于 USER002，USER001 拿不到草稿。"""

    result = _draft(order_id="ORD003", user_id=USER)

    assert result["status"] == "not_accessible"
    assert result["error"] == ORDER_NOT_ACCESSIBLE_MESSAGE
    assert _ticket_count() == 0


def test_draft_rejects_missing_order():
    result = _draft(order_id="ORD999")

    assert result["status"] == "not_accessible"


def test_draft_rejects_empty_reason():
    result = _draft(reason="   ")

    assert "error" in result
    assert "理由" in result["error"]


def test_draft_rejects_overlong_reason():
    """超长理由可能是注入载荷，也可能是模型把整段对话塞进来。"""

    result = _draft(reason="商" * 300)

    assert "error" in result
    assert "过长" in result["error"]


def test_draft_rejects_invalid_type():
    result = _draft(request_type="投诉")

    assert "error" in result
    assert "refund" in result["error"]


def test_draft_rejects_refund_when_not_eligible():
    """ORD002 正在运输中，不符合退款条件。

    这里要在草稿阶段就拒掉，
    不能等到用户点确认之后才告诉他不行。
    """

    result = _draft(order_id="ORD002", request_type="refund")

    assert result["eligible"] is False
    assert result["status"] == "in_transit"
    assert "运输中" in result["error"]
    assert "draft_id" not in result
    assert _ticket_count() == 0


def test_draft_allows_return_for_in_transit_order():
    """运输中的订单不能退款，但可以走退货——不同申请类型规则不同。

    如果这里被退款预检连坐挡住，说明预检写得太粗。
    """

    result = _draft(order_id="ORD002", request_type="return")

    assert result["status"] == "awaiting_confirmation"


# ============================================================
# 确认：闸门
# ============================================================


def test_confirm_blocked_without_explicit_confirmation():
    """用户没说要提交时，confirm 必须被拦住。"""

    draft = _draft()
    _say("商品破损了，怎么处理")

    result = confirm_after_sales_ticket(
        draft_id=draft["draft_id"],
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "confirmation_required"
    assert _ticket_count() == 0


def test_confirm_blocked_when_user_asks_a_question():
    """「建个工单要多久」是提问，不是同意。"""

    draft = _draft()
    _say("建个工单要多久")

    result = confirm_after_sales_ticket(
        draft_id=draft["draft_id"],
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "confirmation_required"
    assert _ticket_count() == 0


def test_confirm_blocked_when_user_declines():
    """用户反悔时不能被建单。"""

    draft = _draft()
    _say("先别建，我再想想")

    result = confirm_after_sales_ticket(
        draft_id=draft["draft_id"],
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "confirmation_required"
    assert _ticket_count() == 0


def test_confirm_blocked_when_user_uncertain():
    """「不太确定」含「确定」，不能因此通过闸门。"""

    draft = _draft()
    _say("我不太确定")

    result = confirm_after_sales_ticket(
        draft_id=draft["draft_id"],
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "confirmation_required"
    assert _ticket_count() == 0


def test_confirm_blocked_when_same_turn_as_draft():
    """用户在提出申请的那句话里就带了「确认」，也不能同一轮提交。

    这是最隐蔽的一个洞：
        用户说「ORD001 破损了，确认退款吧」，
        这句话本身含「确认」，如果只看关键词，
        draft 之后紧接着 confirm 会被放过。
        但用户根本没看到系统整理出来的草稿内容
        ——订单号对不对、理由写成了什么，他都没机会检查。

    兜法：草稿记下生成时最后一条用户消息的序号，
    confirm 要求确认消息的序号更大。
    同一轮里这就是同一条消息，必然被拦。
    """

    _say("ORD001 商品破损了，确认退款吧")

    draft = _draft()

    result = confirm_after_sales_ticket(
        draft_id=draft["draft_id"],
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "confirmation_required"
    assert "草稿" in result["error"]
    assert _ticket_count() == 0


def test_confirm_passes_after_user_replies_in_next_turn():
    """下一轮单独回一句「确认」就能过——闸门不能把正常流程也堵死。"""

    _say("ORD001 商品破损了，确认退款吧")
    draft = _draft()

    _say("确认")

    result = confirm_after_sales_ticket(
        draft_id=draft["draft_id"],
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "submitted"
    assert _ticket_count() == 1


# ============================================================
# 确认：正常路径
# ============================================================


def test_confirm_creates_ticket_after_approval():
    draft = _draft()
    _say("确认")

    result = confirm_after_sales_ticket(
        draft_id=draft["draft_id"],
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "submitted"
    assert result["ticket_id"]
    assert result["ticket_status"] == "pending"
    assert _ticket_count() == 1


def test_confirm_is_idempotent_on_repeat():
    """用户连点两次确认，只能产生一张工单。"""

    draft = _draft()
    _say("确认")

    first = confirm_after_sales_ticket(
        draft_id=draft["draft_id"],
        user_id=USER,
        conversation_id=CONVERSATION,
    )
    second = confirm_after_sales_ticket(
        draft_id=draft["draft_id"],
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert first["ticket_id"] == second["ticket_id"]
    assert _ticket_count() == 1


def test_confirm_replay_is_flagged():
    """重放要明确标记出来，模型才不会跟用户说「又提交了一次」。"""

    draft = _draft()
    _say("确认")

    confirm_after_sales_ticket(
        draft_id=draft["draft_id"],
        user_id=USER,
        conversation_id=CONVERSATION,
    )
    replayed = confirm_after_sales_ticket(
        draft_id=draft["draft_id"],
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert replayed["idempotent_replay"] is True


# ============================================================
# 确认：归属与失效
# ============================================================


def test_confirm_rejects_other_users_draft():
    """别人拿到 draft_id 也确认不了。"""

    draft = _draft()
    _say("确认")

    result = confirm_after_sales_ticket(
        draft_id=draft["draft_id"],
        user_id=OTHER_USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "draft_not_found"
    assert _ticket_count() == 0


def test_confirm_rejects_other_conversation():
    """换个会话，同一张草稿失效。"""

    draft = _draft()
    _say("确认")

    result = confirm_after_sales_ticket(
        draft_id=draft["draft_id"],
        user_id=USER,
        conversation_id="conv-other",
    )

    assert result["status"] == "draft_not_found"
    assert _ticket_count() == 0


def test_confirm_rejects_fake_draft_id():
    """模型编造 draft_id 时必须失败，不能靠猜中。"""

    _say("确认")

    result = confirm_after_sales_ticket(
        draft_id="drf_0000000000000000000000000000",
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "draft_not_found"
    assert _ticket_count() == 0


def test_confirm_rejects_expired_draft():
    from datetime import datetime, timedelta

    draft = _draft()
    _say("确认")

    past = (datetime.now() - timedelta(seconds=1)).isoformat()

    with get_connection() as connection:
        connection.execute(
            "UPDATE ticket_drafts SET expires_at = ? WHERE draft_id = ?",
            (past, draft["draft_id"]),
        )

    result = confirm_after_sales_ticket(
        draft_id=draft["draft_id"],
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "draft_expired"
    assert _ticket_count() == 0


def test_confirm_without_draft_id_uses_latest():
    """省略 draft_id 时自动用最近一张待确认草稿。

    这是真实跑模型踩出来的问题：
        draft_id 只存在于「生成草稿那一轮」的上下文里，
        下一轮模型看不到它（工具返回的 JSON 不落库），
        于是它编一串编号，confirm 报「草稿不存在」，
        用户看到「提交失败」——但草稿其实好好的。
    """

    _say("ORD001 商品破损了，我要退款")
    draft = _draft()
    _say("确认")

    result = confirm_after_sales_ticket(
        draft_id=None,
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "submitted"
    assert result["ticket_id"]
    assert _ticket_count() == 1

    # 用的确实是那张草稿
    loaded = get_draft(draft["draft_id"], USER, CONVERSATION)
    assert loaded.status == "consumed"


def test_confirm_without_any_draft_returns_not_found():
    """没有草稿时确认，要给出明确的下一步，而不是含糊失败。"""

    _say("确认")

    result = confirm_after_sales_ticket(
        draft_id=None,
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "draft_not_found"
    assert "draft_after_sales_ticket" in result["hint"]
    assert _ticket_count() == 0


def test_confirm_with_wrong_draft_id_does_not_fall_back():
    """传了错的编号就报错，不能悄悄回落到最新草稿。

    一旦允许回落，模型每次编造都能成功，
    「draft_id 必须是真实存在的」这条约束就名存实亡了。
    """

    _say("ORD001 商品破损了，我要退款")
    _draft()
    _say("确认")

    result = confirm_after_sales_ticket(
        draft_id="drf_wrong_id",
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "draft_not_found"
    assert "不要自己编造" in result["hint"]
    assert _ticket_count() == 0


def test_confirm_uses_most_recent_of_multiple_drafts():
    """同一会话有多张草稿时，确认的是最新那张。"""

    _say("ORD001 商品破损了，我要退款")
    _draft(reason="商品破损")

    _say("算了还是因为尺码不合适")
    _draft(reason="尺码不合适")

    _say("确认")

    result = confirm_after_sales_ticket(
        draft_id=None,
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "submitted"

    with get_connection() as connection:
        row = connection.execute(
            "SELECT reason FROM after_sales_tickets WHERE ticket_id = ?",
            (result["ticket_id"],),
        ).fetchone()

    assert row["reason"] == "尺码不合适"


# ============================================================
# 完整流程
# ============================================================


def test_draft_reuses_identical_content():
    """同样内容重复 draft，必须返回同一张草稿。

    不这么做会死锁：模型在用户说「确认」的那一轮
    又调一次 draft 重新整理内容，新草稿的把关基准
    就变成这条「确认」消息本身，永远提交不了。
    """

    first = _draft()
    second = _draft()

    assert first["draft_id"] == second["draft_id"]
    assert second.get("reused") is True
    assert _ticket_count() == 0


def test_draft_creates_new_when_content_differs():
    """理由不同就是另一张申请，不能复用同一张草稿。"""

    first = _draft(reason="商品破损")
    second = _draft(reason="尺码不合适")

    assert first["draft_id"] != second["draft_id"]


def test_redraft_in_confirm_turn_still_submits():
    """模型在「确认」那一轮又调了一次 draft，也必须能提交。

    这是真实跑模型时踩到的死锁：
    模型习惯每轮重新整理申请内容，
    新草稿的基准消息变成当前这条「确认」，
    「确认必须晚于草稿」就永远不成立。
    复用完全相同内容的草稿可以解掉。
    """

    _say("ORD001 商品破损了，我要退款")
    first = _draft()

    _say("确认")
    again = _draft()

    assert again["draft_id"] == first["draft_id"]

    result = confirm_after_sales_ticket(
        draft_id=again["draft_id"],
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "submitted"
    assert _ticket_count() == 1


def test_executor_handles_missing_draft_id():
    """模型完全不传 draft_id 时，执行器不能抛 TypeError。

    真实踩过这个坑：
        签名里 draft_id 是必填位置参数，
        模型省略它 → TypeError → 被兜底成「工具暂时不可用」，
        用户看到「系统报错」，而草稿其实好好的。

    这条同时守住「工具签名要容忍模型少传参数」这个约定。
    """

    import json

    from app.agent.tool_executor import execute_tool

    output = execute_tool(
        "confirm_after_sales_ticket",
        {},
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    result = json.loads(output)

    assert result["status"] == "draft_not_found"
    assert "error" in result


def test_executor_blocks_write_without_conversation():
    """写操作没有会话上下文时，必须拒绝而不是随便挑一个会话。"""

    import json

    from app.agent.tool_executor import execute_tool

    output = execute_tool(
        "confirm_after_sales_ticket",
        {},
        user_id=USER,
        conversation_id=None,
    )

    result = json.loads(output)

    assert "会话上下文" in result["error"]


def test_repeat_confirm_after_submit_replays():
    """用户又说了一遍「确认」，要回放原工单，不能说「没有待确认申请」。

    否则用户会以为没提交成功，转头再申请一次。
    """

    _say("ORD001 商品破损了，我要退款")
    _draft()
    _say("确认")

    first = confirm_after_sales_ticket(
        draft_id=None,
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    _say("确认")

    second = confirm_after_sales_ticket(
        draft_id=None,
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert second["status"] == "submitted"
    assert second["ticket_id"] == first["ticket_id"]
    assert second["idempotent_replay"] is True
    assert _ticket_count() == 1


def test_full_flow_draft_show_then_confirm():
    """模拟真实两轮对话：
        助手：（生成草稿）我准备提交这张申请……确认吗？
        用户：确认
        助手：（真正建单）
    """

    _say("ORD001 商品破损了，我要退款")

    draft = _draft(order_id="ORD001", request_type="refund", reason="商品破损")

    assert draft["status"] == "awaiting_confirmation"
    assert _ticket_count() == 0, "草稿阶段不能落库"

    # 第二轮用户明确确认
    _say("确认")

    result = confirm_after_sales_ticket(
        draft_id=draft["draft_id"],
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "submitted"
    assert _ticket_count() == 1

    # 工单内容要和草稿一致，不能被中途篡改
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM after_sales_tickets WHERE ticket_id = ?",
            (result["ticket_id"],),
        ).fetchone()

    assert row["order_id"] == "ORD001"
    assert row["request_type"] == "refund"
    assert row["reason"] == "商品破损"
    assert row["user_id"] == USER


# ============================================================
# 查工单进度
# ============================================================


def _create_ticket(
    order_id: str = "ORD001",
    user_id: str = USER,
    reason: str = "商品破损",
) -> str:
    """直接走服务层建一张工单，跳过草稿流程（查询测试不关心怎么建的）。"""

    from app.models.after_sales import AfterSalesType
    from app.services.after_sales_service import create_after_sales_ticket

    ticket = create_after_sales_ticket(
        order_id=order_id,
        user_id=user_id,
        request_type=AfterSalesType.REFUND,
        reason=reason,
    )

    return ticket.ticket_id


def test_lookup_returns_recent_tickets():
    """用户问「我的申请怎么样了」，没给编号也能查到最近的。"""

    ticket_id = _create_ticket()

    result = get_after_sales_tickets(
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["count"] == 1
    assert result["tickets"][0]["ticket_id"] == ticket_id
    assert result["tickets"][0]["status_label"] == "待人工审核"


def test_lookup_by_order_id():
    ticket_id = _create_ticket(order_id="ORD001")

    result = get_after_sales_tickets(
        user_id=USER,
        conversation_id=CONVERSATION,
        keyword="ORD001",
    )

    assert result["count"] == 1
    assert result["tickets"][0]["ticket_id"] == ticket_id


def test_lookup_by_ticket_id():
    ticket_id = _create_ticket()

    result = get_after_sales_tickets(
        user_id=USER,
        conversation_id=CONVERSATION,
        keyword=ticket_id,
    )

    assert result["count"] == 1
    assert result["tickets"][0]["ticket_id"] == ticket_id


def test_lookup_rejects_other_users_order():
    """他人订单号不能用来问「有没有工单」——这本身也是隐私。"""

    result = get_after_sales_tickets(
        user_id=USER,
        conversation_id=CONVERSATION,
        keyword="ORD003",
    )

    assert result["status"] == "not_accessible"
    assert result["error"] == ORDER_NOT_ACCESSIBLE_MESSAGE


def test_lookup_does_not_leak_other_users_ticket():
    """别人工单号查不到，返回空而不是报错。

    返回「无权限」等于告诉对方「这个工单存在」。
    """

    # ORD003 属于 USER002，用它的身份建一张工单
    other_ticket = _create_ticket(order_id="ORD003", user_id=OTHER_USER)

    result = get_after_sales_tickets(
        user_id=USER,
        conversation_id=CONVERSATION,
        keyword=other_ticket,
    )

    assert result["count"] == 0
    assert result["tickets"] == []


def test_lookup_id_case_insensitive():
    _create_ticket(order_id="ORD001")

    result = get_after_sales_tickets(
        user_id=USER,
        conversation_id=CONVERSATION,
        keyword="ord001",
    )

    assert result["count"] == 1


def test_lookup_reports_pending_draft():
    """有草稿没提交时，必须让模型知道，否则用户以为已经在审核了。"""

    _draft(order_id="ORD001", reason="商品破损")

    result = get_after_sales_tickets(
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["count"] == 0
    assert result["pending_draft"] is not None
    assert result["pending_draft"]["order_id"] == "ORD001"
    assert result["pending_draft"]["request_type_label"] == "退款"
    assert "还没提交" in result["hint"]


def test_lookup_empty_when_nothing_exists():
    result = get_after_sales_tickets(
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["count"] == 0
    assert result["pending_draft"] is None
    assert "编造" in result["hint"]


def test_lookup_shows_both_ticket_and_draft():
    """既有已提交工单、又有新草稿时，两个都要报。"""

    _create_ticket(order_id="ORD001", reason="商品破损")
    _draft(order_id="ORD002", request_type="return", reason="尺码不合适")

    result = get_after_sales_tickets(
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["count"] == 1
    assert result["pending_draft"] is not None
    assert result["pending_draft"]["order_id"] == "ORD002"


# ============================================================
# 与流程其他环节的衔接
# ============================================================


def test_clear_conversation_removes_drafts():
    """清空会话必须一起作废草稿。

    否则会出现这个场景：
        用户说「退 ORD001，商品破损」→ 草稿生成
        用户发现说错了，清空会话想重来
        用户随口说一句「确认」
        系统把清空前那张草稿提交了

    「清空会话」在用户理解里等于「重新开始」，
    只删消息不删草稿，就等于留了个定时炸弹。
    """

    from app.agent.conversation_store import clear_conversation
    from app.agent.ticket_draft import count_drafts

    _draft(order_id="ORD001", reason="商品破损")

    assert count_drafts() == 1

    clear_conversation(CONVERSATION)

    assert count_drafts() == 0


def test_confirm_clears_cached_eligibility():
    """提交成功后，提交前的退款资格缓存必须失效。

    不清会出现这种对话：
        第 1 轮：能退吗 → 缓存「符合退款条件」
        第 2 轮：申请退款 → 工单建好
        第 3 轮：我还能退吗 → 命中第 1 轮缓存，答「可以申请退款」
        用户以为还能再申请一次，其实已经有一张在审核了。
    """

    from app.agent.tool_cache import get_cached_result, store_tool_result

    cache_key = "check_refund_eligibility:ORD001"

    store_tool_result(
        CONVERSATION,
        cache_key,
        '{"order_id": "ORD001", "eligible": true}',
    )

    assert get_cached_result(CONVERSATION, cache_key) is not None

    _say("ORD001 商品破损了，我要退款")
    _draft()
    _say("确认")

    result = confirm_after_sales_ticket(
        draft_id=None,
        user_id=USER,
        conversation_id=CONVERSATION,
    )

    assert result["status"] == "submitted"
    assert get_cached_result(CONVERSATION, cache_key) is None
