"""售后工单的写操作工具：草稿 + 确认。

这两个工具要连起来看：

    draft_after_sales_ticket    只读，冻结参数，返回 draft_id
    confirm_after_sales_ticket  写，凭 draft_id 真正建单

为什么拆成两个而不是一个 create_ticket：

    一个 create_ticket 意味着「模型决定何时落库」。
    模型会被诱导、会误判、会在多轮里串参数，
    每一次误判都是一张真实工单。

    拆开之后，落库的前提变成四个条件同时成立：
        1. 存在一张本会话本用户的、未过期的草稿
        2. 用户最近一条消息里有明确确认表达
        3. 这条确认消息发生在草稿生成之后
        4. 草稿还没被消费过（幂等键兜住并发与重复点击）

    条件 2、3 都是代码判定的，模型绕不过去。
    这是「不把安全完全托付给模型」的具体实现。

为什么 confirm 只收 draft_id：

    如果 confirm 让模型重新传 order_id / request_type，
    模型完全可以在第二步把订单号换成别人的。
    参数只能在 draft 阶段确定一次，之后冻结。

为什么退款要在 draft 阶段预检资格：

    否则用户会走完「看摘要 → 点确认」才被告知不符合条件，
    体验差，而且会让人怀疑系统在骗确认。
    不符合条件时直接不生成草稿，把原因说清楚。
"""

import logging
import uuid

from app.agent.confirmation import is_explicit_confirmation
from app.agent.ticket_draft import (
    DraftBusy,
    DraftExpired,
    DraftNotFound,
    create_draft,
    complete_draft,
    find_latest_pending_draft,
    find_recent_consumed_draft,
    find_reusable_draft,
    get_draft,
    release_draft,
    reserve_draft,
)
from app.agent.conversation_store import get_last_user_message_record
from app.agent.tool_cache import clear_tool_cache
from app.core.identity import ORDER_NOT_ACCESSIBLE_MESSAGE
from app.core.execution_trace import record_execution_event
from app.models.after_sales import AfterSalesType
from app.services.after_sales_service import create_after_sales_ticket
from app.services.order_service import get_order_by_id
from app.services.refund_service import check_refund_eligibility


logger = logging.getLogger(__name__)


# 申请类型的中文名，用于给模型生成可复述的摘要
REQUEST_TYPE_LABELS = {
    AfterSalesType.REFUND: "退款",
    AfterSalesType.RETURN: "退货",
    AfterSalesType.EXCHANGE: "换货",
}

# 草稿编号前缀。
# 加前缀是为了让日志和工单号一眼能区分开：
# 工单号是纯 UUID，草稿是 drf_ 开头。
DRAFT_ID_PREFIX = "drf_"

# 理由长度上限。
# 太长的理由有可能是注入载荷，也可能是模型把整段对话塞进来。
# 真实售后理由超过 200 字的情况极少。
MAX_REASON_LENGTH = 200


def _parse_request_type(raw: object) -> AfterSalesType:
    """把模型传来的类型字符串转成枚举。

    校验器只检查了 type=string，没有检查 enum，
    所以这里必须自己兜底：
        模型可能传「退款」「refund_request」「换货」这类值。
    """

    if not isinstance(raw, str):
        raise ValueError("申请类型必须是字符串")

    text = raw.strip().lower()

    # 容错：模型偶尔会传中文
    aliases = {
        "退款": AfterSalesType.REFUND,
        "退货": AfterSalesType.RETURN,
        "换货": AfterSalesType.EXCHANGE,
        "refund_request": AfterSalesType.REFUND,
        "return_request": AfterSalesType.RETURN,
        "exchange_request": AfterSalesType.EXCHANGE,
    }

    if text in aliases:
        return aliases[text]

    try:
        return AfterSalesType(text)
    except ValueError as exc:
        raise ValueError(
            "申请类型只能是 refund（退款）、return（退货）、"
            "exchange（换货）三者之一"
        ) from exc


def _get_last_user_message(conversation_id: str) -> dict:
    """从存储里取本会话最近一条用户消息。

    返回 {"id": 序号, "content": 正文}；没有用户消息时返回 {"id": 0, "content": ""}。

    为什么读库而不是用调用方传进来的字符串：
        库里的才是用户真正说过、并且会留在历史里的话。
        内存变量可能被上层改过（比如拼接了订单号提示），
        拿它做安全判定不严谨。

    为什么还需要「序号」：
        草稿生成时会记下当时的最后一条用户消息序号，
        确认时用它判断用户是不是在草稿之后才表态。
    """

    record = get_last_user_message_record(conversation_id)

    if record is None:
        return {"id": 0, "content": ""}

    return record


# ============================================================
# 工具 1：生成草稿
# ============================================================


def draft_after_sales_ticket(
    order_id: str,
    request_type: str,
    reason: str,
    user_id: str,
    conversation_id: str,
) -> dict:
    """生成待确认的售后申请草稿。不写业务表。"""

    order_id = (order_id or "").strip().upper()
    reason = (reason or "").strip()

    if not reason:
        return {
            "error": "申请理由不能为空",
            "hint": "请先向用户询问具体原因，例如商品破损、尺码不合适。",
        }

    if len(reason) > MAX_REASON_LENGTH:
        return {
            "error": f"申请理由过长（{len(reason)} 字），上限 {MAX_REASON_LENGTH} 字",
            "hint": "请把理由概括成一句话后重试，不要粘贴整段对话。",
        }

    try:
        request_type_enum = _parse_request_type(request_type)
    except ValueError as exc:
        return {"error": str(exc)}

    # 归属校验：查不到和不是本人的订单走同一个错误，
    # 不泄露订单是否存在。
    order = get_order_by_id(order_id, user_id)

    if order is None:
        record_execution_event(
            event="ticket_draft",
            status="denied",
            details={"reason": "order_not_accessible"},
        )
        return {
            "error": ORDER_NOT_ACCESSIBLE_MESSAGE,
            "status": "not_accessible",
        }

    # 退款类申请先预检资格：
    # 不符合条件就不生成草稿，避免用户确认完才被拒。
    if request_type_enum == AfterSalesType.REFUND:

        eligibility = check_refund_eligibility(order_id, user_id)

        if eligibility.get("eligible") is not True:

            reason_text = eligibility.get("reason") or "暂不满足退款条件"

            record_execution_event(
                event="ticket_draft",
                status="rejected",
                details={"reason": "refund_not_eligible"},
            )

            return {
                "error": reason_text,
                "status": eligibility.get("status") or "not_eligible",
                "eligible": False,
                "hint": "不要生成草稿。把不符合的原因告诉用户，"
                        "如有争议建议联系人工客服。",
            }

    # 已经有内容完全一致的草稿就直接复用。
    #
    # 不这么做会死锁：模型在用户说「确认」的那一轮
    # 又调用一次 draft 重新整理内容，新草稿的把关基准
    # 就变成了这条「确认」消息本身，
    # 于是永远满足不了「确认必须晚于草稿」。
    existing = find_reusable_draft(
        conversation_id=conversation_id,
        user_id=user_id,
        order_id=order_id,
        request_type=request_type_enum.value,
        reason=reason,
    )

    if existing is not None:

        label = REQUEST_TYPE_LABELS[request_type_enum]

        record_execution_event(
            event="ticket_draft",
            status="reused",
            details={"request_type": request_type_enum.value},
        )

        return {
            "draft_id": existing.draft_id,
            "status": "awaiting_confirmation",
            "reused": True,
            "summary": (
                f"订单 {order_id} 的{label}申请，"
                f"理由：{reason}"
            ),
            "order_id": order_id,
            "request_type": request_type_enum.value,
            "request_type_label": label,
            "reason": reason,
            "expires_in_seconds": existing.expires_in_seconds(),
            "hint": (
                "这张草稿之前已经生成过了，内容完全一致，直接沿用。"
                "如果用户还没确认，把内容再复述一遍并询问是否提交；"
                "如果用户已经确认过，直接用同一个 draft_id 调用 "
                "confirm_after_sales_ticket。"
            ),
        }

    draft_id = DRAFT_ID_PREFIX + uuid.uuid4().hex

    # 记下当前会话最后一条用户消息的序号。
    # confirm 时要求「用户表态的那句」序号比它大，
    # 才能证明用户是在草稿生成之后才同意的。
    last_record = _get_last_user_message(conversation_id)

    draft = create_draft(
        draft_id=draft_id,
        conversation_id=conversation_id,
        user_id=user_id,
        order_id=order_id,
        request_type=request_type_enum.value,
        reason=reason,
        created_after_message_id=last_record["id"],
    )

    label = REQUEST_TYPE_LABELS[request_type_enum]

    record_execution_event(
        event="ticket_draft",
        status="success",
        details={
            "request_type": request_type_enum.value,
            "reason_length": len(reason),
        },
    )

    return {
        "draft_id": draft.draft_id,
        "status": "awaiting_confirmation",
        "summary": (
            f"订单 {order_id} 的{label}申请，"
            f"理由：{reason}"
        ),
        "order_id": order_id,
        "request_type": request_type_enum.value,
        "request_type_label": label,
        "reason": reason,
        "expires_in_seconds": draft.expires_in_seconds(),
        "hint": (
            "这是一张待确认草稿，系统还没有创建工单。"
            "请把 summary 的内容完整告诉用户，"
            "并明确询问「是否确认提交」。"
            "在用户明确同意之前，不要调用 confirm_after_sales_ticket。"
        ),
    }


# ============================================================
# 工具 2：确认落库
# ============================================================


def confirm_after_sales_ticket(
    user_id: str,
    conversation_id: str,
    draft_id: str | None = None,
) -> dict:
    """凭草稿真正创建工单。

    draft_id 放在最后且可省略，因为模型经常不传：
        agent 只把最终回答写进会话历史，工具返回的 JSON 不落库，
        下一轮模型看不到上一轮的 draft_id。
        签名里给它默认值，省略时走「最近一张待确认草稿」。

    三道闸门按顺序：
        1. 草稿存在，且属于本人本会话
        2. 用户最近一条消息是明确确认，且晚于草稿生成
        3. 幂等键 draft:<id> 保证重复确认只建一张

    为什么归属检查排在确认检查之前：
        草稿不是你的，就不该进入「请你确认」这个流程。
        先判归属，语义上更干净，也少了无用的库查询。
    """

    draft_id = (draft_id or "").strip()

    # 闸门 1：找到要确认的草稿。
    #
    # draft_id 不是必填。
    # 原因：agent 只把最终回答写进会话历史，工具返回的 JSON 不落库，
    # 所以下一轮模型看不到上一轮的 draft_id。
    # 硬要求它传，它只会编一串不存在的编号，用户看到「提交失败」。
    #
    # 传了就严格按传的找（不回落），避免模型编造后蒙对别人的草稿；
    # 没传就取本会话最近一张待确认的。
    if draft_id:

        try:
            draft = get_draft(
                draft_id=draft_id,
                user_id=user_id,
                conversation_id=conversation_id,
            )

        except DraftNotFound as exc:
            record_execution_event(
                event="ticket_confirm",
                status="denied",
                details={"reason": "draft_not_found"},
            )
            return {
                "error": str(exc),
                "status": "draft_not_found",
                "hint": (
                    "不要自己编造 draft_id。"
                    "省略 draft_id 再调用一次，系统会自动使用最近一张待确认草稿。"
                ),
            }

    else:

        draft = find_latest_pending_draft(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        if draft is None:

            # 再看一眼是不是刚刚提交过。
            # 用户连说两遍「确认」时，第一遍已经建单、草稿变 consumed，
            # 这里要是直接说「没有待确认申请」，
            # 用户会以为没成功，转头再申请一次。
            replayed = find_recent_consumed_draft(
                conversation_id=conversation_id,
                user_id=user_id,
            )

            if replayed is not None:
                record_execution_event(
                    event="ticket_confirm",
                    status="replayed",
                    details={"reason": "recent_consumed_draft"},
                )
                return {
                    "status": "submitted",
                    "ticket_id": replayed.ticket_id,
                    "order_id": replayed.order_id,
                    "request_type": replayed.request_type,
                    "idempotent_replay": True,
                    "hint": (
                        "这张申请刚才已经提交成功了。"
                        "把原工单号告诉用户，说明无需重复提交，不要建新单。"
                    ),
                }

            record_execution_event(
                event="ticket_confirm",
                status="denied",
                details={"reason": "no_pending_draft"},
            )
            return {
                "error": "当前没有待确认的售后申请",
                "status": "draft_not_found",
                "hint": (
                    "请先调用 draft_after_sales_ticket 生成申请内容，"
                    "告诉用户并拿到确认后再提交。"
                ),
            }

        draft_id = draft.draft_id

    # 闸门 2：用户有没有明确说要提交
    last_record = _get_last_user_message(conversation_id)
    confirmed, confirm_reason = is_explicit_confirmation(last_record["content"])

    if not confirmed:
        record_execution_event(
            event="ticket_confirm",
            status="blocked",
            details={"reason": confirm_reason},
        )

        logger.info(
            "确认被闸门拦截 | draft=%s | reason=%s",
            draft_id,
            confirm_reason,
        )

        return {
            "error": "用户还没有明确确认这次申请",
            "status": "confirmation_required",
            "hint": (
                "请先把草稿内容告诉用户，等用户明确表示同意"
                "（例如「确认」「可以」「提交吧」）之后，再调用本工具。"
                "如果用户表示不需要了，就不要再调用。"
            ),
        }

    # 闸门 3：用户必须先看到草稿，再表态。
    #
    # 为什么需要这一条：
    #   「ORD001 破损了，确认退款吧」这句话本身含「确认」，
    #   如果模型在同一轮里 draft 完立刻 confirm，
    #   前面两道闸门都会放过——但用户根本没看到系统整理的草稿内容，
    #   也没机会发现订单号或理由被写错。
    #
    #   用消息序号比时间可靠：同一秒内的两轮也不会判错。
    if last_record["id"] <= draft.created_after_message_id:
        record_execution_event(
            event="ticket_confirm",
            status="blocked",
            details={"reason": "confirmation_before_draft"},
        )

        logger.info(
            "确认早于草稿生成，已拦截 | draft=%s | msg_id=%s | draft_after=%s",
            draft_id,
            last_record["id"],
            draft.created_after_message_id,
        )

        return {
            "error": "用户还没有看到这张草稿",
            "status": "confirmation_required",
            "hint": (
                "草稿刚刚生成。请先把订单号、申请类型、理由告诉用户，"
                "等用户在下一轮单独回复确认之后，再调用本工具。"
                "同一轮里不能既生成草稿又提交。"
            ),
        }

    # 闸门 3：抢占草稿（含过期、并发校验）
    try:
        draft = reserve_draft(
            draft_id=draft_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    except DraftNotFound as exc:
        record_execution_event(
            event="ticket_confirm",
            status="denied",
            details={"reason": "draft_not_found"},
        )
        return {
            "error": str(exc),
            "status": "draft_not_found",
            "hint": "请重新调用 draft_after_sales_ticket 生成草稿。",
        }

    except DraftExpired as exc:
        record_execution_event(
            event="ticket_confirm",
            status="expired",
            details={"reason": "draft_expired"},
        )
        return {
            "error": str(exc),
            "status": "draft_expired",
            "hint": "请重新生成草稿并让用户再确认一次。",
        }

    except DraftBusy as exc:
        record_execution_event(
            event="ticket_confirm",
            status="busy",
            details={"reason": "draft_busy"},
        )
        return {
            "error": str(exc),
            "status": "draft_busy",
            "hint": "稍后用同一个 draft_id 重试即可，不会建出两张工单。",
        }

    # 之前已经建过单：幂等重放，直接返回原工单
    if draft.status == "consumed" and draft.ticket_id:
        record_execution_event(
            event="ticket_confirm",
            status="replayed",
            details={"reason": "already_consumed"},
        )
        return {
            "status": "submitted",
            "ticket_id": draft.ticket_id,
            "order_id": draft.order_id,
            "request_type": draft.request_type,
            "idempotent_replay": True,
            "hint": "这张申请之前已经提交过，直接把原工单号告诉用户，不要重复提交。",
        }

    # 闸门 3：真正建单。
    # 把 draft_id 当幂等键，复用 ticket_store 已有的两层防重。
    try:
        ticket, replayed = create_after_sales_ticket(
            order_id=draft.order_id,
            user_id=user_id,
            request_type=AfterSalesType(draft.request_type),
            reason=draft.reason,
            idempotency_key=f"draft:{draft.draft_id}",
            with_meta=True,
        )

    except Exception as exc:
        # 失败要释放草稿，让用户改完能重新确认，
        # 否则草稿会卡在 consuming，10 分钟后彻底作废。
        release_draft(draft_id)

        record_execution_event(
            event="ticket_confirm",
            status="error",
            details={"error_type": type(exc).__name__},
        )

        logger.warning(
            "建单失败 | draft=%s | error=%s",
            draft_id,
            type(exc).__name__,
        )

        return {
            # 不带 str(exc)：这里可能撞上数据库错误，
            # 异常原文里有表名、SQL 片段甚至文件路径。
            # 用户只需要知道「没提交成功」这一件事。
            "error": "提交时出了故障，这次没有创建工单",
            "status": "failed",
            "error_type": type(exc).__name__,
            "hint": (
                "明确告诉用户这次没有提交成功、系统里没有新工单，"
                "可以稍后重试或联系人工客服。"
                "绝对不要说成「已提交但可能延迟」——用户会以为申请已经进去了。"
            ),
        }

    complete_draft(draft_id, ticket.ticket_id)

    # 写之后要让读失效。
    #
    # 不清缓存会出现这种对话：
    #     第 1 轮：能退吗 → check_refund_eligibility 结果进缓存（TTL 300 秒）
    #     第 2 轮：帮我申请 → 工单建好了
    #     第 3 轮：我还能退吗 → 命中第 1 轮的缓存，答「可以申请退款」
    #     用户以为还能再申请一次，其实已经有一张待审核的工单了
    #
    # 唯一索引能防住重复建单，但挡不住「答非所问」。
    # 这里主动清掉，保证写完之后读到的是新状态。
    clear_tool_cache(conversation_id)

    record_execution_event(
        event="ticket_confirm",
        status="success",
        details={
            "request_type": ticket.request_type.value,
            "replayed": replayed,
        },
    )

    label = REQUEST_TYPE_LABELS.get(ticket.request_type, "售后")

    return {
        "status": "submitted",
        "ticket_id": ticket.ticket_id,
        "order_id": ticket.order_id,
        "request_type": ticket.request_type.value,
        "request_type_label": label,
        "reason": ticket.reason,
        "ticket_status": ticket.status.value,
        "created_at": ticket.created_at.isoformat(),
        "idempotent_replay": replayed,
        "hint": (
            f"工单已创建，编号 {ticket.ticket_id}，当前状态为待人工审核。"
            "请把工单号告诉用户，并说明审核结果需等待人工处理。"
            "不要承诺一定会通过。"
        ),
    }


# ============================================================
# 工具 3：查询工单进度
# ============================================================


# 工单状态的用户可读说法。
# 不直接把枚举值（pending）交给模型：
# 它有时会原样念出来，用户看不懂。
STATUS_LABELS = {
    "pending": "待人工审核",
    "approved": "审核通过",
    "rejected": "审核未通过",
    "processing": "处理中",
    "completed": "已完成",
}


def _ticket_to_dict(ticket) -> dict:
    """把工单对象转成给模型看的精简结构。"""

    return {
        "ticket_id": ticket.ticket_id,
        "order_id": ticket.order_id,
        "request_type": ticket.request_type.value,
        "request_type_label": REQUEST_TYPE_LABELS.get(
            ticket.request_type, "售后"
        ),
        "status": ticket.status.value,
        "status_label": STATUS_LABELS.get(
            ticket.status.value, ticket.status.value
        ),
        "reason": ticket.reason,
        "created_at": ticket.created_at.isoformat(timespec="seconds"),
    }


def get_after_sales_tickets(
    user_id: str,
    conversation_id: str,
    keyword: str | None = None,
) -> dict:
    """查询用户的售后工单进度。

    这是写操作之后原本缺的那一环：
        系统能提交申请，但用户提交完问「我那个退款怎么样了」，
        原来没有任何工具能回答，模型只能说自己看不到。
        流程走到一半就断了。

    参数：
        keyword: 订单号（ORD001）或工单号（UUID）。
                 不传则返回最近的工单。

    为什么用单个 keyword 而不是「订单号 + 工单号」两个参数：
        模型分不清用户给的是哪种编号，更分不清该填哪个参数。
        让它原样把用户说的编号传进来，由代码判断格式，出错率更低。
    """

    from app.services.after_sales_service import get_after_sales_ticket
    from app.services.ticket_store import (
        list_recent_tickets,
        list_tickets_by_order,
    )

    raw = (keyword or "").strip()

    tickets: list = []

    if not raw:

        # 没给编号：查最近的。
        # 「我的退款申请怎么样了」这种问法在真实客服里很常见。
        tickets = list_recent_tickets(user_id, limit=5)

    elif raw.upper().startswith("ORD"):

        order_id = raw.upper()

        # 先校验订单归属。
        # 不加这一步，用户拿别人的订单号就能问出
        # 「这个单有没有售后」——这本身也是隐私。
        if get_order_by_id(order_id, user_id) is None:
            record_execution_event(
                event="ticket_lookup",
                status="denied",
                details={"reason": "order_not_accessible"},
            )
            return {
                "error": ORDER_NOT_ACCESSIBLE_MESSAGE,
                "status": "not_accessible",
            }

        tickets = list_tickets_by_order(order_id, user_id, limit=5)

    else:

        ticket = get_after_sales_ticket(raw, user_id)

        if ticket is not None:
            tickets = [ticket]

    result: dict = {
        "count": len(tickets),
        "tickets": [_ticket_to_dict(item) for item in tickets],
        "pending_draft": None,
    }

    # 把「生成了草稿但还没提交」也一并报给模型。
    #
    # 为什么需要：
    #     用户说「我申请过了」，其实只生成了草稿没确认。
    #     只报工单的话，模型会说「没有查到你的申请」，
    #     用户以为丢了，转头又申请一次。
    pending = find_latest_pending_draft(
        conversation_id=conversation_id,
        user_id=user_id,
    )

    if pending is not None:
        result["pending_draft"] = {
            "draft_id": pending.draft_id,
            "order_id": pending.order_id,
            "request_type": pending.request_type,
            "request_type_label": REQUEST_TYPE_LABELS.get(
                AfterSalesType(pending.request_type), "售后"
            ),
            "reason": pending.reason,
            "expires_in_seconds": pending.expires_in_seconds(),
        }

    if result["count"] == 0 and pending is None:
        result["hint"] = (
            "没有查到已提交的工单，也没有待确认的草稿。"
            "如果用户想申请，可以按流程先生成草稿；"
            "不要凭空说「已为您提交」，也不要编造工单号。"
        )
    elif result["count"] == 0 and pending is not None:
        result["hint"] = (
            "用户有一张还没提交的申请草稿（见 pending_draft）。"
            "告诉用户「申请还没提交」，把内容复述一遍，"
            "问是否确认提交。不要说他「已经申请过了」。"
        )
    else:
        result["hint"] = (
            "把工单号、申请类型、当前状态如实告诉用户。"
            "状态为待人工审核时，说明需要等待人工处理，"
            "不要承诺多久通过，也不要承诺一定能通过。"
        )

    record_execution_event(
        event="ticket_lookup",
        status="success",
        details={
            "count": len(tickets),
            "by": "recent" if not raw else "keyword",
            "has_pending_draft": pending is not None,
        },
    )

    return result


# ============================================================
# 工具 4：撤销售后申请
# ============================================================


def _recently_cancelled_ticket(user_id: str):
    """最近 10 分钟内被取消的工单。

    用于「用户重复说撤销」的场景：
        没有它，第二次会说「没有可以撤销的申请」，
        用户以为第一次没生效，可能反复操作。

    限定时间窗是为了避免翻出一张上周取消的单说
    「你刚撤销过了」。
    """

    from datetime import datetime, timedelta

    from app.services.ticket_store import list_recent_tickets

    cutoff = datetime.now() - timedelta(seconds=600)

    for ticket in list_recent_tickets(user_id, limit=3):
        if (
            ticket.status.value == "cancelled"
            and ticket.updated_at is not None
            and ticket.updated_at >= cutoff
        ):
            return ticket

    return None


def cancel_after_sales_ticket(
    user_id: str,
    conversation_id: str,
    ticket_id: str | None = None,
    order_id: str | None = None,
) -> dict:
    """撤销用户已经提交的售后申请。

    为什么这是一个「有副作用」的工具：
        它会把工单从 pending 改成 cancelled，
        之后审核员不会再处理这张单。
        如果用户其实是想「催一下」而不是「撤销」，
        误操作等于把自己的申请弄丢了。

    ticket_id / order_id 都不传时，自动定位该用户最近一张可撤销的工单。
        理由和 confirm 一样：模型看不到上一轮工具返回的 UUID，
        硬要它传只会编一个不存在的编号。
    """

    from app.services.after_sales_service import (
        cancel_after_sales_ticket as cancel_ticket_service,
    )
    from app.services.after_sales_service import get_after_sales_ticket
    from app.services.ticket_store import (
        InvalidStatusTransition,
        list_recent_tickets,
        list_tickets_by_order,
    )

    target = None
    resolved_by = ""

    # ---- 第一步：定位要撤销的工单（只读，不产生副作用）----
    if ticket_id:

        raw = ticket_id.strip()
        found = get_after_sales_ticket(raw, user_id)

        if found is None:
            record_execution_event(
                event="ticket_cancel",
                status="denied",
                details={"reason": "ticket_not_found"},
            )
            return {
                "error": "没有找到这张工单",
                "status": "ticket_not_found",
                "hint": "请先调用 get_after_sales_tickets 确认工单号，不要自己编造。",
            }

        target = found
        resolved_by = "ticket_id"

    elif order_id:

        normalized = order_id.strip().upper()

        if get_order_by_id(normalized, user_id) is None:
            return {
                "error": ORDER_NOT_ACCESSIBLE_MESSAGE,
                "status": "not_accessible",
            }

        candidates = [
            item
            for item in list_tickets_by_order(normalized, user_id)
            if item.status.value == "pending"
        ]

        if not candidates:
            return {
                "error": "这个订单没有可以撤销的售后申请",
                "status": "no_cancellable_ticket",
                "hint": (
                    "可能已经撤销、已经审核完成，或者还没提交过。"
                    "可以调用 get_after_sales_tickets 查一下当前状态。"
                ),
            }

        target = candidates[0]
        resolved_by = "order_id"

    else:

        candidates = [
            item
            for item in list_recent_tickets(user_id, limit=5)
            if item.status.value == "pending"
        ]

        if not candidates:

            # 用户可能是重复操作（网络重试、不确定时说两遍）。
            # 直接回「没有可撤销的申请」会让他以为操作没生效，
            # 所以先看有没有刚撤销过的单，明确告诉他结果。
            recent_cancelled = _recently_cancelled_ticket(user_id)

            if recent_cancelled is not None:
                return {
                    "status": "already_cancelled",
                    "ticket": _ticket_to_dict(recent_cancelled),
                    "hint": (
                        "这张申请刚刚已经撤销过了，不会进入审核。"
                        "直接把这个结果告诉用户，不要说成又撤销了一次。"
                    ),
                }

            return {
                "error": "没有可以撤销的售后申请",
                "status": "no_cancellable_ticket",
                "hint": (
                    "只有「待人工审核」的申请能撤销。"
                    "如果用户还没提交过，或者已经审核完成，就不需要撤销。"
                ),
            }

        target = candidates[0]
        resolved_by = "latest"

    # ---- 第二步：确认闸门（与建单同一套规则）----
    #
    # 这里要放宽两个词：
    #   「撤销」在撤销语境下是动作本身，不是否定表态，
    #   所以既要让它不触发 DENY，又要让它算作确认。
    #   「取消」不放宽——它太通用（取消订单、取消订阅），
    #   放进来会让「取消订单」这类无关请求直接通过闸门。
    last_record = _get_last_user_message(conversation_id)
    confirmed, confirm_reason = is_explicit_confirmation(
        last_record["content"],
        ignore_deny=("撤销",),
        extra_confirm=("撤销",),
    )

    if not confirmed:

        record_execution_event(
            event="ticket_cancel",
            status="blocked",
            details={"reason": confirm_reason},
        )

        label = REQUEST_TYPE_LABELS.get(target.request_type, "售后")

        return {
            "error": "用户还没有明确确认要撤销",
            "status": "confirmation_required",
            "ticket": _ticket_to_dict(target),
            "hint": (
                f"告诉用户：「你要撤销的是订单 {target.order_id} 的{label}申请"
                f"（理由：{target.reason}），撤销后不会再进入审核。」"
                "然后等用户明确回复「确认」「撤销吧」之后再调用一次。"
                "用户如果只是催进度或者问状态，就不要撤销。"
            ),
        }

    # ---- 第三步：执行 ----
    try:
        updated, changed = cancel_ticket_service(
            target.ticket_id,
            user_id,
        )

    except InvalidStatusTransition as exc:

        record_execution_event(
            event="ticket_cancel",
            status="rejected",
            details={"reason": "invalid_transition"},
        )

        return {
            "error": str(exc),
            "status": "not_cancellable",
            "hint": "如实告诉用户当前状态，不要说他撤销成功了。",
        }

    if updated is None:
        return {
            "error": "没有找到这张工单",
            "status": "ticket_not_found",
        }

    if not changed:
        # 已经是取消状态：重复操作，不算失败
        return {
            "status": "already_cancelled",
            "ticket": _ticket_to_dict(updated),
            "hint": "这张申请之前已经撤销过了，直接告诉用户即可，不要说成又一次撤销。",
        }

    clear_tool_cache(conversation_id)

    record_execution_event(
        event="ticket_cancel",
        status="success",
        details={"resolved_by": resolved_by},
    )

    label = REQUEST_TYPE_LABELS.get(updated.request_type, "售后")

    return {
        "status": "cancelled",
        "ticket": _ticket_to_dict(updated),
        "resolved_by": resolved_by,
        "hint": (
            f"已撤销订单 {updated.order_id} 的{label}申请，"
            f"工单 {updated.ticket_id} 不再进入审核。"
            "告诉用户现在可以重新申请（如果他还需要）。"
            "不要说成「已退款」——撤销申请不等于退款。"
        ),
    }
