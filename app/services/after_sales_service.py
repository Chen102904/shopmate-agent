from uuid import uuid4

from app.models.after_sales import AfterSalesTicket, AfterSalesType, TicketStatus
from app.services.order_service import get_order_by_id
from app.services.refund_service import check_refund_eligibility
from app.services.ticket_store import (
    IdempotencyConflict,
    RequestInProgress,
    build_fingerprint,
    complete_idempotency_key,
    find_recent_active_ticket,
    get_ticket_by_id,
    release_idempotency_key,
    reserve_idempotency_key,
    save_ticket,
)

__all__ = [
    "IdempotencyConflict",
    "RequestInProgress",
    "cancel_after_sales_ticket",
    "create_after_sales_ticket",
    "get_after_sales_ticket",
    "review_after_sales_ticket",
]


def create_after_sales_ticket(
    order_id: str,
    user_id: str,
    request_type: AfterSalesType,
    reason: str,
    idempotency_key: str | None = None,
    with_meta: bool = False,
) -> AfterSalesTicket | tuple[AfterSalesTicket, bool]:
    """创建售后工单。

    重要：
        user_id 必须由服务端身份层提供（请求头 / 令牌），
        不能使用客户端请求体里自报的 user_id。
        否则用户只要把 user_id 填成别人，就能替别人的订单申请售后。

    幂等：
        传了 idempotency_key → 严格幂等，同一 key 只会成功建一张工单。
        没传 → 按 user + order + 类型做业务去重，已有活跃工单直接返回。

        工单背后是人工审核和退款流程，重复建单等于重复退款，
        所以这一层不能省。

    with_meta：
        为 True 时返回 (工单, 是否重复请求)，
        API 层用它决定返回 201（新建）还是 200（幂等重放）。
        默认 False，保持旧调用方式不变。
    """

    fingerprint = build_fingerprint(
        user_id=user_id,
        order_id=order_id,
        request_type=request_type,
        reason=reason,
    )

    reserved_key = False

    if idempotency_key:

        # 重复请求会在这里直接返回上次的工单，不再往下执行业务。
        existing = reserve_idempotency_key(
            idempotency_key=idempotency_key,
            user_id=user_id,
            fingerprint=fingerprint,
        )

        if existing is not None:
            if with_meta:
                return existing, True
            return existing

        reserved_key = True

    else:

        # 没传 key 的兜底：已有同类活跃工单就直接返回
        existing = find_recent_active_ticket(
            order_id=order_id,
            user_id=user_id,
            request_type=request_type,
        )

        if existing is not None:
            if with_meta:
                return existing, True
            return existing

    try:
        ticket = _create_and_validate(
            order_id=order_id,
            user_id=user_id,
            request_type=request_type,
            reason=reason,
            idempotency_key=idempotency_key if reserved_key else None,
        )

    except Exception:
        # 业务失败要释放占位，
        # 否则这个 key 会永久卡在 processing，客户端再也用不了。
        if reserved_key:
            release_idempotency_key(idempotency_key)

        raise

    if reserved_key:
        complete_idempotency_key(
            idempotency_key=idempotency_key,
            ticket=ticket,
        )

    if with_meta:
        return ticket, False

    return ticket


def _create_and_validate(
    order_id: str,
    user_id: str,
    request_type: AfterSalesType,
    reason: str,
    idempotency_key: str | None,
) -> AfterSalesTicket:
    """业务校验 + 落库。"""

    # 1. 查询订单，同时校验归属。
    #    get_order_by_id 内部会把「不存在」和「不属于你」都返回 None，
    #    这里统一按同一个错误处理，不泄露订单是否存在。
    order = get_order_by_id(order_id, user_id)

    if order is None:
        raise ValueError("订单不存在或无权访问")

    # 2. 如果是退款申请，先检查退款资格
    if request_type == AfterSalesType.REFUND:

        eligibility = check_refund_eligibility(order_id, user_id)

        # eligible 为 None 表示「无法判断」（订单查不到这类情况），
        # 不能当成「不符合条件」放过去，也不能当成「符合」放过去。
        if eligibility["eligible"] is not True:

            if eligibility["eligible"] is None:
                raise ValueError(
                    "暂时无法确认该订单的退款资格，请稍后重试或联系人工客服"
                )

            raise ValueError(eligibility["reason"])

    # 3. 创建工单
    ticket = AfterSalesTicket(
        ticket_id=str(uuid4()),
        order_id=order_id,
        user_id=user_id,
        request_type=request_type,
        reason=reason,
    )

    # 4. 保存工单。
    #    并发下唯一索引冲突会返回已有工单，见 ticket_store.save_ticket。
    return save_ticket(ticket, idempotency_key=idempotency_key)


def get_after_sales_ticket(
    ticket_id: str,
    user_id: str,
) -> AfterSalesTicket | None:
    """根据工单编号查询售后工单，并校验归属。

    工单号是 UUID，虽然难猜，
    但一旦通过日志、截图、分享链接泄露，
    没有归属校验就等于把别人的售后记录直接开放。
    """

    return get_ticket_by_id(ticket_id, user_id)


# ============================================================
# 状态变更：用户取消 / 管理员审核
# ============================================================


def cancel_after_sales_ticket(
    ticket_id: str,
    user_id: str,
) -> tuple[AfterSalesTicket | None, bool]:
    """用户撤销自己提交的售后申请。

    返回 (工单, 本次是否真的取消了)：
        工单为 None     → 不存在或不属于当前用户（统一处理，不泄露存在性）
        changed=False   → 已经是取消状态，属于重复操作，不算失败

    为什么不让用户「删除」工单而是「取消」：
        工单是审计凭证。用户撤销后审核员仍应该看得到
        「这张单曾经提交过、后来被用户撤回了」，
        直接删掉会让操作记录消失。
    """

    from app.services.ticket_store import update_ticket_status

    return update_ticket_status(
        ticket_id=ticket_id,
        new_status=TicketStatus.CANCELLED,
        user_id=user_id,
    )


# 管理员审核只允许这两个结论。
# 不允许直接改成 processing / completed：
# 那些是系统内部流转，不该由审核入口决定。
REVIEW_DECISIONS = {
    TicketStatus.APPROVED,
    TicketStatus.REJECTED,
}


def review_after_sales_ticket(
    ticket_id: str,
    decision: str | TicketStatus,
    review_note: str | None = None,
) -> tuple[AfterSalesTicket | None, bool]:
    """人工审核：通过或驳回。

    参数：
        decision: approved / rejected
        review_note: 审核意见。驳回时建议必填——
                     用户被拒却不知道原因，只能再来问客服。

    注意这里**不校验 user_id**：
        审核是管理员行为，要处理所有人的工单。
        所以调用方（API 层）必须先验证管理员身份，
        绝不能让普通用户走到这个函数——
        否则就是「自己审自己的退款」。
    """

    from app.services.ticket_store import update_ticket_status

    if isinstance(decision, str):
        try:
            status = TicketStatus(decision.strip().lower())
        except ValueError as exc:
            raise ValueError(
                "审核结论只能是 approved 或 rejected"
            ) from exc
    else:
        status = decision

    if status not in REVIEW_DECISIONS:
        raise ValueError("审核结论只能是 approved 或 rejected")

    note = (review_note or "").strip() or None

    if status == TicketStatus.REJECTED and note is None:
        # 不强制阻断（有些场景确实来不及写理由），
        # 但补齐一个默认值，避免用户看到「已驳回」却没有任何说明。
        note = "未通过审核，如需了解原因请联系人工客服"

    return update_ticket_status(
        ticket_id=ticket_id,
        new_status=status,
        user_id=None,           # 管理员视角，不限归属
        review_note=note,
    )
