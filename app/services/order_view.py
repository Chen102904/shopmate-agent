"""订单的「对外视图」。

解决的问题：

    `Order` 只描述交易本身（下单、发货、签收），
    但用户看到的「我的订单」还要叠上售后进度：

        ORD001  待发货
        ORD001  退款中（待审核）   ← 提交退款申请之后
        ORD001  已退款             ← 审核通过之后

    这三个状态不在 Order.status 里，而是「订单状态 + 售后工单状态」
    合起来算出来的。

为什么不在创建工单时直接改 ORDERS 里的 status：

    1. data/mock_data.py 的 ORDERS 是模块级全局对象，
       改它会被同进程的测试互相污染（A 用例退了款，B 用例看到的是退款后的状态）。
    2. 退款资格判定、物流查询都读 order.status，
       直接改会让「这个订单原本是待发货」这个事实丢失，
       驳回之后没法还原。
    3. 订单状态和售后状态本来就是两张表的事，
       视图层算一次比在两条写路径上各自维护一致要可靠。

    所以：Order.status 保持「交易状态」原义不动，
    对外展示用这里算出来的 display_status。
"""

from app.models.after_sales import AfterSalesTicket, AfterSalesType, TicketStatus
from app.models.order import Order, OrderStatus
from app.services.refund_service import check_refund_eligibility
from app.services.ticket_store import list_tickets_by_order


# 订单交易状态的中文名。
# 面向用户不用「已付款」，用户关心的是「发没发货」。
ORDER_STATUS_LABELS = {
    OrderStatus.PENDING: "待付款",
    OrderStatus.PAID: "待发货",
    OrderStatus.SHIPPED: "已发货",
    OrderStatus.DELIVERED: "已签收",
    OrderStatus.CANCELLED: "已取消",
}


TICKET_STATUS_LABELS = {
    TicketStatus.PENDING: "待审核",
    TicketStatus.PROCESSING: "处理中",
    TicketStatus.APPROVED: "已通过",
    TicketStatus.COMPLETED: "已完成",
    TicketStatus.REJECTED: "已驳回",
    TicketStatus.CANCELLED: "已撤销",
}


# 售后卡在「还没结论」的阶段：订单对外显示为退款中。
# 用户视角里「我提交了、正在等」就是「退款中」，
# 不需要区分「排队中」和「处理中」。
_STILL_IN_PROGRESS = {TicketStatus.PENDING, TicketStatus.PROCESSING}

# 售后已有结论且钱会退回：订单对外显示为已退款。
_DONE_AND_REFUNDED = {TicketStatus.APPROVED, TicketStatus.COMPLETED}


def _latest_ticket(order_id: str, user_id: str) -> AfterSalesTicket | None:
    """取这个订单下最新的一张工单。

    只看最新的一张：
        用户撤销后重新申请，旧的那张是 cancelled，
        拿它算状态会让订单一直显示「已撤销」。
    """

    tickets = list_tickets_by_order(order_id, user_id, limit=1)

    return tickets[0] if tickets else None


def _derive_display_status(
    order: Order,
    ticket: AfterSalesTicket | None,
) -> str:
    """把「交易状态 + 最新工单状态」合成一个对用户有意义的状态。"""

    # 订单本身已取消：售后单不可能还有效，优先按订单说。
    if order.status == OrderStatus.CANCELLED:
        return OrderStatus.CANCELLED.value

    if ticket is None:
        return order.status.value

    # 换货 / 退货不走「退款中」这个说法，
    # 否则用户会以为钱在退，其实是在等换货。
    is_refund = ticket.request_type == AfterSalesType.REFUND

    if ticket.status in _STILL_IN_PROGRESS:
        return "refunding" if is_refund else "after_sales"

    if ticket.status in _DONE_AND_REFUNDED:
        return "refunded" if is_refund else "after_sales_done"

    # rejected / cancelled：售后没成，订单回到原来的交易状态。
    return order.status.value


# 聚合状态的中文名。键既包含交易状态，也包含售后叠加出来的状态。
DISPLAY_STATUS_LABELS = {
    **ORDER_STATUS_LABELS,
    "refunding": "退款中",
    "refunded": "已退款",
    "after_sales": "售后中",
    "after_sales_done": "售后完成",
}


def build_order_view(order: Order, user_id: str) -> dict:
    """拼出前端要的一张订单卡。

    字段说明：
        status / status_label            原始交易状态，业务逻辑用
        display_status / display_status_label
                                         叠加售后后的展示状态，界面用
        ticket                           最新工单摘要，没有就是 None
        can_request_refund               现在还能不能申请退款
        refund_hint                      为什么能 / 为什么不能，直接给用户看
    """

    ticket = _latest_ticket(order.order_id, user_id)
    display_status = _derive_display_status(order, ticket)

    eligibility = check_refund_eligibility(order.order_id, user_id)

    # eligible 是三态：True 能退、False 不能退、None 查不到。
    # None 在这里基本不会出现（订单是刚查出来的），
    # 但也不能当成能退处理。
    can_refund = eligibility.get("eligible") is True

    ticket_brief = None

    if ticket is not None:
        ticket_brief = {
            "ticket_id": ticket.ticket_id,
            "short_id": ticket.ticket_id[:8],
            "request_type": ticket.request_type.value,
            "status": ticket.status.value,
            "status_label": TICKET_STATUS_LABELS.get(
                ticket.status,
                ticket.status.value,
            ),
            "created_at": ticket.created_at.isoformat(),
            "review_note": ticket.review_note,
        }

    return {
        "order_id": order.order_id,
        "user_id": order.user_id,
        "status": order.status.value,
        "status_label": ORDER_STATUS_LABELS.get(
            order.status,
            order.status.value,
        ),
        "display_status": display_status,
        "display_status_label": DISPLAY_STATUS_LABELS.get(
            display_status,
            display_status,
        ),
        "total_amount": format(order.total_amount, ".2f"),
        "order_time": order.order_time.isoformat(),
        "ticket": ticket_brief,
        "can_request_refund": can_refund,
        "refund_hint": eligibility.get("reason") or "",
    }
